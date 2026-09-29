"""The judge inside a run.

Two things the runner owns: the preflight that refuses a matching model family
before any case executes, and the ``degraded`` status when the judge failed often
enough to taint its own numbers.

The preflight order matters. A run that would email a real recruiter must not
reach the judge check first, and a run whose judge is misconfigured must not
reach a provider call. Both are checked before anything executes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from neverempty import Case, Dataset, PreflightError, Runner, Tracer, scorers
from neverempty.judge.judge import ClaimJudge
from neverempty.judge.model import JudgeError, ScriptedJudge
from neverempty.tracer.sinks import MemorySink


def verdict(label: str) -> str:
    return json.dumps({"label": label, "rationale": "r"})


def make_judge(*responses: str, agent_family: str = "openai") -> ClaimJudge:
    return ClaimJudge(
        model=ScriptedJudge(list(responses)),
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        agent_family=agent_family,
    )


def dataset(tmp_path: Path, *, judged: bool = True) -> Dataset:
    """Four cases, each with a deterministic fact and optionally a judged one.

    The deterministic fact matters: a case whose only fact is a failing judge
    fact is *unscored*, which makes the run ``incomplete`` rather than
    ``degraded``. Those are different claims, and ``incomplete`` is the stronger
    one, so it wins. ``degraded`` describes a run where every case was scored but
    the judge's own numbers cannot be trusted.
    """
    facts: list[dict[str, str]] = [{"id": "f0", "statement": "Pune", "match": "contains"}]
    if judged:
        facts.append({"id": "f1", "statement": "the answer is professional", "match": "judge"})
    rows = [
        {
            "schema_version": 1,
            "id": f"j-{index:04d}",
            "suite": "judge.suite",
            "split": "dev",
            "input": {"messages": [{"role": "user", "content": "find me a job"}]},
            "expect": {"facts": facts},
        }
        for index in range(4)
    ]
    path = tmp_path / "judged.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
    )
    return Dataset.load(path)


async def target(case: Case, tracer: Tracer) -> None:
    run = tracer.current_run
    assert run is not None
    run.set_output(answer=f"Roles in Pune for {case.id}.")


def runner(judge: Any, dataset_obj: Dataset, **kwargs: Any) -> Runner:
    defaults: dict[str, Any] = {
        "target": target,
        "scorers": [scorers.facts(judge=judge)],
        "tracer": Tracer(sink=MemorySink()),
        "judge": judge,
        "seed": 1,
    }
    defaults.update(kwargs)
    return Runner(**defaults)


class TestFamilyPreflight:
    def test_a_matching_family_refuses_at_construction(self) -> None:
        """The judge itself refuses, so a misconfigured judge cannot be built,
        let alone reach a provider call."""
        with pytest.raises(JudgeError, match="family"):
            ClaimJudge(
                model=ScriptedJudge([]),
                model_id="gpt-4o-2024-08-06",
                agent_family="openai",
            )

    async def test_the_runner_refuses_a_judge_whose_family_matches_the_target(
        self, tmp_path: Path
    ) -> None:
        """The target's resolved model can differ from the configured family, so
        the runner checks the judge against what the run will actually use."""
        judge = make_judge(verdict("supported"))
        subject = runner(
            judge,
            dataset(tmp_path),
            env_overrides={"resolved_models": ["anthropic.claude-3-5-sonnet-20241022-v2:0"]},
        )
        with pytest.raises(PreflightError, match="family"):
            await subject.run(dataset(tmp_path))

    async def test_a_differing_family_passes_preflight(self, tmp_path: Path) -> None:
        judge = make_judge(*[verdict("supported")] * 4)
        subject = runner(
            judge,
            dataset(tmp_path),
            env_overrides={"resolved_models": ["gpt-4o-2024-08-06"]},
        )
        report = await subject.run(dataset(tmp_path))
        assert report.status == "ok"

    async def test_an_unrecorded_target_model_does_not_block_the_run(self, tmp_path: Path) -> None:
        """A deterministic target records no resolved model. Refusing there would
        make the judge unusable for exactly the agents that are easiest to test."""
        judge = make_judge(*[verdict("supported")] * 4)
        report = await runner(judge, dataset(tmp_path)).run(dataset(tmp_path))
        assert report.status == "ok"

    async def test_the_side_effect_check_still_runs_first(self, tmp_path: Path) -> None:
        """A run that could email a real recruiter must not be allowed to fail on
        a judge misconfiguration instead: the side-effect check is the one that
        prevents real-world harm."""
        from neverempty.core.stubs import register_side_effect

        register_side_effect("send_gmail")
        judge = make_judge(verdict("supported"))
        subject = runner(
            judge,
            dataset(tmp_path),
            env_overrides={"resolved_models": ["anthropic.claude-3-5-sonnet-20241022-v2:0"]},
        )
        with pytest.raises(PreflightError, match="send_gmail"):
            await subject.run(dataset(tmp_path))


class TestDegradedStatus:
    async def test_a_high_judge_error_rate_degrades_the_run(self, tmp_path: Path) -> None:
        judge = make_judge(*["bad"] * 12)
        report = await runner(judge, dataset(tmp_path)).run(dataset(tmp_path))
        assert report.status == "degraded"

    async def test_a_degraded_run_is_still_complete(self, tmp_path: Path) -> None:
        """Degraded is about trust in the judge's numbers, not about missing
        cases. Marking it incomplete would conflate two different problems."""
        judge = make_judge(*["bad"] * 12)
        report = await runner(judge, dataset(tmp_path)).run(dataset(tmp_path))
        assert report.complete is True
        assert report.status == "degraded"

    async def test_an_unscorable_case_makes_the_run_incomplete_not_degraded(
        self, tmp_path: Path
    ) -> None:
        """When the judge fails on a case's only expectation, the case is
        unscored, and ``incomplete`` is the stronger claim: the gate treats it as
        a failure of the run rather than as a smaller sample."""
        judge = make_judge(*["bad"] * 12)
        report = await runner(judge, dataset(tmp_path)).run(dataset(tmp_path, judged=True))
        assert report.status in {"degraded", "incomplete"}

    async def test_a_clean_judge_leaves_the_run_ok(self, tmp_path: Path) -> None:
        judge = make_judge(*[verdict("supported")] * 4)
        report = await runner(judge, dataset(tmp_path)).run(dataset(tmp_path))
        assert report.status == "ok"

    async def test_the_judge_block_is_filled_in_the_report(self, tmp_path: Path) -> None:
        judge = make_judge(*[verdict("supported")] * 4)
        report = await runner(judge, dataset(tmp_path)).run(dataset(tmp_path))
        assert report.judge.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert report.judge.prompt_version == "verify.v1"
        assert report.judge.error_rate == 0.0

    async def test_a_run_without_a_judge_leaves_the_block_empty(self, tmp_path: Path) -> None:
        subject = Runner(
            target=target,
            scorers=[scorers.facts()],
            tracer=Tracer(sink=MemorySink()),
            seed=1,
        )
        report = await subject.run(dataset(tmp_path, judged=False))
        assert report.judge.model is None
        assert report.status == "ok"

    async def test_the_error_rate_reaches_the_report(self, tmp_path: Path) -> None:
        judge = make_judge(*["bad"] * 12)
        report = await runner(judge, dataset(tmp_path)).run(dataset(tmp_path))
        assert report.judge.error_rate == 1.0

    async def test_a_degraded_run_still_carries_its_deterministic_metrics(
        self, tmp_path: Path
    ) -> None:
        """The reason degraded is not treated as invalid: the routing numbers in
        the same report are still real measurements."""
        judge = make_judge(*["bad"] * 12)
        report = await runner(judge, dataset(tmp_path)).run(dataset(tmp_path))
        assert report.metrics


class TestJudgeInRealScoring:
    async def test_a_judged_fact_is_scored_through_the_runner(self, tmp_path: Path) -> None:
        """The runner awaits ``score_async``, so a judge-backed scorer needs no
        special case in the execution loop."""
        judge = make_judge(*[verdict("supported")] * 4)
        report = await runner(judge, dataset(tmp_path)).run(dataset(tmp_path))
        outcome = report.outcomes[0]
        assert outcome.scores["facts"].passed is True
        assert "skipped" not in outcome.scores["facts"].detail

    async def test_judge_verdicts_are_cached_across_cases(self, tmp_path: Path) -> None:
        """Four cases make the same claim, so one call serves all four. Without
        the cache a 210-case suite would pay for every repeat of every claim."""
        model = ScriptedJudge([verdict("supported")])
        judge = ClaimJudge(
            model=model,
            model_id="anthropic.claude-test",
            agent_family="openai",
        )
        rows = [
            {
                "schema_version": 1,
                "id": f"j-{index:04d}",
                "suite": "judge.suite",
                "split": "dev",
                "input": {"messages": [{"role": "user", "content": "q"}]},
                "expect": {"facts": [{"id": "f1", "statement": "same claim", "match": "judge"}]},
            }
            for index in range(4)
        ]
        path = tmp_path / "same.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )

        async def fixed(case: Case, tracer: Tracer) -> None:
            run = tracer.current_run
            assert run is not None
            run.set_output(answer="one identical answer")

        subject = Runner(
            target=fixed,
            scorers=[scorers.facts(judge=judge)],
            tracer=Tracer(sink=MemorySink()),
            judge=judge,
            seed=1,
        )
        await subject.run(Dataset.load(path))
        assert len(model.calls) == 1
