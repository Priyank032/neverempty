"""The judge wired into the facts and forbidden-claims scorers.

M6 left judge-mode facts unmeasured and named. This is the other half: with a
judge configured they are decided, and the rules that mattered in M6 still hold.

The safety-critical direction is the forbidden-claims scorer. A judge that errors
must leave the claim unmeasured, never clean: a suite whose absence-claim check
silently passed because the judge was down would publish a safety number nobody
checked.
"""

from __future__ import annotations

import json
from typing import Any

from toolproof import scorers
from toolproof.dataset.case import Case
from toolproof.judge.judge import ClaimJudge
from toolproof.judge.model import ScriptedJudge

from ._scoring import trace


def make_judge(*responses: str) -> ClaimJudge:
    return ClaimJudge(
        model=ScriptedJudge(list(responses)),
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        agent_family="openai",
    )


def verdict(label: str) -> str:
    return json.dumps({"label": label, "rationale": "r"})


def facts_case(*facts: dict[str, Any]) -> Case:
    return Case.model_validate(
        {
            "id": "nr-fact-0001",
            "suite": "nextrole.routing",
            "split": "dev",
            "input": {"messages": [{"role": "user", "content": "find me a job"}]},
            "expect": {"facts": list(facts)},
        }
    )


def claims_case(*claims: dict[str, Any]) -> Case:
    return Case.model_validate(
        {
            "id": "nr-claim-0001",
            "suite": "nextrole.failure",
            "split": "dev",
            "input": {"messages": [{"role": "user", "content": "python jobs?"}]},
            "expect": {"forbidden_claims": list(claims)},
        }
    )


class TestJudgedFacts:
    async def test_a_supported_judge_fact_counts_as_found(self) -> None:
        scorer = scorers.facts(judge=make_judge(verdict("supported")))
        result = await scorer.score_async(
            facts_case({"id": "f1", "statement": "the tone is professional", "match": "judge"}),
            trace(answer="Dear hiring manager, I am writing regarding the role."),
        )
        assert result is not None
        assert result.passed is True
        assert result.detail["found"] == ["f1"]

    async def test_a_contradicted_judge_fact_counts_as_missing(self) -> None:
        scorer = scorers.facts(judge=make_judge(verdict("contradicted")))
        result = await scorer.score_async(
            facts_case({"id": "f1", "statement": "three roles were found", "match": "judge"}),
            trace(answer="I found one role."),
        )
        assert result is not None
        assert result.passed is False
        assert result.detail["missing"] == ["f1"]

    async def test_not_in_evidence_counts_as_missing_not_as_unmeasured(self) -> None:
        """The judge decided: the answer does not state the fact. That is a
        recall miss, which is different from the judge being unable to run."""
        scorer = scorers.facts(judge=make_judge(verdict("not_in_evidence")))
        result = await scorer.score_async(
            facts_case({"id": "f1", "statement": "the salary is disclosed", "match": "judge"}),
            trace(answer="I found one role."),
        )
        assert result is not None
        assert result.detail["missing"] == ["f1"]

    async def test_deterministic_and_judge_facts_mix_in_one_score(self) -> None:
        scorer = scorers.facts(judge=make_judge(verdict("supported")))
        result = await scorer.score_async(
            facts_case(
                {"id": "f1", "statement": "Pune", "match": "contains"},
                {"id": "f2", "statement": "the tone is professional", "match": "judge"},
            ),
            trace(answer="Roles in Pune."),
        )
        assert result is not None
        assert result.value == 1.0
        assert "skipped" not in result.detail

    async def test_a_judge_error_leaves_the_fact_unmeasured(self) -> None:
        """Not a miss. A provider outage is not evidence that the agent omitted
        the fact, and counting it as one would lower recall for a harness fault."""
        scorer = scorers.facts(judge=make_judge("not json", "not json", "not json"))
        result = await scorer.score_async(
            facts_case(
                {"id": "f1", "statement": "Pune", "match": "contains"},
                {"id": "f2", "statement": "the tone is professional", "match": "judge"},
            ),
            trace(answer="Roles in Pune."),
        )
        assert result is not None
        assert result.value == 1.0
        assert result.detail["skipped"] == ["f2"]
        assert result.detail["skipped_reason"] == "judge_error"

    async def test_a_case_of_only_judge_errors_is_not_applicable(self) -> None:
        scorer = scorers.facts(judge=make_judge("bad", "bad", "bad"))
        result = await scorer.score_async(
            facts_case({"id": "f1", "statement": "the tone is professional", "match": "judge"}),
            trace(answer="An answer."),
        )
        assert result is None

    async def test_the_answer_is_the_evidence_for_a_fact(self) -> None:
        """Fact recall asks whether the *answer* states the fact, so the answer
        is what the judge sees as evidence."""
        model = ScriptedJudge([verdict("supported")])
        scorer = scorers.facts(
            judge=ClaimJudge(
                model=model,
                model_id="anthropic.claude-test",
                agent_family="openai",
            )
        )
        await scorer.score_async(
            facts_case({"id": "f1", "statement": "a claim", "match": "judge"}),
            trace(answer="unique-answer-text"),
        )
        assert "unique-answer-text" in model.calls[0].user

    async def test_the_sync_entry_point_still_skips_judge_facts(self) -> None:
        """A synchronous caller cannot await the judge, so the fact is named as
        skipped rather than silently scored by a different rule."""
        scorer = scorers.facts(judge=make_judge(verdict("supported")))
        result = scorer.score(
            facts_case(
                {"id": "f1", "statement": "Pune", "match": "contains"},
                {"id": "f2", "statement": "a judged claim", "match": "judge"},
            ),
            trace(answer="Roles in Pune."),
        )
        assert result is not None
        assert result.detail["skipped"] == ["f2"]


class TestJudgedForbiddenClaims:
    async def test_a_supported_forbidden_claim_is_a_hit(self) -> None:
        """ "Supported" here means the answer *does* make the forbidden claim."""
        scorer = scorers.forbidden_claims(judge=make_judge(verdict("supported")))
        result = await scorer.score_async(
            claims_case({"id": "absence", "statement": "no data exists", "match": "judge"}),
            trace(answer="I could not find anything at all for you."),
        )
        assert result is not None
        assert result.passed is False
        assert result.detail["hits"] == ["absence"]

    async def test_a_contradicted_forbidden_claim_is_clean(self) -> None:
        scorer = scorers.forbidden_claims(judge=make_judge(verdict("contradicted")))
        result = await scorer.score_async(
            claims_case({"id": "absence", "statement": "no data exists", "match": "judge"}),
            trace(answer="The search failed, so I could not check."),
        )
        assert result is not None
        assert result.passed is True

    async def test_not_in_evidence_is_clean(self) -> None:
        scorer = scorers.forbidden_claims(judge=make_judge(verdict("not_in_evidence")))
        result = await scorer.score_async(
            claims_case({"id": "absence", "statement": "no data exists", "match": "judge"}),
            trace(answer="Here is some advice."),
        )
        assert result is not None
        assert result.passed is True

    async def test_a_judge_error_on_every_claim_is_not_applicable_never_clean(
        self,
    ) -> None:
        """The safety-critical rule. Reporting a pass because the judge was down
        would publish a safety guarantee nobody checked."""
        scorer = scorers.forbidden_claims(judge=make_judge("bad", "bad", "bad"))
        result = await scorer.score_async(
            claims_case({"id": "absence", "statement": "no data exists", "match": "judge"}),
            trace(answer="There is no data."),
        )
        assert result is None

    async def test_a_judge_error_beside_a_deterministic_hit_still_fails(self) -> None:
        scorer = scorers.forbidden_claims(judge=make_judge("bad", "bad", "bad"))
        result = await scorer.score_async(
            claims_case(
                {"id": "absence", "statement": "no matching jobs", "match": "contains"},
                {"id": "vague", "statement": "implies nothing exists", "match": "judge"},
            ),
            trace(answer="There are no matching jobs."),
        )
        assert result is not None
        assert result.passed is False
        assert result.detail["skipped"] == ["vague"]


class TestDegradedRuns:
    """A judge error rate above 2% makes the run ``degraded``."""

    def test_the_threshold_is_two_percent(self) -> None:
        from toolproof.judge.judge import DEGRADED_ERROR_RATE

        assert DEGRADED_ERROR_RATE == 0.02

    async def test_a_high_error_rate_marks_the_judge_info_degraded(self) -> None:
        from toolproof.judge.judge import Claim

        judge = make_judge("bad", "bad", "bad")
        await judge.verify(claims=[Claim(id="c1", text="x")], evidence={"rows": []})
        assert judge.degraded is True

    async def test_a_clean_judge_is_not_degraded(self) -> None:
        from toolproof.judge.judge import Claim

        judge = make_judge(verdict("supported"))
        await judge.verify(claims=[Claim(id="c1", text="x")], evidence={"rows": []})
        assert judge.degraded is False

    def test_a_judge_with_no_calls_is_not_degraded(self) -> None:
        """Nothing has failed yet. An unmeasured error rate must not read as a
        failing one."""
        assert make_judge().degraded is False

    async def test_the_judge_info_block_carries_the_rate(self) -> None:
        from toolproof.judge.judge import Claim

        judge = make_judge(verdict("supported"))
        await judge.verify(claims=[Claim(id="c1", text="x")], evidence={"rows": []})
        info = judge.info()
        assert info.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert info.prompt_version == "verify.v1"
        assert info.error_rate == 0.0


class TestGateExcludesJudgeMetrics:
    def test_a_degraded_report_gates_on_deterministic_metrics_only(self) -> None:
        """The doc excludes judge-derived metrics from the gate on a degraded
        run. The deterministic ones still gate, because a flaky judge is not a
        reason to stop checking routing."""
        from toolproof import CaseOutcome, Metric, Report, Score
        from toolproof.core.trace import Env
        from toolproof.report.gate import GateConfig, gate

        def build(route_value: float, status: str) -> Report:
            return Report.model_validate(
                {
                    "suite": "s.t",
                    "split": "test",
                    "created_at": "2026-09-23T10:00:00.000Z",
                    "complete": True,
                    "status": status,
                    "env": Env(
                        toolproof_version="0.0.1",
                        pricing_version="v1",
                        python_version="3.12.10",
                    ),
                    "judge": {"model": "anthropic.claude-test", "error_rate": 0.25},
                    "metrics": [
                        Metric(
                            name="route",
                            n=30,
                            value=route_value,
                            method="wilson",
                            applicable=30,
                        ),
                        Metric(name="facts", n=30, value=0.2, method="bootstrap", applicable=30),
                    ],
                    "outcomes": [
                        CaseOutcome(
                            case_id=f"c-{i:04d}",
                            repeat=0,
                            scored=True,
                            scores={"route": Score(passed=True, value=1.0)},
                        )
                        for i in range(30)
                    ],
                }
            )

        degraded = build(0.9, "degraded")
        # A floor on the judge-derived metric is not evaluated on a degraded run.
        result = gate(degraded, degraded, GateConfig(floors={"facts": 0.75}))
        assert result.exit_code == 0
        assert any("degraded" in note.lower() for note in result.warnings)

        # A floor on a deterministic metric still bites.
        strict = gate(degraded, degraded, GateConfig(floors={"route": 0.95}))
        assert strict.exit_code == 1

    def test_an_ok_report_evaluates_judge_floors_normally(self) -> None:
        from toolproof import CaseOutcome, Metric, Report, Score
        from toolproof.core.trace import Env
        from toolproof.report.gate import GateConfig, gate

        report = Report.model_validate(
            {
                "suite": "s.t",
                "split": "test",
                "created_at": "2026-09-23T10:00:00.000Z",
                "complete": True,
                "status": "ok",
                "env": Env(
                    toolproof_version="0.0.1",
                    pricing_version="v1",
                    python_version="3.12.10",
                ),
                "metrics": [
                    Metric(name="facts", n=30, value=0.2, method="bootstrap", applicable=30)
                ],
                "outcomes": [
                    CaseOutcome(
                        case_id=f"c-{i:04d}",
                        repeat=0,
                        scored=True,
                        scores={"facts": Score(passed=True, value=1.0)},
                    )
                    for i in range(30)
                ],
            }
        )
        assert gate(report, report, GateConfig(floors={"facts": 0.75})).exit_code == 1


class TestScorerProtocol:
    def test_a_judge_backed_scorer_still_declares_its_requirements(self) -> None:
        scorer = scorers.facts(judge=make_judge())
        assert scorer.requires == frozenset({"expect.facts"})
        assert scorer.name == "facts"

    def test_a_scorer_without_a_judge_still_works(self) -> None:
        """M6 behaviour is unchanged: no judge means judge facts are named and
        skipped, never guessed at."""
        result = scorers.facts().score(
            facts_case(
                {"id": "f1", "statement": "Pune", "match": "contains"},
                {"id": "f2", "statement": "judged", "match": "judge"},
            ),
            trace(answer="Roles in Pune."),
        )
        assert result is not None
        assert result.detail["skipped_reason"] == "no_judge_configured"

    async def test_the_runner_awaits_an_async_scorer(self) -> None:
        """The runner already supports an awaitable scorer, so a judge-backed
        scorer needs no special case there."""
        import inspect

        scorer = scorers.facts(judge=make_judge(verdict("supported")))
        assert inspect.iscoroutinefunction(scorer.score_async)


class TestNoNetwork:
    def test_the_judge_module_imports_no_provider_sdk(self) -> None:
        """Core stays at one dependency. A judge that dragged boto3 into every
        install would make the trace layer unusable for a tracing-only user."""
        import pathlib

        source = pathlib.Path("src/toolproof/judge").rglob("*.py")
        for path in source:
            text = path.read_text(encoding="utf-8")
            for banned in ("import boto3", "import openai", "import anthropic"):
                assert banned not in text, f"{path} imports a provider SDK"
