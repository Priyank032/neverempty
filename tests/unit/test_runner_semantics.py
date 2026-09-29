"""Acceptance row for ``runner``.

    case timeout, target exception, scorer exception each produce the
    documented status; budget abort writes a valid incomplete report; replay
    cache miss errors; resume merges without duplicating; identical output
    ordering across two runs

One test class per row of the runner-semantics table. Every ambiguous situation
has exactly one defined behaviour, because the alternative is a report that
silently means something different from run to run.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from neverempty import (
    Case,
    Dataset,
    MemorySink,
    Ok,
    Runner,
    Score,
    Tracer,
    tool,
)
from neverempty.runner.runner import PreflightError

from .test_dataset import case_dict, write_jsonl


def dataset(tmp_path: Path, count: int = 2, **overrides: Any) -> Dataset:
    cases = [case_dict(f"c-{i:04d}", **overrides) for i in range(count)]
    return Dataset.load(write_jsonl(tmp_path / "d.jsonl", cases))


async def noop_target(case: Case, tracer: Tracer) -> None:
    run = tracer.current_run
    if run is not None:
        run.set_output(answer="ok", route="job_search")


def always_pass(case: Case, trace: Any) -> Score | None:
    return Score(passed=True, value=1.0, detail={})


def build(**kwargs: Any) -> Runner:
    kwargs.setdefault("target", noop_target)
    kwargs.setdefault("scorers", [always_pass])
    kwargs.setdefault("tracer", Tracer(sink=MemorySink()))
    return Runner(**kwargs)


class TestHappyPath:
    async def test_every_case_is_scored(self, tmp_path: Path) -> None:
        result = await build().run(dataset(tmp_path, 3))
        assert result.counts.cases == 3
        assert result.counts.scored == 3
        assert result.counts.unscored == 0
        assert result.complete is True
        assert result.status == "ok"

    async def test_repeats_produce_one_outcome_each(self, tmp_path: Path) -> None:
        result = await build(repeats=3).run(dataset(tmp_path, 2))
        assert len(result.outcomes) == 6
        assert result.counts.repeats == 3

    async def test_each_case_gets_its_own_trace(self, tmp_path: Path) -> None:
        sink = MemorySink()
        await build(tracer=Tracer(sink=sink)).run(dataset(tmp_path, 3))
        assert len({trace.trace_id for trace in sink.traces}) == 3

    async def test_the_case_id_and_repeat_reach_the_trace(self, tmp_path: Path) -> None:
        sink = MemorySink()
        await build(tracer=Tracer(sink=sink), repeats=2).run(dataset(tmp_path, 1))
        pairs = {(t.case_id, t.repeat) for t in sink.traces}
        assert pairs == {("c-0000", 0), ("c-0000", 1)}


class TestCaseTimeout:
    """Row: trace status=timeout, scorers not run, case unscored, listed by id."""

    async def test_a_slow_case_is_marked_timeout_and_unscored(self, tmp_path: Path) -> None:
        async def slow(case: Case, tracer: Tracer) -> None:
            await asyncio.sleep(5)

        result = await build(target=slow, case_timeout_s=0.05).run(dataset(tmp_path, 1))

        outcome = result.outcomes[0]
        assert outcome.trace_status == "timeout"
        assert outcome.scored is False
        assert result.counts.unscored == 1

    async def test_scorers_are_not_run_on_a_timed_out_case(self, tmp_path: Path) -> None:
        calls: list[str] = []

        async def slow(case: Case, tracer: Tracer) -> None:
            await asyncio.sleep(5)

        def spy(case: Case, trace: Any) -> Score | None:
            calls.append(case.id)
            return Score(passed=True, value=1.0, detail={})

        await build(target=slow, scorers=[spy], case_timeout_s=0.05).run(dataset(tmp_path, 1))
        assert calls == []

    async def test_the_timed_out_case_is_listed_by_id(self, tmp_path: Path) -> None:
        async def slow(case: Case, tracer: Tracer) -> None:
            await asyncio.sleep(5)

        result = await build(target=slow, case_timeout_s=0.05).run(dataset(tmp_path, 2))
        assert set(result.unscored_ids()) == {"c-0000", "c-0001"}

    async def test_the_default_timeout_is_120_seconds(self) -> None:
        assert build().case_timeout_s == 120

    async def test_one_slow_case_does_not_fail_the_others(self, tmp_path: Path) -> None:
        async def mixed(case: Case, tracer: Tracer) -> None:
            if case.id == "c-0000":
                await asyncio.sleep(5)
            run = tracer.current_run
            if run is not None:
                run.set_output(answer="ok")

        result = await build(target=mixed, case_timeout_s=0.05).run(dataset(tmp_path, 2))
        assert result.counts.scored == 1
        assert result.counts.unscored == 1


class TestTargetRaises:
    """Row: trace status=target_error, type and message recorded, case unscored."""

    async def test_a_raising_target_is_recorded_and_unscored(self, tmp_path: Path) -> None:
        async def boom(case: Case, tracer: Tracer) -> None:
            raise RuntimeError("target blew up")

        result = await build(target=boom).run(dataset(tmp_path, 1))

        outcome = result.outcomes[0]
        assert outcome.trace_status == "target_error"
        assert outcome.scored is False
        assert outcome.error is not None
        assert outcome.error.kind == "RuntimeError"
        assert "target blew up" in outcome.error.message

    async def test_the_error_message_is_capped(self, tmp_path: Path) -> None:
        async def boom(case: Case, tracer: Tracer) -> None:
            raise RuntimeError("x" * 5000)

        result = await build(target=boom).run(dataset(tmp_path, 1))
        assert result.outcomes[0].error is not None
        assert len(result.outcomes[0].error.message.encode()) <= 2048

    async def test_no_traceback_reaches_the_report(self, tmp_path: Path) -> None:
        async def boom(case: Case, tracer: Tracer) -> None:
            raise RuntimeError("failed at /home/me/secret.py")

        result = await build(target=boom).run(dataset(tmp_path, 1))
        assert "Traceback" not in result.model_dump_json()

    async def test_a_run_of_all_failures_is_incomplete_not_ok(self, tmp_path: Path) -> None:
        async def boom(case: Case, tracer: Tracer) -> None:
            raise RuntimeError("x")

        result = await build(target=boom).run(dataset(tmp_path, 2))
        assert result.complete is False
        assert result.status == "incomplete"


class TestScorerRaises:
    """Row: case unscored for that scorer only, other scorers still apply."""

    async def test_one_raising_scorer_does_not_stop_the_others(self, tmp_path: Path) -> None:
        def broken(case: Case, trace: Any) -> Score | None:
            raise ValueError("cannot decide")

        broken.name = "broken"  # type: ignore[attr-defined]
        good = always_pass
        good.name = "good"  # type: ignore[attr-defined]

        result = await build(scorers=[broken, good]).run(dataset(tmp_path, 1))

        outcome = result.outcomes[0]
        assert outcome.scores["good"].passed is True
        assert "broken" in outcome.scorer_errors
        assert "cannot decide" in outcome.scorer_errors["broken"]

    async def test_a_case_with_one_working_scorer_still_counts_as_scored(
        self, tmp_path: Path
    ) -> None:
        def broken(case: Case, trace: Any) -> Score | None:
            raise ValueError("nope")

        result = await build(scorers=[broken, always_pass]).run(dataset(tmp_path, 1))
        assert result.counts.scored == 1

    async def test_a_case_where_every_scorer_raises_is_unscored(self, tmp_path: Path) -> None:
        def broken(case: Case, trace: Any) -> Score | None:
            raise ValueError("nope")

        result = await build(scorers=[broken]).run(dataset(tmp_path, 1))
        assert result.counts.unscored == 1
        assert result.complete is False

    async def test_a_scorer_returning_none_is_not_applicable_never_a_failure(
        self, tmp_path: Path
    ) -> None:
        """The rule the library exists to enforce, applied to itself."""

        def cannot_decide(case: Case, trace: Any) -> Score | None:
            return None

        cannot_decide.name = "cannot_decide"  # type: ignore[attr-defined]

        result = await build(scorers=[cannot_decide, always_pass]).run(dataset(tmp_path, 1))
        outcome = result.outcomes[0]
        assert "cannot_decide" not in outcome.scores
        assert "cannot_decide" in outcome.not_applicable
        assert outcome.scorer_errors == {}

    async def test_a_scorer_may_return_a_passed_none_numeric_score(self, tmp_path: Path) -> None:
        def numeric_only(case: Case, trace: Any) -> Score | None:
            return Score(passed=None, value=0.42, detail={})

        numeric_only.name = "latency"  # type: ignore[attr-defined]

        result = await build(scorers=[numeric_only]).run(dataset(tmp_path, 1))
        assert result.outcomes[0].scores["latency"].value == 0.42
        assert result.outcomes[0].scores["latency"].passed is None

    async def test_cancellation_inside_a_scorer_propagates(self, tmp_path: Path) -> None:
        def cancel(case: Case, trace: Any) -> Score | None:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await build(scorers=[cancel]).run(dataset(tmp_path, 1))

    async def test_scorers_may_be_async(self, tmp_path: Path) -> None:
        async def slow_scorer(case: Case, trace: Any) -> Score | None:
            await asyncio.sleep(0)
            return Score(passed=True, value=1.0, detail={})

        slow_scorer.name = "async_scorer"  # type: ignore[attr-defined]

        result = await build(scorers=[slow_scorer]).run(dataset(tmp_path, 1))
        assert result.outcomes[0].scores["async_scorer"].passed is True


class TestScorerRequirements:
    async def test_a_scorer_is_skipped_when_its_expectation_is_absent(self, tmp_path: Path) -> None:
        """Skipped is not-applicable, never pass."""

        def needs_tools(case: Case, trace: Any) -> Score | None:
            raise AssertionError("should not have been called")

        needs_tools.name = "tool_selection"  # type: ignore[attr-defined]
        needs_tools.requires = frozenset({"expect.tool_calls"})  # type: ignore[attr-defined]

        result = await build(scorers=[needs_tools, always_pass]).run(dataset(tmp_path, 1))
        assert "tool_selection" in result.outcomes[0].not_applicable

    async def test_a_scorer_runs_when_its_expectation_is_present(self, tmp_path: Path) -> None:
        def needs_route(case: Case, trace: Any) -> Score | None:
            return Score(passed=True, value=1.0, detail={})

        needs_route.name = "route"  # type: ignore[attr-defined]
        needs_route.requires = frozenset({"expect.route"})  # type: ignore[attr-defined]

        result = await build(scorers=[needs_route]).run(dataset(tmp_path, 1))
        assert result.outcomes[0].scores["route"].passed is True


class TestBudget:
    """Row: pending tasks cancelled, completed kept, status=aborted_budget,
    complete=false."""

    async def test_exceeding_the_budget_aborts_with_the_documented_status(
        self, tmp_path: Path
    ) -> None:
        result = await build(max_cost_usd=0.0, _force_cost=1.0).run(dataset(tmp_path, 4))

        assert result.status == "aborted_budget"
        assert result.complete is False

    async def test_completed_cases_are_kept(self, tmp_path: Path) -> None:
        result = await build(max_cost_usd=0.02, _force_cost=0.01, concurrency=1).run(
            dataset(tmp_path, 6)
        )

        assert 0 < len(result.outcomes) < 6
        assert all(o.scored for o in result.outcomes)

    async def test_the_spent_total_is_recorded(self, tmp_path: Path) -> None:
        result = await build(max_cost_usd=0.02, _force_cost=0.01, concurrency=1).run(
            dataset(tmp_path, 6)
        )
        assert result.costs.total_usd is not None
        assert result.costs.total_usd > 0

    async def test_a_run_within_budget_is_not_aborted(self, tmp_path: Path) -> None:
        result = await build(max_cost_usd=100.0, _force_cost=0.01).run(dataset(tmp_path, 2))
        assert result.status == "ok"

    async def test_an_unknown_cost_does_not_silently_count_as_zero(self, tmp_path: Path) -> None:
        """With the empty pricing table every cost is null. A budget that
        treated null as 0 would never fire, which is the missing-vs-zero bug."""
        result = await build(max_cost_usd=0.01).run(dataset(tmp_path, 2))
        assert result.costs.unknown_count == 2
        assert result.costs.total_usd is None
        assert result.status == "ok"

    async def test_a_budget_abort_still_writes_a_valid_report(self, tmp_path: Path) -> None:
        from neverempty import Report

        result = await build(max_cost_usd=0.0, _force_cost=1.0).run(dataset(tmp_path, 3))
        path = tmp_path / "report.json"
        result.save(path)
        assert Report.model_validate_json(path.read_text(encoding="utf-8")).complete is False


class TestInterrupt:
    """Row: partial report to <path>.partial.json, never to the final path.

    A real ``KeyboardInterrupt`` aborts the pytest session itself, so these use
    ``CancelledError``, which travels the same uncatchable path in the runner.
    The KeyboardInterrupt path is asserted directly in
    ``test_keyboard_interrupt_takes_the_same_path``, outside pytest's runner.
    """

    async def test_an_interrupt_writes_a_partial_report(self, tmp_path: Path) -> None:
        async def interrupt(case: Case, tracer: Tracer) -> None:
            if case.id == "c-0001":
                raise asyncio.CancelledError
            run = tracer.current_run
            if run is not None:
                run.set_output(answer="ok")

        path = tmp_path / "report.json"
        runner = build(target=interrupt, concurrency=1, output_path=path)

        with pytest.raises(asyncio.CancelledError):
            await runner.run(dataset(tmp_path, 4))

        partial = tmp_path / "report.partial.json"
        assert partial.is_file()
        assert not path.exists(), "the final path must never hold a partial report"

    async def test_the_partial_report_is_marked_incomplete(self, tmp_path: Path) -> None:
        async def interrupt(case: Case, tracer: Tracer) -> None:
            raise asyncio.CancelledError

        path = tmp_path / "report.json"
        runner = build(target=interrupt, concurrency=1, output_path=path)
        with pytest.raises(asyncio.CancelledError):
            await runner.run(dataset(tmp_path, 2))

        payload = json.loads((tmp_path / "report.partial.json").read_text(encoding="utf-8"))
        assert payload["complete"] is False

    async def test_the_partial_keeps_the_cases_that_did_finish(self, tmp_path: Path) -> None:
        async def interrupt(case: Case, tracer: Tracer) -> None:
            if case.id == "c-0002":
                raise asyncio.CancelledError
            run = tracer.current_run
            if run is not None:
                run.set_output(answer="ok")

        path = tmp_path / "report.json"
        with pytest.raises(asyncio.CancelledError):
            await build(target=interrupt, concurrency=1, output_path=path).run(dataset(tmp_path, 5))

        payload = json.loads((tmp_path / "report.partial.json").read_text(encoding="utf-8"))
        assert payload["counts"]["scored"] == 2

    async def test_the_interrupt_still_propagates(self, tmp_path: Path) -> None:
        async def interrupt(case: Case, tracer: Tracer) -> None:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await build(target=interrupt, output_path=tmp_path / "r.json").run(dataset(tmp_path, 1))

    def test_keyboard_interrupt_takes_the_same_path(self, tmp_path: Path) -> None:
        """Asserted in a subprocess: pytest treats a real KeyboardInterrupt as a
        session abort, so it cannot be caught inside a test."""
        import subprocess
        import sys

        script = f"""
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, {str(Path.cwd() / "src")!r})
sys.path.insert(0, {str(Path.cwd())!r})
from neverempty import Case, Dataset, MemorySink, Score, Tracer, Runner

tmp = Path({str(tmp_path)!r})
case = {{
    "schema_version": 1, "id": "c-0001", "suite": "nextrole.routing",
    "split": "dev", "input": {{"messages": [{{"role": "user", "content": "q"}}]}},
    "expect": {{"route": {{"label": "job_search"}}}},
}}
data_path = tmp / "ki.jsonl"
data_path.write_text(json.dumps(case) + chr(10), encoding="utf-8")

async def boom(c, t):
    raise KeyboardInterrupt

runner = Runner(
    target=boom, scorers=[lambda c, t: Score(passed=True)],
    tracer=Tracer(sink=MemorySink()), output_path=tmp / "ki.json",
)
try:
    asyncio.run(runner.run(Dataset.load(data_path)))
except KeyboardInterrupt:
    partial = tmp / "ki.partial.json"
    assert partial.is_file(), "no partial report written"
    assert not (tmp / "ki.json").exists(), "final path was written"
    print("OK")
"""
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        assert "OK" in proc.stdout, proc.stdout + proc.stderr


class TestDuplicateIds:
    """Row: validation error before any execution."""

    async def test_duplicate_ids_fail_before_the_target_runs(self, tmp_path: Path) -> None:
        from neverempty import DatasetError

        calls: list[str] = []

        async def spy(case: Case, tracer: Tracer) -> None:
            calls.append(case.id)

        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001"), case_dict("c-0001")])
        with pytest.raises(DatasetError, match="duplicate"):
            Dataset.load(path)
        assert calls == []


class TestReplayCacheMiss:
    """Row: hard error naming the missing key; replay never calls a provider."""

    async def test_a_replay_miss_is_a_hard_error_naming_the_key(self, tmp_path: Path) -> None:
        from neverempty.runner.cache import CacheMissError, ResponseCache

        cache = ResponseCache(tmp_path / "cache", mode="replay")
        with pytest.raises(CacheMissError) as exc:
            cache.get({"provider": "openai", "model": "gpt-4o", "messages": []})
        message = str(exc.value)
        assert "replay" in message.lower()
        assert len([part for part in message.split() if len(part) == 64]) == 1

    async def test_replay_returns_a_recorded_response(self, tmp_path: Path) -> None:
        from neverempty.runner.cache import ResponseCache

        key = {"provider": "openai", "model": "gpt-4o", "messages": [{"role": "user"}]}
        recorder = ResponseCache(tmp_path / "cache", mode="record")
        recorder.put(key, {"text": "hello"}, usage={"input_tokens": 5, "output_tokens": 2})

        replayer = ResponseCache(tmp_path / "cache", mode="replay")
        entry = replayer.get(key)
        assert entry is not None
        assert entry.response == {"text": "hello"}
        assert entry.usage == {"input_tokens": 5, "output_tokens": 2}

    async def test_record_mode_reports_a_miss_without_raising(self, tmp_path: Path) -> None:
        from neverempty.runner.cache import ResponseCache

        cache = ResponseCache(tmp_path / "cache", mode="record")
        assert cache.get({"provider": "openai", "messages": []}) is None


class TestOrdering:
    """Determinism: results sorted by (case_id, repeat), so runs are diffable."""

    async def test_outcomes_are_sorted_by_case_id_then_repeat(self, tmp_path: Path) -> None:
        cases = [case_dict(f"c-{i:04d}") for i in reversed(range(5))]
        data = Dataset.load(write_jsonl(tmp_path / "d.jsonl", cases))

        result = await build(repeats=2, concurrency=8).run(data)
        keys = [(o.case_id, o.repeat) for o in result.outcomes]
        assert keys == sorted(keys)

    async def test_two_runs_produce_identical_ordering(self, tmp_path: Path) -> None:
        data = dataset(tmp_path, 6)
        first = await build(concurrency=8).run(data)
        second = await build(concurrency=8).run(data)
        assert [(o.case_id, o.repeat) for o in first.outcomes] == [
            (o.case_id, o.repeat) for o in second.outcomes
        ]

    async def test_concurrency_does_not_change_the_result_order(self, tmp_path: Path) -> None:
        data = dataset(tmp_path, 8)
        serial = await build(concurrency=1).run(data)
        parallel = await build(concurrency=8).run(data)
        assert [o.case_id for o in serial.outcomes] == [o.case_id for o in parallel.outcomes]

    async def test_the_seed_is_recorded_in_env(self, tmp_path: Path) -> None:
        result = await build(seed=20260921).run(dataset(tmp_path, 1))
        assert result.env.seed == 20260921

    async def test_concurrency_is_recorded_in_env(self, tmp_path: Path) -> None:
        """p95 under 8-way concurrency is not production p95."""
        result = await build(concurrency=4).run(dataset(tmp_path, 1))
        assert result.env.concurrency == 4


class TestConcurrency:
    async def test_the_semaphore_caps_in_flight_cases(self, tmp_path: Path) -> None:
        in_flight = 0
        peak = 0

        async def watcher(case: Case, tracer: Tracer) -> None:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1

        await build(target=watcher, concurrency=3).run(dataset(tmp_path, 12))
        assert peak <= 3

    async def test_concurrent_cases_do_not_cross_attach_spans(self, tmp_path: Path) -> None:
        @tool(never_empty=True)
        async def work(marker: str) -> str:
            await asyncio.sleep(0)
            return marker

        async def target(case: Case, tracer: Tracer) -> None:
            await work(marker=case.id)

        sink = MemorySink()
        await build(target=target, tracer=Tracer(sink=sink), concurrency=8).run(
            dataset(tmp_path, 20)
        )

        for trace in sink.traces:
            args = json.loads(str(trace.spans[0].attributes["tool.args"]))
            assert args["marker"] == trace.case_id


class TestFaultInjection:
    async def test_declared_faults_are_applied_for_the_case(self, tmp_path: Path) -> None:
        from neverempty import Err

        seen: list[Any] = []

        @tool(never_empty=True)
        async def search_jobs() -> str:
            return "real data"

        async def target(case: Case, tracer: Tracer) -> None:
            seen.append(await search_jobs())

        cases = [
            case_dict("c-0001", faults=[{"tool": "search_jobs", "kind": "timeout"}]),
            case_dict("c-0002"),
        ]
        data = Dataset.load(write_jsonl(tmp_path / "d.jsonl", cases))

        await build(target=target, concurrency=1).run(data)

        assert isinstance(seen[0], Err)
        assert isinstance(seen[1], Ok)

    async def test_the_fault_profile_is_recorded_in_env(self, tmp_path: Path) -> None:
        cases = [case_dict("c-0001", faults=[{"tool": "t", "kind": "timeout"}])]
        data = Dataset.load(write_jsonl(tmp_path / "d.jsonl", cases))
        result = await build().run(data)
        assert result.env.fault_profile is not None


class TestSideEffectPreflight:
    """Preflight 2: every tool tagged side_effect=True has a stub bound."""

    async def test_the_runner_refuses_to_start_with_an_unstubbed_side_effect_tool(
        self, tmp_path: Path
    ) -> None:
        from neverempty.core.tool import clear_side_effect_registry, register_side_effect

        clear_side_effect_registry()
        register_side_effect("send_gmail")
        try:
            with pytest.raises(PreflightError) as exc:
                await build().run(dataset(tmp_path, 1))
            assert "send_gmail" in str(exc.value)
        finally:
            clear_side_effect_registry()

    async def test_a_stubbed_side_effect_tool_passes_preflight(self, tmp_path: Path) -> None:
        from neverempty.core.tool import clear_side_effect_registry, register_side_effect

        clear_side_effect_registry()
        register_side_effect("send_gmail")
        try:
            result = await build(stubs={"send_gmail": lambda **kw: "stubbed"}).run(
                dataset(tmp_path, 1)
            )
            assert result.status == "ok"
        finally:
            clear_side_effect_registry()

    async def test_nothing_runs_when_preflight_fails(self, tmp_path: Path) -> None:
        from neverempty.core.tool import clear_side_effect_registry, register_side_effect

        calls: list[str] = []

        async def spy(case: Case, tracer: Tracer) -> None:
            calls.append(case.id)

        clear_side_effect_registry()
        register_side_effect("send_gmail")
        try:
            with pytest.raises(PreflightError):
                await build(target=spy).run(dataset(tmp_path, 1))
            assert calls == []
        finally:
            clear_side_effect_registry()

    async def test_a_stub_replaces_the_real_tool_body(self, tmp_path: Path) -> None:
        from neverempty import stub_scope

        called = False

        @tool(never_empty=True, side_effect=True)
        async def send_gmail(to: str) -> str:
            nonlocal called
            called = True
            return "REAL SEND"

        async def stub(**kwargs: Any) -> str:
            return "stubbed"

        with stub_scope({"send_gmail": stub}):
            result = await send_gmail(to="a@b.com")

        assert isinstance(result, Ok)
        assert result.value == "stubbed"
        assert called is False, "the real side-effecting body ran"


class TestPreflightOutputPath:
    """Preflight 6: not overwriting an existing report unless forced."""

    async def test_an_existing_report_is_not_overwritten(self, tmp_path: Path) -> None:
        path = tmp_path / "report.json"
        path.write_text("{}", encoding="utf-8")

        with pytest.raises(PreflightError, match="force"):
            await build(output_path=path).run(dataset(tmp_path, 1))

    async def test_force_allows_the_overwrite(self, tmp_path: Path) -> None:
        path = tmp_path / "report.json"
        path.write_text("{}", encoding="utf-8")

        result = await build(output_path=path, force=True).run(dataset(tmp_path, 1))
        assert result.status == "ok"
        assert path.read_text(encoding="utf-8") != "{}"

    async def test_an_unwritable_directory_fails_preflight(self, tmp_path: Path) -> None:
        target = tmp_path / "missing" / "deep" / "report.json"
        result = await build(output_path=target).run(dataset(tmp_path, 1))
        assert target.is_file()
        assert result.status == "ok"


class TestPreflightSplitHash:
    """Preflight 1: test-split hash matches what config recorded."""

    async def test_a_mismatched_split_hash_refuses_to_start(self, tmp_path: Path) -> None:
        data = dataset(tmp_path, 1, split="test")
        with pytest.raises(PreflightError, match="split hash"):
            await build(expect_split_hash="f" * 64).run(data)

    async def test_a_matching_split_hash_starts(self, tmp_path: Path) -> None:
        data = dataset(tmp_path, 1, split="test")
        digest = data.split_hash()
        assert digest is not None
        result = await build(expect_split_hash=digest).run(data)
        assert result.status == "ok"


class TestResume:
    """Row: re-runs only missing or unscored cases, merges, records resumed_from."""

    async def test_resume_reruns_only_the_unscored_cases(self, tmp_path: Path) -> None:
        attempts: list[str] = []
        failed_once = False

        async def flaky(case: Case, tracer: Tracer) -> None:
            nonlocal failed_once
            attempts.append(case.id)
            if case.id == "c-0001" and not failed_once:
                failed_once = True
                raise RuntimeError("transient")
            run = tracer.current_run
            if run is not None:
                run.set_output(answer="ok")

        data = dataset(tmp_path, 3)
        path = tmp_path / "report.json"
        first = await build(target=flaky, output_path=path, concurrency=1).run(data)
        assert first.counts.unscored == 1

        attempts.clear()
        resumed = await build(target=flaky, concurrency=1).resume(first, data)

        assert attempts == ["c-0001"]
        assert resumed.counts.unscored == 0
        assert resumed.complete is True

    async def test_resume_does_not_duplicate_outcomes(self, tmp_path: Path) -> None:
        async def flaky(case: Case, tracer: Tracer) -> None:
            if case.id == "c-0001":
                raise RuntimeError("transient")
            run = tracer.current_run
            if run is not None:
                run.set_output(answer="ok")

        data = dataset(tmp_path, 3)
        first = await build(target=flaky, concurrency=1).run(data)
        resumed = await build(concurrency=1).resume(first, data)

        assert len(resumed.outcomes) == 3
        assert len({(o.case_id, o.repeat) for o in resumed.outcomes}) == 3

    async def test_resume_records_what_it_resumed_from(self, tmp_path: Path) -> None:
        async def boom(case: Case, tracer: Tracer) -> None:
            raise RuntimeError("x")

        data = dataset(tmp_path, 1)
        first = await build(target=boom).run(data)
        resumed = await build().resume(first, data)
        assert resumed.resumed_from == first.report_id

    async def test_a_resumed_report_is_complete_only_when_all_cases_scored(
        self, tmp_path: Path
    ) -> None:
        async def boom(case: Case, tracer: Tracer) -> None:
            raise RuntimeError("x")

        data = dataset(tmp_path, 2)
        first = await build(target=boom).run(data)
        still_broken = await build(target=boom).resume(first, data)
        assert still_broken.complete is False

    async def test_resume_keeps_the_ordering_invariant(self, tmp_path: Path) -> None:
        async def flaky(case: Case, tracer: Tracer) -> None:
            if case.id in ("c-0001", "c-0003"):
                raise RuntimeError("x")
            run = tracer.current_run
            if run is not None:
                run.set_output(answer="ok")

        data = dataset(tmp_path, 5)
        first = await build(target=flaky, concurrency=1).run(data)
        resumed = await build(concurrency=1).resume(first, data)
        keys = [(o.case_id, o.repeat) for o in resumed.outcomes]
        assert keys == sorted(keys)

    async def test_resume_loads_a_partial_report_from_disk(self, tmp_path: Path) -> None:
        from neverempty import Report

        async def interrupt(case: Case, tracer: Tracer) -> None:
            if case.id == "c-0002":
                raise asyncio.CancelledError
            run = tracer.current_run
            if run is not None:
                run.set_output(answer="ok")

        data = dataset(tmp_path, 4)
        path = tmp_path / "report.json"
        with pytest.raises(asyncio.CancelledError):
            await build(target=interrupt, concurrency=1, output_path=path).run(data)

        partial = Report.model_validate_json(
            (tmp_path / "report.partial.json").read_text(encoding="utf-8")
        )
        resumed = await build(concurrency=1).resume(partial, data)
        assert resumed.complete is True
        assert len(resumed.outcomes) == 4


class TestEnvBlock:
    async def test_the_reproducibility_block_is_populated(self, tmp_path: Path) -> None:
        result = await build(
            env_overrides={"target_git_sha": "a" * 40, "prompt_hashes": {"P": "b" * 64}}
        ).run(dataset(tmp_path, 1))

        assert result.env.neverempty_version
        assert result.env.target_git_sha == "a" * 40
        assert result.env.prompt_hashes["P"] == "b" * 64
        assert result.env.pricing_version

    async def test_the_mode_is_recorded(self, tmp_path: Path) -> None:
        result = await build(mode="replay").run(dataset(tmp_path, 1))
        assert result.env.mode == "replay"

    async def test_the_suite_and_split_are_recorded(self, tmp_path: Path) -> None:
        result = await build().run(dataset(tmp_path, 1, split="test"))
        assert result.suite == "nextrole.routing"
        assert result.split == "test"

    async def test_a_dataset_spanning_two_suites_is_rejected(self, tmp_path: Path) -> None:
        """One report describes one suite; averaging two would hide both."""
        cases = [case_dict("c-0001"), case_dict("c-0002", suite="nextrole.failure")]
        data = Dataset.load(write_jsonl(tmp_path / "d.jsonl", cases))
        with pytest.raises(PreflightError, match="one suite"):
            await build().run(data)


class TestLatencyAndCost:
    async def test_per_trace_latencies_are_kept(self, tmp_path: Path) -> None:
        result = await build(repeats=2).run(dataset(tmp_path, 3))
        assert len(result.latency.samples_ms) == 6

    async def test_the_unknown_cost_count_is_reported(self, tmp_path: Path) -> None:
        result = await build().run(dataset(tmp_path, 3))
        assert result.costs.unknown_count == 3
        assert result.costs.total_usd is None


class TestArgumentValidation:
    def test_zero_repeats_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="repeats"):
            build(repeats=0)

    def test_zero_concurrency_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="concurrency"):
            build(concurrency=0)

    async def test_replay_mode_without_a_cache_is_allowed_when_not_required(
        self, tmp_path: Path
    ) -> None:
        """A target with no provider calls is a legitimate replay run."""
        result = await build(mode="replay").run(dataset(tmp_path, 1))
        assert result.env.mode == "replay"

    async def test_replay_mode_requiring_a_cache_refuses_without_one(self, tmp_path: Path) -> None:
        with pytest.raises(PreflightError, match="cache"):
            await build(mode="replay", require_cache=True).run(dataset(tmp_path, 1))
