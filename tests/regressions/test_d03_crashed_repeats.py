"""D3: crashed repeats disappear and the report claims to be complete.

A target that raises on 2 of every 3 calls produced ``complete=True``,
``status="ok"``, ``route=1.0``, ``unstable=0``, and a gate verdict of pass.
Ten of fifteen repeats had crashed. The rendered markdown mentioned none of it.

The "leaves the denominator" rule is right -- a repeat that never produced an
answer cannot be scored as a wrong answer -- but applied alone it let a 67%
crash rate report as a flawless run. Three separate things were wrong:

- ``counts`` recorded no crash count at all, so the fact was unrecoverable
  from the report's summary.
- A case whose repeats disagree about *whether they ran* is not stable, but
  ``unstable`` counted only disagreement about the answer.
- ``status`` stayed ``ok``, so nothing downstream had a reason to look closer.

The measurement is still honest about what it measured: ``route=1.0`` over the
repeats that ran is a true statement. What was missing was any signal that most
of the run did not happen.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from pathlib import Path

import pytest

from neverempty import Dataset, Report, Runner, Tracer, scorers
from neverempty.tracer.sinks import MemorySink


def _dataset(tmp_path: Path, *, cases: int = 5) -> Dataset:
    lines = [
        json.dumps(
            {
                "schema_version": 1,
                "id": f"demo-{index:04d}",
                "suite": "demo.routing",
                "split": "dev",
                "input": {"messages": [{"role": "user", "content": "q"}]},
                "expect": {"route": {"label": "job_search"}},
            }
        )
        for index in range(cases)
    ]
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Dataset.load(path, split="dev")


def _run(dataset: Dataset, target: object, *, repeats: int = 3) -> Report:
    runner = Runner(
        target=target,  # type: ignore[arg-type]
        scorers=[scorers.route()],
        tracer=Tracer(sink=MemorySink()),
        repeats=repeats,
        seed=1,
    )
    return asyncio.run(runner.run(dataset))


def _flaky(every: int = 3):  # type: ignore[no-untyped-def]
    """Crashes on every call whose index is not a multiple of ``every``."""
    counter = itertools.count()

    async def target(case: object, tracer: object) -> None:
        if next(counter) % every != 0:
            raise RuntimeError("agent crashed")
        tracer.current_run.set_output(answer="a", route="job_search")  # type: ignore[attr-defined]

    return target


class TestTheCrashCountReachesTheReport:
    def test_counts_records_the_crashed_repeats(self, tmp_path: Path) -> None:
        report = _run(_dataset(tmp_path), _flaky())
        assert report.counts.crashed == 10

    def test_a_clean_run_records_zero(self, tmp_path: Path) -> None:
        async def clean(case: object, tracer: object) -> None:
            tracer.current_run.set_output(answer="a", route="job_search")  # type: ignore[attr-defined]

        assert _run(_dataset(tmp_path), clean).counts.crashed == 0


class TestARunThatMostlyCrashedFailsTheBuild:
    """``status`` stays ``ok``: the doc fixes that enum to
    ``ok | degraded | aborted_budget | incomplete`` (line 834) and scopes
    ``degraded`` to the judge error rate (line 791), which the gate keys judge-
    metric exclusion off. Widening it here would silently stop evaluating judge
    floors for an unrelated reason.

    The crash signal reaches the gate through instability instead, which is
    what it is: repeats of one input that disagree about whether the agent ran.
    """

    def test_the_gate_no_longer_passes(self, tmp_path: Path) -> None:
        """The defect at its narrowest: 10 of 15 repeats crashed, exit 0."""
        from neverempty import gate
        from neverempty.report.gate import GateConfig

        dataset = _dataset(tmp_path)

        async def clean(case: object, tracer: object) -> None:
            tracer.current_run.set_output(answer="a", route="job_search")  # type: ignore[attr-defined]

        result = gate(_run(dataset, clean), _run(dataset, _flaky()), config=GateConfig())
        assert result.exit_code != 0
        assert result.verdict == "inconclusive"

    def test_a_clean_run_still_passes(self, tmp_path: Path) -> None:
        from neverempty import gate
        from neverempty.report.gate import GateConfig

        dataset = _dataset(tmp_path)

        async def clean(case: object, tracer: object) -> None:
            tracer.current_run.set_output(answer="a", route="job_search")  # type: ignore[attr-defined]

        base, candidate = _run(dataset, clean), _run(dataset, clean)
        assert gate(base, candidate, config=GateConfig()).exit_code == 0

    def test_the_rendered_report_says_so(self, tmp_path: Path) -> None:
        """The reviewer's point: the markdown mentioned crashes nowhere."""
        from neverempty import render_markdown

        rendered = render_markdown(_run(_dataset(tmp_path), _flaky()))
        assert "10 repeat(s) crashed" in rendered
        assert "67% of repeats" in rendered

    def test_a_clean_run_says_nothing_about_crashes(self, tmp_path: Path) -> None:
        from neverempty import render_markdown

        async def clean(case: object, tracer: object) -> None:
            tracer.current_run.set_output(answer="a", route="job_search")  # type: ignore[attr-defined]

        assert "crashed" not in render_markdown(_run(_dataset(tmp_path), clean))


class TestRepeatsThatDisagreeAboutRunningAreUnstable:
    """A case where one repeat answered and two crashed is not a stable case:
    the same input produced an answer and an exception."""

    def test_such_cases_count_as_unstable(self, tmp_path: Path) -> None:
        report = _run(_dataset(tmp_path), _flaky())
        assert report.counts.unstable == 5

    def test_a_clean_run_has_none(self, tmp_path: Path) -> None:
        async def clean(case: object, tracer: object) -> None:
            tracer.current_run.set_output(answer="a", route="job_search")  # type: ignore[attr-defined]

        assert _run(_dataset(tmp_path), clean).counts.unstable == 0

    def test_a_case_whose_every_repeat_crashed_is_not_unstable(self, tmp_path: Path) -> None:
        """Consistent failure is consistent. It is unscored, not unstable."""

        async def always_crash(case: object, tracer: object) -> None:
            raise RuntimeError("agent crashed")

        report = _run(_dataset(tmp_path), always_crash)
        assert report.counts.unstable == 0
        assert report.counts.crashed == 15
        assert report.complete is False


class TestTheMeasurementItselfStaysHonest:
    """The repeats that ran are still measured; only the silence is fixed."""

    def test_the_metric_over_surviving_repeats_is_unchanged(self, tmp_path: Path) -> None:
        report = _run(_dataset(tmp_path), _flaky())
        route = next(m for m in report.metrics if m.name == "route")
        assert route.value == pytest.approx(1.0)

    def test_the_crashed_repeats_keep_their_error(self, tmp_path: Path) -> None:
        report = _run(_dataset(tmp_path), _flaky())
        errored = [o for o in report.outcomes if o.error is not None]
        assert len(errored) == 10
        assert all(o.error is not None and o.error.kind == "RuntimeError" for o in errored)
