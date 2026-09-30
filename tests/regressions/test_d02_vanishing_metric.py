"""D2: a floor is silently skipped when its metric stops being measured.

An agent that stops producing output has ``route`` go from 1.0 to "not
measured". ``_floor_breaches`` then skips the floor and the gate exits 0 --
a pass for an agent that answered nothing.

The existing comment is right about the principle:

    Not measured cannot be a breach. Treating it as one would be this
    library's own headline bug: missing must never look like failure.

But it never asks *why* the metric is missing, and there are two different
facts wearing the same shape:

- The suite never declared the expectation. Nothing was measured, nothing
  regressed, and failing the build would be inventing a result. This must
  keep passing.
- The baseline measured it and the candidate does not. Something the agent
  used to do, it has stopped doing. That is a regression with a missing
  number, not a missing measurement -- and silently skipping the floor is the
  same bug the comment warns about, pointed the other way: a failure
  rendered as an absence.

The gate has a baseline in hand, so it can tell these apart.
"""

from __future__ import annotations

from neverempty.core.trace import Env
from neverempty.report.gate import GateConfig, gate
from neverempty.report.report import CaseOutcome, Counts, Metric, Report, Score


def _env() -> Env:
    return Env(
        neverempty_version="0.1.0",
        pricing_version="p-2026-09-23",
        python_version="3.12.10",
    )


def _report(*, measured: bool, passes: int = 5, total: int = 5) -> Report:
    """A report whose ``route`` metric is either measured or absent."""
    outcomes: list[CaseOutcome] = []
    for index in range(total):
        scores = (
            {"route": Score(passed=index < passes, value=1.0 if index < passes else 0.0)}
            if measured
            else {}
        )
        outcomes.append(CaseOutcome(case_id=f"c-{index:04d}", repeat=0, scored=True, scores=scores))

    metrics = (
        [
            Metric(
                name="route",
                n=total,
                applicable=total,
                value=passes / total,
                method="wilson",
            )
        ]
        if measured
        else [Metric(name="route", n=0, applicable=0, note="not measured")]
    )
    return Report(
        suite="nextrole.routing",
        split="test",
        created_at="2026-10-01T00:00:00.000Z",
        complete=True,
        status="ok",
        env=_env(),
        outcomes=outcomes,
        counts=Counts(cases=total, scored=total),
        metrics=metrics,
    )


FLOOR = GateConfig(floors={"route": 0.95})


class TestAMetricThatDisappearsFailsTheGate:
    def test_the_gate_does_not_pass(self) -> None:
        """The defect at its narrowest: exit 0 for an agent that stopped answering."""
        result = gate(_report(measured=True), _report(measured=False), config=FLOOR)
        assert result.exit_code != 0

    def test_the_reason_names_the_metric_and_says_it_vanished(self) -> None:
        result = gate(_report(measured=True), _report(measured=False), config=FLOOR)
        assert "route" in result.reason
        assert "no longer measured" in result.reason

    def test_it_is_reported_as_invalid_input_not_a_statistical_regression(self) -> None:
        """No significance test produced this, so it must not claim one."""
        result = gate(_report(measured=True), _report(measured=False), config=FLOOR)
        assert result.exit_code == 4
        assert result.verdict == "invalid"

    def test_a_ceiling_floor_also_fires(self) -> None:
        config = GateConfig(floors={"route_max": 0.10})
        result = gate(_report(measured=True), _report(measured=False), config=config)
        assert result.exit_code == 4


class TestButAMetricNeverMeasuredStillPasses:
    """The principle the existing comment defends, kept intact."""

    def test_unmeasured_in_both_reports_is_not_a_breach(self) -> None:
        result = gate(_report(measured=False), _report(measured=False), config=FLOOR)
        assert result.exit_code == 0

    def test_it_is_still_reported_as_a_warning(self) -> None:
        result = gate(_report(measured=False), _report(measured=False), config=FLOOR)
        assert any("could not be evaluated" in w for w in result.warnings)

    def test_a_floor_with_no_baseline_measurement_does_not_fail(self) -> None:
        """A newly added floor must not fail the first build that carries it."""
        config = GateConfig(floors={"facts": 0.9})
        result = gate(_report(measured=True), _report(measured=True), config=config)
        assert result.exit_code == 0


class TestAMeasuredMetricIsUnaffected:
    def test_a_met_floor_still_passes(self) -> None:
        both = _report(measured=True)
        assert gate(both, both, config=FLOOR).exit_code == 0

    def test_a_breached_floor_still_exits_one(self) -> None:
        result = gate(
            _report(measured=True),
            _report(measured=True, passes=3),
            config=FLOOR,
        )
        assert result.exit_code == 1
        assert "below floor" in result.reason
