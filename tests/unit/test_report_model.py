"""The Report model.

M7 fills ``metrics``, ``confusion`` and ``judge``; this covers the structure the
runner produces and the honesty rules baked into the model itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from neverempty import CaseOutcome, Metric, Report, Score
from neverempty.core.trace import Env


def env() -> Env:
    return Env(
        neverempty_version="0.0.1", pricing_version="empty-2026-09-23", python_version="3.12.10"
    )


def report(**overrides: object) -> Report:
    base: dict[str, object] = {
        "suite": "nextrole.routing",
        "split": "test",
        "created_at": "2026-09-23T10:00:00.000Z",
        "env": env(),
    }
    base.update(overrides)
    return Report.model_validate(base)


class TestMetricHonesty:
    def test_a_metric_with_nothing_to_measure_carries_no_value(self) -> None:
        """ "Not measured" must never render as 0: the same rule the library
        enforces on agents, applied to its own reports."""
        with pytest.raises(ValidationError, match="not measured"):
            Metric(name="route_strict", n=0, value=0.0, applicable=0)

    def test_an_inapplicable_metric_with_a_null_value_is_fine(self) -> None:
        metric = Metric(name="route_strict", n=0, applicable=0, note="no route expectations")
        assert metric.value is None

    def test_an_applicable_metric_may_carry_a_value(self) -> None:
        metric = Metric(
            name="route_strict",
            n=210,
            value=0.85,
            ci_low=0.80,
            ci_high=0.89,
            method="wilson",
            applicable=210,
        )
        assert metric.value == 0.85

    def test_a_genuinely_zero_metric_is_expressible(self) -> None:
        """0% misreport rate is a result; it needs a positive denominator."""
        metric = Metric(name="misreport_as_empty", n=30, value=0.0, applicable=30)
        assert metric.value == 0.0


class TestOrdering:
    def test_outcomes_are_sorted_on_construction(self) -> None:
        unsorted = [
            CaseOutcome(case_id="c-0002", repeat=0),
            CaseOutcome(case_id="c-0001", repeat=1),
            CaseOutcome(case_id="c-0001", repeat=0),
        ]
        keys = [o.key for o in report(outcomes=unsorted).outcomes]
        assert keys == [("c-0001", 0), ("c-0001", 1), ("c-0002", 0)]

    def test_serialization_is_stable_across_a_round_trip(self) -> None:
        original = report(
            outcomes=[
                CaseOutcome(case_id="c-0002", repeat=0, scored=True),
                CaseOutcome(case_id="c-0001", repeat=0, scored=True),
            ]
        )
        once = original.to_json()
        assert Report.model_validate_json(once).to_json() == once


class TestUnscoredIds:
    def test_only_cases_with_no_scored_outcome_are_listed(self) -> None:
        subject = report(
            outcomes=[
                CaseOutcome(case_id="c-0001", repeat=0, scored=True),
                CaseOutcome(case_id="c-0002", repeat=0, scored=False),
            ]
        )
        assert subject.unscored_ids() == ["c-0002"]

    def test_a_case_scored_on_one_repeat_is_not_unscored(self) -> None:
        subject = report(
            outcomes=[
                CaseOutcome(case_id="c-0001", repeat=0, scored=False),
                CaseOutcome(case_id="c-0001", repeat=1, scored=True),
            ]
        )
        assert subject.unscored_ids() == []


class TestScore:
    def test_a_numeric_only_score_has_no_pass_verdict(self) -> None:
        score = Score(passed=None, value=0.42)
        assert score.passed is None
        assert score.value == 0.42

    def test_a_score_is_frozen(self) -> None:
        score = Score(passed=True, value=1.0)
        mutable: Any = score
        with pytest.raises((ValueError, TypeError)):
            mutable.passed = False

    def test_detail_defaults_to_an_empty_dict(self) -> None:
        assert Score(passed=True).detail == {}


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        path = report(complete=True, status="ok").save(tmp_path / "r.json")
        assert Report.load(path).status == "ok"

    def test_save_creates_missing_directories(self, tmp_path: Path) -> None:
        path = report().save(tmp_path / "nested" / "deep" / "r.json")
        assert path.is_file()

    def test_the_written_file_ends_with_a_newline(self, tmp_path: Path) -> None:
        path = report().save(tmp_path / "r.json")
        assert path.read_text(encoding="utf-8").endswith("\n")

    def test_traces_are_not_written_into_the_report(self, tmp_path: Path) -> None:
        """Traces live in their own JSONL; embedding them would bloat a
        committed report a README links to."""
        path = report().save(tmp_path / "r.json")
        assert "traces" not in path.read_text(encoding="utf-8")

    def test_the_default_status_is_incomplete_not_ok(self) -> None:
        """A report that never finished must not read as a clean run."""
        assert report().status == "incomplete"
        assert report().complete is False


class TestAnUnmeasuredDurationIsNullNotZero:
    """``CaseOutcome.duration_ms`` defaulted to 0, so a case whose trace was
    never produced reported 0 ms -- indistinguishable from a case that really
    finished in under a millisecond.

    That is the library's own rule ("missing must never look like zero")
    broken inside its own report model, and it is not only cosmetic: the
    fabricated 0 entered the sorted sample and dragged p50 down, so the
    published latency described a run that never happened. ``Costs`` directly
    above it in the same constructor already gets this right, reporting
    ``None`` with an ``unknown_count`` rather than a zero.
    """

    def test_the_default_is_none(self) -> None:
        assert CaseOutcome(case_id="c-0001", repeat=0).duration_ms is None

    def test_a_measured_zero_is_preserved(self) -> None:
        """A real sub-millisecond case records 0, and 0 must stay a measurement."""
        assert CaseOutcome(case_id="c-0001", repeat=0, duration_ms=0).duration_ms == 0

    def test_a_negative_duration_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            CaseOutcome(case_id="c-0001", repeat=0, duration_ms=-1)

    def test_an_unmeasured_duration_round_trips_as_null(self) -> None:
        outcome = CaseOutcome(case_id="c-0001", repeat=0)
        assert CaseOutcome.model_validate_json(outcome.model_dump_json()).duration_ms is None
