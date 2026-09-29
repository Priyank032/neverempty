"""Building the curve from a report, and rendering it.

The curve is built from *collapsed* scores, one per case, for the same reason
every other metric is: three repeats of seventy cases is a seventy-case sample,
and bucketing it as two hundred and ten would shrink every interval by a factor
the sample never earned.
"""

from __future__ import annotations

import pytest

from neverempty.metrics.reliability import curve_from_outcomes, reliability_curve
from neverempty.report.render import render_reliability
from neverempty.report.report import CaseOutcome, Score


def outcome(
    case_id: str,
    *,
    repeat: int = 0,
    confidence: float | None = None,
    correct: bool | None = None,
    scorer: str = "calibration",
) -> CaseOutcome:
    scores: dict[str, Score] = {}
    if confidence is not None:
        detail: dict[str, object] = {"confidence": confidence}
        if correct is not None:
            detail["correct"] = correct
        scores[scorer] = Score(passed=None, value=confidence, detail=detail)
    return CaseOutcome(case_id=case_id, repeat=repeat, scored=True, scores=scores)


class TestBuildingFromOutcomes:
    def test_one_pair_per_case_not_per_repeat(self) -> None:
        """Three repeats of one case contribute one prediction, not three."""
        outcomes = [outcome("c1", repeat=i, confidence=0.9, correct=True) for i in range(3)]
        curve = curve_from_outcomes(outcomes)
        assert curve.n == 1

    def test_median_confidence_over_repeats(self) -> None:
        """The collapsed value is the median, so near-identical repeats survive."""
        outcomes = [
            outcome("c1", repeat=0, confidence=0.90, correct=True),
            outcome("c1", repeat=1, confidence=0.91, correct=True),
            outcome("c1", repeat=2, confidence=0.89, correct=True),
        ]
        curve = curve_from_outcomes(outcomes)
        assert curve.n == 1
        assert curve.buckets[0].mean_confidence == pytest.approx(0.90)

    def test_repeats_disagreeing_on_correctness_drop_out(self) -> None:
        """An unstable answer has no single event for a confidence to be right
        about, so the case leaves the curve rather than being averaged in."""
        outcomes = [
            outcome("c1", repeat=0, confidence=0.9, correct=True),
            outcome("c1", repeat=1, confidence=0.9, correct=False),
            outcome("c2", repeat=0, confidence=0.8, correct=True),
        ]
        curve = curve_from_outcomes(outcomes)
        assert curve.n == 1
        assert curve.buckets[0].low == pytest.approx(0.8)

    def test_cases_without_the_scorer_are_absent_not_zero(self) -> None:
        outcomes = [
            outcome("c1", confidence=0.9, correct=True),
            outcome("c2"),
            outcome("c3"),
        ]
        curve = curve_from_outcomes(outcomes)
        assert curve.n == 1

    def test_no_calibration_scores_gives_an_unmeasured_curve(self) -> None:
        curve = curve_from_outcomes([outcome("c1"), outcome("c2")])
        assert curve.n == 0
        assert curve.ece is None

    def test_a_score_without_correctness_is_skipped(self) -> None:
        """A confidence with no correctness attached measures nothing, and must
        not enter the curve as a silent miss."""
        outcomes = [outcome("c1", confidence=0.9, correct=None)]
        curve = curve_from_outcomes(outcomes)
        assert curve.n == 0

    def test_reads_the_configured_scorer_name(self) -> None:
        outcomes = [outcome("c1", confidence=0.9, correct=True, scorer="my_calibration")]
        assert curve_from_outcomes(outcomes).n == 0
        assert curve_from_outcomes(outcomes, scorer="my_calibration").n == 1


class TestRendering:
    @pytest.fixture
    def curve(self) -> object:
        return reliability_curve(
            [(0.95, True)] * 4
            + [(0.95, False)]
            + [(0.85, True)] * 2
            + [(0.85, False)] * 2
            + [(0.65, False)] * 2
        )

    def test_reports_ece_and_sample_size(self, curve: object) -> None:
        text = render_reliability(curve)  # type: ignore[arg-type]
        assert "ECE" in text
        assert "n=11" in text

    def test_every_non_empty_bucket_is_a_row(self, curve: object) -> None:
        text = render_reliability(curve)  # type: ignore[arg-type]
        assert "0.9-1.0" in text
        assert "0.8-0.9" in text
        assert "0.6-0.7" in text

    def test_empty_buckets_are_not_rows(self, curve: object) -> None:
        """A bucket nobody predicted into has no accuracy; printing 0% would
        read as a band where the agent is always wrong."""
        text = render_reliability(curve)  # type: ignore[arg-type]
        assert "0.0-0.1" not in text
        assert "0.7-0.8" not in text

    def test_small_buckets_are_marked_rather_than_quoted_as_rates(self, curve: object) -> None:
        """Two predictions do not support a percentage, and the renderer says so
        instead of printing 0%, matching how every other metric is suppressed."""
        text = render_reliability(curve)  # type: ignore[arg-type]
        lines = [line for line in text.splitlines() if "0.6-0.7" in line]
        assert len(lines) == 1
        assert "too few" in lines[0].lower()

    def test_unmeasured_curve_says_so(self) -> None:
        text = render_reliability(reliability_curve([]))
        assert "not measured" in text.lower()
        assert "0.0" not in text
        assert "0%" not in text

    def test_overconfidence_direction_is_named(self, curve: object) -> None:
        """A signed number alone invites a sign error in the reading."""
        text = render_reliability(curve)  # type: ignore[arg-type]
        assert "overconfident" in text.lower()

    def test_underconfidence_is_named_differently(self) -> None:
        curve = reliability_curve([(0.3, True)] * 10 + [(0.3, False)])
        text = render_reliability(curve)
        assert "underconfident" in text.lower()

    def test_a_perfectly_calibrated_curve_is_not_called_either(self) -> None:
        curve = reliability_curve([(1.0, True)] * 5 + [(0.0, False)] * 5)
        text = render_reliability(curve)
        assert "overconfident" not in text.lower()
        assert "underconfident" not in text.lower()

    def test_ece_depends_on_the_bucket_count_and_the_table_says_so(self, curve: object) -> None:
        """ECE is not comparable across bucketings, so the number never appears
        without the bucketing that produced it."""
        text = render_reliability(curve)  # type: ignore[arg-type]
        assert "10 buckets" in text
