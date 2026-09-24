"""The reliability curve: buckets, per-bucket accuracy, ECE.

The acceptance row for M12 asks for two things: the curve matches a
hand-computed fixture, and it works for any model returning a confidence. The
fixture below is computed by hand in exact rationals (see the module docstring
of the fixture class) so that a failure here is a bug in the code and never a
disagreement about floating point.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import ClassVar

import pytest

from toolproof.metrics.reliability import (
    BUCKET_COUNT,
    Bucket,
    ReliabilityCurve,
    reliability_curve,
)


class TestBucketing:
    """Where a confidence lands, and why the edges are where they are."""

    def test_ten_buckets_of_equal_width(self) -> None:
        assert BUCKET_COUNT == 10

    def test_half_open_low_inclusive(self) -> None:
        """0.8 belongs to [0.8, 0.9), not to [0.7, 0.8)."""
        curve = reliability_curve([(0.8, True)])
        assert curve.buckets[0].low == pytest.approx(0.8)
        assert curve.buckets[0].high == pytest.approx(0.9)

    def test_one_point_zero_lands_in_the_top_bucket(self) -> None:
        """A model that says 1.0 must have a home; the top bucket is closed."""
        curve = reliability_curve([(1.0, True)])
        assert len(curve.buckets) == 1
        assert curve.buckets[0].low == pytest.approx(0.9)
        assert curve.buckets[0].n == 1

    def test_zero_lands_in_the_bottom_bucket(self) -> None:
        curve = reliability_curve([(0.0, False)])
        assert curve.buckets[0].low == pytest.approx(0.0)

    def test_a_prediction_lands_in_exactly_one_bucket(self) -> None:
        """No double-counting: the bucket sizes sum to the sample size."""
        data = [(i / 100.0, i % 3 == 0) for i in range(101)]
        curve = reliability_curve(data)
        assert sum(bucket.n for bucket in curve.buckets) == len(data)
        assert curve.n == len(data)

    def test_empty_buckets_are_omitted(self) -> None:
        """A bucket with no predictions has no accuracy, so it carries none."""
        curve = reliability_curve([(0.95, True), (0.15, False)])
        assert [bucket.n for bucket in curve.buckets] == [1, 1]
        assert len(curve.buckets) == 2

    def test_buckets_are_ordered_low_to_high(self) -> None:
        curve = reliability_curve([(0.95, True), (0.15, False), (0.55, True)])
        lows = [bucket.low for bucket in curve.buckets]
        assert lows == sorted(lows)


class TestHandComputedFixture:
    """The acceptance fixture, computed independently in exact rationals.

    20 predictions:

    ======================  ====  ========  ========  =====
    bucket                  n     accuracy  mean conf  gap
    ======================  ====  ========  ========  =====
    [0.3, 0.4)              2     1/2       3/10      1/5
    [0.5, 0.6)              4     1/4       11/20     3/10
    [0.6, 0.7)              2     0         13/20     13/20
    [0.7, 0.8)              3     1         3/4       1/4
    [0.8, 0.9)              4     1/2       17/20     7/20
    [0.9, 1.0]              5     4/5       19/20     3/20
    ======================  ====  ========  ========  =====

    ECE = sum (n_b/20) * gap_b = 1/50 + 3/50 + 13/200 + 3/80 + 7/100 + 3/80
        = 29/100 exactly.
    MCE = max gap = 13/20. Brier = 41/160. accuracy = 11/20,
    mean confidence = 29/40, so overconfidence = 29/40 - 11/20 = 7/40.
    """

    DATA: ClassVar[list[tuple[float, bool]]] = [
        (0.95, True),
        (0.95, True),
        (0.95, True),
        (0.95, True),
        (0.95, False),
        (0.85, True),
        (0.85, True),
        (0.85, False),
        (0.85, False),
        (0.75, True),
        (0.75, True),
        (0.75, True),
        (0.65, False),
        (0.65, False),
        (0.55, True),
        (0.55, False),
        (0.55, False),
        (0.55, False),
        (0.30, True),
        (0.30, False),
    ]

    @pytest.fixture
    def curve(self) -> ReliabilityCurve:
        return reliability_curve(self.DATA)

    def test_ece_is_exactly_twenty_nine_hundredths(self, curve: ReliabilityCurve) -> None:
        assert curve.ece == pytest.approx(0.29)

    def test_mce_is_the_largest_gap(self, curve: ReliabilityCurve) -> None:
        assert curve.mce == pytest.approx(0.65)

    def test_brier_score(self, curve: ReliabilityCurve) -> None:
        assert curve.brier == pytest.approx(41 / 160)

    def test_accuracy_and_mean_confidence(self, curve: ReliabilityCurve) -> None:
        assert curve.accuracy == pytest.approx(11 / 20)
        assert curve.mean_confidence == pytest.approx(29 / 40)

    def test_overconfidence_is_confidence_minus_accuracy(self, curve: ReliabilityCurve) -> None:
        assert curve.overconfidence == pytest.approx(7 / 40)

    def test_six_non_empty_buckets(self, curve: ReliabilityCurve) -> None:
        assert len(curve.buckets) == 6
        assert [bucket.n for bucket in curve.buckets] == [2, 4, 2, 3, 4, 5]

    def test_every_bucket_row(self, curve: ReliabilityCurve) -> None:
        expected = [
            (0.3, 2, 1, 1 / 2, 3 / 10),
            (0.5, 4, 1, 1 / 4, 11 / 20),
            (0.6, 2, 0, 0.0, 13 / 20),
            (0.7, 3, 3, 1.0, 3 / 4),
            (0.8, 4, 2, 1 / 2, 17 / 20),
            (0.9, 5, 4, 4 / 5, 19 / 20),
        ]
        for bucket, (low, n, hits, accuracy, confidence) in zip(
            curve.buckets, expected, strict=True
        ):
            assert bucket.low == pytest.approx(low)
            assert bucket.n == n
            assert bucket.hits == hits
            assert bucket.accuracy == pytest.approx(accuracy)
            assert bucket.mean_confidence == pytest.approx(confidence)
            assert bucket.gap == pytest.approx(abs(accuracy - confidence))

    def test_sample_weighting_is_not_a_plain_mean_of_gaps(self, curve: ReliabilityCurve) -> None:
        """The two ECE definitions give different numbers on this fixture.

        Sample-weighted is 29/100; an unweighted mean of the six bucket gaps is
        19/60. The fixture is built so they disagree, otherwise the test would
        pass under either definition and pin neither.
        """
        unweighted = sum(bucket.gap for bucket in curve.buckets) / len(curve.buckets)
        assert unweighted == pytest.approx(19 / 60)
        assert curve.ece == pytest.approx(29 / 100)


class TestNothingToMeasure:
    """An empty or unusable sample has no calibration error, not a zero."""

    def test_no_predictions_gives_none(self) -> None:
        curve = reliability_curve([])
        assert curve.n == 0
        assert curve.ece is None
        assert curve.mce is None
        assert curve.brier is None
        assert curve.accuracy is None
        assert curve.mean_confidence is None
        assert curve.overconfidence is None
        assert curve.buckets == ()

    def test_zero_ece_is_reachable_and_distinct_from_none(self) -> None:
        """A perfectly calibrated model scores 0.0; that must not read as
        'unmeasured', which is why the empty case returns None instead."""
        data = [(1.0, True)] * 4 + [(0.0, False)] * 4
        curve = reliability_curve(data)
        assert curve.ece == pytest.approx(0.0)
        assert curve.ece is not None


class TestAnyModelReturningConfidence:
    """The acceptance row's second half: no assumption about the producer."""

    def test_confidence_out_of_range_is_rejected(self) -> None:
        for bad in (-0.01, 1.01, 2.0, -1.0):
            with pytest.raises(ValueError, match="confidence"):
                reliability_curve([(bad, True)])

    def test_nan_confidence_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            reliability_curve([(math.nan, True)])

    def test_infinite_confidence_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            reliability_curve([(math.inf, True)])

    def test_integer_confidence_is_accepted(self) -> None:
        """A model returning 1 rather than 1.0 is the same statement."""
        curve = reliability_curve([(1, True), (0, False)])
        assert curve.ece == pytest.approx(0.0)

    def test_percentage_scale_is_not_silently_rescaled(self) -> None:
        """A model returning 95 for 95% is a bug in the caller, and the curve
        says so rather than dividing by 100 and publishing a number."""
        with pytest.raises(ValueError, match="confidence"):
            reliability_curve([(95, True)])

    def test_every_bucket_at_full_resolution(self) -> None:
        """All ten buckets populated, as a model spanning the range would."""
        data = [(i / 10.0 + 0.05, i % 2 == 0) for i in range(10)]
        curve = reliability_curve(data)
        assert len(curve.buckets) == BUCKET_COUNT


class TestBucketModel:
    def test_bucket_is_frozen(self) -> None:
        bucket = Bucket(
            low=0.9, high=1.0, n=2, hits=1, accuracy=0.5, mean_confidence=0.95, gap=0.45
        )
        with pytest.raises(FrozenInstanceError):
            bucket.n = 3  # type: ignore[misc]

    def test_curve_is_frozen(self) -> None:
        curve = reliability_curve([(0.5, True)])
        with pytest.raises(FrozenInstanceError):
            curve.ece = 0.0  # type: ignore[misc]

    def test_label_reads_as_an_interval(self) -> None:
        curve = reliability_curve([(0.95, True), (0.05, False)])
        assert curve.buckets[-1].label == "0.9-1.0"
        assert curve.buckets[0].label == "0.0-0.1"
