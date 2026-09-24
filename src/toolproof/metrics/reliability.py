"""The reliability curve: does a stated confidence mean what it says?

A model that reports 0.9 on a hundred predictions should be right about ninety
times. The gap between stated confidence and measured accuracy is calibration
error, and it is invisible to accuracy alone: two models with identical accuracy
can differ completely in whether their confidence can be acted on.

That matters here because a confidence is only useful if something downstream
branches on it. Routing a low-confidence prediction to a clarify branch, or
escalating a cheap judge to an expensive one below a threshold, are both
decisions that are worse than useless when the confidence is miscalibrated: the
threshold gets set from the vendor's claim rather than from measurement.

Nothing in this module knows what produced the confidence. It takes
``(confidence, correct)`` pairs, so it works for a classifier's softmax, a
judge's self-reported certainty, or a typed-decision API, and the definition of
"correct" stays with the scorer that decided it.

Two conventions are fixed here because the doc names the metric but not its
edges, and an unstated convention is a number nobody can reproduce:

- Ten buckets of width 0.1, half-open ``[low, high)``, with the top bucket
  closed so a confidence of exactly 1.0 has a home and no prediction can land
  in two buckets.
- ECE is sample-weighted over non-empty buckets. An unweighted mean of bucket
  gaps lets a bucket holding two predictions move the headline number as much
  as one holding two hundred.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from toolproof.metrics.aggregate import collapsed_scores
from toolproof.report.report import CaseOutcome

BUCKET_COUNT = 10
"""Buckets of width 0.1.

Fixed rather than configurable: ECE depends on the bin count, so two reports
using different bucketings are not comparable, and the number is published.
"""


@dataclass(frozen=True)
class Bucket:
    """One confidence band, with what was claimed and what happened.

    ``mean_confidence`` is the mean of the stated confidences that landed here,
    not the midpoint of the band. A bucket holding five predictions all at 0.95
    is claiming 0.95, and scoring it against 0.85 would invent an error.
    """

    low: float
    high: float
    n: int
    hits: int
    accuracy: float
    mean_confidence: float
    gap: float

    @property
    def label(self) -> str:
        """The band as it appears in a report, e.g. ``0.9-1.0``."""
        return f"{self.low:.1f}-{self.high:.1f}"


@dataclass(frozen=True)
class ReliabilityCurve:
    """The curve and its summary numbers.

    Every summary is ``None`` when there was nothing to measure, never ``0.0``.
    A zero ECE is a real and excellent result -- perfect calibration -- so it
    must not share a representation with "no predictions carried a confidence".
    """

    buckets: tuple[Bucket, ...]
    n: int
    ece: float | None = None
    mce: float | None = None
    brier: float | None = None
    accuracy: float | None = None
    mean_confidence: float | None = None
    overconfidence: float | None = None
    """Mean confidence minus accuracy. Positive is overconfident.

    Signed on purpose: ECE and MCE are magnitudes, and an underconfident model
    is a different problem from an overconfident one. A model that understates
    its certainty wastes escalations; one that overstates it ships wrong
    answers with authority.
    """

    @property
    def measured(self) -> bool:
        return self.n > 0


def _bucket_index(confidence: float) -> int:
    """Which band a confidence belongs to.

    The ``min`` is what closes the top bucket: ``1.0 * 10 == 10`` would be an
    eleventh band of width zero holding only the single value 1.0.
    """
    return min(int(confidence * BUCKET_COUNT), BUCKET_COUNT - 1)


def _check(confidence: float) -> float:
    """Reject anything that is not a probability.

    A model returning 95 for 95% is a caller bug, and rescaling it silently
    would publish a calibration number for a scale nobody declared. NaN is
    rejected for the same reason: it would propagate into the ECE and print as
    ``nan`` in a report that claims to be reproducible.
    """
    value = float(confidence)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"confidence must be a finite probability in [0, 1], got {confidence!r}; "
            f"a value on another scale is not rescaled, because a curve computed "
            f"from an undeclared scale is not reproducible"
        )
    return value


def reliability_curve(
    predictions: Sequence[tuple[float, bool]],
) -> ReliabilityCurve:
    """Bucket predictions by stated confidence and measure accuracy per bucket.

    ``predictions`` pairs each stated confidence with whether the prediction was
    correct. Correctness is the caller's decision; this function never infers it.
    """
    pairs = [(_check(confidence), bool(correct)) for confidence, correct in predictions]
    if not pairs:
        return ReliabilityCurve(buckets=(), n=0)

    grouped: dict[int, list[tuple[float, bool]]] = {}
    for confidence, correct in pairs:
        grouped.setdefault(_bucket_index(confidence), []).append((confidence, correct))

    total = len(pairs)
    buckets: list[Bucket] = []
    ece = 0.0
    mce = 0.0
    for index in sorted(grouped):
        items = grouped[index]
        count = len(items)
        hits = sum(1 for _, correct in items if correct)
        accuracy = hits / count
        mean_confidence = math.fsum(confidence for confidence, _ in items) / count
        gap = abs(accuracy - mean_confidence)
        ece += (count / total) * gap
        mce = max(mce, gap)
        buckets.append(
            Bucket(
                low=index / BUCKET_COUNT,
                high=(index + 1) / BUCKET_COUNT,
                n=count,
                hits=hits,
                accuracy=accuracy,
                mean_confidence=mean_confidence,
                gap=gap,
            )
        )

    accuracy = sum(1 for _, correct in pairs if correct) / total
    mean_confidence = math.fsum(confidence for confidence, _ in pairs) / total
    brier = (
        math.fsum((confidence - (1.0 if correct else 0.0)) ** 2 for confidence, correct in pairs)
        / total
    )
    return ReliabilityCurve(
        buckets=tuple(buckets),
        n=total,
        ece=ece,
        mce=mce,
        brier=brier,
        accuracy=accuracy,
        mean_confidence=mean_confidence,
        overconfidence=mean_confidence - accuracy,
    )


def curve_from_outcomes(
    outcomes: Sequence[CaseOutcome],
    *,
    scorer: str = "calibration",
) -> ReliabilityCurve:
    """Build the curve from a run's case outcomes.

    Repeats collapse first, so a suite of 70 cases run 3 times contributes 70
    predictions rather than 210. Bucketing the repeats individually would shrink
    every bucket's apparent noise without the sample having earned it.

    A collapsed score with no ``correct`` field contributes nothing. That is a
    case whose repeats disagreed about whether the agent was right, and an
    unstable answer gives its stated confidence nothing to have been right
    about; silently counting it either way would put a guess in the curve.
    """
    predictions: list[tuple[float, bool]] = []
    for scores in collapsed_scores(outcomes).values():
        score = scores.get(scorer)
        if score is None or score.value is None:
            continue
        correct = score.detail.get("correct")
        if not isinstance(correct, bool):
            continue
        predictions.append((score.value, correct))
    return reliability_curve(predictions)


__all__ = [
    "BUCKET_COUNT",
    "Bucket",
    "ReliabilityCurve",
    "curve_from_outcomes",
    "reliability_curve",
]
