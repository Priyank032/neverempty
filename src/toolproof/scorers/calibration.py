"""The calibration scorer: pair a stated confidence with a correctness verdict.

This scorer measures nothing by itself. It contributes one
``(confidence, correct)`` pair per case, and the reliability curve turns the
collection of pairs into buckets and an ECE. A single prediction cannot be
miscalibrated; only a distribution can.

Correctness is deliberately delegated. A confidence has to be scored against the
same ground truth the rest of the report uses, so this scorer names an existing
scorer as its correctness source rather than inventing a second notion of
"right". If the route metric and the calibration curve could disagree about
whether case 42 was correct, neither number would be worth publishing.

``passed`` is always ``None``. A confident wrong answer is already a failure of
the route scorer, and returning ``passed=False`` here as well would gate twice
on one event and make the suite look worse than it is.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from toolproof.dataset.case import Case
from toolproof.report.report import Score
from toolproof.runner.runner import ScorerError
from toolproof.scorers.reading import route_prediction, structured_of

CONFIDENCE_KEYS: tuple[str, ...] = (
    "confidence",
    "llm_confidence",
    "route_confidence",
    "score",
)
"""Keys read from ``final_output.structured``, in priority order.

``confidence`` is the documented name. The others are spellings real agents
already use -- YojanaKhoj emits ``llm_confidence`` -- and reading them keeps the
scorer usable without rewriting an agent's output schema. The order is fixed so
that an agent emitting two keys produces the same number on every run.
"""

CORRECTNESS_SOURCES: tuple[str, ...] = ("route",)
"""Scorers this one can take its ground truth from.

Only ``route`` today, because it is the one the doc's reliability curve is
specified over (a 7-way intent route with a stated confidence). Adding a source
means deciding what "correct" means for it, which is a decision to make
explicitly rather than by generalising this list.
"""


def stated_confidence(structured: Mapping[str, Any] | None) -> float | None:
    """The confidence the agent stated, or ``None`` if it stated none.

    ``None`` and ``0.0`` are kept strictly apart: an agent that said nothing has
    not claimed to be uncertain, and folding the two together would invent a
    prediction in the bottom bucket.

    A boolean is rejected even though Python makes ``True == 1``: an agent
    emitting ``confidence: true`` has a bug, and scoring it as perfect certainty
    would bury it. A string is rejected for the same reason -- ``"high"`` is not
    a probability, and neither is ``"0.9"`` until someone decides it is.
    """
    if not structured:
        return None
    for key in CONFIDENCE_KEYS:
        if key not in structured:
            continue
        value = structured[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)
    return None


class CalibrationScorer:
    """Does this agent's stated confidence mean what it says?

    One pair per case: the confidence it stated, and whether the prediction was
    right according to ``correctness``. The curve is built in
    ``toolproof.metrics.reliability`` from the collected pairs.
    """

    name = "calibration"

    def __init__(self, *, correctness: str = "route", lenient: bool = False) -> None:
        if correctness not in CORRECTNESS_SOURCES:
            raise ValueError(
                f"unknown correctness source {correctness!r}; "
                f"choose one of {sorted(CORRECTNESS_SOURCES)}. Calibration takes its "
                f"ground truth from a scorer that already measured correctness, so "
                f"the curve and that scorer's metric can never disagree."
            )
        self.correctness = correctness
        self.lenient = lenient
        self.requires = frozenset({f"expect.{correctness}"})

    def score(self, case: Case, trace: object) -> Score | None:
        expectation = case.expect.route
        if expectation is None:
            # No ground truth for this case, so there is nothing to calibrate
            # against. Not applicable, which is not a calibration failure.
            return None

        confidence = stated_confidence(structured_of(trace))  # type: ignore[arg-type]
        if confidence is None:
            return None

        if not 0.0 <= confidence <= 1.0:
            raise ScorerError(
                f"case {case.id!r} stated a confidence of {confidence!r}, which is not "
                f"a probability in [0, 1]. It is not rescaled: a curve computed from an "
                f"undeclared scale would publish a calibration number nobody can "
                f"reproduce. The case is unscored, which the gate treats as a failure "
                f"of the run rather than of the agent."
            )

        prediction = route_prediction(trace)  # type: ignore[arg-type]
        if prediction is None:
            raise ScorerError(
                f"case {case.id!r} stated a confidence but no route prediction could "
                f"be read, so there is nothing to score the confidence against."
            )

        predicted, source, _span = prediction
        correct = predicted == expectation.label
        if self.lenient:
            correct = correct or predicted in expectation.acceptable

        return Score(
            passed=None,
            value=confidence,
            detail={
                "confidence": confidence,
                "correct": correct,
                "correctness": self.correctness,
                "predicted": predicted,
                "expected": expectation.label,
                "source": source,
                "lenient": self.lenient,
            },
        )


def calibration(*, correctness: str = "route", lenient: bool = False) -> CalibrationScorer:
    """The calibration scorer.

    ``lenient`` follows the route scorer's own distinction: strict by default,
    so a confidence is not flattered by the acceptable-branch allowance unless
    the caller asks for the lenient ground truth explicitly.
    """
    return CalibrationScorer(correctness=correctness, lenient=lenient)


__all__ = [
    "CONFIDENCE_KEYS",
    "CORRECTNESS_SOURCES",
    "CalibrationScorer",
    "calibration",
    "stated_confidence",
]
