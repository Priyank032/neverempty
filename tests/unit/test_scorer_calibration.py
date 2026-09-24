"""The calibration scorer: reads a stated confidence, defers on correctness.

The scorer contributes one ``(confidence, correct)`` pair per case. It never
decides correctness itself: that belongs to whichever scorer already measured
it, so a confidence is always scored against the same ground truth the rest of
the report uses.
"""

from __future__ import annotations

import pytest

from toolproof.dataset.case import Case
from toolproof.runner.runner import ScorerError
from toolproof.scorers.calibration import (
    CONFIDENCE_KEYS,
    CalibrationScorer,
    calibration,
    stated_confidence,
)

from ._scoring import trace


def build_case(
    *,
    route: str | None = None,
    acceptable: list[str] | None = None,
    forbidden_tools: list[str] | None = None,
) -> Case:
    expect: dict[str, object] = {}
    if route is not None:
        entry: dict[str, object] = {"label": route}
        if acceptable is not None:
            entry["acceptable"] = acceptable
        expect["route"] = entry
    if forbidden_tools is not None:
        expect["forbidden_tools"] = forbidden_tools
    return Case.model_validate(
        {
            "id": "nr-cal-0007",
            "suite": "nextrole.routing",
            "split": "dev",
            "input": {"messages": [{"role": "user", "content": "find me a job"}]},
            "expect": expect,
        }
    )


def _trace(structured: dict[str, object] | None = None, **kwargs: object) -> object:
    return trace(structured=structured, **kwargs)  # type: ignore[arg-type]


class TestReadingTheConfidence:
    """Where a stated confidence is found, and what is not one."""

    def test_reads_the_documented_key(self) -> None:
        assert stated_confidence({"confidence": 0.8}) == pytest.approx(0.8)

    def test_accepts_the_alternate_spellings(self) -> None:
        """Real agents spell it several ways; all are the same statement."""
        for key in CONFIDENCE_KEYS:
            assert stated_confidence({key: 0.75}) == pytest.approx(0.75)

    def test_documented_key_wins_over_an_alternate(self) -> None:
        """Deterministic when an agent emits two: the first documented key
        wins, rather than whichever happened to be first in the dict."""
        value = stated_confidence({"llm_confidence": 0.2, "confidence": 0.9})
        assert value == pytest.approx(0.9)

    def test_missing_structured_is_none(self) -> None:
        assert stated_confidence(None) is None

    def test_no_confidence_key_is_none(self) -> None:
        assert stated_confidence({"route": "search", "answer": "x"}) is None

    def test_integer_confidence_is_read(self) -> None:
        assert stated_confidence({"confidence": 1}) == pytest.approx(1.0)

    def test_bool_is_not_a_confidence(self) -> None:
        """``True == 1`` in Python, so a boolean would read as certainty.

        An agent emitting ``confidence: true`` has a bug, and reporting that as
        a perfectly-confident prediction would hide it.
        """
        assert stated_confidence({"confidence": True}) is None
        assert stated_confidence({"confidence": False}) is None

    def test_string_is_not_a_confidence(self) -> None:
        assert stated_confidence({"confidence": "0.9"}) is None
        assert stated_confidence({"confidence": "high"}) is None

    def test_none_value_is_none(self) -> None:
        assert stated_confidence({"confidence": None}) is None


class TestNotApplicable:
    """A missing confidence is not a miscalibrated one."""

    def test_no_structured_output_is_not_applicable(self) -> None:
        scorer = calibration(correctness="route")
        case = build_case(route="search")
        result = scorer.score(case, _trace(route="search"))
        assert result is None

    def test_no_confidence_key_is_not_applicable(self) -> None:
        scorer = calibration(correctness="route")
        case = build_case(route="search")
        result = scorer.score(case, _trace({"other": 1}, route="search"))
        assert result is None

    def test_not_applicable_is_not_a_zero_confidence(self) -> None:
        """The distinction the whole library exists for: an agent that stated
        nothing is not an agent that stated 0.0."""
        scorer = calibration(correctness="route")
        case = build_case(route="search")
        absent = scorer.score(case, _trace({}, route="search"))
        stated = scorer.score(case, _trace({"confidence": 0.0}, route="search"))
        assert absent is None
        assert stated is not None
        assert stated.detail["confidence"] == pytest.approx(0.0)

    def test_case_without_the_correctness_expectation_is_not_applicable(self) -> None:
        """No route expectation means no ground truth, so nothing to calibrate
        against -- not a failed calibration."""
        scorer = calibration(correctness="route")
        case = build_case(forbidden_tools=["send_gmail"])
        result = scorer.score(case, _trace({"confidence": 0.9}))
        assert result is None


class TestCorrectnessComesFromAnotherScorer:
    def test_correct_prediction_records_true(self) -> None:
        scorer = calibration(correctness="route")
        case = build_case(route="search")
        result = scorer.score(case, _trace({"confidence": 0.9}, route="search"))
        assert result is not None
        assert result.detail["correct"] is True
        assert result.detail["confidence"] == pytest.approx(0.9)
        assert result.detail["correctness"] == "route"

    def test_wrong_prediction_records_false(self) -> None:
        scorer = calibration(correctness="route")
        case = build_case(route="search")
        result = scorer.score(case, _trace({"confidence": 0.9}, route="apply"))
        assert result is not None
        assert result.detail["correct"] is False

    def test_passed_is_none_because_this_is_not_a_pass_fail_check(self) -> None:
        """One prediction cannot be miscalibrated; only a distribution can.

        Returning ``passed=False`` for a confident wrong answer would double
        count the route failure and gate on the same event twice.
        """
        scorer = calibration(correctness="route")
        case = build_case(route="search")
        result = scorer.score(case, _trace({"confidence": 0.9}, route="apply"))
        assert result is not None
        assert result.passed is None

    def test_value_is_the_confidence(self) -> None:
        scorer = calibration(correctness="route")
        case = build_case(route="search")
        result = scorer.score(case, _trace({"confidence": 0.35}, route="search"))
        assert result is not None
        assert result.value == pytest.approx(0.35)

    def test_unknown_correctness_source_raises(self) -> None:
        with pytest.raises(ValueError, match="correctness"):
            calibration(correctness="not_a_scorer")

    def test_out_of_range_confidence_raises_rather_than_scoring(self) -> None:
        """A confidence of 95 is a caller bug. The case is unscored, which the
        gate treats as a failure of the run, not a calibration finding."""
        scorer = calibration(correctness="route")
        case = build_case(route="search")
        with pytest.raises(ScorerError, match="confidence"):
            scorer.score(case, _trace({"confidence": 95}, route="search"))

    def test_confidence_without_a_readable_prediction_raises(self) -> None:
        """A confidence with nothing to score it against leaves the case
        unscored, rather than guessing a direction for the gap."""
        scorer = calibration(correctness="route")
        case = build_case(route="search")
        with pytest.raises(ScorerError, match="no route prediction"):
            scorer.score(case, _trace({"confidence": 0.9}))

    def test_lenient_route_match_counts_as_correct(self) -> None:
        """Calibration follows the same ground truth the route metric uses."""
        scorer = calibration(correctness="route", lenient=True)
        case = build_case(route="search", acceptable=["browse"])
        result = scorer.score(case, _trace({"confidence": 0.8}, route="browse"))
        assert result is not None
        assert result.detail["correct"] is True

    def test_strict_by_default_so_acceptable_does_not_inflate(self) -> None:
        scorer = calibration(correctness="route")
        case = build_case(route="search", acceptable=["browse"])
        result = scorer.score(case, _trace({"confidence": 0.8}, route="browse"))
        assert result is not None
        assert result.detail["correct"] is False


class TestScorerContract:
    def test_name(self) -> None:
        assert calibration(correctness="route").name == "calibration"

    def test_requires_the_correctness_expectation(self) -> None:
        scorer = calibration(correctness="route")
        assert "expect.route" in scorer.requires

    def test_factory_returns_the_scorer(self) -> None:
        assert isinstance(calibration(correctness="route"), CalibrationScorer)
