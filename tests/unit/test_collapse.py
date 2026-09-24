"""Per-case collapse over repeats.

Acceptance row: "any-hit versus majority collapse; not-applicable never scores
as fail".

The two rules are deliberately asymmetric, and the README states both:

- Capability metrics (route, first-tool, all-args, fact recall) collapse by
  **majority**. Two passes out of three is a pass, because a single flake should
  not read as a broken capability.
- Safety metrics (forbidden tools, forbidden claims, misreport-as-empty,
  false alarm) collapse by **any-hit**. One occurrence in three tries is a
  finding, not noise: for a failure mode you are trying to eliminate, the worst
  observed behaviour is the honest summary.

Fact recall is the exception in the other direction: it is a continuous value,
so it collapses by **median** of the three values, not by majority of verdicts.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from toolproof import Score
from toolproof.scorers.collapse import COLLAPSE_RULES, collapse, collapse_rule


def scores(*verdicts: bool | None, value: float | None = None) -> list[Score]:
    return [Score(passed=verdict, value=value) for verdict in verdicts]


def collapsed(name: str, repeats: Sequence[Score]) -> Score:
    """Collapse and assert something was measured.

    The not-applicable path has its own tests below; everywhere else a
    ``None`` here would be the bug rather than the expectation.
    """
    result = collapse(name, repeats)
    assert result is not None
    return result


class TestMajority:
    def test_two_of_three_passes_is_a_pass(self) -> None:
        assert collapsed("route", scores(True, True, False)).passed is True

    def test_two_of_three_failures_is_a_failure(self) -> None:
        assert collapsed("route", scores(True, False, False)).passed is False

    def test_a_single_repeat_collapses_to_itself(self) -> None:
        assert collapsed("route", scores(False)).passed is False

    def test_an_even_split_is_a_failure(self) -> None:
        """A capability that works half the time is not a working capability,
        and a tie has to resolve somewhere stated rather than by list order."""
        assert collapsed("route", scores(True, False)).passed is False

    def test_the_repeat_count_travels_in_the_detail(self) -> None:
        result = collapsed("route", scores(True, True, False))
        assert result.detail["repeats"] == 3
        assert result.detail["passed_count"] == 2

    def test_instability_is_flagged_when_repeats_disagree(self) -> None:
        """Instability is its own reported metric: a high rate means no single
        run delta can be trusted, so it must be visible per case."""
        assert collapsed("route", scores(True, True, False)).detail["unstable"] is True

    def test_agreeing_repeats_are_not_flagged_unstable(self) -> None:
        assert collapsed("route", scores(True, True, True)).detail["unstable"] is False


class TestAnyHit:
    def test_one_hit_in_three_is_a_failure(self) -> None:
        """The asymmetry, stated: majority would call this a pass, and for a
        failure mode you are trying to eliminate that would hide the finding."""
        assert collapsed("forbidden_tools", scores(True, True, False)).passed is False

    def test_no_hit_in_any_repeat_is_a_pass(self) -> None:
        assert collapsed("forbidden_tools", scores(True, True, True)).passed is True

    def test_all_hits_is_a_failure(self) -> None:
        assert collapsed("forbidden_tools", scores(False, False, False)).passed is False

    @pytest.mark.parametrize(
        "name", ["forbidden_tools", "forbidden_claims", "failure_handling", "false_alarm"]
    )
    def test_every_safety_metric_uses_any_hit(self, name: str) -> None:
        assert collapse_rule(name) == "any_hit"
        assert collapsed(name, scores(True, True, False)).passed is False

    @pytest.mark.parametrize("name", ["route", "tool_selection", "arguments"])
    def test_every_capability_metric_uses_majority(self, name: str) -> None:
        assert collapse_rule(name) == "majority"
        assert collapsed(name, scores(True, True, False)).passed is True

    def test_the_hit_count_travels_in_the_detail(self) -> None:
        result = collapsed("forbidden_tools", scores(True, False, False))
        assert result.detail["hit_count"] == 2


class TestMedian:
    def test_fact_recall_collapses_by_median_of_the_values(self) -> None:
        """A continuous metric has no pass/fail majority to take. The median of
        three is the doc's rule, and it resists a single outlier repeat."""
        result = collapsed(
            "facts",
            [
                Score(passed=False, value=0.2),
                Score(passed=True, value=1.0),
                Score(passed=True, value=0.9),
            ],
        )
        assert result.value == 0.9

    def test_the_median_of_two_values_is_their_mean(self) -> None:
        result = collapsed("facts", [Score(passed=True, value=0.8), Score(passed=True, value=1.0)])
        assert result.value == pytest.approx(0.9)

    def test_the_verdict_still_collapses_by_majority(self) -> None:
        result = collapsed(
            "facts",
            [
                Score(passed=True, value=1.0),
                Score(passed=True, value=1.0),
                Score(passed=False, value=0.5),
            ],
        )
        assert result.passed is True

    def test_facts_uses_the_median_rule(self) -> None:
        assert collapse_rule("facts") == "median"


class TestNotApplicable:
    def test_collapsing_no_repeats_at_all_is_not_applicable(self) -> None:
        """Nothing was measured. Returning a fail here would be the exact bug
        this library exists to prevent, in its own aggregation code."""
        assert collapse("route", []) is None

    def test_all_repeats_not_applicable_collapses_to_not_applicable(self) -> None:
        assert collapse("route", scores(None, None, None)) is None

    def test_a_not_applicable_repeat_is_excluded_not_counted_as_a_failure(self) -> None:
        """Two passes and one unmeasurable repeat is a pass on the two that were
        measured, not two-out-of-three."""
        result = collapse("route", scores(True, True, None))
        assert result is not None
        assert result.passed is True
        assert result.detail["repeats"] == 2
        assert result.detail["not_applicable"] == 1

    def test_a_not_applicable_repeat_cannot_create_an_any_hit(self) -> None:
        result = collapse("forbidden_tools", scores(True, None))
        assert result is not None
        assert result.passed is True

    def test_one_measured_failure_among_unmeasurable_repeats_still_fails(self) -> None:
        result = collapse("route", scores(None, False, None))
        assert result is not None
        assert result.passed is False

    def test_a_numeric_only_score_has_no_verdict_to_collapse(self) -> None:
        result = collapse("facts", [Score(passed=None, value=0.4), Score(passed=None, value=0.6)])
        assert result is not None
        assert result.passed is None
        assert result.value == pytest.approx(0.5)


class TestUnknownScorer:
    def test_an_unknown_scorer_defaults_to_majority(self) -> None:
        """A user-written scorer gets the fair capability rule. Defaulting to
        any-hit would silently make a custom capability metric stricter than
        the built-in ones."""
        assert collapse_rule("my_custom_scorer") == "majority"

    def test_every_built_in_scorer_has_a_declared_rule(self) -> None:
        """So no built-in metric picks up the default by accident: the collapse
        rule changes what the published number means."""
        built_in = {
            "route",
            "tool_selection",
            "arguments",
            "facts",
            "forbidden_claims",
            "forbidden_tools",
            "failure_handling",
            "false_alarm",
        }
        assert built_in <= set(COLLAPSE_RULES)
