"""Aggregating outcomes into report metrics.

Acceptance row (report and gate): "renderer refuses a metric with
``applicable=0``; percentages suppressed below n=10".

The collapse rules were tested in M6 against hand-built scores. This is the
layer above: many cases, each with repeats, becoming the ``metrics``,
``confusion`` and ``counts`` blocks that the renderer and the gate read. The
rule that matters most is that a metric with nothing to measure carries no
value at all, so no downstream consumer can print it as 0.
"""

from __future__ import annotations

import pytest

from neverempty import CaseOutcome, Score
from neverempty.metrics.aggregate import (
    CONFUSION_UNSCORED,
    build_confusion,
    build_metrics,
    case_verdicts,
    count_unstable,
)


def outcome(
    case_id: str,
    repeat: int = 0,
    *,
    scores: dict[str, Score] | None = None,
    scored: bool = True,
    must_pass: bool = False,
) -> CaseOutcome:
    return CaseOutcome(
        case_id=case_id,
        repeat=repeat,
        scored=scored,
        scores=scores or {},
        must_pass=must_pass,
    )


def verdict(passed: bool | None, value: float | None = None, **detail: object) -> Score:
    return Score(passed=passed, value=value, detail=detail)


class TestBuildMetrics:
    def test_a_single_scorer_becomes_one_metric(self) -> None:
        outcomes = [
            outcome("c-1", scores={"route": verdict(True, 1.0)}),
            outcome("c-2", scores={"route": verdict(False, 0.0)}),
        ]
        metrics = {m.name: m for m in build_metrics(outcomes, seed=1)}
        assert metrics["route"].n == 2
        assert metrics["route"].value == pytest.approx(0.5)
        assert metrics["route"].applicable == 2

    def test_a_wilson_interval_is_attached_to_a_rate(self) -> None:
        outcomes = [
            outcome(f"c-{i}", scores={"route": verdict(i < 85, 1.0 if i < 85 else 0.0)})
            for i in range(100)
        ]
        metric = next(m for m in build_metrics(outcomes, seed=1) if m.name == "route")
        assert metric.method == "wilson"
        assert metric.ci_low == pytest.approx(0.77, abs=0.01)
        assert metric.ci_high == pytest.approx(0.91, abs=0.01)

    def test_fact_recall_gets_a_bootstrap_interval(self) -> None:
        """A mean of per-case fractions, not a count of successes, so Wilson
        does not apply and the doc specifies a seeded bootstrap."""
        outcomes = [
            outcome(f"c-{i}", scores={"facts": verdict(True, 0.5 + i / 100)}) for i in range(20)
        ]
        metric = next(m for m in build_metrics(outcomes, seed=20260921) if m.name == "facts")
        assert metric.method == "bootstrap"
        assert metric.ci_low is not None
        assert metric.ci_high is not None

    def test_the_bootstrap_is_reproducible_under_the_seed(self) -> None:
        outcomes = [
            outcome(f"c-{i}", scores={"facts": verdict(True, 0.5 + i / 100)}) for i in range(20)
        ]
        first = build_metrics(outcomes, seed=7)
        second = build_metrics(outcomes, seed=7)
        assert [m.model_dump() for m in first] == [m.model_dump() for m in second]

    def test_repeats_collapse_before_aggregation(self) -> None:
        """One case with three repeats is one observation, not three. Counting
        repeats as independent cases would inflate n and shrink every CI."""
        outcomes = [
            outcome("c-1", 0, scores={"route": verdict(True, 1.0)}),
            outcome("c-1", 1, scores={"route": verdict(True, 1.0)}),
            outcome("c-1", 2, scores={"route": verdict(False, 0.0)}),
        ]
        metric = next(m for m in build_metrics(outcomes, seed=1) if m.name == "route")
        assert metric.n == 1
        assert metric.value == 1.0  # majority of 2-of-3

    def test_a_safety_metric_collapses_by_any_hit_across_repeats(self) -> None:
        outcomes = [
            outcome("c-1", 0, scores={"forbidden_tools": verdict(True, 1.0)}),
            outcome("c-1", 1, scores={"forbidden_tools": verdict(True, 1.0)}),
            outcome("c-1", 2, scores={"forbidden_tools": verdict(False, 0.0)}),
        ]
        metric = next(m for m in build_metrics(outcomes, seed=1) if m.name == "forbidden_tools")
        assert metric.value == 0.0

    def test_metrics_are_sorted_by_name(self) -> None:
        """So two runs produce diffable JSON regardless of scorer order."""
        outcomes = [
            outcome(
                "c-1",
                scores={
                    "route": verdict(True, 1.0),
                    "arguments": verdict(True, 1.0),
                    "facts": verdict(True, 1.0),
                },
            )
        ]
        names = [m.name for m in build_metrics(outcomes, seed=1)]
        assert names == sorted(names)


class TestNotMeasured:
    def test_a_metric_no_case_could_measure_has_applicable_zero(self) -> None:
        """The whole point of the rule: nothing to measure is not 0%."""
        outcomes = [outcome("c-1", scores={}), outcome("c-2", scores={})]
        metrics = build_metrics(outcomes, seed=1, scorer_names=["route"])
        metric = next(m for m in metrics if m.name == "route")
        assert metric.applicable == 0
        assert metric.value is None
        assert metric.note is not None

    def test_a_metric_with_no_value_carries_no_interval_either(self) -> None:
        outcomes = [outcome("c-1", scores={})]
        metric = next(m for m in build_metrics(outcomes, seed=1, scorer_names=["route"]))
        assert metric.ci_low is None
        assert metric.ci_high is None

    def test_partially_applicable_metrics_count_only_what_applied(self) -> None:
        """Three cases, one of which the scorer could not decide. The metric is
        2 out of 2, not 2 out of 3: an undecidable case is not a failure."""
        outcomes = [
            outcome("c-1", scores={"route": verdict(True, 1.0)}),
            outcome("c-2", scores={"route": verdict(True, 1.0)}),
            outcome("c-3", scores={}),
        ]
        metric = next(m for m in build_metrics(outcomes, seed=1) if m.name == "route")
        assert metric.n == 2
        assert metric.applicable == 2
        assert metric.value == 1.0

    def test_a_numeric_only_metric_reports_its_mean_without_a_verdict(self) -> None:
        outcomes = [
            outcome("c-1", scores={"confidence": verdict(None, 0.4)}),
            outcome("c-2", scores={"confidence": verdict(None, 0.6)}),
        ]
        metric = next(m for m in build_metrics(outcomes, seed=1) if m.name == "confidence")
        assert metric.value == pytest.approx(0.5)


class TestUnstable:
    def test_a_case_whose_repeats_disagree_is_unstable(self) -> None:
        outcomes = [
            outcome("c-1", 0, scores={"route": verdict(True, 1.0)}),
            outcome("c-1", 1, scores={"route": verdict(False, 0.0)}),
        ]
        assert count_unstable(outcomes) == 1

    def test_agreeing_repeats_are_stable(self) -> None:
        outcomes = [
            outcome("c-1", 0, scores={"route": verdict(True, 1.0)}),
            outcome("c-1", 1, scores={"route": verdict(True, 1.0)}),
        ]
        assert count_unstable(outcomes) == 0

    def test_instability_in_any_scorer_makes_the_case_unstable(self) -> None:
        outcomes = [
            outcome(
                "c-1",
                0,
                scores={"route": verdict(True, 1.0), "facts": verdict(True, 1.0)},
            ),
            outcome(
                "c-1",
                1,
                scores={"route": verdict(True, 1.0), "facts": verdict(False, 0.0)},
            ),
        ]
        assert count_unstable(outcomes) == 1

    def test_a_single_repeat_is_never_unstable(self) -> None:
        """With one observation there is nothing to disagree with. Calling it
        unstable would make every repeats=1 run inconclusive."""
        assert count_unstable([outcome("c-1", scores={"route": verdict(True, 1.0)})]) == 0


class TestCaseVerdicts:
    def test_a_case_passes_when_every_applicable_scorer_passes(self) -> None:
        outcomes = [
            outcome("c-1", scores={"route": verdict(True, 1.0), "facts": verdict(True, 1.0)})
        ]
        assert case_verdicts(outcomes) == {"c-1": True}

    def test_one_failing_scorer_fails_the_case(self) -> None:
        outcomes = [
            outcome("c-1", scores={"route": verdict(True, 1.0), "facts": verdict(False, 0.0)})
        ]
        assert case_verdicts(outcomes) == {"c-1": False}

    def test_an_unscored_case_has_no_verdict_rather_than_a_failure(self) -> None:
        """``gate`` treats an unscored case as a failure of the *run*, which is
        a different thing from the agent failing, and is decided there."""
        assert case_verdicts([outcome("c-1", scored=False, scores={})]) == {"c-1": None}

    def test_a_numeric_only_score_does_not_decide_a_case(self) -> None:
        outcomes = [outcome("c-1", scores={"confidence": verdict(None, 0.4)})]
        assert case_verdicts(outcomes) == {"c-1": None}

    def test_a_named_primary_metric_decides_the_case_alone(self) -> None:
        """For the paired test the doc compares one outcome per case. Gating on
        a conjunction of every metric would mix a routing regression with a
        fact-recall one and attribute both to whichever changed."""
        outcomes = [
            outcome("c-1", scores={"route": verdict(True, 1.0), "facts": verdict(False, 0.0)})
        ]
        assert case_verdicts(outcomes, primary="route") == {"c-1": True}

    def test_a_primary_metric_absent_from_a_case_gives_no_verdict(self) -> None:
        outcomes = [outcome("c-1", scores={"facts": verdict(True, 1.0)})]
        assert case_verdicts(outcomes, primary="route") == {"c-1": None}


class TestConfusion:
    def test_rows_are_labels_and_columns_are_predictions(self) -> None:
        outcomes = [
            outcome(
                "c-1",
                scores={"route": verdict(True, 1.0, expected="job_search", predicted="job_search")},
            ),
            outcome(
                "c-2",
                scores={"route": verdict(False, 0.0, expected="followup", predicted="email_draft")},
            ),
        ]
        matrix = build_confusion(outcomes)
        assert matrix["job_search"]["job_search"] == 1
        assert matrix["followup"]["email_draft"] == 1

    def test_unscored_cases_get_their_own_column(self) -> None:
        """The doc requires an extra column for ``unscored``: a case the harness
        could not read is not a misrouting, and hiding it would make the rows
        sum to less than the case count with no explanation."""
        outcomes = [
            outcome(
                "c-1",
                scores={"route": verdict(True, 1.0, expected="job_search", predicted="job_search")},
            ),
            outcome("c-2", scored=False, scores={}, must_pass=False),
        ]
        matrix = build_confusion(outcomes, expected_labels={"c-2": "followup"})
        assert matrix["followup"][CONFUSION_UNSCORED] == 1

    def test_counts_accumulate_across_cases(self) -> None:
        outcomes = [
            outcome(
                f"c-{i}",
                scores={"route": verdict(True, 1.0, expected="job_search", predicted="job_search")},
            )
            for i in range(3)
        ]
        assert build_confusion(outcomes)["job_search"]["job_search"] == 3

    def test_the_matrix_is_built_from_collapsed_outcomes(self) -> None:
        """Three repeats of one case contribute one cell, not three."""
        outcomes = [
            outcome(
                "c-1",
                repeat,
                scores={"route": verdict(True, 1.0, expected="job_search", predicted="job_search")},
            )
            for repeat in range(3)
        ]
        assert build_confusion(outcomes)["job_search"]["job_search"] == 1

    def test_an_empty_run_gives_an_empty_matrix(self) -> None:
        assert build_confusion([]) == {}

    def test_a_case_without_route_detail_is_not_in_the_matrix(self) -> None:
        """The confusion matrix is about routing. A suite with no route
        expectation has no matrix rather than a matrix of empty strings."""
        outcomes = [outcome("c-1", scores={"facts": verdict(True, 1.0)})]
        assert build_confusion(outcomes) == {}
