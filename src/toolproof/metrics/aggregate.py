"""Turning case outcomes into the report's metrics, confusion and counts.

The layer between the scorers and the report. Two rules govern everything here:

- Repeats collapse to one observation per case *before* aggregation. Counting
  three repeats as three cases would inflate every n and shrink every interval,
  making a 70-case suite look like a 210-case one.
- A metric with nothing to measure carries no value. ``Metric`` enforces that
  too, but it is enforced here first so the failure is a missing number rather
  than a validation error deep in a run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from toolproof.metrics.stats import bootstrap_ci, wilson_interval
from toolproof.report.report import CaseOutcome, Metric, Score
from toolproof.scorers.collapse import collapse, collapse_rule

CONFUSION_UNSCORED = "unscored"
"""The extra confusion column the doc requires.

A case the harness could not read is not a misrouting. Folding it into a
prediction cell would make the rows sum to fewer than the case count with no
visible reason.
"""

CONTINUOUS_METRICS = frozenset({"facts"})
"""Metrics that are a mean of per-case fractions rather than a rate of successes.

Wilson describes a binomial proportion, so it does not apply to these; the doc
specifies a seeded bootstrap instead.
"""


def _by_case(outcomes: Sequence[CaseOutcome]) -> dict[str, list[CaseOutcome]]:
    grouped: dict[str, list[CaseOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.case_id, []).append(outcome)
    return grouped


def collapsed_scores(outcomes: Sequence[CaseOutcome]) -> dict[str, dict[str, Score]]:
    """One score per (case, scorer), collapsed over that case's repeats.

    A scorer absent from the result for a case is one that never applied there,
    which is what keeps it out of the denominator.
    """
    result: dict[str, dict[str, Score]] = {}
    for case_id, repeats in _by_case(outcomes).items():
        names = {name for repeat in repeats for name in repeat.scores}
        per_scorer: dict[str, Score] = {}
        for name in names:
            verdicts = [repeat.scores[name] for repeat in repeats if name in repeat.scores]
            merged = collapse(name, verdicts)
            if merged is not None:
                per_scorer[name] = merged
        result[case_id] = per_scorer
    return result


def build_metrics(
    outcomes: Sequence[CaseOutcome],
    *,
    seed: int,
    scorer_names: Iterable[str] | None = None,
) -> list[Metric]:
    """One metric per scorer, sorted by name so two runs diff cleanly.

    ``scorer_names`` declares the scorers that were configured, so a scorer that
    applied to no case still appears with ``applicable=0``. Omitting it entirely
    would let a suite silently stop measuring something.
    """
    collapsed = collapsed_scores(outcomes)
    names = set(scorer_names or ())
    names.update(name for scores in collapsed.values() for name in scores)

    metrics: list[Metric] = []
    for name in sorted(names):
        scores = [scores[name] for scores in collapsed.values() if name in scores]
        metrics.append(_metric_for(name, scores, seed=seed))
    return metrics


def _metric_for(name: str, scores: Sequence[Score], *, seed: int) -> Metric:
    if not scores:
        return Metric(
            name=name,
            n=0,
            applicable=0,
            note="not measured: no case in this run declared the expectation",
        )

    verdicts = [score.passed for score in scores if score.passed is not None]
    values = [score.value for score in scores if score.value is not None]

    if name in CONTINUOUS_METRICS or not verdicts:
        interval = bootstrap_ci(values, seed=seed) if values else None
    else:
        interval = wilson_interval(sum(1 for v in verdicts if v), len(verdicts))

    if interval is None:
        # Scores existed but carried neither a verdict nor a value, so there is
        # still nothing to report as a number.
        return Metric(
            name=name,
            n=len(scores),
            applicable=0,
            note="not measured: no scorer produced a verdict or a value",
        )

    return Metric(
        name=name,
        n=interval.n,
        value=interval.value,
        ci_low=interval.low,
        ci_high=interval.high,
        method=interval.method,
        applicable=len(scores),
        note=f"collapse={collapse_rule(name)}",
    )


def count_unstable(outcomes: Sequence[CaseOutcome]) -> int:
    """Cases whose outcome differs across repeats, in any scorer.

    High instability blocks trusting any single-run delta, which is why it is a
    reported number and a gate condition rather than something smoothed away.
    """
    unstable = 0
    for repeats in _by_case(outcomes).values():
        if len(repeats) < 2:
            continue
        names = {name for repeat in repeats for name in repeat.scores}
        for name in names:
            verdicts = {repeat.scores[name].passed for repeat in repeats if name in repeat.scores}
            if len(verdicts - {None}) > 1:
                unstable += 1
                break
    return unstable


def case_verdicts(
    outcomes: Sequence[CaseOutcome], *, primary: str | None = None
) -> dict[str, bool | None]:
    """One pass/fail per case, for the paired test and for ``must_pass``.

    With ``primary`` set, that metric alone decides: the paired test compares one
    outcome per case, and a conjunction over every metric would mix a routing
    regression with a fact-recall one and attribute both to whichever changed.

    Without it, a case passes only when every applicable scorer passed, which is
    what ``must_pass`` means.

    ``None`` means undecidable, never a failure. Whether an undecidable case
    fails the *run* is the gate's decision, not this function's.
    """
    collapsed = collapsed_scores(outcomes)
    scored_cases = {o.case_id for o in outcomes if o.scored}

    verdicts: dict[str, bool | None] = {}
    for case_id, scores in collapsed.items():
        if case_id not in scored_cases:
            verdicts[case_id] = None
            continue
        if primary is not None:
            score = scores.get(primary)
            verdicts[case_id] = None if score is None else score.passed
            continue
        decided = [score.passed for score in scores.values() if score.passed is not None]
        verdicts[case_id] = all(decided) if decided else None
    return verdicts


def build_confusion(
    outcomes: Sequence[CaseOutcome],
    *,
    expected_labels: Mapping[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    """Rows are labels, columns are predictions, built from collapsed outcomes.

    ``expected_labels`` supplies the label for a case that produced no route
    score, so an unscored case still appears in its own row rather than vanishing
    from the matrix.
    """
    collapsed = collapsed_scores(outcomes)
    labels = dict(expected_labels or {})
    matrix: dict[str, dict[str, int]] = {}

    for case_id, scores in collapsed.items():
        route = scores.get("route")
        if route is not None:
            expected = route.detail.get("expected")
            predicted = route.detail.get("predicted")
            if isinstance(expected, str) and isinstance(predicted, str):
                matrix.setdefault(expected, {}).setdefault(predicted, 0)
                matrix[expected][predicted] += 1
                continue

        expected_label = labels.get(case_id)
        if isinstance(expected_label, str):
            matrix.setdefault(expected_label, {}).setdefault(CONFUSION_UNSCORED, 0)
            matrix[expected_label][CONFUSION_UNSCORED] += 1

    return {label: dict(sorted(row.items())) for label, row in sorted(matrix.items())}


__all__ = [
    "CONFUSION_UNSCORED",
    "CONTINUOUS_METRICS",
    "build_confusion",
    "build_metrics",
    "case_verdicts",
    "collapsed_scores",
    "count_unstable",
]
