"""Markdown rendering for reports and gate results.

The renderer reads only the report structure: metrics, counts, confusion, costs,
latency and env. It never touches outcomes to recompute a number. That is the
constraint that makes a README trustworthy, because a table in a README can only
contain numbers that are in a committed report.

Three refusals, all of them about not printing a number that cannot be defended:

- ``applicable=0`` prints "not measured", never 0%.
- ``n < 10`` prints the counts without a percentage. At n=9 a percentage carries
  almost no information and invites a claim nobody can support.
- ``n < 50`` prints the percentage labelled indicative, which is what the doc
  requires for per-branch figures.
"""

from __future__ import annotations

from collections.abc import Sequence

from toolproof.metrics.reliability import (
    BUCKET_COUNT,
    ReliabilityCurve,
    curve_from_outcomes,
)
from toolproof.report.gate import GateResult
from toolproof.report.report import Metric, Report

SUPPRESS_BELOW_N = 10
"""Below this n, a percentage is not printed at all."""

INDICATIVE_MIN_N = 50
"""Below this n, a percentage is printed but labelled indicative."""

NOT_MEASURED = "not measured"
TOO_FEW = "too few to report a rate"

OVERCONFIDENCE_EPSILON = 0.005
"""Below half a percentage point, the curve is not called either way.

Naming a direction for a gap that rounds to 0.0% would invite a reader to act on
floating-point noise.
"""


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _metric_row(metric: Metric) -> tuple[str, str, str, str]:
    """One table row: name, value, interval, note."""
    if metric.applicable == 0 or metric.value is None:
        note = metric.note or "no case in this run declared the expectation"
        return metric.name, NOT_MEASURED, "—", note

    if metric.n < SUPPRESS_BELOW_N:
        successes = round(metric.value * metric.n)
        return (
            metric.name,
            f"{successes}/{metric.n}",
            "—",
            f"{TOO_FEW} (n={metric.n})",
        )

    interval = (
        f"{_percent(metric.ci_low)} to {_percent(metric.ci_high)}"
        if metric.ci_low is not None and metric.ci_high is not None
        else "—"
    )
    notes: list[str] = []
    if metric.method:
        notes.append(metric.method)
    if metric.n < INDICATIVE_MIN_N:
        notes.append(f"indicative only at n={metric.n}")
    if metric.note:
        notes.append(metric.note)

    return metric.name, _percent(metric.value), interval, "; ".join(notes) or "—"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _reproducibility(report: Report) -> list[str]:
    """The block that lets someone else reproduce the number.

    Every field that is absent says so. A blank would read as "clean", which is
    the opposite of what an unrecorded git sha means.
    """
    env = report.env
    sha = env.target_git_sha or "not recorded"
    if env.target_git_sha and env.target_dirty:
        sha = f"{env.target_git_sha} (dirty working tree: not reproducible)"

    models = ", ".join(sorted(env.resolved_models)) if env.resolved_models else "not recorded"
    rows = [
        ("target commit", sha),
        ("resolved model", models),
        ("toolproof", env.toolproof_version),
        ("pricing table", env.pricing_version),
        ("python", env.python_version),
        ("mode", env.mode),
        ("seed", str(env.seed) if env.seed is not None else "not set"),
        ("concurrency", str(env.concurrency)),
    ]
    if env.fault_profile:
        rows.append(("fault profile", env.fault_profile))
    if report.split_hash:
        rows.append(("test split hash", report.split_hash))

    return ["## Reproducibility", "", *_table(("field", "value"), rows), ""]


def _confusion(report: Report) -> list[str]:
    """Rows are labels, columns are predictions.

    A row with fewer than ten cases gets counts only: a 2-of-3 row printed as
    67% is a number that reads like a measurement and is not one.
    """
    if not report.confusion:
        return []

    columns = sorted({column for row in report.confusion.values() for column in row})
    rows: list[list[str]] = []
    for label in sorted(report.confusion):
        row = report.confusion[label]
        total = sum(row.values())
        cells = [label]
        for column in columns:
            count = row.get(column, 0)
            if total >= SUPPRESS_BELOW_N and count:
                cells.append(f"{count} ({_percent(count / total)})")
            else:
                cells.append(str(count))
        cells.append(str(total))
        rows.append(cells)

    return [
        "## Confusion matrix",
        "",
        "Rows are labels, columns are predictions. Rows with fewer than "
        f"{SUPPRESS_BELOW_N} cases show counts only.",
        "",
        *_table(("label", *columns, "n"), rows),
        "",
    ]


def _cost_and_latency(report: Report) -> list[str]:
    costs = report.costs
    if costs.total_usd is None:
        cost_line = (
            f"Cost unknown for {costs.unknown_count} trace(s); no total is reported, "
            f"because an unknown cost is never counted as zero."
        )
    else:
        cost_line = f"Total ${costs.total_usd:.4f}"
        if costs.mean_usd is not None:
            cost_line += f", mean ${costs.mean_usd:.4f} per trace"
        if costs.unknown_count:
            cost_line += f" ({costs.unknown_count} trace(s) of unknown cost excluded)"

    latency = report.latency
    p50 = f"{latency.p50_ms}ms" if latency.p50_ms is not None else NOT_MEASURED
    p95 = f"{latency.p95_ms}ms" if latency.p95_ms is not None else NOT_MEASURED
    latency_line = (
        f"p50 {p50}, p95 {p95} over {len(latency.samples_ms)} trace(s) at "
        f"concurrency {latency.concurrency}. Concurrency is stated because p95 "
        f"under {latency.concurrency}-way concurrency is not production p95."
    )

    return ["## Cost and latency", "", cost_line, "", latency_line, ""]


def render_markdown(report: Report) -> str:
    """Render one report as Markdown."""
    counts = report.counts
    lines: list[str] = [
        f"# {report.suite} ({report.split})",
        "",
        f"Run {report.created_at} · suite version {report.suite_version} · "
        f"status **{report.status}**",
        "",
    ]

    if not report.complete:
        lines += [
            f"> **Incomplete run.** Status `{report.status}`: "
            f"{counts.unscored} case(s) were not scored. The gate treats an "
            f"incomplete run as a failure, never as a smaller sample.",
            "",
        ]

    lines += [
        f"{counts.cases} case(s), {counts.repeats} repeat(s) each, "
        f"{counts.scored} scored, {counts.unscored} unscored, "
        f"{counts.unstable} unstable across repeats.",
        "",
    ]

    unscored = report.unscored_ids()
    if unscored:
        shown = ", ".join(f"`{case_id}`" for case_id in unscored[:20])
        suffix = f" (and {len(unscored) - 20} more)" if len(unscored) > 20 else ""
        lines += [f"Unscored cases: {shown}{suffix}", ""]

    if report.metrics:
        rows = [_metric_row(metric) for metric in sorted(report.metrics, key=lambda m: m.name)]
        lines += [
            "## Metrics",
            "",
            *_table(("metric", "value", "95% CI", "notes"), rows),
            "",
        ]

    lines += _confusion(report)

    curve = curve_from_outcomes(report.outcomes)
    if curve.measured:
        # Rendered only when some case stated a confidence. A "not measured"
        # calibration section on every report that never asked for one would
        # train the reader to skip the section that matters.
        lines += render_reliability(curve).splitlines()

    lines += _cost_and_latency(report)

    if report.judge.model:
        judge = report.judge
        kappa = f"{judge.kappa:.3f}" if judge.kappa is not None else NOT_MEASURED
        lines += [
            "## Judge",
            "",
            f"Model `{judge.model}`, prompt version `{judge.prompt_version}`, "
            f"agreement with human labels (Cohen's kappa) {kappa}.",
            "",
        ]

    lines += _reproducibility(report)
    return "\n".join(lines).rstrip("\n") + "\n"


_VERDICT_HEADLINES: dict[str, str] = {
    "pass": "Gate passed",
    "regression": "Gate failed: significant regression",
    "must_pass_failed": "Gate failed: a must-pass case did not pass",
    "inconclusive": "Gate inconclusive: rerun or reduce noise",
    "invalid": "Gate could not run: invalid input (infrastructure failure)",
}


def render_gate(result: GateResult) -> str:
    """Render a gate verdict for a PR comment or a CI job summary.

    The wording distinguishes the four failure codes on purpose. An inconclusive
    run says "rerun", and invalid input says "infrastructure", so neither gets
    read as the agent having got worse.
    """
    lines: list[str] = [
        f"# {_VERDICT_HEADLINES.get(result.verdict, result.verdict)}",
        "",
        f"`exit {result.exit_code}` — {result.reason}",
        "",
    ]

    if result.mcnemar is not None:
        test = result.mcnemar
        lines += [
            "## Paired test (exact McNemar, one-sided)",
            "",
            f"{test.b} case(s) went pass to fail, {test.c} went fail to pass. "
            f"Exact p = {test.p_value:.4g} against alpha {test.alpha}. "
            f"{'Significant.' if test.significant else 'Not significant.'}",
            "",
        ]

    if result.must_pass_failures:
        shown = ", ".join(f"`{case_id}`" for case_id in result.must_pass_failures[:20])
        lines += ["## Must-pass failures", "", shown, ""]

    if result.breached_floors:
        lines += [
            "## Floors breached",
            "",
            *(f"- {breach}" for breach in result.breached_floors),
            "",
        ]

    if result.comparison is not None:
        comparison = result.comparison
        lines += [
            "## Flips",
            "",
            f"{comparison.regressed_count} regressed, "
            f"{comparison.improved_count} improved, "
            f"over {comparison.paired} paired case(s).",
            "",
        ]
        if comparison.regressed_shown:
            lines += [
                "Regressed: " + ", ".join(f"`{case_id}`" for case_id in comparison.regressed_shown),
                "",
            ]

    if result.warnings:
        lines += [
            "## Warnings (not build failures)",
            "",
            *(f"- {note}" for note in result.warnings),
            "",
        ]

    lines += [
        f"Unstable rate {result.unstable_rate:.1%}.",
        "",
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def render_reliability(curve: ReliabilityCurve) -> str:
    """The reliability curve as a Markdown section.

    Two conventions carried over from the metric table, for the same reasons:

    - A bucket below ``SUPPRESS_BELOW_N`` prints its hit count, not a
      percentage. "0%" from two predictions reads as a band where the agent is
      always wrong, and it is not.
    - Empty buckets are omitted rather than printed as zero. A band nobody
      predicted into has no accuracy at all.

    The bucket count travels with the ECE because ECE is a function of the
    bucketing: two runs bucketed differently are not comparable, and a bare
    number invites exactly that comparison.
    """
    lines = ["## Calibration", ""]
    if not curve.measured or curve.ece is None or curve.overconfidence is None:
        lines += [
            f"Calibration: {NOT_MEASURED} — no case in this run stated a confidence.",
            "",
        ]
        return "\n".join(lines)

    direction = ""
    if abs(curve.overconfidence) >= OVERCONFIDENCE_EPSILON:
        word = "overconfident" if curve.overconfidence > 0 else "underconfident"
        direction = (
            f" The agent is {word} by "
            f"{_percent(abs(curve.overconfidence))} on average"
            f" (stated {_percent(curve.mean_confidence or 0.0)},"
            f" accurate {_percent(curve.accuracy or 0.0)})."
        )

    lines += [
        f"Expected calibration error (ECE): **{curve.ece:.3f}** "
        f"over {BUCKET_COUNT} buckets, n={curve.n}." + direction,
        "",
    ]

    rows: list[list[str]] = []
    for bucket in curve.buckets:
        if bucket.n < SUPPRESS_BELOW_N:
            accuracy = f"{bucket.hits}/{bucket.n}"
            note = f"{TOO_FEW} (n={bucket.n})"
        else:
            accuracy = _percent(bucket.accuracy)
            note = "—"
        rows.append(
            [
                bucket.label,
                str(bucket.n),
                accuracy,
                _percent(bucket.mean_confidence),
                f"{bucket.gap:.3f}",
                note,
            ]
        )

    lines += _table(
        ["confidence", "n", "accuracy", "stated", "gap", "note"],
        rows,
    )
    lines += [
        "",
        f"Maximum bucket gap (MCE): {curve.mce:.3f}. Brier score: {curve.brier:.4f}.",
        "",
        "ECE depends on the bucketing, so it is comparable only against another "
        f"curve built with the same {BUCKET_COUNT} buckets.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "INDICATIVE_MIN_N",
    "NOT_MEASURED",
    "SUPPRESS_BELOW_N",
    "TOO_FEW",
    "render_gate",
    "render_markdown",
    "render_reliability",
]
