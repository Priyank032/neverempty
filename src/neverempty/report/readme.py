"""Rendering the README's numbers from committed reports.

The doc's M11 row: "every README number links to a committed report". This makes
that a property of the code rather than of anyone's discipline -- the section is
generated from report files, and a number with no report behind it cannot be
produced at all, because there is nothing to produce it from.

Every rule the report renderer already enforces is inherited here, plus two that
only matter once a number leaves the repository:

- A judge-derived number is printed only when kappa clears the publishing
  threshold. Below it the doc is explicit: those numbers are cut and only the
  deterministic checks are published.
- Every row carries its sample size, interval, resolved model, date and git sha,
  and links to the report it came from. A number without a path back to the run
  that produced it is a claim, not a measurement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from neverempty.judge.calibration import KAPPA_PUBLISH_THRESHOLD
from neverempty.report.gate import JUDGE_DERIVED_METRICS
from neverempty.report.render import (
    INDICATIVE_MIN_N,
    SUPPRESS_BELOW_N,
    TOO_FEW,
)
from neverempty.report.report import Report

NO_NUMBERS = (
    "No measured numbers yet. This section is generated from committed reports, "
    "so it stays empty until a run produces one."
)
"""What the section says when there is nothing to publish.

Stated rather than left blank: an empty section reads as an oversight, and this
is a deliberate refusal.
"""


@dataclass(frozen=True)
class PublishedRow:
    """One number, with everything needed to check it."""

    suite: str
    metric: str
    value: str
    interval: str
    n: int
    report_path: str
    note: str


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _rows(report: Report, path: str) -> list[PublishedRow]:
    kappa = report.judge.kappa
    judge_publishable = kappa is not None and kappa >= KAPPA_PUBLISH_THRESHOLD

    rows: list[PublishedRow] = []
    for metric in sorted(report.metrics, key=lambda m: m.name):
        if metric.applicable == 0 or metric.value is None:
            # Not measured. It is reported inside the repo by the report
            # renderer; the README publishes results, not absences.
            continue

        if metric.name in JUDGE_DERIVED_METRICS and not judge_publishable:
            # The doc is explicit: below kappa 0.6 the judge-derived numbers are
            # cut and only the deterministic checks are published.
            continue

        if metric.n < SUPPRESS_BELOW_N:
            value = f"{round(metric.value * metric.n)}/{metric.n}"
            interval = "—"
            note = f"{TOO_FEW} (n={metric.n})"
        else:
            value = _percent(metric.value)
            interval = (
                # An en dash is the range separator a published table wants; a
                # hyphen reads as a minus sign next to percentages.
                f"{_percent(metric.ci_low)} – {_percent(metric.ci_high)}"  # noqa: RUF001
                if metric.ci_low is not None and metric.ci_high is not None
                else "—"
            )
            note = metric.method or "—"
            if metric.n < INDICATIVE_MIN_N:
                note = f"{note}; indicative only at n={metric.n}"

        if metric.name in JUDGE_DERIVED_METRICS and kappa is not None:
            note = f"{note}; judge kappa {kappa:.3f}"

        rows.append(
            PublishedRow(
                suite=report.suite,
                metric=metric.name,
                value=value,
                interval=interval,
                n=metric.n,
                report_path=path,
                note=note,
            )
        )
    return rows


def _provenance(report: Report, path: str) -> list[str]:
    env = report.env
    sha = env.target_git_sha or "unrecorded"
    dirty = " (dirty)" if env.target_dirty else ""
    models = ", ".join(f"`{model}`" for model in _models(report)) or "unrecorded"
    return [
        f"- **{report.suite}** ({report.split}, suite version "
        f"{report.suite_version}) — [report]({path}), run {report.created_at}, "
        f"target `{sha}`{dirty}, models {models}, "
        f"pricing `{env.pricing_version}`, neverempty `{env.neverempty_version}`.",
    ]


def _models(report: Report) -> list[str]:
    """The model ids the run actually resolved to.

    A declared field on ``Env``, not an extra: a published number has to name
    the snapshot it was measured against, so this is part of the schema rather
    than something an adapter may or may not attach.
    """
    return [str(model) for model in report.env.resolved_models]


def render_readme_numbers(
    reports: Sequence[tuple[Report, str]],
) -> str:
    """The README's Numbers section, built from committed reports.

    ``reports`` pairs each report with the repository-relative path it is
    committed at, which becomes the link. The path is passed rather than derived
    so that a report loaded from anywhere still has to declare where a reader can
    find it: a link to a file nobody committed is worse than no link.
    """
    lines = ["## Numbers", ""]

    publishable = [
        (report, path)
        for report, path in reports
        if report.complete and report.status != "incomplete"
    ]
    refused = [(report, path) for report, path in reports if (report, path) not in publishable]

    rows: list[PublishedRow] = []
    for report, path in publishable:
        rows.extend(_rows(report, path))

    if not rows:
        lines += [NO_NUMBERS, ""]
    else:
        lines += [
            "Every number below is rendered from a committed report and links "
            "back to it. Each carries its sample size and 95% interval.",
            "",
            "| suite | metric | value | 95% CI | n | notes |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            lines.append(
                f"| [{row.suite}]({row.report_path}) | `{row.metric}` | "
                f"{row.value} | {row.interval} | {row.n} | {row.note} |"
            )
        lines += ["", "### Reproducibility", ""]
        for report, path in publishable:
            lines += _provenance(report, path)
        lines.append("")

    if refused:
        lines += [
            "### Not published",
            "",
        ]
        for report, path in refused:
            lines.append(
                f"- **{report.suite}** ({path}): status `{report.status}`, "
                f"{report.counts.unscored} case(s) unscored. An incomplete run is "
                f"a failure of the run, never a smaller sample, so nothing from it "
                f"is published."
            )
        lines.append("")

    return "\n".join(lines)


def load_reports(paths: Sequence[str | Path]) -> list[tuple[Report, str]]:
    """Load each report, pairing it with the path it was read from."""
    loaded: list[tuple[Report, str]] = []
    for path in paths:
        target = Path(path)
        loaded.append((Report.load(target), target.as_posix()))
    return loaded


__all__ = [
    "NO_NUMBERS",
    "PublishedRow",
    "load_reports",
    "render_readme_numbers",
]
