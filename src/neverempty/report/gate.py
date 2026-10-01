"""Compare two reports, and decide whether a build fails.

The gate fails a build for three reasons only: a must-pass case failed, the
paired test shows a significant regression, or a hard floor was breached.
Latency, cost and small accuracy wobbles are warnings. That restraint is the
design: a gate that fires on noise gets disabled within a month, which leaves
the project with no gate at all.

Exit codes, and why they are ordered this way:

- ``4`` invalid input, and it outranks everything. A quality verdict computed
  from a broken input is worse than no verdict.
- ``2`` a must-pass case failed. A stated safety requirement, so it outranks a
  statistical claim nobody can argue with a p-value about.
- ``3`` inconclusive. If the run is too noisy to attribute a delta to the
  change, the delta must not be reported as a regression.
- ``1`` a significant regression or a breached floor.
- ``0`` pass.
"""

from __future__ import annotations

import json
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from neverempty.metrics.aggregate import build_metrics, case_verdicts, count_unstable
from neverempty.metrics.stats import McNemarResult, mcnemar_exact
from neverempty.report.report import Metric, Report

FLIP_LIST_CAP = 20
"""The doc's cap on printed flip lists. The *counts* are never capped."""

JUDGE_DERIVED_METRICS = frozenset({"facts", "forbidden_claims"})
"""Metrics a judge can decide.

On a ``degraded`` run these are excluded from the gate, per the doc. The run
still gates on its deterministic metrics: a flaky judge is not a reason to stop
checking routing, and treating it as one would make one bad provider call block
a build whose real measurements were fine.
"""

MAX_SUFFIX = "_max"
"""Marks a ceiling in ``floors``.

The doc's TOML writes ``misreport_as_empty_max = 0.10`` beside
``route_strict = 0.75``, so the suffix is the documented convention and a config
copied from the doc has to work. Everything without it is a minimum.
"""

Verdict: TypeAlias = Literal["pass", "regression", "must_pass_failed", "inconclusive", "invalid"]

EXIT_CODES: dict[Verdict, int] = {
    "pass": 0,
    "regression": 1,
    "must_pass_failed": 2,
    "inconclusive": 3,
    "invalid": 4,
}


class GateConfig(BaseModel):
    """Gate thresholds, from ``[gate]`` in the config file."""

    model_config = ConfigDict(extra="forbid")

    paired_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    max_unstable_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    floors: dict[str, float] = Field(default_factory=dict)
    """Hard levels. ``name`` is a minimum, ``name_max`` is a maximum."""
    warn: dict[str, float] = Field(default_factory=dict)
    must_pass: Literal["fail_on_any"] = "fail_on_any"  # noqa: S105 - a policy, not a secret
    primary: str | None = None
    """The metric the paired test compares.

    Without it, a case passes only when every applicable scorer passed. With it,
    that one metric decides: McNemar compares one outcome per case, and a
    conjunction over eight metrics would mix a routing regression with a
    fact-recall one and attribute both to whichever changed.
    """
    seed: int = 20260921


class MetricDelta(BaseModel):
    """One metric, before and after, with both intervals."""

    model_config = ConfigDict(extra="forbid")

    name: str
    base_value: float | None = None
    candidate_value: float | None = None
    delta: float | None = None
    base_ci_low: float | None = None
    base_ci_high: float | None = None
    candidate_ci_low: float | None = None
    candidate_ci_high: float | None = None
    base_n: int = 0
    candidate_n: int = 0


class CompareResult(BaseModel):
    """What ``neverempty compare`` produces."""

    model_config = ConfigDict(extra="forbid")

    refused: bool = False
    refusal_reason: str = ""
    paired: int = 0
    unpaired_base: int = 0
    unpaired_candidate: int = 0
    deltas: list[MetricDelta] = Field(default_factory=list)
    mcnemar: McNemarDTO | None = None
    regressed: list[str] = Field(default_factory=list)
    improved: list[str] = Field(default_factory=list)
    cost_delta_usd: float | None = None
    p95_delta_ms: int | None = None

    @property
    def regressed_count(self) -> int:
        return len(self.regressed)

    @property
    def improved_count(self) -> int:
        return len(self.improved)

    @property
    def regressed_shown(self) -> list[str]:
        """Capped for printing. ``regressed_count`` stays the true number, so a
        reader cannot conclude that only 20 cases regressed."""
        return self.regressed[:FLIP_LIST_CAP]

    @property
    def improved_shown(self) -> list[str]:
        return self.improved[:FLIP_LIST_CAP]


class McNemarDTO(BaseModel):
    """The paired test's numbers, serializable into a report or CI summary."""

    model_config = ConfigDict(extra="forbid")

    b: int
    c: int
    p_value: float
    significant: bool
    alpha: float

    @classmethod
    def of(cls, result: McNemarResult) -> McNemarDTO:
        return cls(
            b=result.b,
            c=result.c,
            p_value=result.p_value,
            significant=result.significant,
            alpha=result.alpha,
        )


class GateResult(BaseModel):
    """The gate's verdict, with everything a CI summary needs to explain it."""

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    exit_code: int
    reason: str
    mcnemar: McNemarDTO | None = None
    unstable_rate: float = 0.0
    must_pass_failures: list[str] = Field(default_factory=list)
    breached_floors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    comparison: CompareResult | None = None

    def to_json(self) -> str:
        payload = json.loads(self.model_dump_json())
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _resolved_models(report: Report) -> list[str]:
    return sorted(report.env.resolved_models)


def _pairing_problem(base: Report, candidate: Report, *, allow_model_change: bool) -> str:
    """Why these two reports cannot be compared, or ``""`` when they can."""
    if base.suite != candidate.suite:
        return f"suite differs: {base.suite!r} vs {candidate.suite!r}"
    if base.suite_version != candidate.suite_version:
        return (
            f"suite_version differs: {base.suite_version} vs {candidate.suite_version}. "
            f"Editing the test split bumps the version, which invalidates the baseline."
        )
    if not allow_model_change:
        base_models, candidate_models = _resolved_models(base), _resolved_models(candidate)
        if base_models and candidate_models and base_models != candidate_models:
            return (
                f"resolved model differs: {base_models} vs {candidate_models}. "
                f"Pass --allow-model-change to compare anyway; the delta will "
                f"include the model change as well as the change under review."
            )
    return ""


def _incomplete_problem(base: Report, candidate: Report) -> str:
    """Refuse an incomplete run, and refuse one that only claims to be complete.

    A report is a committed artifact handed between CI jobs, and ``split_hash``
    and ``config_hash`` are null unless a runner filled them, so ``complete``
    has no integrity check behind it. A hand-edit or a mangled merge that sets
    it true made the gate compare unscored pairs against each other and count
    them as non-regressions -- a pass, exit 0, on a run that scored two cases
    out of five. So the flag is checked against the outcomes it summarizes
    instead of being believed.
    """
    for label, report in (("baseline", base), ("candidate", candidate)):
        if not report.complete:
            return (
                f"{label} report is not complete (status={report.status!r}); "
                f"an incomplete run is a failure, never a smaller sample"
            )
        # An unscored ``must_pass`` case is left to ``_must_pass_failures``,
        # which exits 2: a safety requirement the harness could not prove is a
        # failure on the merits, not an infrastructure complaint. The doc scopes
        # code 4 to a report that *declares* complete=false, so downgrading a
        # must_pass failure to 4 would hide the more serious verdict.
        must_pass_ids = {o.case_id for o in report.outcomes if o.must_pass}
        unscored = [cid for cid in report.unscored_ids() if cid not in must_pass_ids]
        if unscored:
            shown = ", ".join(unscored[:5])
            more = f" and {len(unscored) - 5} more" if len(unscored) > 5 else ""
            return (
                f"{label} report says complete=true but {len(unscored)} case(s) "
                f"have no scored outcome ({shown}{more}); the report contradicts "
                f"itself, so no verdict is computed from it"
            )
        claimed = report.counts.unscored
        if claimed and not must_pass_ids:
            return (
                f"{label} report says complete=true but counts.unscored is "
                f"{claimed}; the report contradicts itself, so no verdict is "
                f"computed from it"
            )
    return ""


def compare(
    base: Report,
    candidate: Report,
    *,
    allow_model_change: bool = False,
    primary: str | None = None,
    alpha: float = 0.05,
    seed: int = 20260921,
) -> CompareResult:
    """Pair two reports on ``case_id`` and diff them."""
    problem = _incomplete_problem(base, candidate) or _pairing_problem(
        base, candidate, allow_model_change=allow_model_change
    )
    if problem:
        return CompareResult(refused=True, refusal_reason=problem)

    base_verdicts = case_verdicts(base.outcomes, primary=primary)
    candidate_verdicts = case_verdicts(candidate.outcomes, primary=primary)
    shared = sorted(set(base_verdicts) & set(candidate_verdicts))

    if not shared:
        return CompareResult(
            refused=True,
            refusal_reason=(
                "no cases to pair: the two reports share no case_id, so the "
                "paired test cannot be computed"
            ),
        )

    regressed = [
        case_id
        for case_id in shared
        if base_verdicts[case_id] is True and candidate_verdicts[case_id] is False
    ]
    improved = [
        case_id
        for case_id in shared
        if base_verdicts[case_id] is False and candidate_verdicts[case_id] is True
    ]

    return CompareResult(
        paired=len(shared),
        unpaired_base=len(set(base_verdicts) - set(candidate_verdicts)),
        unpaired_candidate=len(set(candidate_verdicts) - set(base_verdicts)),
        deltas=_deltas(base, candidate, seed=seed),
        mcnemar=McNemarDTO.of(mcnemar_exact(len(regressed), len(improved), alpha=alpha)),
        regressed=regressed,
        improved=improved,
        cost_delta_usd=_delta_or_none(base.costs.total_usd, candidate.costs.total_usd),
        p95_delta_ms=_int_delta_or_none(base.latency.p95_ms, candidate.latency.p95_ms),
    )


def _delta_or_none(before: float | None, after: float | None) -> float | None:
    """A delta only when both sides are known.

    ``None`` minus ``None`` is not a zero change: an unknown cost on both sides
    means the delta is unknown, and reporting 0 would claim the cost held steady.
    """
    if before is None or after is None:
        return None
    return after - before


def _int_delta_or_none(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before


def _deltas(base: Report, candidate: Report, *, seed: int) -> list[MetricDelta]:
    base_metrics = {m.name: m for m in _metrics_of(base, seed=seed)}
    candidate_metrics = {m.name: m for m in _metrics_of(candidate, seed=seed)}

    deltas: list[MetricDelta] = []
    for name in sorted(set(base_metrics) | set(candidate_metrics)):
        before = base_metrics.get(name)
        after = candidate_metrics.get(name)
        deltas.append(
            MetricDelta(
                name=name,
                base_value=before.value if before else None,
                candidate_value=after.value if after else None,
                delta=_delta_or_none(
                    before.value if before else None, after.value if after else None
                ),
                base_ci_low=before.ci_low if before else None,
                base_ci_high=before.ci_high if before else None,
                candidate_ci_low=after.ci_low if after else None,
                candidate_ci_high=after.ci_high if after else None,
                base_n=before.n if before else 0,
                candidate_n=after.n if after else 0,
            )
        )
    return deltas


def _metrics_of(report: Report, *, seed: int) -> list[Metric]:
    """The report's own metrics, or freshly computed ones.

    A report written by this library already carries them. One assembled by hand,
    or by an older version, may not, and recomputing from outcomes keeps the
    comparison possible rather than refusing on a technicality.
    """
    if report.metrics:
        return report.metrics
    return build_metrics(report.outcomes, seed=seed)


def _must_pass_failures(report: Report, *, primary: str | None) -> list[str]:
    """Must-pass cases that did not demonstrably pass.

    An *unscored* must-pass case counts as a failure: the harness could not prove
    the requirement held, and a safety requirement unproven is unmet.
    """
    flagged = {o.case_id for o in report.outcomes if o.must_pass}
    if not flagged:
        return []
    verdicts = case_verdicts(report.outcomes, primary=primary)
    return sorted(case_id for case_id in flagged if verdicts.get(case_id) is not True)


def _floor_breaches(
    report: Report, config: GateConfig, *, base: Report | None = None
) -> tuple[list[str], list[str], list[str]]:
    """Breached floors, floors that vanished, and floors not evaluated.

    ``base`` lets a floor tell two different facts apart that otherwise wear
    the same shape -- see ``vanished`` below.
    """
    metrics = {m.name: m for m in _metrics_of(report, seed=config.seed)}
    base_metrics = (
        {m.name: m for m in _metrics_of(base, seed=config.seed)} if base is not None else {}
    )
    breaches: list[str] = []
    vanished: list[str] = []
    warnings: list[str] = []
    degraded = report.status == "degraded"

    for key, threshold in sorted(config.floors.items()):
        is_max = key.endswith(MAX_SUFFIX)
        name = key[: -len(MAX_SUFFIX)] if is_max else key
        metric = metrics.get(name)

        if degraded and name in JUDGE_DERIVED_METRICS:
            # The judge errored often enough to taint its own numbers, so this
            # floor is not evaluated. The deterministic floors still apply.
            warnings.append(
                f"floor {key!r} was not evaluated: the run is degraded "
                f"(judge error rate {report.judge.error_rate}), so judge-derived "
                f"metrics are excluded from the gate"
            )
            continue

        if metric is None or metric.value is None:
            base_metric = base_metrics.get(name)
            if base_metric is not None and base_metric.value is not None:
                # The baseline measured it and the candidate does not. Something
                # the agent used to do, it has stopped doing -- an agent that
                # produces no route has failed routing, which is not the same
                # claim as "this suite does not test routing". Skipping the floor
                # here is the headline bug pointed the other way: a failure
                # rendered as an absence, passing a build at exit 0.
                vanished.append(
                    f"{name!r} was measured in the baseline "
                    f"({base_metric.value:.4f}) and is no longer measured in the "
                    f"candidate, so floor {key!r} cannot be checked"
                )
                continue
            # Never measured on either side: nothing was measured and nothing
            # regressed, so failing the build would invent a result. Treating
            # this as a breach would be this library's own headline bug:
            # missing must never look like failure.
            warnings.append(
                f"floor {key!r} could not be evaluated: metric {name!r} was not measured"
            )
            continue

        if is_max and metric.value > threshold:
            breaches.append(f"{name}={metric.value:.4f} exceeds max {threshold}")
        elif not is_max and metric.value < threshold:
            breaches.append(f"{name}={metric.value:.4f} below floor {threshold}")

    return breaches, vanished, warnings


def _threshold_warnings(base: Report, candidate: Report, config: GateConfig) -> list[str]:
    warnings: list[str] = []

    latency_limit = config.warn.get("p95_latency_increase")
    if latency_limit is not None:
        before, after = base.latency.p95_ms, candidate.latency.p95_ms
        if before and after and after > before * (1 + latency_limit):
            warnings.append(
                f"p95 latency rose from {before}ms to {after}ms "
                f"(above the {latency_limit:.0%} warning threshold)"
            )

    cost_limit = config.warn.get("cost_increase")
    if cost_limit is not None:
        before_cost, after_cost = base.costs.total_usd, candidate.costs.total_usd
        if before_cost and after_cost and after_cost > before_cost * (1 + cost_limit):
            warnings.append(
                f"cost rose from ${before_cost:.4f} to ${after_cost:.4f} "
                f"(above the {cost_limit:.0%} warning threshold)"
            )

    return warnings


def gate(
    base: Report,
    candidate: Report,
    config: GateConfig,
    *,
    allow_model_change: bool = False,
) -> GateResult:
    """Decide whether the candidate fails the build."""
    problem = _incomplete_problem(base, candidate) or _pairing_problem(
        base, candidate, allow_model_change=allow_model_change
    )
    if problem:
        return _invalid(problem)

    comparison = compare(
        base,
        candidate,
        allow_model_change=allow_model_change,
        primary=config.primary,
        alpha=config.paired_alpha,
        seed=config.seed,
    )
    if comparison.refused:
        return _invalid(comparison.refusal_reason)

    breaches, vanished, floor_warnings = _floor_breaches(candidate, config, base=base)
    warnings = [*floor_warnings, *_threshold_warnings(base, candidate, config)]

    cases = len({o.case_id for o in candidate.outcomes})
    unstable_rate = count_unstable(candidate.outcomes) / cases if cases else 0.0

    must_pass_failures = _must_pass_failures(candidate, primary=config.primary)
    if must_pass_failures:
        return GateResult(
            verdict="must_pass_failed",
            exit_code=EXIT_CODES["must_pass_failed"],
            reason=(
                f"{len(must_pass_failures)} must_pass case(s) did not pass: "
                f"{', '.join(must_pass_failures[:FLIP_LIST_CAP])}"
            ),
            mcnemar=comparison.mcnemar,
            unstable_rate=unstable_rate,
            must_pass_failures=must_pass_failures,
            breached_floors=breaches,
            warnings=warnings,
            comparison=comparison,
        )

    if vanished:
        # After must_pass, which is a proven failure of a named case, and before
        # the statistical verdict, which cannot be computed for a metric that is
        # not there. Reported as invalid input rather than a regression because
        # no significance test produced it: the honest statement is "the number
        # the floor guards is gone", not "the agent got worse by this much".
        return _invalid(
            "a floored metric is no longer measured: " + "; ".join(vanished),
            mcnemar=comparison.mcnemar,
            unstable_rate=unstable_rate,
            breaches=breaches,
            warnings=warnings,
            comparison=comparison,
        )

    if unstable_rate > config.max_unstable_rate:
        return GateResult(
            verdict="inconclusive",
            exit_code=EXIT_CODES["inconclusive"],
            reason=(
                f"inconclusive, rerun or reduce noise: {unstable_rate:.1%} of cases "
                f"are unstable across repeats, above the "
                f"{config.max_unstable_rate:.1%} max_unstable_rate limit. No delta "
                f"can be attributed to this change. With 3 repeats this fires at "
                f"roughly 3.5% per-call nondeterminism, so an agent above that "
                f"cannot pass: set temperature=0 and a fixed seed, stub the "
                f"nondeterministic dependency, or raise max_unstable_rate "
                f"deliberately and record that the numbers are noisier."
            ),
            mcnemar=comparison.mcnemar,
            unstable_rate=unstable_rate,
            breached_floors=breaches,
            warnings=warnings,
            comparison=comparison,
        )

    regressed = comparison.mcnemar is not None and comparison.mcnemar.significant
    if regressed or breaches:
        reasons: list[str] = []
        if regressed and comparison.mcnemar is not None:
            reasons.append(
                f"significant regression: {comparison.mcnemar.b} case(s) went "
                f"pass to fail and {comparison.mcnemar.c} the other way "
                f"(exact p={comparison.mcnemar.p_value:.4g} < {config.paired_alpha})"
            )
        reasons.extend(f"floor breached: {breach}" for breach in breaches)
        return GateResult(
            verdict="regression",
            exit_code=EXIT_CODES["regression"],
            reason="; ".join(reasons),
            mcnemar=comparison.mcnemar,
            unstable_rate=unstable_rate,
            breached_floors=breaches,
            warnings=warnings,
            comparison=comparison,
        )

    return GateResult(
        verdict="pass",
        exit_code=EXIT_CODES["pass"],
        reason=(
            f"no significant regression across {comparison.paired} paired case(s); "
            f"{comparison.regressed_count} regressed, {comparison.improved_count} improved"
        ),
        mcnemar=comparison.mcnemar,
        unstable_rate=unstable_rate,
        warnings=warnings,
        comparison=comparison,
    )


def _invalid(
    reason: str,
    *,
    mcnemar: McNemarDTO | None = None,
    unstable_rate: float = 0.0,
    breaches: list[str] | None = None,
    warnings: list[str] | None = None,
    comparison: CompareResult | None = None,
) -> GateResult:
    """An infrastructure failure, exit 4.

    The optional fields carry whatever was computed before the run was judged
    invalid. A refusal that arrives before any comparison leaves them empty; one
    that arrives after should not discard the diagnostics it already has.
    """
    return GateResult(
        verdict="invalid",
        exit_code=EXIT_CODES["invalid"],
        reason=reason,
        mcnemar=mcnemar,
        unstable_rate=unstable_rate,
        breached_floors=breaches or [],
        warnings=warnings or [],
        comparison=comparison,
    )


__all__ = [
    "EXIT_CODES",
    "FLIP_LIST_CAP",
    "MAX_SUFFIX",
    "CompareResult",
    "GateConfig",
    "GateResult",
    "McNemarDTO",
    "MetricDelta",
    "Verdict",
    "compare",
    "gate",
]
