"""The Report model: one run of one suite.

Structured exactly as the doc's top-level key list, so M7 fills ``metrics``,
``confusion`` and ``judge`` without the runner changing. Two rules matter more
than the shape:

- A metric that could not be computed is reported with ``applicable=0`` and
  printed as "not measured", never as ``0``.
- ``complete`` is true only when every case has a scored outcome. A run with
  any unscored case is ``incomplete``, and the gate treats that as a failure of
  the run rather than as a smaller sample.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neverempty.core.trace import Env, Trace, TraceError, TraceStatus

REPORT_SCHEMA_VERSION = 1

ReportStatus: TypeAlias = Literal["ok", "degraded", "aborted_budget", "incomplete"]


class Score(BaseModel):
    """One scorer's verdict on one trace.

    ``passed=None`` is a numeric-only metric, not a failure. A scorer that
    cannot decide returns ``None`` from ``score()`` instead of constructing
    this, so "not applicable" never reaches a report as a false.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool | None = None
    value: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class CaseOutcome(BaseModel):
    """Everything known about one case at one repeat."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    repeat: int = Field(ge=0)
    trace_id: str | None = None
    trace_status: TraceStatus = "ok"
    scored: bool = False
    scores: dict[str, Score] = Field(default_factory=dict)
    not_applicable: list[str] = Field(default_factory=list)
    """Scorers that returned ``None``: the expectation was absent or undecidable."""
    scorer_errors: dict[str, str] = Field(default_factory=dict)
    """Scorers that raised. The case is unscored for these only."""
    error: TraceError | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    """``None`` when no trace was produced, never 0: an unmeasured case and a
    sub-millisecond one are different facts, and a fabricated 0 would enter the
    latency sample and drag p50 below anything that was actually observed."""
    cost_usd: float | None = None
    cost_unknown_reason: str | None = None
    must_pass: bool = False
    tags: list[str] = Field(default_factory=list)

    @property
    def key(self) -> tuple[str, int]:
        return (self.case_id, self.repeat)


class Counts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: int = 0
    repeats: int = 1
    scored: int = 0
    unscored: int = 0
    unstable: int = 0
    crashed: int = 0
    """Repeats whose target raised before producing an answer.

    A crashed repeat leaves the denominator -- it cannot be scored as a wrong
    answer -- but without this count that rule let a run where 10 of 15 repeats
    crashed report ``complete=True``, ``status="ok"`` and a metric of 1.0. The
    surviving repeats were measured honestly; what was missing was any signal
    that most of the run never happened."""


class Metric(BaseModel):
    """One reported number, with everything needed to read it honestly."""

    model_config = ConfigDict(extra="forbid")

    name: str
    n: int = Field(ge=0)
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    method: str | None = None
    applicable: int = Field(default=0, ge=0)
    note: str | None = None

    @model_validator(mode="after")
    def _inapplicable_has_no_value(self) -> Metric:
        if self.applicable == 0 and self.value is not None:
            raise ValueError(
                f"metric {self.name!r} has applicable=0 but a value of {self.value}; "
                f"a metric with nothing to measure is 'not measured', never a number"
            )
        return self


class Costs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_usd: float | None = None
    mean_usd: float | None = None
    unknown_count: int = Field(default=0, ge=0)
    pricing_version: str = ""
    budget_usd: float | None = None


class Latency(BaseModel):
    """Per-trace wall-clock, kept rather than collapsed.

    ``concurrency`` travels with these, because p95 under 8-way concurrency is
    not production p95 and a report that omits it invites the confusion.
    """

    model_config = ConfigDict(extra="forbid")

    samples_ms: list[int] = Field(default_factory=list)
    p50_ms: int | None = None
    p95_ms: int | None = None
    concurrency: int = 1


class JudgeInfo(BaseModel):
    """Filled in M8. Kappa travels next to every judge-derived number."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    prompt_version: str | None = None
    kappa: float | None = None
    error_rate: float | None = None


class Report(BaseModel):
    """One run of one suite."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = REPORT_SCHEMA_VERSION
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    suite: str
    suite_version: int = 1
    split: str
    created_at: str
    complete: bool = False
    status: ReportStatus = "incomplete"
    env: Env
    config_hash: str | None = None
    counts: Counts = Field(default_factory=Counts)
    metrics: list[Metric] = Field(default_factory=list)
    outcomes: list[CaseOutcome] = Field(default_factory=list)
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    judge: JudgeInfo = Field(default_factory=JudgeInfo)
    costs: Costs = Field(default_factory=Costs)
    latency: Latency = Field(default_factory=Latency)
    resumed_from: str | None = None
    split_hash: str | None = None
    traces: list[Trace] = Field(default_factory=list, exclude=True)
    """In-memory only. Traces live in their own JSONL, not inside the report."""

    @model_validator(mode="after")
    def _ordering_is_canonical(self) -> Report:
        """Sorted by ``(case_id, repeat)``, so two runs produce diffable JSON."""
        self.outcomes.sort(key=lambda outcome: outcome.key)
        return self

    def unscored_ids(self) -> list[str]:
        """Case ids with no scored outcome, listed in the report as the doc requires."""
        scored = {o.case_id for o in self.outcomes if o.scored}
        return sorted({o.case_id for o in self.outcomes} - scored)

    def save(self, path: str | Path) -> Path:
        """Write the report as indented JSON with a trailing newline."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8", newline="\n")
        return target

    def to_json(self) -> str:
        payload = json.loads(self.model_dump_json())
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    @classmethod
    def load(cls, path: str | Path) -> Report:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "CaseOutcome",
    "Costs",
    "Counts",
    "JudgeInfo",
    "Latency",
    "Metric",
    "Report",
    "ReportStatus",
    "Score",
]
