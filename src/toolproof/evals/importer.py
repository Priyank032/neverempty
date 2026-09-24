"""Importing traces produced by another language.

The Node exporter writes Trace v1 JSONL. This reads it back, validates every
line against the same pydantic models the Python tracer uses, and pairs each
trace with its case. That shared schema is the whole cross-language contract:
the Node side validates against the committed JSON Schema in its own test, and
this validates again on the way in, so a drift is caught twice and never reaches
a published number.

Two checks live here rather than in the scorer, because they are properties of
the *export* rather than of the agent:

- Cache bypass. A cached explanation is written for a bucketed age and income
  range, so one persona can be served another persona's text. An import that
  cannot prove the cache was bypassed refuses, rather than quietly measuring
  contaminated data.
- LLM error rate. Above the ceiling, the export is not trustworthy enough to
  publish from, which is a different failure from the agent being wrong.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from toolproof.core.trace import Trace
from toolproof.dataset.case import Case
from toolproof.dataset.loader import Dataset, DatasetError

LLM_ERROR_RATE_CEILING = 0.10
"""Above this share of traces carrying an LLM error, the import refuses.

The same ceiling the Node exporter exits 3 on. Checked on both sides because an
export can be produced once and imported many times, and the second reader
should not have to trust the first one's exit code.
"""


@dataclass(frozen=True)
class ImportReport:
    """What was imported, and what was refused."""

    traces: list[Trace] = field(default_factory=list)
    cases: list[Case] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    llm_errors: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def llm_error_rate(self) -> float | None:
        """``None`` when nothing was imported, never 0.0.

        An empty import has no error rate; reporting zero would read as a clean
        run rather than as no run at all.
        """
        if not self.traces:
            return None
        return self.llm_errors / len(self.traces)

    def render(self) -> str:
        lines = [
            f"Imported {len(self.traces)} trace(s) and {len(self.cases)} case(s).",
        ]
        rate = self.llm_error_rate
        if rate is None:
            lines.append("LLM error rate: not measured (no traces).")
        else:
            lines.append(f"LLM error rate: {rate:.1%} ({self.llm_errors}/{len(self.traces)}).")
        if self.problems:
            lines += ["", "Problems:"] + [f"- {problem}" for problem in self.problems]
        return "\n".join(lines)


def _read_lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                yield number, stripped


def load_traces(path: str | Path) -> tuple[list[Trace], list[str]]:
    """Validate every line, collecting problems rather than stopping at the first.

    A malformed export usually has one systematic mistake repeated, and naming
    every line makes that obvious in one pass instead of five.
    """
    target = Path(path)
    if not target.exists():
        return [], [f"{target}: no such file"]

    traces: list[Trace] = []
    problems: list[str] = []
    for number, line in _read_lines(target):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"{target}:{number}: not valid JSON: {exc.msg}")
            continue
        try:
            traces.append(Trace.model_validate(payload))
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first["loc"])
            problems.append(f"{target}:{number}: {location}: {first['msg']}")

    if not traces and not problems:
        problems.append(f"{target}: holds no traces")
    return traces, problems


def _has_llm_error(trace: Trace) -> bool:
    return any(span.kind == "llm" and span.status == "error" for span in trace.spans)


def import_export(
    traces_path: str | Path,
    *,
    cases_path: str | Path | None = None,
    error_ceiling: float = LLM_ERROR_RATE_CEILING,
    require_cache_bypass: bool = True,
) -> ImportReport:
    """Read an exporter's output and decide whether it may be scored.

    ``require_cache_bypass`` is on by default and refuses an export that does
    not record the bypass. That is deliberately awkward: a cached run silently
    mixes personas within an age and income bucket, and the resulting
    consistency rate would look fine.
    """
    traces, problems = load_traces(traces_path)

    cases: list[Case] = []
    if cases_path is not None:
        try:
            cases = Dataset.load(cases_path).cases
        except DatasetError as exc:
            problems.append(f"{cases_path}: {exc}")

    llm_errors = sum(1 for trace in traces if _has_llm_error(trace))

    if traces:
        rate = llm_errors / len(traces)
        if rate > error_ceiling:
            problems.append(
                f"LLM error rate {rate:.1%} is above the {error_ceiling:.0%} ceiling "
                f"({llm_errors}/{len(traces)} traces). The export is not trustworthy "
                f"enough to publish from; that is a failure of the export, not of "
                f"the agent."
            )

    if require_cache_bypass and traces:
        # ``case_id`` is optional on a Trace, so a trace without one is named by
        # its trace_id rather than sorted alongside a None.
        missing = [_label(trace) for trace in traces if not _bypassed_cache(trace)]
        if missing:
            shown = ", ".join(repr(case_id) for case_id in sorted(set(missing))[:5])
            problems.append(
                f"{len(missing)} trace(s) do not record a cache bypass: {shown}. "
                f"MatchCache keys on bucketed age and income, so a cached run can "
                f"serve one persona an explanation written for another, which "
                f"contaminates exactly the rate being measured. Re-export with "
                f"--no-cache."
            )

    if cases:
        problems.extend(_pairing_problems(traces, cases))

    return ImportReport(
        traces=traces,
        cases=cases,
        problems=problems,
        llm_errors=llm_errors,
    )


def _bypassed_cache(trace: Trace) -> bool:
    """True when the export recorded that the cache was not consulted.

    Read from the env block or a run-level attribute. Absence is treated as "not
    proven" rather than "fine": the check exists precisely because a cached run
    looks identical to an uncached one in the numbers.
    """
    env_extra = trace.env.model_extra or {}
    if env_extra.get("cache_bypassed") is True:
        return True
    return any(
        span.attributes.get("cache.bypassed") is True for span in trace.spans if span.kind == "llm"
    )


def _label(trace: Trace) -> str:
    """How a trace is named in a problem message.

    A trace with no ``case_id`` still has to be identifiable, and it cannot be
    sorted next to a ``None``.
    """
    return trace.case_id or f"<trace {trace.trace_id}>"


def _pairing_problems(traces: Sequence[Trace], cases: Sequence[Case]) -> list[str]:
    case_ids = {case.id for case in cases}
    trace_ids = {trace.case_id for trace in traces if trace.case_id is not None}

    problems: list[str] = []
    orphan_traces = sorted(trace_ids - case_ids)
    orphan_cases = sorted(case_ids - trace_ids)
    if orphan_traces:
        shown = ", ".join(repr(case_id) for case_id in orphan_traces[:5])
        problems.append(
            f"{len(orphan_traces)} trace(s) reference a case that was not exported: "
            f"{shown}. A trace with no ground truth cannot be scored."
        )
    if orphan_cases:
        shown = ", ".join(repr(case_id) for case_id in orphan_cases[:5])
        problems.append(
            f"{len(orphan_cases)} case(s) have no trace: {shown}. Those cases are "
            f"unscored, which the gate treats as a failure of the run."
        )
    return problems


__all__ = [
    "LLM_ERROR_RATE_CEILING",
    "ImportReport",
    "import_export",
    "load_traces",
]
