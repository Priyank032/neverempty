"""The arguments scorer: per-argument and all-args-exact accuracy.

Seven match modes, declared per argument in the dataset. There is deliberately
no ``judge`` mode: an argument that needs semantic matching means the dataset is
underspecified, and a judge would paper over that rather than fix it.

Two things must not read as agent errors, and both are handled by returning
"not applicable" with a reason:

- An adapter that recorded no arguments, or recorded them unparseably. That is
  a harness gap.
- A value the redactor replaced. Redaction is a configuration choice, and
  scoring it as a mismatch would blame the agent for the harness.
"""

from __future__ import annotations

from typing import Any

from neverempty.dataset.case import ArgExpectation, Case
from neverempty.report.report import Score
from neverempty.scorers.match import (
    as_datetime,
    as_member_list,
    as_number,
    is_redacted,
    matches,
    normalize,
    same_members,
)
from neverempty.scorers.reading import first_call_of, recorded_args

_MISSING = object()
"""Sentinel for an absent argument.

``None`` cannot serve here: an argument explicitly passed as null is a real
value that a case may expect, and conflating the two would make ``present``
unable to tell "omitted" from "passed as null".
"""


class ArgumentsScorer:
    """Did the agent pass the right arguments to the right tool?"""

    name = "arguments"
    requires = frozenset({"expect.tool_calls"})

    def score(self, case: Case, trace: object) -> Score | None:
        expectation = case.expect.tool_calls
        if expectation is None:
            return None

        expected_call = next((call for call in expectation.calls if call.args), None)
        if expected_call is None:
            # The case asked which tool, not which arguments. A different
            # scorer answers that question.
            return None

        span = first_call_of(trace, expected_call.tool)  # type: ignore[arg-type]
        if span is None:
            # A tool that was never called cannot have had correct arguments.
            # This is a genuine failure, not a missing measurement.
            return Score(
                passed=False,
                value=0.0,
                detail={"tool": expected_call.tool, "tool_called": False},
            )

        actual = recorded_args(span)
        if actual is None:
            return None

        results: dict[str, dict[str, Any]] = {}
        skipped: list[str] = []
        for name, expected in expected_call.args.items():
            value = actual.get(name, _MISSING)
            verdict, note = _compare(expected, value)
            entry: dict[str, Any] = {"match": expected.match, "passed": verdict}
            if value is not _MISSING:
                entry["actual"] = value
            if note is not None:
                entry["note"] = note
            results[name] = entry
            if verdict is None:
                skipped.append(name)

        judged = [entry["passed"] for entry in results.values() if entry["passed"] is not None]
        if not judged:
            # Every scored argument was unreadable, so nothing was measured.
            return None

        all_exact = all(judged)
        detail: dict[str, Any] = {
            "tool": expected_call.tool,
            "tool_called": True,
            "arguments": results,
            "all_args_exact": all_exact,
        }
        if skipped:
            detail["skipped"] = skipped

        # Redacted and unreadable arguments leave the denominator rather than
        # counting as failures: otherwise turning on redaction would silently
        # lower every score in the suite.
        return Score(
            passed=all_exact,
            value=sum(1 for verdict in judged if verdict) / len(judged),
            detail=detail,
        )


def _compare(expected: ArgExpectation, value: object) -> tuple[bool | None, str | None]:
    """One argument, under one mode. ``None`` means undecidable, never a fail."""
    if expected.match == "present":
        # A question about the key, not about the value being interesting:
        # "", [] and 0 are all present. Falsiness inference is exactly what
        # this library exists to forbid.
        if value is _MISSING:
            return False, "absent"
        if value is None:
            return False, "null"
        return True, None

    if value is _MISSING:
        return False, "absent"
    if is_redacted(value):
        return None, "redacted"

    if expected.match == "exact":
        return value == expected.value, None

    if expected.match == "normalized":
        return normalize(expected.value) == normalize(value), None

    if expected.match == "set":
        members = as_member_list(value)
        if members is None:
            return False, "not a list"
        wanted = as_member_list(expected.value) or []
        return same_members(wanted, members), None

    if expected.match == "numeric":
        actual_number = as_number(value)
        if actual_number is None:
            return False, "not numeric"
        wanted_number = as_number(expected.value)
        if wanted_number is None:
            return False, "expected value is not numeric"
        tolerance = expected.tol if expected.tol is not None else 0.0
        return abs(actual_number - wanted_number) <= tolerance, None

    if expected.match == "regex":
        text = value if isinstance(value, str) else str(value)
        return matches(str(expected.value), text), None

    # date: parsed and timezone-aware, so 05:30 IST equals 00:00 UTC.
    actual_date = as_datetime(value)
    if actual_date is None:
        return False, "not a date"
    wanted_date = as_datetime(expected.value)
    if wanted_date is None:
        return False, "expected value is not a date"
    return actual_date == wanted_date, None


def arguments() -> ArgumentsScorer:
    """The arguments scorer."""
    return ArgumentsScorer()


__all__ = ["ArgumentsScorer", "arguments"]
