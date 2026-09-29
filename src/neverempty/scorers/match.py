"""Matching primitives shared by the scorers.

Normalisation lives here rather than in each scorer because the arguments
scorer and the facts scorer must agree about what "the same string" means. If
they disagreed, a suite could report a fact as found and its argument as wrong
for the same value, and nobody would be able to say which number was right.

NFKC is the doc's choice, not NFC. It folds compatibility characters, so a
full-width digit and an ASCII digit compare equal, and a decomposed Devanagari
nukta matches its precomposed form. Those are the differences that actually
appear between a human-typed label and a model-produced argument.
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

_WHITESPACE = re.compile(r"\s+")

REDACTED = re.compile(r"^\[REDACTED:[0-9a-f]{6}\]$")
"""The tracer's placeholder shape, exactly.

Matched strictly so a value that merely mentions redaction stays a value. A
loose check would silently stop scoring arguments whose content is a quoted
log line.
"""


def is_redacted(value: object) -> bool:
    """Whether a recorded value was replaced by the redactor."""
    return isinstance(value, str) and REDACTED.match(value) is not None


def normalize(value: object) -> str:
    """Casefold, NFKC-normalise and collapse whitespace.

    ``casefold`` rather than ``lower``: it is the Unicode-correct operation and
    handles scripts where the two differ.
    """
    text = value if isinstance(value, str) else str(value)
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", folded).strip()


def as_number(value: object) -> float | None:
    """Parse a number, or ``None`` when the value is not one.

    ``bool`` is rejected deliberately. Python says ``True == 1``, which would
    make a boolean argument pass a numeric expectation of 1 and hide a real
    type confusion in the agent's tool call.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def as_datetime(value: object) -> datetime | None:
    """Parse an ISO 8601 date or datetime into an aware UTC datetime.

    A naive datetime is read as UTC rather than as local time. A scorer whose
    verdict depends on the CI runner's timezone is not a measurement.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def as_member_list(value: object) -> list[Any] | None:
    """A list for ``set`` comparison, or ``None`` when the value is not one.

    A string is refused even though it is iterable: comparing ``"python"`` to
    ``["python"]`` by characters would produce a confidently wrong verdict.
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _member_key(item: object) -> str:
    """A comparison key that works for unhashable members.

    Dataset values arrive from JSON, so a list member can be a dict. Sorting
    the canonical form keeps the comparison order-free without requiring the
    members to be hashable.
    """
    if isinstance(item, dict):
        return "{" + ",".join(f"{key}={_member_key(item[key])}" for key in sorted(item)) + "}"
    if isinstance(item, (list, tuple)):
        return "[" + ",".join(sorted(_member_key(entry) for entry in item)) + "]"
    return f"{type(item).__name__}:{item}"


def same_members(expected: list[Any], actual: list[Any]) -> bool:
    """Order-free, duplicate-insensitive comparison of two lists."""
    return {_member_key(item) for item in expected} == {_member_key(item) for item in actual}


def contains(needle: str, haystack: str) -> bool:
    """Substring search after normalisation, so Hindi encodings agree."""
    return normalize(needle) in normalize(haystack)


def matches(pattern: str, text: str) -> bool:
    """Case-insensitive regex search.

    The pattern compiled at dataset load time, so a bad pattern already failed
    validation. This can still raise nothing: a compiled pattern searches.
    """
    return re.search(pattern, text, re.IGNORECASE | re.UNICODE) is not None


__all__ = [
    "REDACTED",
    "as_datetime",
    "as_member_list",
    "as_number",
    "contains",
    "is_redacted",
    "matches",
    "normalize",
    "same_members",
]
