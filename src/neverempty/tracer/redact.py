"""Redaction, applied before anything reaches a sink.

Never after. A sink that briefly sees a real email has already written it
somewhere, and "we redact on export" is how user data ends up in a committed
fixture.

A matched value becomes ``[REDACTED:<6 hex>]``, where the suffix is a truncated
SHA-256 of the value. That keeps one useful property: two spans holding the
same value show the same suffix, so a cache-key or persona-bleed bug stays
debuggable without the value ever being written. The key itself is kept, so the
trace never claims a field was absent when it was only hidden.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

Redactor: TypeAlias = Callable[[Any], Any]
"""Takes a JSON-shaped value, returns it with sensitive values replaced."""

PLACEHOLDER_PREFIX = "[REDACTED"

DEFAULT_KEYS: tuple[str, ...] = (
    "email",
    "phone",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "aadhaar",
    "pan",
    "ssn",
    "credit_card",
)
"""Defaults, not a guarantee. Reviewing what you commit is still your job."""

_MAX_DEPTH = 20
"""Deeply nested arguments are truncated rather than recursed forever."""


def placeholder(value: Any) -> str:
    """The replacement for one redacted value, stable for equal values."""
    digest = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:6]
    return f"{PLACEHOLDER_PREFIX}:{digest}]"


def keys(*names: str, extend_defaults: bool = False) -> Redactor:
    """Redact any value whose key contains one of ``names``.

    Matching is case-insensitive and on substrings, so ``email`` also catches
    ``userEmail`` and ``EMAIL_ADDRESS``. That is deliberately broad: a missed
    key writes real data to disk, while an over-redacted one costs only
    debuggability.

    Args:
        names: Key fragments to redact.
        extend_defaults: Also apply :data:`DEFAULT_KEYS`.
    """
    wanted = {name.casefold() for name in names}
    if extend_defaults:
        wanted |= {name.casefold() for name in DEFAULT_KEYS}

    def matches(key: str) -> bool:
        folded = key.casefold()
        return any(name in folded for name in wanted)

    def redactor(value: Any) -> Any:
        return _walk(value, matches, depth=0)

    return redactor


def defaults() -> Redactor:
    """Redact :data:`DEFAULT_KEYS`."""
    return keys(*DEFAULT_KEYS)


def nothing() -> Redactor:
    """Redact nothing. Explicit, so an absent redactor is never a silent one."""
    return lambda value: value


def _walk(value: Any, matches: Callable[[str], bool], depth: int) -> Any:
    if depth > _MAX_DEPTH:
        return "[depth capped]"
    if isinstance(value, dict):
        return {
            key: placeholder(item)
            if isinstance(key, str) and matches(key)
            else _walk(item, matches, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_walk(item, matches, depth + 1) for item in value]
    return value


def combine(*redactors: Redactor) -> Redactor:
    """Apply several redactors in order."""

    def redactor(value: Any) -> Any:
        for step in redactors:
            value = step(value)
        return value

    return redactor


def assert_clean(payload: str, secrets: Iterable[str]) -> None:
    """Raise if any of ``secrets`` appears in ``payload``.

    Exposed so a target repo can assert the same property over its own
    committed traces in its own test suite.
    """
    found = [secret for secret in secrets if secret and secret in payload]
    if found:
        raise AssertionError(
            f"{len(found)} unredacted value(s) reached serialized output; "
            f"redaction must run before the sink"
        )


__all__ = [
    "DEFAULT_KEYS",
    "PLACEHOLDER_PREFIX",
    "Redactor",
    "assert_clean",
    "combine",
    "defaults",
    "keys",
    "nothing",
    "placeholder",
]
