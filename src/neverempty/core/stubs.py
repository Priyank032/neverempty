"""Recording stubs for side-effecting tools.

The runner refuses to start unless every tool tagged ``side_effect=True`` has a
stub bound. That is a safety preflight, not a convenience: an end-to-end eval
run must never be able to email a real recruiter or write to a real database.

The mechanism mirrors ``fault_scope`` deliberately — a registry plus a
contextvar the wrapper consults before the body — so a stub needs no change to
agent code and cannot leak between concurrent cases.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

_side_effect_tools: set[str] = set()

_active: contextvars.ContextVar[Mapping[str, Callable[..., Any]] | None] = contextvars.ContextVar(
    "neverempty_stubs", default=None
)


def register_side_effect(name: str) -> None:
    """Record that ``name`` touches the outside world.

    Called by the ``@tool`` decorator at definition time, so the runner knows
    the full set before any case executes.
    """
    _side_effect_tools.add(name)


def registered_side_effect_tools() -> frozenset[str]:
    """Every tool tagged ``side_effect=True`` that has been imported."""
    return frozenset(_side_effect_tools)


def clear_side_effect_registry() -> None:
    """Empty the registry. For tests; never call this from a runner."""
    _side_effect_tools.clear()


@contextmanager
def stub_scope(stubs: Mapping[str, Callable[..., Any]]) -> Iterator[None]:
    """Replace the named tools' bodies for the duration of the block.

    Per-context, so concurrent eval cases cannot see each other's stubs, and a
    crash mid-run cannot leave a real tool swapped out.
    """
    token = _active.set(dict(stubs))
    try:
        yield
    finally:
        _active.reset(token)


def find_stub(name: str) -> Callable[..., Any] | None:
    """The stub bound to ``name`` in this context, if any."""
    stubs = _active.get()
    return None if stubs is None else stubs.get(name)


__all__ = [
    "clear_side_effect_registry",
    "find_stub",
    "register_side_effect",
    "registered_side_effect_tools",
    "stub_scope",
]
