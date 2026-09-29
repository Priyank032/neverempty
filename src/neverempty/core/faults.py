"""Declared tool faults, applied by the ``@tool`` wrapper.

Real timeouts almost never happen inside a 280-case eval run, so the natural
misreport-as-empty rate is measured on zero opportunities. Faults are how that
number becomes measurable: they are declared in the case, applied through a
context variable, and require no change to agent code.

A fault that produces an error never calls the real dependency, so a fault run
costs nothing extra and touches no external system.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

FaultKind: TypeAlias = Literal["timeout", "upstream", "rate_limit", "empty", "truncated"]
"""Matches ``faults[].kind`` in the case schema."""

_ERROR_FAULTS: dict[str, tuple[str, bool, str]] = {
    # kind -> (ErrorKind, retryable, message)
    "timeout": ("timeout", True, "Injected fault: the tool timed out."),
    "upstream": ("upstream", True, "Injected fault: the upstream service failed."),
    "rate_limit": ("rate_limit", True, "Injected fault: the tool was rate limited."),
}


@dataclass(frozen=True)
class FaultSpec:
    """One declared fault. Mirrors a ``faults[]`` entry in a case."""

    tool: str
    kind: FaultKind
    after_calls: int = 0
    """Let this many calls through first. 0 hits the very first call."""

    def __post_init__(self) -> None:
        if self.after_calls < 0:
            raise ValueError(f"after_calls must be >= 0, got {self.after_calls}")


@dataclass
class _FaultState:
    """Active faults plus per-tool call counters for one scope."""

    specs: dict[str, FaultSpec] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def take(self, tool_name: str) -> FaultSpec | None:
        """Record a call to ``tool_name`` and return the fault to apply, if any."""
        spec = self.specs.get(tool_name)
        if spec is None:
            return None
        index = self.counts.get(tool_name, 0)
        self.counts[tool_name] = index + 1
        return spec if index >= spec.after_calls else None


_active: contextvars.ContextVar[_FaultState | None] = contextvars.ContextVar(
    "neverempty_faults", default=None
)


@contextmanager
def fault_scope(specs: Sequence[FaultSpec]) -> Iterator[None]:
    """Apply ``specs`` for the duration of the block.

    State is per-context, so concurrent eval cases keep separate call counters
    and never contaminate each other. Counters start fresh on every entry, so
    re-running the same case is repeatable.
    """
    state = _FaultState(specs={spec.tool: spec for spec in specs})
    token = _active.set(state)
    try:
        yield
    finally:
        _active.reset(token)


def next_fault(tool_name: str) -> FaultSpec | None:
    """Return the fault to apply to this call of ``tool_name``, if any.

    Called once per tool invocation by the wrapper, before the function body
    runs. Advances that tool's call counter as a side effect.
    """
    state = _active.get()
    return None if state is None else state.take(tool_name)


def error_fault_details(kind: FaultKind) -> tuple[str, bool, str] | None:
    """The ``(ErrorKind, retryable, message)`` an error-shaped fault produces.

    Returns ``None`` for ``empty`` and ``truncated``, which are not errors.
    """
    return _ERROR_FAULTS.get(kind)


__all__ = ["FaultKind", "FaultSpec", "error_fault_details", "fault_scope", "next_fault"]
