"""The seam between ``@tool`` and the tracer.

``core`` must not import ``tracer``: a tool decorator that drags the tracing
stack in at import time would make wrapper rule 7 ("outside a tracer context
the wrapper still works and simply does not record") impossible to honour
cheaply. So the lookup is deferred to first use and cached.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

_open_span: Callable[[str], Any] | None = None
_redactor_source: Callable[[], Any] | None = None


def open_tool_span(name: str) -> tuple[Any, Any] | None:
    """Open a tool span, or ``None`` when nothing is recording.

    Returns ``(handle, finish)``. The wrapper calls ``finish(result, attempt)``
    once the outcome is known, including on the failure and fault paths, so a
    span is never left open.
    """
    global _open_span
    if _open_span is None:
        from neverempty.tracer.tracer import open_tool_span as impl

        _open_span = impl
    result: tuple[Any, Any] | None = _open_span(name)
    return result


def bind_arguments(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Name the positional arguments, so a trace shows ``{"city": "Pune"}``
    rather than ``["Pune"]``.

    Falls back to a positional map if the signature cannot be bound, because a
    tracing helper must never be the thing that breaks a tool call.
    """
    try:
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        named: dict[str, Any] = {f"arg{i}": value for i, value in enumerate(args)}
        named.update(kwargs)
        return named


def redacted_args(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Arguments as a named map.

    Redaction itself happens in the tracer, immediately before the sink, so
    there is exactly one place where a value can escape rather than two.
    """
    return bind_arguments(func, args, kwargs)


__all__ = ["bind_arguments", "open_tool_span", "redacted_args"]
