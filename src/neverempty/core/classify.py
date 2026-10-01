"""The default exception-to-``Err`` classification table.

Core cannot import httpx, openai or any provider SDK, so HTTP status codes are
read structurally: any exception carrying a ``status_code`` or ``status``
integer attribute is treated as an HTTP error. That covers httpx, requests
(via ``response.status_code``), and most provider SDKs without a dependency.
"""

from __future__ import annotations

import asyncio
import socket
import ssl
from collections.abc import Callable
from typing import TypeAlias

from neverempty.core.results import Err, ErrorKind

ClassifyError: TypeAlias = Callable[[BaseException], "Err | None"]
"""Override hook. Returning ``None`` falls through to the default table."""

# Exceptions that must never be converted into a result. Cancelling a run is
# not a tool failure, and turning a budget abort into an Err would corrupt
# every number computed downstream.
UNCATCHABLE: tuple[type[BaseException], ...] = (
    asyncio.CancelledError,
    KeyboardInterrupt,
    SystemExit,
    # Not in the doc's non-negotiable list, which names the three above, but it
    # belongs with them: Python throws GeneratorExit into a generator or
    # coroutine during teardown, so turning it into an Err corrupts the cleanup
    # it is part of. A tool cancelled mid-await that raised it while unwinding
    # returned Err(kind="exception") and the cancellation was lost.
    #
    # A *custom* BaseException subclass is deliberately not here: someone
    # subclassing BaseException for a domain error should still get a typed
    # failure, and intent cannot be read except by naming the ones Python
    # itself uses for control flow.
    GeneratorExit,
)

_RETRYABLE: frozenset[ErrorKind] = frozenset({"timeout", "upstream", "rate_limit"})


def is_uncatchable(exc: BaseException) -> bool:
    """True when ``exc`` must propagate untouched."""
    return isinstance(exc, UNCATCHABLE)


def _http_status(exc: BaseException) -> int | None:
    """Best-effort HTTP status, read structurally to avoid SDK dependencies."""
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int) and 100 <= status <= 599:
            return status
    return None


def _kind_from_status(status: int) -> ErrorKind:
    if status == 429:
        return "rate_limit"
    if status in (401, 403):
        return "permission"
    if 500 <= status <= 599:
        return "upstream"
    return "validation"


def _kind_from_type(exc: BaseException) -> ErrorKind:
    # TimeoutError and asyncio.TimeoutError are the same class on 3.11+.
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "timeout"
    if isinstance(exc, (ConnectionError, socket.gaierror, socket.herror, ssl.SSLError)):
        return "upstream"
    if isinstance(exc, PermissionError):
        return "permission"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        # pydantic's ValidationError subclasses ValueError, so it lands here.
        return "validation"
    if isinstance(exc, OSError):
        # After the specific OSError subclasses above: a socket-level failure.
        return "upstream"
    return "exception"


def classify(exc: BaseException) -> Err:
    """Map an exception onto ``Err`` using the default table.

    Records the exception's class name as ``cause``. The traceback is never
    recorded: it carries file paths and sometimes argument values.
    """
    if is_uncatchable(exc):  # pragma: no cover - callers check first
        raise exc

    status = _http_status(exc)
    kind = _kind_from_status(status) if status is not None else _kind_from_type(exc)

    return Err(
        kind=kind,
        message=_message_for(exc, status),
        retryable=kind in _RETRYABLE,
        cause=type(exc).__name__,
    )


def safe_str(exc: BaseException) -> str:
    """``str(exc)`` for an exception that may not survive being stringified.

    A wrapped dependency can raise an exception whose ``__str__`` itself
    raises -- ORM and gRPC error types that build their message lazily do this
    when the failure interrupted whatever the message needed. Letting that
    escape breaks the wrapper's one contract: a tool failure always becomes a
    typed failure. The type name is a worse message than the real one and a far
    better one than an unhandled ``RuntimeError`` from inside the guard.
    """
    try:
        return str(exc).strip()
    except BaseException as inner:
        if is_uncatchable(inner):
            raise
        return ""


def _message_for(exc: BaseException, status: int | None) -> str:
    text = safe_str(exc) or type(exc).__name__
    if status is not None and str(status) not in text:
        return f"HTTP {status}: {text}"
    return text


__all__ = ["UNCATCHABLE", "ClassifyError", "classify", "is_uncatchable", "safe_str"]
