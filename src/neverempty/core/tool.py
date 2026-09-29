"""The ``@tool`` decorator.

The wrapper behaves identically in production and under eval. That is not a
convenience: if the two differ, the eval numbers describe a different system
from the one users hit.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, Generic, Protocol, TypeVar, cast

from neverempty.core.classify import ClassifyError, classify, is_uncatchable
from neverempty.core.faults import FaultSpec, error_fault_details, next_fault
from neverempty.core.results import Empty, Err, Ok, ToolResult
from neverempty.core.stubs import (
    clear_side_effect_registry,
    find_stub,
    register_side_effect,
    registered_side_effect_tools,
)
from neverempty.core.tracing import open_tool_span, redacted_args

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class AmbiguousEmptyError(RuntimeError):
    """A tool returned an empty-looking value without declaring what empty means.

    This is an authoring bug, raised at the point where the original bug was
    born: the moment a tool can return ``[]`` without its author having decided
    whether that means "no matches" or "something went wrong".

    Fix it by declaring the intent:

    - ``@tool(empty_when=lambda rows: len(rows) == 0)`` when empty is a real
      outcome the model should be told about, or
    - ``@tool(never_empty=True)`` when an empty container is an ordinary value.
    """


class ToolCallable(Protocol, Generic[T_co]):
    """What ``@tool`` returns: the original callable, plus tool metadata."""

    neverempty_name: str
    neverempty_side_effect: bool
    __wrapped__: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def _is_empty_looking(value: object) -> bool:
    """Whether a value is the shape that makes the bug possible.

    Deliberately not ``not value``: ``0``, ``False`` and ``""`` are legitimate
    values and must never trip strict mode.
    """
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return len(value) == 0
    return False


def _predicate_error(name: str, exc: BaseException) -> Err:
    """A raising predicate is a validation error, never a silent Ok."""
    return Err(
        kind="validation",
        message=f"{name} predicate raised {type(exc).__name__}: {exc}",
        retryable=False,
        cause=type(exc).__name__,
    )


class _ToolSpec:
    """Per-tool configuration, resolved once at decoration time."""

    __slots__ = (
        "classify_error",
        "empty_when",
        "name",
        "never_empty",
        "retries",
        "retry_backoff_s",
        "side_effect",
        "strict",
        "timeout_s",
        "truncated_when",
    )

    def __init__(
        self,
        *,
        name: str,
        empty_when: Callable[[Any], bool] | None,
        truncated_when: Callable[[Any], bool] | None,
        never_empty: bool,
        strict: bool,
        timeout_s: float | None,
        retries: int,
        retry_backoff_s: float,
        side_effect: bool,
        classify_error: ClassifyError | None,
    ) -> None:
        self.name = name
        self.empty_when = empty_when
        self.truncated_when = truncated_when
        self.never_empty = never_empty
        self.strict = strict
        self.timeout_s = timeout_s
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s
        self.side_effect = side_effect
        self.classify_error = classify_error

    def classify(self, exc: BaseException) -> Err:
        if self.classify_error is not None:
            override = self.classify_error(exc)
            if override is not None:
                return override
        return classify(exc)

    def interpret(self, value: Any) -> ToolResult[Any]:
        """Turn a plain return value into a result.

        An explicit ``Ok``, ``Empty`` or ``Err`` passes through untouched.
        """
        if isinstance(value, (Ok, Empty, Err)):
            return value

        if self.empty_when is not None:
            try:
                is_empty = self.empty_when(value)
            except BaseException as exc:
                if is_uncatchable(exc):
                    raise
                return _predicate_error("empty_when", exc)
            if is_empty:
                return Empty()
        elif not self.never_empty and _is_empty_looking(value):
            message = (
                f"Tool {self.name!r} returned {value!r} but does not declare what "
                f"empty means. Add empty_when=... if an empty result is a real "
                f"outcome, or never_empty=True if it is an ordinary value."
            )
            if self.strict:
                raise AmbiguousEmptyError(message)
            # Non-strict: an empty-looking value is not silently promoted to
            # Empty. Claiming absence the author never declared is the bug.

        truncated = False
        if self.truncated_when is not None:
            try:
                truncated = bool(self.truncated_when(value))
            except BaseException as exc:
                if is_uncatchable(exc):
                    raise
                return _predicate_error("truncated_when", exc)

        return Ok(value=value, truncated=truncated)

    def apply_fault(self, spec: FaultSpec) -> ToolResult[Any] | None:
        """The result an injected fault produces, or ``None`` to run the tool.

        ``truncated`` returns ``None``: truncation is a property of a real
        result, so the tool runs and its output is flagged afterwards.
        """
        details = error_fault_details(spec.kind)
        if details is not None:
            kind, retryable, message = details
            return Err(
                kind=cast(Any, kind),
                message=message,
                retryable=retryable,
                cause="InjectedFault",
                fault_injected=True,
            )
        if spec.kind == "empty":
            return Empty(reason="Injected fault: no matching records.", fault_injected=True)
        return None

    def backoff_for(self, attempt: int) -> float:
        """Exponential backoff with jitter, so retries do not synchronise."""
        if self.retry_backoff_s <= 0:
            return 0.0
        jitter: float = 0.5 + random.random() / 2  # noqa: S311
        delay: float = self.retry_backoff_s * float(2**attempt) * jitter
        return delay


def tool(
    *,
    name: str | None = None,
    empty_when: Callable[[Any], bool] | None = None,
    truncated_when: Callable[[Any], bool] | None = None,
    never_empty: bool = False,
    strict: bool = True,
    timeout_s: float | None = None,
    retries: int = 0,
    retry_backoff_s: float = 0.1,
    side_effect: bool = False,
    classify_error: ClassifyError | None = None,
) -> Callable[[Callable[..., Any]], Any]:
    """Wrap a function so it returns a ``ToolResult`` and never raises.

    The wrapped function returns ``Ok``, ``Empty`` or ``Err``. Ordinary
    exceptions become ``Err``; ``CancelledError``, ``KeyboardInterrupt`` and
    ``SystemExit`` propagate untouched.

    Args:
        name: Tool name for traces and fault matching. Defaults to the
            function's name.
        empty_when: Predicate deciding whether a return value means "no
            matching records". This is the only way to produce ``Empty`` from a
            plain value: falsiness is never inferred, because ``0``, ``False``
            and ``""`` are legitimate values.
        truncated_when: Predicate marking a result as capped, so the model is
            told the set is partial rather than complete.
        never_empty: Declares that an empty container is an ordinary value for
            this tool. Mutually exclusive with ``empty_when``.
        strict: When true (the default), returning ``None``, ``[]`` or ``{}``
            without declaring ``empty_when`` or ``never_empty`` raises
            ``AmbiguousEmptyError``. Set false in production to keep serving;
            the ambiguity is then recorded on the span instead. Behaviour is
            otherwise identical, because an eval of a differently configured
            tool measures the wrong system.
        timeout_s: Deadline for one attempt. Async tools get a real timeout via
            ``asyncio.wait_for``. For sync tools the timeout is **advisory**:
            the wrapper returns ``Err(kind="timeout")`` once the deadline
            passes, but the function keeps running in its thread, because
            Python cannot safely interrupt arbitrary synchronous code.
        retries: Retry attempts after the first. Only ``retryable`` errors are
            retried; each attempt gets its own span.
        retry_backoff_s: Base delay for exponential backoff with jitter.
        side_effect: Marks a tool that touches the outside world. The runner
            refuses to start an eval unless every such tool is stubbed.
        classify_error: Per-tool override mapping an exception to ``Err``.
            Returning ``None`` falls through to the default table.

    Raises:
        ValueError: If ``empty_when`` and ``never_empty`` are both given.
    """
    if empty_when is not None and never_empty:
        raise ValueError(
            "Pass either empty_when or never_empty=True, not both: they are two "
            "answers to the same question."
        )
    if retries < 0:
        raise ValueError(f"retries must be >= 0, got {retries}")

    def decorate(func: Callable[..., Any]) -> Any:
        spec = _ToolSpec(
            name=name or func.__name__,
            empty_when=empty_when,
            truncated_when=truncated_when,
            never_empty=never_empty,
            strict=strict,
            timeout_s=timeout_s,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
            side_effect=side_effect,
            classify_error=classify_error,
        )
        wrapper: Any = (
            _build_async_wrapper(func, spec)
            if inspect.iscoroutinefunction(func)
            else _build_sync_wrapper(func, spec)
        )
        wrapper.neverempty_name = spec.name
        wrapper.neverempty_side_effect = spec.side_effect
        if spec.side_effect:
            register_side_effect(spec.name)
        return wrapper

    return decorate


def _build_async_wrapper(
    func: Callable[..., Awaitable[Any]], spec: _ToolSpec
) -> Callable[..., Awaitable[ToolResult[Any]]]:
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> ToolResult[Any]:
        stub = find_stub(spec.name)
        if stub is not None:
            return await _run_async_stub(spec, func, stub, args, kwargs)

        fault = next_fault(spec.name)
        if fault is not None:
            injected = spec.apply_fault(fault)
            if injected is not None:
                opened = open_tool_span(spec.name)
                if opened is not None:
                    handle, finish = opened
                    handle.set("tool.args", redacted_args(func, args, kwargs))
                    finish(injected, 0)
                return injected

        last: Err | None = None
        for attempt in range(spec.retries + 1):
            opened = open_tool_span(spec.name)
            handle, finish = opened if opened is not None else (None, None)
            if handle is not None:
                handle.set("tool.args", redacted_args(func, args, kwargs))
            try:
                if spec.timeout_s is None:
                    value = await func(*args, **kwargs)
                else:
                    value = await asyncio.wait_for(func(*args, **kwargs), spec.timeout_s)
            except BaseException as exc:
                if is_uncatchable(exc):
                    if finish is not None:
                        finish(Err(kind="exception", message="cancelled", retryable=False), attempt)
                    raise
                last = spec.classify(exc)
                if finish is not None:
                    finish(last, attempt)
                if not last.retryable or attempt == spec.retries:
                    return last
                await asyncio.sleep(spec.backoff_for(attempt))
                continue

            try:
                result = _flag_fault(spec.interpret(value), fault)
            except BaseException:
                if finish is not None:
                    finish(
                        Err(kind="validation", message="ambiguous empty", retryable=False),
                        attempt,
                    )
                raise
            if finish is not None:
                finish(result, attempt)
            return result

        return cast(Err, last)  # pragma: no cover - loop always returns

    return wrapper


def _build_sync_wrapper(
    func: Callable[..., Any], spec: _ToolSpec
) -> Callable[..., ToolResult[Any]]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> ToolResult[Any]:
        stub = find_stub(spec.name)
        if stub is not None:
            return _run_sync_stub(spec, func, stub, args, kwargs)

        fault = next_fault(spec.name)
        if fault is not None:
            injected = spec.apply_fault(fault)
            if injected is not None:
                opened = open_tool_span(spec.name)
                if opened is not None:
                    handle, finish = opened
                    handle.set("tool.args", redacted_args(func, args, kwargs))
                    finish(injected, 0)
                return injected

        last: Err | None = None
        for attempt in range(spec.retries + 1):
            started = time.monotonic()
            opened = open_tool_span(spec.name)
            handle, finish = opened if opened is not None else (None, None)
            if handle is not None:
                handle.set("tool.args", redacted_args(func, args, kwargs))
            try:
                value = func(*args, **kwargs)
            except BaseException as exc:
                if is_uncatchable(exc):
                    if finish is not None:
                        finish(Err(kind="exception", message="cancelled", retryable=False), attempt)
                    raise
                last = spec.classify(exc)
                if finish is not None:
                    finish(last, attempt)
                if not last.retryable or attempt == spec.retries:
                    return last
                time.sleep(spec.backoff_for(attempt))
                continue

            # Advisory only: the call already finished, but it overran its
            # deadline, and a caller that set one is entitled to know.
            if spec.timeout_s is not None and time.monotonic() - started > spec.timeout_s:
                overran = Err(
                    kind="timeout",
                    message=(
                        f"Tool {spec.name!r} exceeded its advisory timeout of "
                        f"{spec.timeout_s}s. The call ran to completion; sync "
                        f"timeouts cannot interrupt it."
                    ),
                    retryable=True,
                    cause="TimeoutError",
                )
                if finish is not None:
                    finish(overran, attempt)
                return overran

            try:
                result = _flag_fault(spec.interpret(value), fault)
            except BaseException:
                if finish is not None:
                    finish(
                        Err(kind="validation", message="ambiguous empty", retryable=False),
                        attempt,
                    )
                raise
            if finish is not None:
                finish(result, attempt)
            return result

        return cast(Err, last)  # pragma: no cover - loop always returns

    return wrapper


async def _run_async_stub(
    spec: _ToolSpec,
    func: Callable[..., Any],
    stub: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ToolResult[Any]:
    """Run a stub in place of the real body, traced the same way.

    A stubbed call is still a tool call: it gets a span, its arguments are
    recorded, and its result goes through the same interpretation. Otherwise an
    eval run would measure a system with fewer spans than production has.
    """
    opened = open_tool_span(spec.name)
    handle, finish = opened if opened is not None else (None, None)
    if handle is not None:
        handle.set("tool.args", redacted_args(func, args, kwargs))
        handle.set("tool.stubbed", True)
    try:
        value = stub(*args, **kwargs)
        if inspect.isawaitable(value):
            value = await value
    except BaseException as exc:
        if is_uncatchable(exc):
            if finish is not None:
                finish(Err(kind="exception", message="cancelled", retryable=False), 0)
            raise
        error = spec.classify(exc)
        if finish is not None:
            finish(error, 0)
        return error

    result = spec.interpret(value)
    if finish is not None:
        finish(result, 0)
    return result


def _run_sync_stub(
    spec: _ToolSpec,
    func: Callable[..., Any],
    stub: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ToolResult[Any]:
    """Synchronous counterpart of :func:`_run_async_stub`."""
    opened = open_tool_span(spec.name)
    handle, finish = opened if opened is not None else (None, None)
    if handle is not None:
        handle.set("tool.args", redacted_args(func, args, kwargs))
        handle.set("tool.stubbed", True)
    try:
        value = stub(*args, **kwargs)
    except BaseException as exc:
        if is_uncatchable(exc):
            if finish is not None:
                finish(Err(kind="exception", message="cancelled", retryable=False), 0)
            raise
        error = spec.classify(exc)
        if finish is not None:
            finish(error, 0)
        return error

    result = spec.interpret(value)
    if finish is not None:
        finish(result, 0)
    return result


def _flag_fault(result: ToolResult[Any], fault: FaultSpec | None) -> ToolResult[Any]:
    """Mark a real result that a ``truncated`` fault applied to."""
    if fault is None or fault.kind != "truncated":
        return result
    if isinstance(result, Ok):
        return result.model_copy(update={"truncated": True, "fault_injected": True})
    return result.model_copy(update={"fault_injected": True})


__all__ = [
    "AmbiguousEmptyError",
    "ToolCallable",
    "clear_side_effect_registry",
    "register_side_effect",
    "registered_side_effect_tools",
    "tool",
]
