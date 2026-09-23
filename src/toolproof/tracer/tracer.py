"""The tracer: contextvars, so nothing needs plumbing through call signatures.

Any ``@tool`` call or instrumented LLM call inside ``tracer.run()`` attaches to
the right trace and the right parent span, including under asyncio concurrency
with hundreds of runs in flight. That property is what makes this usable in
production code rather than only in eval scripts, and it is the reason the
current span lives in a ``ContextVar`` rather than on the Tracer.
"""

from __future__ import annotations

import contextvars
import json
import logging
import random
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Literal

from toolproof.core.classify import is_uncatchable
from toolproof.core.results import Empty, Err, Ok
from toolproof.core.trace import (
    TOOL_ARGS_CAP_BYTES,
    Cost,
    Env,
    FinalOutput,
    Span,
    SpanKind,
    SpanStatus,
    Trace,
    TraceError,
    TraceStatus,
    Usage,
    truncate_utf8,
)
from toolproof.tracer.pricing import Pricing
from toolproof.tracer.redact import Redactor, nothing
from toolproof.tracer.sinks import Sink

logger = logging.getLogger("toolproof.tracer")

_current_run: contextvars.ContextVar[_RunState | None] = contextvars.ContextVar(
    "toolproof_run", default=None
)
_current_parent: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "toolproof_parent", default=None
)


def _utc_now_rfc3339() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class SpanHandle:
    """A span while it is open. Adapters write attributes through this."""

    __slots__ = ("_run", "attributes", "kind", "name", "span_id", "start_ns", "status")

    def __init__(
        self, run: _RunState, span_id: str, kind: SpanKind, name: str, start_ns: int
    ) -> None:
        self._run = run
        self.span_id = span_id
        self.kind = kind
        self.name = name
        self.start_ns = start_ns
        self.status: SpanStatus = "ok"
        self.attributes: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Set one attribute. Values are redacted on the way to the sink."""
        self.attributes[key] = value

    def update(self, values: dict[str, Any]) -> None:
        self.attributes.update(values)

    def record_usage(
        self,
        *,
        model: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cached_input_tokens: int | None = None,
        resolved_model: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        finish_reason: str | None = None,
    ) -> None:
        """Record LLM usage using OTel GenAI attribute names.

        ``resolved_model`` is the id the provider returned, which is the only
        way model drift becomes detectable: a config asking for ``gpt-4o`` and
        a response from ``gpt-4o-2024-11-20`` are different experiments.

        Token counts come from provider usage fields only. Passing ``None``
        records null, never a zero and never an estimate.
        """
        self.attributes["gen_ai.request.model"] = model
        if resolved_model is not None:
            self.attributes["gen_ai.response.model"] = resolved_model
        if input_tokens is not None:
            self.attributes["gen_ai.usage.input_tokens"] = input_tokens
        if output_tokens is not None:
            self.attributes["gen_ai.usage.output_tokens"] = output_tokens
        if cached_input_tokens is not None:
            self.attributes["gen_ai.usage.cached_input_tokens"] = cached_input_tokens
        if temperature is not None:
            self.attributes["gen_ai.request.temperature"] = temperature
        if seed is not None:
            self.attributes["gen_ai.request.seed"] = seed
        if finish_reason is not None:
            self.attributes["gen_ai.response.finish_reason"] = finish_reason

        self._run.record_usage(
            model=resolved_model or model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

    def record_tool_result(self, result: Ok[Any] | Empty | Err, *, attempt: int = 0) -> None:
        """Copy a ``ToolResult`` onto the span, using the fixed attribute names."""
        self.attributes["tool.name"] = self.name
        self.attributes["tool.status"] = result.status
        self.attributes["tool.attempt"] = attempt
        self.attributes["tool.fault_injected"] = result.fault_injected

        if isinstance(result, Ok):
            self.attributes["tool.truncated"] = result.truncated
            self.status = "ok"
        elif isinstance(result, Empty):
            self.status = "empty"
        else:
            self.attributes["tool.error_kind"] = result.kind
            self.attributes["tool.retryable"] = result.retryable
            self.status = "error"


class _RunState:
    """Mutable state for one in-flight trace."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        case_id: str | None,
        repeat: int,
        suite: str | None,
        tags: dict[str, str],
        sampled: bool,
    ) -> None:
        self.tracer = tracer
        self.trace_id = str(uuid.uuid4())
        self.case_id = case_id
        self.repeat = repeat
        self.suite = suite
        self.tags = dict(tags)
        self.sampled = sampled

        self.started_at = _utc_now_rfc3339()
        self.start_ns = time.monotonic_ns()
        self.spans: list[Span] = []
        self.output = FinalOutput()
        self.status: TraceStatus = "ok"
        self.error: TraceError | None = None

        # Fallback parent for work that runs in a context the tracer never
        # entered. LangGraph executes each node in a fresh context, so a @tool
        # called inside a node cannot see the node's span through a ContextVar.
        # The adapter publishes the innermost open span here instead.
        self.adapter_parent: str | None = None

        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._cached_tokens: int | None = None
        self._models: list[str] = []
        self._next_span = 0
        self._lock = threading.Lock()

    def new_span_id(self) -> str:
        with self._lock:
            self._next_span += 1
            return f"s{self._next_span:04d}"

    def add_span(self, span: Span) -> None:
        with self._lock:
            self.spans.append(span)

    def record_usage(
        self,
        *,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_input_tokens: int | None,
    ) -> None:
        with self._lock:
            if model and model not in self._models:
                self._models.append(model)
            if input_tokens is not None:
                self._input_tokens = (self._input_tokens or 0) + input_tokens
            if output_tokens is not None:
                self._output_tokens = (self._output_tokens or 0) + output_tokens
            if cached_input_tokens is not None:
                self._cached_tokens = (self._cached_tokens or 0) + cached_input_tokens

    def usage(self) -> Usage:
        return Usage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cached_input_tokens=self._cached_tokens,
        )

    def models(self) -> list[str]:
        return list(self._models)


class TraceRun:
    """The handle ``async with tracer.run()`` yields."""

    __slots__ = ("_state", "trace")

    def __init__(self, state: _RunState) -> None:
        self._state = state
        self.trace: Trace | None = None

    @property
    def sampled(self) -> bool:
        return self._state.sampled

    @property
    def trace_id(self) -> str:
        return self._state.trace_id

    def set_output(
        self,
        *,
        answer: str | None = None,
        route: str | None = None,
        structured: dict[str, Any] | None = None,
    ) -> None:
        """Record what the agent produced. Redacted before it reaches a sink."""
        current = self._state.output
        self._state.output = FinalOutput(
            answer=answer if answer is not None else current.answer,
            route=route if route is not None else current.route,
            structured=structured if structured is not None else current.structured,
        )

    def set_tag(self, key: str, value: str) -> None:
        self._state.tags[key] = value

    def set_status(self, status: TraceStatus) -> None:
        """Override the trace status.

        The runner uses this for ``timeout``, which the tracer cannot detect on
        its own: a target that overran its deadline did not raise.
        """
        self._state.status = status

    def set_error(self, kind: str, message: str) -> None:
        """Record why a run failed. The message is capped; no traceback."""
        self._state.error = TraceError(kind=kind, message=message)

    def set_adapter_parent(self, span_id: str | None) -> None:
        """Publish a fallback parent for work in a context the tracer never entered.

        Used by framework adapters; agent code should not need this.
        """
        self._state.adapter_parent = span_id


class Tracer:
    """Captures spans into traces and hands them to a sink.

    Args:
        sink: Where finished traces go.
        pricing: Versioned pricing table. Defaults to the empty table, so every
            cost is null with a reason until you supply real prices.
        redact: Applied to span attributes and outputs before the sink sees
            them. Defaults to redacting nothing, explicitly.
        sample_rate: Fraction of runs recorded. ``0.0`` records nothing and
            breaks nothing; tools and spans still work.
        seed: Makes sampling reproducible.
        env_overrides: Reproducibility fields the runner supplies, such as
            ``target_git_sha``, ``prompt_hashes``, ``concurrency`` and ``mode``.
    """

    def __init__(
        self,
        *,
        sink: Sink,
        pricing: Pricing | None = None,
        redact: Redactor | None = None,
        sample_rate: float = 1.0,
        seed: int | None = None,
        env_overrides: dict[str, Any] | None = None,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be between 0.0 and 1.0, got {sample_rate}")

        self.sink = sink
        self.pricing = pricing or Pricing.default()
        self.redact: Redactor = redact or nothing()
        self.sample_rate = sample_rate
        self.env_overrides = dict(env_overrides or {})

        self.dropped_traces = 0
        self.last_sink_error: BaseException | None = None

        self._random = random.Random(seed)  # noqa: S311 - sampling, not crypto
        self._sample_lock = threading.Lock()

    @property
    def current_run(self) -> TraceRun | None:
        """The run active in this context, if any."""
        state = _current_run.get()
        return None if state is None else TraceRun(state)

    def _should_sample(self) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        with self._sample_lock:
            return self._random.random() < self.sample_rate

    def run(
        self,
        *,
        case_id: str | None = None,
        repeat: int = 0,
        suite: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> _RunContext:
        """Open a trace. Works as ``with`` and as ``async with``.

        A trace is always finished, whatever happens in the body: an ordinary
        exception marks ``target_error``, and a cancellation marks
        ``budget_abort``. Both re-raise untouched, because turning a budget
        abort into an agent failure would corrupt every number downstream.
        """
        state = _RunState(
            tracer=self,
            case_id=case_id,
            repeat=repeat,
            suite=suite,
            tags=tags or {},
            sampled=self._should_sample(),
        )
        return _RunContext(self, state)

    @contextmanager
    def span(self, kind: SpanKind, *, name: str) -> Iterator[SpanHandle]:
        """Open a span parented to whatever span is active in this context."""
        state = _current_run.get()
        if state is None or not state.sampled:
            yield _DetachedSpan()  # type: ignore[misc]
            return

        span_id = state.new_span_id()
        handle = SpanHandle(state, span_id, kind, name, time.monotonic_ns())
        parent_id = _current_parent.get()
        token = _current_parent.set(span_id)
        try:
            yield handle
        except BaseException as exc:
            if not is_uncatchable(exc):
                handle.status = "error"
                handle.set("error.kind", type(exc).__name__)
            raise
        finally:
            _current_parent.reset(token)
            state.add_span(
                Span(
                    span_id=span_id,
                    parent_id=parent_id,
                    kind=kind,
                    name=name,
                    start_ns=handle.start_ns,
                    end_ns=time.monotonic_ns(),
                    status=handle.status,
                    attributes=self._clean_attributes(handle.attributes),
                )
            )

    def langchain_handler(self) -> Any:
        """A ``BaseCallbackHandler`` emitting node, tool and LLM spans.

        Needs the langgraph extra. Works for any LangChain runnable, not only
        LangGraph.
        """
        from toolproof.integrations.langchain import build_handler

        return build_handler(self)

    def instrument_openai(self, client: Any) -> Any:
        """Capture usage, resolved model, latency and finish reason.

        Patches the client instance by duck typing, so no provider SDK is
        imported. Never records prompt or completion text.
        """
        from toolproof.integrations.providers import instrument_openai

        return instrument_openai(self, client)

    def instrument_bedrock(self, client: Any) -> Any:
        """Capture usage from ``converse`` responses."""
        from toolproof.integrations.providers import instrument_bedrock

        return instrument_bedrock(self, client)

    def open_span(
        self, kind: SpanKind, *, name: str, parent_id: str | None = None
    ) -> tuple[Any, Any] | None:
        """Open a span without a ``with`` block, for callback-driven adapters.

        Returns ``(handle, finish)``, or ``None`` when nothing is recording.
        ``finish(error)`` closes it. Callbacks cannot use a context manager,
        because start and end arrive as separate events.
        """
        return open_named_span(kind, name, parent_id=parent_id)

    def _clean_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        """Redact, then flatten to the scalar map the schema allows."""
        redacted = self.redact(attributes)
        flat: dict[str, Any] = {}
        for key, value in redacted.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                flat[key] = value
            else:
                flat[key] = truncate_utf8(
                    json.dumps(value, ensure_ascii=False, default=str),
                    TOOL_ARGS_CAP_BYTES,
                )
        return flat

    def _finish(self, state: _RunState) -> Trace | None:
        if not state.sampled:
            return None

        usage = state.usage()
        cost: Cost = self.pricing.cost_for(usage, state.models())
        output = state.output
        redacted_structured = (
            self.redact(output.structured) if output.structured is not None else None
        )

        trace = Trace(
            trace_id=state.trace_id,
            case_id=state.case_id,
            repeat=state.repeat,
            suite=state.suite,
            started_at=state.started_at,
            duration_ms=max((time.monotonic_ns() - state.start_ns) // 1_000_000, 0),
            spans=state.spans,
            final_output=FinalOutput(
                answer=output.answer,
                route=output.route,
                structured=redacted_structured,
            ),
            usage=usage,
            cost=cost,
            status=state.status,
            error=state.error,
            env=self._build_env(state),
            tags=state.tags,
        )
        self._emit(trace)
        return trace

    def _build_env(self, state: _RunState) -> Env:
        from toolproof import __version__

        fields: dict[str, Any] = {
            "toolproof_version": __version__,
            "pricing_version": self.pricing.version,
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
            "resolved_models": state.models(),
        }
        fields.update(self.env_overrides)
        return Env(**fields)

    def _emit(self, trace: Trace) -> None:
        """Hand the trace to the sink. Never raises into the agent."""
        try:
            self.sink.write(trace)
        except BaseException as exc:
            if is_uncatchable(exc):
                raise
            self.dropped_traces += 1
            self.last_sink_error = exc
            # Counted and logged, never silent: losing a trace without saying so
            # is the same bug class this library exists to prevent.
            logger.error(
                "sink write failed, trace %s dropped (%d total): %s",
                trace.trace_id,
                self.dropped_traces,
                exc,
            )


class _RunContext:
    """One open trace, enterable with either ``with`` or ``async with``.

    The doc's API uses ``async with``, but a sync agent must be traceable too,
    and ``@contextmanager`` gives only the sync half. Nothing here awaits, so
    both protocols share one implementation.
    """

    __slots__ = ("_handle", "_parent_token", "_run_token", "_state", "_tracer")

    def __init__(self, tracer: Tracer, state: _RunState) -> None:
        self._tracer = tracer
        self._state = state
        self._handle = TraceRun(state)
        self._run_token: contextvars.Token[_RunState | None] | None = None
        self._parent_token: contextvars.Token[str | None] | None = None

    @property
    def trace(self) -> Trace | None:
        """The finished trace, available once the block has exited."""
        return self._handle.trace

    def _enter(self) -> TraceRun:
        self._run_token = _current_run.set(self._state)
        self._parent_token = _current_parent.set(None)
        return self._handle

    def _exit(self, exc: BaseException | None) -> None:
        state = self._state
        if exc is not None:
            state.status = "budget_abort" if is_uncatchable(exc) else "target_error"
            state.error = TraceError(kind=type(exc).__name__, message=str(exc))
        if self._parent_token is not None:
            _current_parent.reset(self._parent_token)
        if self._run_token is not None:
            _current_run.reset(self._run_token)
        self._handle.trace = self._tracer._finish(state)

    def __enter__(self) -> TraceRun:
        return self._enter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self._exit(exc)
        return False

    async def __aenter__(self) -> TraceRun:
        return self._enter()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self._exit(exc)
        return False


class _DetachedSpan:
    """No-op span for unsampled runs and calls outside any run.

    Adapters call ``record_usage`` and ``set`` unconditionally, so this has to
    accept everything and record nothing.
    """

    __slots__ = ("attributes", "status")

    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.status: SpanStatus = "ok"

    def set(self, key: str, value: Any) -> None:
        return None

    def update(self, values: dict[str, Any]) -> None:
        return None

    def record_usage(self, **kwargs: Any) -> None:
        return None

    def record_tool_result(self, result: Any, *, attempt: int = 0) -> None:
        return None

    def __enter__(self) -> _DetachedSpan:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


def current_span_context() -> tuple[_RunState | None, str | None]:
    """The active run and parent span id. Used by the ``@tool`` wrapper."""
    return _current_run.get(), _current_parent.get()


def open_named_span(
    kind: SpanKind, name: str, *, parent_id: str | None = None
) -> tuple[Any, Any] | None:
    """Open a span of any kind, closed by calling the returned finisher.

    Used by adapters whose start and end arrive as separate callbacks, so a
    ``with`` block is not available. Returns ``None`` outside a sampled run.

    Args:
        parent_id: Parent span, when the adapter tracks nesting itself. Omit to
            inherit whatever span is active in this context.
    """
    state = _current_run.get()
    if state is None or not state.sampled:
        return None

    span_id = state.new_span_id()
    handle = SpanHandle(state, span_id, kind, name, time.monotonic_ns())
    resolved_parent = parent_id if parent_id is not None else _current_parent.get()
    # Deliberately no ContextVar token here. A LangChain callback's start and
    # end can arrive in different contexts, and resetting a token across
    # contexts raises. Adapters that need nesting pass the parent explicitly.
    closed = False

    def finish(error: BaseException | None = None) -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        if error is not None:
            handle.status = "error"
            handle.set("error.kind", type(error).__name__)
        state.add_span(
            Span(
                span_id=span_id,
                parent_id=resolved_parent,
                kind=kind,
                name=name,
                start_ns=handle.start_ns,
                end_ns=time.monotonic_ns(),
                status=handle.status,
                attributes=state.tracer._clean_attributes(handle.attributes),
            )
        )

    return handle, finish


def open_tool_span(name: str) -> tuple[Any, Any] | None:
    """Open a span for a tool call, or ``None`` outside a sampled run.

    Returns ``(handle, finisher)``; the wrapper calls ``finisher(result)`` once
    the outcome is known.
    """
    state = _current_run.get()
    if state is None or not state.sampled:
        return None

    span_id = state.new_span_id()
    handle = SpanHandle(state, span_id, "tool", name, time.monotonic_ns())
    parent_id = _current_parent.get() or state.adapter_parent
    token = _current_parent.set(span_id)

    def finish(result: Any, attempt: int = 0) -> None:
        _current_parent.reset(token)
        handle.record_tool_result(result, attempt=attempt)
        state.add_span(
            Span(
                span_id=span_id,
                parent_id=parent_id,
                kind="tool",
                name=name,
                start_ns=handle.start_ns,
                end_ns=time.monotonic_ns(),
                status=handle.status,
                attributes=state.tracer._clean_attributes(handle.attributes),
            )
        )

    return handle, finish


__all__ = [
    "SpanHandle",
    "TraceRun",
    "Tracer",
    "current_span_context",
    "open_named_span",
    "open_tool_span",
]
