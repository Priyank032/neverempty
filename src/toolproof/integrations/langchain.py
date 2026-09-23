"""LangChain callback handler: node, tool and LLM spans.

Two rules govern everything here:

1. **Telemetry never breaks the agent.** Every callback is wrapped, because a
   LangChain version skew that changes a payload shape must cost a span, not a
   production request. LangChain swallows handler errors by default; this does
   not rely on that.
2. **No prompt or completion text is recorded.** Usage, models and finish
   reasons only. Message content carries user data, and a tracer is not a
   prompt logger.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

logger = logging.getLogger("toolproof.tracer")

_INSTALL_HINT = "LangChain support needs the langgraph extra: pip install 'toolproof[langgraph]'"


def _base_handler_class() -> Any:
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError as exc:  # pragma: no cover - exercised via the extra test
        raise ImportError(_INSTALL_HINT) from exc
    return BaseCallbackHandler


def build_handler(tracer: Any) -> Any:
    """Return a ``BaseCallbackHandler`` bound to ``tracer``.

    Built at call time rather than at import, so ``toolproof`` imports cleanly
    without the extra installed.
    """
    base = _base_handler_class()

    class ToolproofCallbackHandler(base):  # type: ignore[misc, valid-type]
        """Emits toolproof spans for LangChain node, tool and LLM events."""

        # LangChain inspects these; both default to False on the base class.
        raise_error = False
        run_inline = True

        def __init__(self) -> None:
            super().__init__()
            self._tracer = tracer
            self._spans: dict[UUID, Any] = {}
            self._span_ids: dict[UUID, str] = {}
            self._node_stack: list[tuple[UUID, str]] = []

        # -- lifecycle ----------------------------------------------------

        def _open(
            self, run_id: UUID, kind: str, name: str, parent_run_id: UUID | None = None
        ) -> None:
            # Parenting comes from LangChain's own run tree, not contextvars: a
            # callback's start and end can arrive in different contexts.
            parent_span = self._span_ids.get(parent_run_id) if parent_run_id else None
            context = self._tracer.open_span(kind, name=name, parent_id=parent_span)
            if context is not None:
                self._spans[run_id] = context
                self._span_ids[run_id] = context[0].span_id
                if kind == "node":
                    # So a @tool called inside this node parents to it, even
                    # though LangGraph runs the node in a fresh context.
                    self._node_stack.append((run_id, context[0].span_id))
                    self._publish_parent()

        def _close(self, run_id: UUID, *, error: BaseException | None = None) -> Any:
            context = self._spans.pop(run_id, None)
            self._span_ids.pop(run_id, None)
            if self._node_stack and any(entry[0] == run_id for entry in self._node_stack):
                self._node_stack = [e for e in self._node_stack if e[0] != run_id]
                self._publish_parent()
            if context is None:
                return None
            handle, finish = context
            finish(error)
            return handle

        def _publish_parent(self) -> None:
            """Expose the innermost open node span as the adapter fallback."""
            run = self._tracer.current_run
            if run is None:
                return
            run.set_adapter_parent(self._node_stack[-1][1] if self._node_stack else None)

        def _handle(self, run_id: UUID) -> Any:
            context = self._spans.get(run_id)
            return None if context is None else context[0]

        # -- chain and graph nodes ----------------------------------------

        def on_chain_start(
            self,
            serialized: dict[str, Any] | None,
            inputs: dict[str, Any] | None,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            with _Guard("on_chain_start"):
                name = _node_name(serialized, kwargs)
                if name is None:
                    return
                self._open(run_id, "node", name, kwargs.get("parent_run_id"))
                handle = self._handle(run_id)
                if handle is not None:
                    handle.set("graph.node", name)

        def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
            with _Guard("on_chain_end"):
                self._close(run_id)

        def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
            with _Guard("on_chain_error"):
                self._close(run_id, error=error)

        # -- LLM calls ----------------------------------------------------

        def on_chat_model_start(
            self,
            serialized: dict[str, Any] | None,
            messages: Any,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            with _Guard("on_chat_model_start"):
                self._start_llm(serialized, run_id, kwargs)

        def on_llm_start(
            self,
            serialized: dict[str, Any] | None,
            prompts: list[str] | None,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            with _Guard("on_llm_start"):
                self._start_llm(serialized, run_id, kwargs)

        def _start_llm(
            self, serialized: dict[str, Any] | None, run_id: UUID, kwargs: dict[str, Any]
        ) -> None:
            params = kwargs.get("invocation_params") or {}
            name = params.get("model") or params.get("model_name") or _class_name(serialized)
            self._open(run_id, "llm", name or "llm", kwargs.get("parent_run_id"))
            handle = self._handle(run_id)
            if handle is None:
                return
            # Recorded eagerly: a call that errors still had a requested model.
            if name:
                handle.set("gen_ai.request.model", name)
            for key, attribute in (
                ("temperature", "gen_ai.request.temperature"),
                ("seed", "gen_ai.request.seed"),
            ):
                if params.get(key) is not None:
                    handle.set(attribute, params[key])

        def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
            with _Guard("on_llm_end"):
                handle = self._handle(run_id)
                if handle is not None:
                    _record_llm_usage(handle, response)
                self._close(run_id)

        def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
            with _Guard("on_llm_error"):
                self._close(run_id, error=error)

        # -- tools --------------------------------------------------------

        def on_tool_start(
            self,
            serialized: dict[str, Any] | None,
            input_str: str,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            with _Guard("on_tool_start"):
                name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
                self._open(run_id, "tool", name, kwargs.get("parent_run_id"))
                handle = self._handle(run_id)
                if handle is not None:
                    handle.set("tool.name", name)

        def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
            with _Guard("on_tool_end"):
                handle = self._handle(run_id)
                if handle is not None:
                    status = getattr(output, "status", None)
                    if status in ("ok", "empty", "error"):
                        handle.set("tool.status", status)
                        handle.status = status
                self._close(run_id)

        def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
            with _Guard("on_tool_error"):
                self._close(run_id, error=error)

    return ToolproofCallbackHandler()


class _Guard:
    """Swallow and log any exception from a callback body.

    A handler that raises would turn a telemetry problem into an agent failure.
    Cancellation still propagates, as everywhere else in this library.
    """

    __slots__ = ("label",)

    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        if exc is None:
            return False
        from toolproof.core.classify import is_uncatchable

        if is_uncatchable(exc):
            return False
        logger.warning("toolproof callback %s failed, span dropped: %s", self.label, exc)
        return True


def _node_name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str | None:
    """A LangGraph node's name, or ``None`` for plumbing chains.

    LangGraph emits a chain event for the graph itself and for internal
    channel writers. Those are not nodes, and a span per writer would bury the
    real trace.
    """
    name = kwargs.get("name") or (serialized or {}).get("name")
    if not name:
        return None
    if name.startswith("_") or name in _NOT_A_NODE:
        return None
    return str(name)


_NOT_A_NODE = frozenset(
    {
        "LangGraph",
        "RunnableSequence",
        "RunnableParallel",
        "RunnableLambda",
        "ChannelWrite",
        "ChannelRead",
        "__start__",
        "__end__",
    }
)


def _class_name(serialized: dict[str, Any] | None) -> str | None:
    identifier = (serialized or {}).get("id")
    if isinstance(identifier, list) and identifier:
        return str(identifier[-1])
    return None


def _record_llm_usage(handle: Any, response: Any) -> None:
    """Pull usage off an ``LLMResult``, from whichever field carries it.

    Token counts come from provider usage fields only, never an estimate, so a
    response that reports nothing records null.
    """
    usage: dict[str, Any] = {}
    resolved: str | None = None
    finish_reason: str | None = None

    generations = getattr(response, "generations", None) or []
    for batch in generations:
        for generation in batch or []:
            message = getattr(generation, "message", None)
            if message is not None:
                usage = getattr(message, "usage_metadata", None) or usage
                metadata = getattr(message, "response_metadata", None) or {}
                resolved = resolved or metadata.get("model_name") or metadata.get("model")
                finish_reason = finish_reason or metadata.get("finish_reason")
            info = getattr(generation, "generation_info", None) or {}
            finish_reason = finish_reason or info.get("finish_reason")

    llm_output = getattr(response, "llm_output", None) or {}
    if not usage:
        raw = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if raw:
            usage = {
                "input_tokens": raw.get("prompt_tokens") or raw.get("input_tokens"),
                "output_tokens": raw.get("completion_tokens") or raw.get("output_tokens"),
            }
    resolved = resolved or llm_output.get("model_name") or llm_output.get("model")

    requested = handle.attributes.get("gen_ai.request.model") or resolved or "unknown"
    handle.record_usage(
        model=str(requested),
        resolved_model=resolved,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cached_input_tokens=(usage.get("input_token_details") or {}).get("cache_read"),
        finish_reason=finish_reason,
    )


__all__ = ["build_handler"]
