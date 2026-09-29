"""Usage capture from provider SDKs, by duck typing.

Neither `openai` nor `boto3` is imported. Instrumentation patches the method on
the client instance and reads the response shape, so core stays at one
dependency and the same code path is exercised by a fake with the real shape.

**No prompt or completion text is ever recorded.** Usage, models, finish
reasons and latency only. Messages carry user data, and a tracer is not a
prompt logger — redaction cannot help with something that should never have
been captured.
"""

from __future__ import annotations

import functools
from typing import Any

_MARKER = "__neverempty_instrumented__"


def instrument_openai(tracer: Any, client: Any) -> Any:
    """Wrap ``client.chat.completions.create`` to emit an LLM span.

    Idempotent: instrumenting the same client twice does not double count.
    Returns the client, so it can be used inline.
    """
    completions = _reach(client, ("chat", "completions"), "chat.completions.create")
    original = getattr(completions, "create", None)
    if original is None:
        raise TypeError(
            f"{type(client).__name__} has no chat.completions.create; this does not "
            f"look like an OpenAI client"
        )
    if getattr(original, _MARKER, False):
        return client

    @functools.wraps(original)
    def create(*args: Any, **kwargs: Any) -> Any:
        return _call(
            tracer,
            original,
            args,
            kwargs,
            name=str(kwargs.get("model") or "openai"),
            requested_model=kwargs.get("model"),
            temperature=kwargs.get("temperature"),
            seed=kwargs.get("seed"),
            extract=_extract_openai,
        )

    setattr(create, _MARKER, True)
    completions.create = create
    return client


def instrument_bedrock(tracer: Any, client: Any) -> Any:
    """Wrap ``client.converse`` to emit an LLM span."""
    original = getattr(client, "converse", None)
    if original is None:
        raise TypeError(
            f"{type(client).__name__} has no converse method; pass a boto3 bedrock-runtime client"
        )
    if getattr(original, _MARKER, False):
        return client

    @functools.wraps(original)
    def converse(*args: Any, **kwargs: Any) -> Any:
        inference = kwargs.get("inferenceConfig") or {}
        return _call(
            tracer,
            original,
            args,
            kwargs,
            name=str(kwargs.get("modelId") or "bedrock"),
            requested_model=kwargs.get("modelId"),
            temperature=inference.get("temperature"),
            seed=None,
            extract=_extract_bedrock,
        )

    setattr(converse, _MARKER, True)
    client.converse = converse
    return client


def _call(
    tracer: Any,
    original: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    name: str,
    requested_model: Any,
    temperature: Any,
    seed: Any,
    extract: Any,
) -> Any:
    """Run the provider call inside a span, recording only metadata."""
    from neverempty.tracer.tracer import open_named_span

    context = open_named_span("llm", name)
    if context is None:
        return original(*args, **kwargs)

    handle, finish = context
    try:
        response = original(*args, **kwargs)
    except BaseException as exc:
        finish(exc)
        raise

    try:
        resolved, usage, finish_reason = extract(response)
        handle.record_usage(
            model=str(requested_model or resolved or "unknown"),
            resolved_model=resolved,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cached_input_tokens=usage.get("cached_input_tokens"),
            temperature=temperature,
            seed=seed,
            finish_reason=finish_reason,
        )
    finally:
        finish(None)
    return response


def _extract_openai(response: Any) -> tuple[str | None, dict[str, Any], str | None]:
    resolved = getattr(response, "model", None)
    usage_obj = getattr(response, "usage", None)

    usage: dict[str, Any] = {}
    if usage_obj is not None:
        usage["input_tokens"] = getattr(usage_obj, "prompt_tokens", None)
        usage["output_tokens"] = getattr(usage_obj, "completion_tokens", None)
        details = getattr(usage_obj, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        if cached is not None:
            usage["cached_input_tokens"] = cached

    choices = getattr(response, "choices", None) or []
    finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
    return (str(resolved) if resolved else None, usage, finish_reason)


def _extract_bedrock(response: Any) -> tuple[str | None, dict[str, Any], str | None]:
    if not isinstance(response, dict):  # pragma: no cover - defensive
        return None, {}, None

    raw = response.get("usage") or {}
    usage: dict[str, Any] = {}
    if raw:
        usage["input_tokens"] = raw.get("inputTokens")
        usage["output_tokens"] = raw.get("outputTokens")
        cached = raw.get("cacheReadInputTokens")
        if cached is not None:
            usage["cached_input_tokens"] = cached

    # converse echoes no model id, so the request's modelId is the best available.
    return None, usage, response.get("stopReason")


def _reach(client: Any, path: tuple[str, ...], label: str) -> Any:
    current = client
    for attribute in path:
        current = getattr(current, attribute, None)
        if current is None:
            raise TypeError(
                f"{type(client).__name__} has no {label}; this does not look like an OpenAI client"
            )
    return current


__all__ = ["instrument_bedrock", "instrument_openai"]
