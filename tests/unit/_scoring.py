"""Builders for the scorer tests.

Scorers read a ``Trace``, not an agent, so every test here constructs the exact
span shape a real adapter would have written. That is the point of the trace
being a contract: a scorer test needs no LLM, no graph and no network.
"""

from __future__ import annotations

import itertools
from typing import Any

from neverempty.core.trace import Cost, Env, FinalOutput, Span, Trace, Usage

_ids = itertools.count(1)


def span_id() -> str:
    return f"s{next(_ids):04d}"


def env() -> Env:
    return Env(
        neverempty_version="0.0.1",
        pricing_version="empty-2026-09-23",
        python_version="3.12.10",
    )


def node_span(name: str, *, start_ns: int, next_node: str | None = None, **extra: Any) -> Span:
    attributes: dict[str, Any] = {"graph.node": name}
    if next_node is not None:
        attributes["graph.next"] = next_node
    attributes.update(extra)
    return Span(
        span_id=span_id(),
        kind="node",
        name=name,
        start_ns=start_ns,
        end_ns=start_ns + 1_000,
        attributes=attributes,
    )


def probe_span(*, start_ns: int, route: str, **extra: Any) -> Span:
    """What ``route_probe`` writes: a custom span carrying ``graph.next``."""
    attributes: dict[str, Any] = {"graph.next": route}
    attributes.update(extra)
    return Span(
        span_id=span_id(),
        kind="custom",
        name="route_probe",
        start_ns=start_ns,
        end_ns=start_ns + 1_000,
        attributes=attributes,
    )


def tool_span(
    name: str,
    *,
    start_ns: int,
    args: str | None = None,
    status: str = "ok",
    error_kind: str | None = None,
    fault_injected: bool = False,
    # ``attrs`` carries attribute names that are not valid Python identifiers,
    # such as "tool.stubbed", which ** unpacking cannot express.
    attrs: dict[str, Any] | None = None,
    **extra: Any,
) -> Span:
    attributes: dict[str, Any] = {
        "tool.name": name,
        "tool.status": status,
        "tool.fault_injected": fault_injected,
    }
    if args is not None:
        attributes["tool.args"] = args
    if error_kind is not None:
        attributes["tool.error_kind"] = error_kind
    attributes.update(extra)
    if attrs is not None:
        attributes.update(attrs)
    return Span(
        span_id=span_id(),
        kind="tool",
        name=name,
        start_ns=start_ns,
        end_ns=start_ns + 1_000,
        status="error" if status == "error" else ("empty" if status == "empty" else "ok"),
        attributes=attributes,
    )


def trace(
    *,
    spans: list[Span] | None = None,
    answer: str | None = None,
    route: str | None = None,
    structured: dict[str, Any] | None = None,
    status: str = "ok",
    case_id: str = "c-0001",
) -> Trace:
    return Trace(
        trace_id="11111111-1111-4111-8111-111111111111",
        case_id=case_id,
        started_at="2026-09-23T10:00:00.000Z",
        duration_ms=10,
        spans=spans or [],
        final_output=FinalOutput(answer=answer, route=route, structured=structured),
        usage=Usage(),
        cost=Cost(pricing_version="empty-2026-09-23"),
        status=status,  # type: ignore[arg-type]
        env=env(),
    )
