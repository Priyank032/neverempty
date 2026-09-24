"""Reading a trace the way every scorer must read it.

Centralised so two scorers cannot disagree about what "the tools this agent
called" means. A trace is a cross-language contract, so these helpers assume
only the documented attribute names and never that a Python adapter wrote it.
"""

from __future__ import annotations

import json
from typing import Any

from toolproof.core.trace import Span, Trace

FINALIZE_NODES = frozenset({"finalize", "__end__", "end"})
"""Nodes that mark the end of routing rather than a branch."""

PLUMBING_NODES = frozenset(
    {
        "safety_check",
        "classify_intent",
        "__start__",
        "start",
        "router",
        "route",
        "agent",
    }
)
"""Nodes that run on every path, so none of them can be the chosen branch.

Named here rather than inferred: a scorer guessing which node was a branch
would silently change a published number when a graph gained a node.
"""


def tool_spans(trace: Trace) -> list[Span]:
    """Tool spans in call order, by start time.

    Ordered by ``start_ns`` rather than by list position, because a concurrent
    agent flushes spans in completion order and "the first tool called" has to
    mean the first one started.
    """
    return sorted(
        (span for span in trace.spans if span.kind == "tool"),
        key=lambda span: (span.start_ns, span.span_id),
    )


def node_spans(trace: Trace) -> list[Span]:
    """Node spans in execution order."""
    return sorted(
        (span for span in trace.spans if span.kind == "node"),
        key=lambda span: (span.start_ns, span.span_id),
    )


def tool_names(trace: Trace) -> list[str]:
    """The tools called, in order, with duplicates kept.

    ``tool.name`` is preferred over the span name: the attribute is the
    documented field, and an adapter may name the span differently.
    """
    names: list[str] = []
    for span in tool_spans(trace):
        recorded = span.attributes.get("tool.name")
        names.append(str(recorded) if isinstance(recorded, str) and recorded else span.name)
    return names


def first_call_of(trace: Trace, tool: str) -> Span | None:
    """The first span for one tool, or ``None`` if it was never called."""
    for span, name in zip(tool_spans(trace), tool_names(trace), strict=True):
        if name == tool:
            return span
    return None


def recorded_args(span: Span) -> dict[str, Any] | None:
    """The call's arguments, or ``None`` when they cannot be read.

    ``None`` covers two harness gaps that must not read as agent errors: an
    adapter that records no arguments, and a payload the JSON encoder mangled.
    Both are "not measured", never "wrong".
    """
    raw = span.attributes.get("tool.args")
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def route_prediction(trace: Trace) -> tuple[str, str, Span | None] | None:
    """The predicted route, its source, and the span it came from.

    The doc fixes the order: ``graph.next`` from the probe span, then the last
    node span before ``finalize``, then ``final_output.route``. The order is not
    a preference; the three can disagree, and a scorer that chose differently
    would make two runs of the same suite mean different things.
    """
    for span in trace.spans:
        candidate = span.attributes.get("graph.next")
        if isinstance(candidate, str) and candidate:
            return candidate, "graph.next", span

    branch = _last_branch_node(trace)
    if branch is not None:
        return branch.name, "node_span", branch

    route = trace.final_output.route
    if isinstance(route, str) and route:
        return route, "final_output.route", None
    return None


def _last_branch_node(trace: Trace) -> Span | None:
    """The last node that was a branch, ignoring plumbing and anything after
    ``finalize``."""
    candidates: list[Span] = []
    for span in node_spans(trace):
        name = _node_name(span)
        if name in FINALIZE_NODES:
            break
        if name in PLUMBING_NODES:
            continue
        candidates.append(span)
    return candidates[-1] if candidates else None


def _node_name(span: Span) -> str:
    recorded = span.attributes.get("graph.node")
    return str(recorded) if isinstance(recorded, str) and recorded else span.name


def failed_tool_spans(trace: Trace) -> list[Span]:
    """Tool spans that errored, whether injected or genuine.

    A real upstream failure during a live run is the same measurement as an
    injected one; the fault flag records how the failure was *caused*, not
    whether it counts.
    """
    return [span for span in tool_spans(trace) if span.attributes.get("tool.status") == "error"]


def empty_tool_spans(trace: Trace) -> list[Span]:
    """Tool spans that truthfully returned nothing.

    Available as a separate question from ``failed_tool_spans`` only because
    ``Empty`` and ``Err`` are different statuses. That distinction is what makes
    the false-alarm rate measurable at all.
    """
    return [span for span in tool_spans(trace) if span.attributes.get("tool.status") == "empty"]


def answer_of(trace: Trace) -> str | None:
    """The agent's final answer, or ``None`` when the field was never set.

    ``None`` and ``""`` are kept apart on purpose: a missing answer is a harness
    gap, and an empty answer is an agent that replied with nothing. The first
    must not be scored; the second must.
    """
    return trace.final_output.answer


def structured_of(trace: Trace) -> dict[str, Any] | None:
    """The agent's structured output, or ``None`` when it produced none.

    Read through a helper for the same reason as ``answer_of``: the field is
    part of a cross-language contract, and a scorer reaching into the model
    directly would be one more place to update when the shape moves.
    """
    return trace.final_output.structured


__all__ = [
    "FINALIZE_NODES",
    "PLUMBING_NODES",
    "answer_of",
    "empty_tool_spans",
    "failed_tool_spans",
    "first_call_of",
    "node_spans",
    "recorded_args",
    "route_prediction",
    "structured_of",
    "tool_names",
    "tool_spans",
]
