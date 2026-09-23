"""``route_probe``: read the route without executing a branch.

This is a safety requirement, not an optimisation. NextRole's ``email_draft``
and ``followup`` branches sit next to Gmail send, so a routing eval that
executed a branch could email a real recruiter. The probe compiles the graph
with ``interrupt_before`` on every branch node and an in-memory checkpointer,
runs until the interrupt, and reads the pending next node.

The route is what the graph *did* — the edge it was about to take — not the
``intent`` field in state. A field and an edge can disagree, the edge is what
users experience, and a disagreement is recorded as its own finding rather
than resolved silently.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

_INSTALL_HINT = "route_probe needs the langgraph extra: pip install 'toolproof[langgraph]'"

_LANGGRAPH: Any = None


class RouteProbeError(RuntimeError):
    """The probe could not be built, or the graph does not match the branches."""


@dataclass(frozen=True)
class RouteResult:
    """What the probe observed.

    Attributes:
        route: The branch the graph was about to enter, or ``None`` when it
            finished without reaching one. ``None`` is not-applicable, never a
            guess: a scorer must be able to tell the difference.
        state_route: The route according to the state field, when present.
        disagreement: True when the edge and the state field disagree.
        reached_end: True when the graph completed without interrupting.
        state: The final state, for scorers that need more than the route.
    """

    route: str | None
    state_route: str | None
    disagreement: bool
    reached_end: bool
    state: dict[str, Any]

    @property
    def next_node(self) -> str | None:
        """Alias matching the doc's ``result.next_node``."""
        return self.route


def _require_langgraph() -> Any:
    global _LANGGRAPH
    if _LANGGRAPH is None:
        try:
            from langgraph.checkpoint.memory import MemorySaver
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        _LANGGRAPH = {"MemorySaver": MemorySaver}
    return _LANGGRAPH


class RouteProbe:
    """A graph compiled so that no branch node can run."""

    def __init__(
        self,
        graph: Any,
        branch_nodes: Sequence[str],
        *,
        state_route_key: str = "intent",
    ) -> None:
        self.branch_nodes = list(branch_nodes)
        self.state_route_key = state_route_key
        modules = _require_langgraph()
        self._compiled = graph.compile(
            checkpointer=modules["MemorySaver"](),
            interrupt_before=self.branch_nodes,
        )

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> RouteResult:
        """Run until the interrupt and report the pending branch."""
        merged = self._thread_config(config)
        final_state = await self._compiled.ainvoke(state, config=merged)
        snapshot = await self._compiled.aget_state(merged)
        return self._result(final_state, snapshot, config)

    def invoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> RouteResult:
        """Synchronous counterpart, for sync targets."""
        merged = self._thread_config(config)
        final_state = self._compiled.invoke(state, config=merged)
        snapshot = self._compiled.get_state(merged)
        return self._result(final_state, snapshot, config)

    def _thread_config(self, config: dict[str, Any] | None) -> dict[str, Any]:
        """A fresh checkpointer thread per call.

        Without this, two concurrent eval cases would resume each other's
        graph, which is the same class of bug as cross-attached spans.
        """
        merged: dict[str, Any] = dict(config or {})
        configurable = dict(merged.get("configurable") or {})
        configurable.setdefault("thread_id", f"toolproof-{uuid.uuid4()}")
        merged["configurable"] = configurable
        return merged

    def _result(
        self,
        final_state: Any,
        snapshot: Any,
        config: dict[str, Any] | None,
    ) -> RouteResult:
        pending = tuple(getattr(snapshot, "next", ()) or ())
        route = next((node for node in pending if node in self.branch_nodes), None)

        state = dict(final_state) if isinstance(final_state, dict) else {}
        raw_state_route = state.get(self.state_route_key)
        state_route = str(raw_state_route) if raw_state_route else None

        # Only a comparison of two known values can disagree. An absent state
        # field is not evidence of anything.
        disagreement = route is not None and state_route is not None and route != state_route

        self._record(route, state_route, disagreement, config)
        return RouteResult(
            route=route,
            state_route=state_route,
            disagreement=disagreement,
            reached_end=not pending,
            state=state,
        )

    def _record(
        self,
        route: str | None,
        state_route: str | None,
        disagreement: bool,
        config: dict[str, Any] | None,
    ) -> None:
        """Write the observation onto a span, when a tracer is recording.

        The disagreement travels in the trace rather than raising: it is a
        finding about the agent, and failing the case would discard the very
        run that produced the evidence.
        """
        from toolproof.tracer.tracer import open_named_span

        context = open_named_span("node", "route_probe")
        if context is None:
            return
        handle, finish = context
        try:
            if route is not None:
                handle.set("graph.next", route)
            if state_route is not None:
                handle.set("graph.state_route", state_route)
            handle.set("graph.route_disagreement", disagreement)
            handle.set("graph.branches_executed", 0)
        finally:
            finish(None)


def route_probe(
    graph: Any,
    *,
    branch_nodes: Sequence[str],
    state_route_key: str = "intent",
) -> RouteProbe:
    """Compile ``graph`` so every node in ``branch_nodes`` is interrupted before.

    Args:
        graph: An uncompiled ``StateGraph``.
        branch_nodes: The branches that must never execute. Every name must
            exist in the graph; a renamed branch fails here rather than quietly
            scoring zero forever.
        state_route_key: State field holding the agent's own view of the route,
            compared against the edge for the disagreement finding.

    Raises:
        RouteProbeError: If ``branch_nodes`` is empty or names a node the graph
            does not have.
        ImportError: If the langgraph extra is not installed.
    """
    _require_langgraph()

    if not branch_nodes:
        raise RouteProbeError(
            "branch_nodes must name at least one node: probing with no interrupts "
            "would execute every branch, which is what the probe exists to prevent"
        )

    available = _graph_nodes(graph)
    missing = [name for name in branch_nodes if name not in available]
    if missing:
        raise RouteProbeError(
            f"branch node(s) {missing} are not in the graph. Available nodes: "
            f"{sorted(available)}. A renamed branch must fail here, because an "
            f"interrupt that never fires lets the branch run for real."
        )

    return RouteProbe(graph, branch_nodes, state_route_key=state_route_key)


def _graph_nodes(graph: Any) -> set[str]:
    """The node names of a compiled or uncompiled graph."""
    nodes = getattr(graph, "nodes", None)
    if nodes is None:
        builder = getattr(graph, "builder", None)
        nodes = getattr(builder, "nodes", None)
    if nodes is None:
        raise RouteProbeError(
            f"cannot read nodes from {type(graph).__name__}; pass an uncompiled StateGraph"
        )
    return {str(name) for name in nodes}


def assert_branches_exist(graph: Any, branch_nodes: Sequence[str]) -> None:
    """Raise unless every branch exists. For a target's import-time check.

    The doc's adapter asserts this at import so a renamed branch fails loudly
    rather than scoring zero for a whole run.
    """
    available = _graph_nodes(graph)
    missing = [name for name in branch_nodes if name not in available]
    if missing:
        raise RouteProbeError(
            f"branch node(s) {missing} are not in the graph. Available: {sorted(available)}"
        )


__all__ = [
    "RouteProbe",
    "RouteProbeError",
    "RouteResult",
    "assert_branches_exist",
    "route_probe",
]
