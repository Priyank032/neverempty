"""A tiny LangGraph agent with fake LLM responses.

Stands in for NextRole's orchestrator: ``safety_check`` and
``classify_intent`` run for real, then seven branches that a routing eval must
never execute. Every node appends to :data:`execution_log`, so a test can
assert what did *not* run — which is the whole point of the route probe.

No network, no provider SDK, no real LLM. Used by the integration and replay
tests so they prove the harness rather than the model.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from neverempty import Ok, ToolResult, tool

BRANCHES: list[str] = [
    "job_search",
    "email_draft",
    "blog_search",
    "resume_query",
    "followup",
    "general",
    "clarify",
]
"""The seven branch nodes, mirroring the doc's NextRole list."""

execution_log: list[str] = []
"""Append-only record of which nodes actually ran. Cleared per test."""


class AgentState(TypedDict, total=False):
    query: str
    intent: str
    response: str
    blocked: bool


def _classify(query: str) -> str:
    """Deterministic stand-in for an intent classifier."""
    lowered = query.lower()
    if "recruiter" in lowered or "write to" in lowered:
        return "email_draft"
    if "update" in lowered or "applied" in lowered or "application" in lowered:
        return "followup"
    if "resume" in lowered:
        return "resume_query"
    if "blog" in lowered:
        return "blog_search"
    if "job" in lowered:
        return "job_search"
    if len(query.split()) <= 1:
        return "clarify"
    return "general"


def build_orchestrator(*, lying_state: bool = False, block_everything: bool = False) -> Any:
    """The orchestrator graph, uncompiled.

    Args:
        lying_state: Make ``classify_intent`` write a state field that disagrees
            with the edge it takes. Exercises the doc's rule that the edge is
            authoritative and the disagreement is its own finding.
        block_everything: Make ``safety_check`` reject, so the graph reaches
            ``finalize`` without entering any branch. The probe must then report
            no route rather than guessing one.
    """

    def safety_check(state: AgentState) -> dict[str, Any]:
        execution_log.append("safety_check")
        return {"blocked": block_everything}

    def classify_intent(state: AgentState) -> dict[str, Any]:
        execution_log.append("classify_intent")
        intent = _classify(state.get("query", ""))
        # The state field is deliberately wrong here, while the edge below still
        # uses the real intent: a field and an edge can disagree in production.
        return {"intent": "general" if lying_state else intent}

    def route(state: AgentState) -> str:
        if state.get("blocked"):
            return "finalize"
        return _classify(state.get("query", ""))

    def finalize(state: AgentState) -> dict[str, Any]:
        execution_log.append("finalize")
        return {"response": "blocked" if state.get("blocked") else "done"}

    graph = StateGraph(AgentState)
    graph.add_node("safety_check", safety_check)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("finalize", finalize)

    for branch in BRANCHES:
        graph.add_node(branch, _make_branch(branch))

    graph.set_entry_point("safety_check")
    graph.add_edge("safety_check", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route,
        {name: name for name in [*BRANCHES, "finalize"]},
    )
    for branch in BRANCHES:
        graph.add_edge(branch, "finalize")
    graph.add_edge("finalize", END)
    return graph


def _make_branch(name: str) -> Any:
    def branch(state: AgentState) -> dict[str, Any]:
        execution_log.append(name)
        if name in ("email_draft", "followup"):
            # Stands in for the Gmail send that sits next to these branches.
            execution_log.append(f"SENT EMAIL from {name}")
        return {"response": f"{name} ran"}

    return branch


class FakeChat(GenericFakeChatModel):
    """A fake chat model that reports usage the way a real one does."""

    def _generate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult

        message = AIMessage(
            content="fake answer",
            usage_metadata={"input_tokens": 11, "output_tokens": 5, "total_tokens": 16},
            response_metadata={"model_name": "fake-model-2026-09-01"},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def build_llm_graph() -> Any:
    """A one-node graph that calls the fake LLM, compiled."""
    model = FakeChat(messages=iter([AIMessage(content="unused")]))

    async def answer(state: AgentState) -> dict[str, Any]:
        execution_log.append("answer")
        result = await model.ainvoke(state.get("query", ""))
        return {"response": str(result.content)}

    graph = StateGraph(AgentState)
    graph.add_node("answer", answer)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)
    return graph.compile()


@tool(empty_when=lambda rows: len(rows) == 0)
async def search_jobs(query: str) -> ToolResult[list[dict[str, str]]]:
    """A traced tool, so a test can assert span parenting through a node."""
    return Ok(value=[{"title": "Backend Engineer", "city": "Pune"}])


@tool(never_empty=True, side_effect=True)
async def send_gmail(to: str, body: str) -> str:
    """Tagged ``side_effect``, so the runner preflight refuses to run unstubbed."""
    raise AssertionError("send_gmail must be stubbed in eval mode")


def build_tool_graph() -> Any:
    """A one-node graph that calls a traced tool, compiled."""

    async def find(state: AgentState) -> dict[str, Any]:
        execution_log.append("find")
        await search_jobs(query=state.get("query", ""))
        return {"response": "searched"}

    graph = StateGraph(AgentState)
    graph.add_node("find", find)
    graph.set_entry_point("find")
    graph.add_edge("find", END)
    return graph.compile()


__all__ = [
    "BRANCHES",
    "AgentState",
    "FakeChat",
    "build_llm_graph",
    "build_orchestrator",
    "build_tool_graph",
    "execution_log",
    "search_jobs",
    "send_gmail",
]
