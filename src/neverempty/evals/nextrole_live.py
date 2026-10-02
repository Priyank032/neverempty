"""Routing target for the router NextRole actually serves.

``NextRoleAdapter`` drives a LangGraph graph, and the only graph in the agent
with intent branches is ``NextRoleOrchestrator`` -- which nothing in the app
imports. The live chat endpoint (``app/api/v1/chat.py``) classifies with
``IntentRouterAgent.classify_intent`` and dispatches with an if/elif chain, and
that orchestrator graph has nodes for only six of the eleven intents. Measuring
it would score dead code, so this target calls the live router directly, with
the arguments the endpoint passes:

- ``message`` is the final user turn.
- ``conversation_history`` is every turn *including* that one: the endpoint
  saves the user message before it loads history, so the router sees it twice.
- ``user_profile`` is ``None`` and ``context`` is ``None``: the labels were
  written about the message, not about a particular user's profile or page.

Routing only. A failure case has to execute a branch, and the live branches are
not wrapped with ``@tool`` yet, so a fault there would never fire.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from neverempty.dataset.case import Case

ROUTER_ERROR_FALLBACKS: frozenset[str] = frozenset(
    {
        "I encountered an issue understanding your request. Could you try again?",
        "I'm not sure I understood that. Could you please rephrase your request?",
    }
)
"""The clarification texts ``classify_intent`` returns from its own except paths.

It returns ``general`` on any exception (bad key, outage, unparseable JSON)
rather than raising. Scored as a route, every such failure would count as a
correct answer on the thirty ``general`` cases and a wrong one everywhere else:
an outage reported as a measurement. They are raised here instead, so the case
errors and the runner reports it as an error.
"""


class RouterFailedError(RuntimeError):
    """The live router fell back to its error answer instead of classifying."""


def _final_user_message(case: Case) -> str:
    for message in reversed(case.input.messages or []):
        if message.role == "user":
            return message.content
    raise ValueError(f"case {case.id!r} has no user message to route")


def _history(case: Case) -> list[Any]:
    # The router reads only ``.role`` and ``.content``.
    return [
        SimpleNamespace(role=message.role, content=message.content)
        for message in (case.input.messages or [])
    ]


async def route_live(case: Case, router: Any) -> tuple[str, float]:
    """``(intent, confidence)`` from the live router for one case."""
    if not case.input.messages:
        raise ValueError(f"case {case.id!r}: the live router target needs input.messages")
    result = await router.classify_intent(
        message=_final_user_message(case),
        user_profile=None,
        conversation_history=_history(case),
        context=None,
    )
    if result.requires_clarification and result.clarification_question in ROUTER_ERROR_FALLBACKS:
        raise RouterFailedError(
            f"case {case.id!r}: the router returned its error fallback "
            f"({result.clarification_question!r}); check the LLM key and logs"
        )
    intent = getattr(result.intent, "value", result.intent)
    return str(intent), float(result.confidence)


__all__ = ["ROUTER_ERROR_FALLBACKS", "RouterFailedError", "route_live"]
