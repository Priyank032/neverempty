"""The entrypoint ``evals/neverempty.toml`` names, and the recording stubs.

``run_nextrole_router`` is what ``[target].entrypoint`` resolves to today; see
:mod:`neverempty.evals.nextrole_live` for why. ``run_nextrole`` is the graph
target, kept for an agent whose live path is a LangGraph graph. It is deliberately
thin: all the behaviour is in :class:`~neverempty.evals.nextrole.NextRoleAdapter`,
which is unit-tested without langgraph installed. This module is the seam where
the real agent gets imported, so importing it fails cleanly when the agent is not
on the path rather than at some point mid-run.

``STUBS`` binds every tool in ``side_effect_tools``. The runner refuses to start
if one is missing, which is the check that stops a fault-injection run from
emailing a real recruiter.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from neverempty.core.results import Ok, ToolResult
from neverempty.dataset.case import Case
from neverempty.evals.nextrole import SIDE_EFFECT_TOOLS, NextRoleAdapter

_GRAPH_FACTORY_ENV = "NEXTROLE_GRAPH_FACTORY"
"""Where to import the agent's graph builder from, as ``module:name``.

An environment variable rather than a hard import, because neverempty must not
depend on the agent repo being importable: its own test suite, its wheel build
and ``neverempty validate`` all have to work without ai-career-copilot present.
"""


class RecordingStub:
    """A stub that records its calls and never touches the outside world.

    Returns an explicit ``Ok`` rather than ``None``: a stub returning ``None``
    under strict mode would raise ``AmbiguousEmptyError``, and a stub returning a
    falsy value would be the very ambiguity this library exists to remove. What a
    stubbed side effect means is "this would have happened", which is a fact, not
    an absence.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> ToolResult[dict[str, Any]]:
        self.calls.append(dict(kwargs))
        return Ok(value={"stubbed": self.name, "call_index": len(self.calls) - 1})

    async def acall(self, **kwargs: Any) -> ToolResult[dict[str, Any]]:
        return self(**kwargs)

    def reset(self) -> None:
        self.calls.clear()


STUBS: dict[str, RecordingStub] = {name: RecordingStub(name) for name in SIDE_EFFECT_TOOLS}
"""One recording stub per side-effecting tool.

Built eagerly so ``[target].stubs`` resolves to a populated mapping at preflight,
before any case runs.
"""


def _graph_factory() -> Callable[..., Any]:
    """Import the agent's graph builder, named by the environment.

    Raises with the variable name in the message: a missing agent is a setup
    problem with an obvious fix, and it should not read as a neverempty bug.
    """
    spec = os.environ.get(_GRAPH_FACTORY_ENV)
    if not spec:
        raise RuntimeError(
            f"set {_GRAPH_FACTORY_ENV} to the agent's graph builder as "
            f"'module:name' (for example "
            f"'app.agents.graph:build_orchestrator'). neverempty does not import "
            f"the agent repo by default, so its own tests and wheel build do not "
            f"depend on it being present."
        )
    if ":" not in spec:
        raise RuntimeError(f"{_GRAPH_FACTORY_ENV} must be 'module:name', got {spec!r}")
    module_name, _, attribute = spec.partition(":")
    from importlib import import_module

    module = import_module(module_name)
    factory = getattr(module, attribute, None)
    if factory is None:
        raise RuntimeError(f"{module_name!r} has no attribute {attribute!r}")
    if not callable(factory):
        raise RuntimeError(f"{spec!r} is not callable")
    resolved: Callable[..., Any] = factory
    return resolved


_adapter: NextRoleAdapter | None = None


def _resolved_adapter() -> NextRoleAdapter:
    """The adapter, built once per process.

    Cached because ``probe_graph`` asserts every branch exists against the
    compiled graph, and compiling is the expensive part of a routing run.
    """
    global _adapter
    if _adapter is None:
        _adapter = NextRoleAdapter(_graph_factory(), stubs=STUBS)
    return _adapter


async def run_nextrole(case: Case, tracer: Any) -> None:
    """The runner's target: ``async (Case, Tracer) -> None``."""
    await _resolved_adapter()(case, tracer)


_router: Any = None


def _live_router() -> Any:
    """The agent's ``IntentRouterAgent``, built once per process.

    Imported here rather than at module import for the same reason as
    ``_graph_factory``: neverempty must import without the agent repo present.
    """
    global _router
    if _router is None:
        try:
            from app.agents.intent_router import (  # type: ignore[import-not-found]
                IntentRouterAgent,
            )
        except ImportError as exc:
            raise RuntimeError(
                "cannot import app.agents.intent_router: put ai-career-copilot/backend "
                "on PYTHONPATH (and install its requirements) to run the live router"
            ) from exc
        _router = IntentRouterAgent()
    return _router


async def run_nextrole_router(case: Case, tracer: Any) -> None:
    """Routing target for the live router. See :mod:`neverempty.evals.nextrole_live`."""
    if not case.suite.endswith(".routing"):
        raise RuntimeError(
            f"run_nextrole_router only runs routing suites, got {case.suite!r}: a failure "
            f"case executes a branch, and the live branches are not wrapped with @tool"
        )
    from neverempty.evals.nextrole_live import route_live

    intent, confidence = await route_live(case, _live_router())
    run = tracer.current_run
    if run is None:
        raise RuntimeError("no active tracer run: the runner opens one for every case")
    run.set_output(route=intent, structured={"confidence": confidence})


__all__ = ["STUBS", "RecordingStub", "run_nextrole", "run_nextrole_router"]
