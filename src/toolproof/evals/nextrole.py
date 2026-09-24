"""The NextRole target adapter and suite specs.

The doc places this file in the ai-career-copilot repo. It lives here instead so
that toolproof ships the adapter and its tests, and the agent repo needs only a
config pointing at it -- one import rather than a second copy of this logic to
keep in step.

``BRANCHES`` is the eleven intents the live router actually returns. The doc
assumed seven and included a ``clarify`` branch that does not exist; the router's
own prompt is the authority, and a branch list that disagrees with it would score
a whole intent zero forever without ever looking like an error. That is why
``assert_branches_exist`` runs at adapter construction rather than being trusted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from toolproof.dataset.case import Case
from toolproof.evals.suites import MIN_PER_BRANCH, SuiteSpec

BRANCHES: tuple[str, ...] = (
    "job_search",
    "email_draft",
    "resume_query",
    "blog_search",
    "followup",
    "profile_update",
    "interview_prep",
    "salary_estimate",
    "skill_gap",
    "career_planning",
    "general",
)
"""The router's eleven intents, in the order its prompt lists them.

Taken from ``INTENT_CLASSIFICATION_PROMPT`` in the agent, not from the design
doc: the doc predates the router growing four intents, and an eval measuring a
branch set the agent does not have is measuring nothing.
"""

SIDE_EFFECT_TOOLS: tuple[str, ...] = ("send_gmail", "save_application", "send_email")
"""Tools that touch the outside world.

The runner refuses to start unless each of these has a stub bound. A routing
suite never executes a branch so it cannot reach them, but the failure suite
does, and a run that could email a real recruiter must fail before it starts
rather than after.
"""

CONFIDENCE_KEY = "confidence"
"""The router returns a stated confidence alongside the intent.

Which means the calibration scorer has a real producer in this agent today, with
no change to the agent: its classification JSON already carries ``confidence``.
"""


ROUTING_SUITE = SuiteSpec(
    name="nextrole.routing",
    branches=BRANCHES,
    min_per_branch=MIN_PER_BRANCH,
    split="test",
)
"""The frozen routing split: 30 per branch across 11 branches, so 330 cases.

The doc sized this as 210 on the assumption of seven branches. Eleven branches
at the doc's own per-branch minimum is 330; keeping 210 would leave 19 per
branch, which its own Wilson table calls noise.
"""

ROUTING_DEV_SUITE = SuiteSpec(
    name="nextrole.routing",
    branches=BRANCHES,
    min_per_branch=10,
    split="dev",
)
"""The dev split, where prompt iteration is allowed.

Ten per branch is enough to see a prompt change move something. Iterating here
and publishing from the test split is the whole of the doc's trap #1.
"""

FAILURE_SUITE = SuiteSpec(
    name="nextrole.failure",
    branches=BRANCHES,
    min_per_branch=3,
    split="test",
)
"""The fault-injection split, which is about tools rather than branches.

Three per branch: the measured quantity is the misreport-as-empty rate over
injected faults, pooled across branches, not a per-branch rate. The branch floor
here only stops the suite from testing one intent's error handling and calling it
the agent's.
"""

SUITES: tuple[SuiteSpec, ...] = (ROUTING_SUITE, ROUTING_DEV_SUITE, FAILURE_SUITE)


class _Tracer(Protocol):
    """Only what the adapter uses, so tests need no real Tracer.

    ``current_run`` rather than the doc example's ``tracer.current``: the
    shipped ``Tracer`` names it ``current_run``, and the adapter follows the
    code, not the prose.
    """

    @property
    def current_run(self) -> Any: ...

    def langchain_handler(self) -> Any: ...


class NextRoleAdapter:
    """Turns a compiled NextRole graph into the runner's target callable.

    Two paths, decided by the suite name rather than by inspecting the case:

    - A ``.routing`` case goes through ``route_probe``, which reads the branch
      the graph would take without executing it. That is what makes a 330-case
      routing run cost eleven classification calls per case instead of eleven
      full branch executions, and it is why a routing suite cannot send an email.
    - Anything else executes the graph with stubs bound.

    The split is on the suite, not on whether ``expect.route`` is set, because a
    failure case also carries a route expectation and must still execute.
    """

    def __init__(
        self,
        build_graph: Callable[..., Any],
        *,
        branches: Sequence[str] = BRANCHES,
        stubs: Mapping[str, Any] | None = None,
        state_from: Callable[[Case], dict[str, Any]] | None = None,
    ) -> None:
        self.build_graph = build_graph
        self.branches = tuple(branches)
        self.stubs = dict(stubs or {})
        self.state_from = state_from or _default_state

    def probe_graph(self) -> Any:
        """The graph used for routing, with branch existence asserted.

        Imported lazily so that ``toolproof.evals.nextrole`` can be imported --
        and its suite specs read by ``validate`` -- without langgraph installed.
        """
        from toolproof.integrations.langgraph import assert_branches_exist, route_probe

        graph = self.build_graph()
        assert_branches_exist(graph, self.branches)
        return route_probe(graph, branch_nodes=self.branches)

    async def __call__(self, case: Case, tracer: _Tracer) -> None:
        if case.suite.endswith(".routing"):
            probe = self.probe_graph()
            result = await probe.ainvoke(
                self.state_from(case),
                config={"callbacks": [tracer.langchain_handler()]},
            )
            # No answer is set: a probe never runs a branch, so there is nothing
            # for a fact or failure scorer to read. Those scorers report
            # not-applicable, which is correct -- not a failure to answer.
            run = _run_of(tracer)
            run.set_output(route=result.next_node, structured=_structured(result))
            return

        graph = self.build_graph(stubs=self.stubs) if self.stubs else self.build_graph()
        out = await graph.ainvoke(
            self.state_from(case),
            config={"callbacks": [tracer.langchain_handler()]},
        )
        mapping = out if isinstance(out, Mapping) else {}
        run = _run_of(tracer)
        run.set_output(
            answer=mapping.get("response"),
            route=mapping.get("intent"),
            structured=_structured_from_state(mapping),
        )


def _run_of(tracer: _Tracer) -> Any:
    """The active run, or a clear error rather than an AttributeError later.

    A target invoked outside ``tracer.run(...)`` has nowhere to record output,
    and the case would silently produce a trace with no final output at all.
    """
    run = tracer.current_run
    if run is None:
        raise RuntimeError(
            "no active tracer run: the NextRole adapter must be invoked inside "
            "'async with tracer.run(...)', which the Runner does for every case"
        )
    return run


def _default_state(case: Case) -> dict[str, Any]:
    """The graph's input state for one case.

    Messages are passed as plain dicts rather than LangChain message objects, so
    the adapter does not depend on which message classes the agent uses.
    """
    messages = [
        {"role": message.role, "content": message.content}
        for message in (case.input.messages or [])
    ]
    state: dict[str, Any] = {"messages": messages}
    if case.input.payload:
        state.update(case.input.payload)
    return state


def _structured(result: Any) -> dict[str, Any] | None:
    """The probe's structured output: the stated confidence, when there is one.

    Read defensively. A router that stops emitting a confidence makes the
    calibration metric not-applicable, which is the honest outcome; it must not
    make the routing run fail.
    """
    state = getattr(result, "state", None)
    if isinstance(state, Mapping):
        return _structured_from_state(state)
    return None


def _structured_from_state(state: Mapping[str, Any]) -> dict[str, Any] | None:
    value = state.get(CONFIDENCE_KEY)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return {CONFIDENCE_KEY: float(value)}


def adapter(
    build_graph: Callable[..., Any],
    *,
    branches: Sequence[str] = BRANCHES,
    stubs: Mapping[str, Any] | None = None,
    state_from: Callable[[Case], dict[str, Any]] | None = None,
) -> NextRoleAdapter:
    """Build the NextRole target adapter."""
    return NextRoleAdapter(build_graph, branches=branches, stubs=stubs, state_from=state_from)


__all__ = [
    "BRANCHES",
    "CONFIDENCE_KEY",
    "FAILURE_SUITE",
    "ROUTING_DEV_SUITE",
    "ROUTING_SUITE",
    "SIDE_EFFECT_TOOLS",
    "SUITES",
    "NextRoleAdapter",
    "adapter",
]
