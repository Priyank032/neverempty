"""The NextRole adapter: routing goes through the probe, faults execute.

M9's acceptance row needs the suites to run at all, which means the adapter has
to put a routing case through ``route_probe`` (zero branch executions, so no
email can be sent) and a failure case through the real graph with stubs bound.
"""

from __future__ import annotations

from typing import Any

import pytest

from neverempty.dataset.case import Case
from neverempty.evals.nextrole import (
    BRANCHES,
    FAILURE_SUITE,
    ROUTING_SUITE,
    SIDE_EFFECT_TOOLS,
    adapter,
)


def case(suite: str = "nextrole.routing", content: str = "find me a job") -> Case:
    return Case.model_validate(
        {
            "id": "nr-route-0001",
            "suite": suite,
            "split": "test",
            "input": {"messages": [{"role": "user", "content": content}]},
            "expect": {"route": {"label": "job_search"}},
            "provenance": {"method": "human", "labeller": "priyank"},
        }
    )


class FakeOutput:
    def __init__(self) -> None:
        self.answer: str | None = None
        self.route: str | None = None
        self.structured: dict[str, Any] | None = None

    def set_output(
        self,
        *,
        answer: str | None = None,
        route: str | None = None,
        structured: dict[str, Any] | None = None,
    ) -> None:
        if answer is not None:
            self.answer = answer
        if route is not None:
            self.route = route
        if structured is not None:
            self.structured = structured


class FakeTracer:
    def __init__(self) -> None:
        self.current_run = FakeOutput()
        self.handlers = 0

    def langchain_handler(self) -> object:
        self.handlers += 1
        return object()


class FakeResult:
    def __init__(self, route: str | None, state: dict[str, Any] | None = None) -> None:
        self.route = route
        self.next_node = route
        self.state = state or {}


class FakeProbe:
    def __init__(self, route: str | None, state: dict[str, Any] | None = None) -> None:
        self.route = route
        self.state = state or {}
        self.calls = 0

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> FakeResult:
        self.calls += 1
        self.seen_state = state
        self.seen_config = config
        return FakeResult(self.route, self.state)


class FakeGraph:
    """A graph that records whether it was executed, and with what."""

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {"response": "here are jobs", "intent": "job_search"}
        self.calls = 0
        self.built_with_stubs: dict[str, Any] | None = None

    async def ainvoke(
        self, state: dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.calls += 1
        self.seen_state = state
        return self.response


class TestBranches:
    def test_eleven_real_intents_not_the_docs_seven(self) -> None:
        """The router's own prompt is the authority, not the design doc."""
        assert len(BRANCHES) == 11
        assert "clarify" not in BRANCHES
        for intent in (
            "profile_update",
            "interview_prep",
            "salary_estimate",
            "skill_gap",
            "career_planning",
        ):
            assert intent in BRANCHES

    def test_routing_suite_is_thirty_per_branch(self) -> None:
        assert ROUTING_SUITE.min_per_branch == 30
        assert ROUTING_SUITE.expected_total == 330

    def test_side_effect_tools_are_declared(self) -> None:
        """The runner refuses to start unless each of these has a stub."""
        assert "send_gmail" in SIDE_EFFECT_TOOLS
        assert "save_application" in SIDE_EFFECT_TOOLS


class TestRoutingPath:
    async def test_routing_uses_the_probe_and_never_executes_the_graph(self) -> None:
        graph = FakeGraph()
        probe = FakeProbe("job_search")
        subject = adapter(lambda **_: graph)
        subject.probe_graph = lambda: probe  # type: ignore[method-assign]

        tracer = FakeTracer()
        await subject(case(), tracer)

        assert probe.calls == 1
        assert graph.calls == 0, "a routing case must never execute a branch"
        assert tracer.current_run.route == "job_search"
        assert tracer.current_run.answer is None

    async def test_no_answer_is_set_on_a_routing_case(self) -> None:
        """A probe runs no branch, so there is no answer. Leaving it None makes
        the fact scorers report not-applicable rather than scoring an empty."""
        probe = FakeProbe("general")
        subject = adapter(lambda **_: FakeGraph())
        subject.probe_graph = lambda: probe  # type: ignore[method-assign]
        tracer = FakeTracer()
        await subject(case(), tracer)
        assert tracer.current_run.answer is None

    async def test_the_tracers_handler_is_passed_to_the_probe(self) -> None:
        probe = FakeProbe("job_search")
        subject = adapter(lambda **_: FakeGraph())
        subject.probe_graph = lambda: probe  # type: ignore[method-assign]
        tracer = FakeTracer()
        await subject(case(), tracer)
        assert tracer.handlers == 1
        assert "callbacks" in (probe.seen_config or {})

    async def test_a_stated_confidence_reaches_structured_output(self) -> None:
        """This is what makes the calibration scorer work on this agent with no
        change to the agent: its classifier already returns a confidence."""
        probe = FakeProbe("job_search", {"confidence": 0.82})
        subject = adapter(lambda **_: FakeGraph())
        subject.probe_graph = lambda: probe  # type: ignore[method-assign]
        tracer = FakeTracer()
        await subject(case(), tracer)
        assert tracer.current_run.structured == {"confidence": 0.82}

    async def test_a_missing_confidence_is_absent_not_zero(self) -> None:
        probe = FakeProbe("job_search", {})
        subject = adapter(lambda **_: FakeGraph())
        subject.probe_graph = lambda: probe  # type: ignore[method-assign]
        tracer = FakeTracer()
        await subject(case(), tracer)
        assert tracer.current_run.structured is None

    async def test_a_non_numeric_confidence_is_dropped(self) -> None:
        """A router returning a string or a bool has a bug; recording it as a
        number would put a fabricated prediction in the reliability curve."""
        for bad in ("high", True, None):
            probe = FakeProbe("job_search", {"confidence": bad})
            subject = adapter(lambda **_: FakeGraph())

            def probe_graph(bound: FakeProbe = probe) -> FakeProbe:
                return bound

            subject.probe_graph = probe_graph  # type: ignore[method-assign]
            tracer = FakeTracer()
            await subject(case(), tracer)
            assert tracer.current_run.structured is None

    async def test_an_unreached_route_is_none_not_a_guess(self) -> None:
        probe = FakeProbe(None)
        subject = adapter(lambda **_: FakeGraph())
        subject.probe_graph = lambda: probe  # type: ignore[method-assign]
        tracer = FakeTracer()
        await subject(case(), tracer)
        assert tracer.current_run.route is None

    async def test_messages_are_passed_as_plain_dicts(self) -> None:
        """So the adapter does not depend on the agent's message classes."""
        probe = FakeProbe("job_search")
        subject = adapter(lambda **_: FakeGraph())
        subject.probe_graph = lambda: probe  # type: ignore[method-assign]
        await subject(case(content="hello there"), FakeTracer())
        assert probe.seen_state["messages"] == [{"role": "user", "content": "hello there"}]


class TestExecutionPath:
    async def test_a_failure_case_executes_the_graph(self) -> None:
        graph = FakeGraph()
        subject = adapter(lambda **_: graph)
        tracer = FakeTracer()
        await subject(case(suite="nextrole.failure"), tracer)
        assert graph.calls == 1
        assert tracer.current_run.answer == "here are jobs"
        assert tracer.current_run.route == "job_search"

    async def test_stubs_are_bound_when_executing(self) -> None:
        seen: dict[str, Any] = {}

        def build(**kwargs: Any) -> FakeGraph:
            seen.update(kwargs)
            return FakeGraph()

        subject = adapter(build, stubs={"send_gmail": object()})
        await subject(case(suite="nextrole.failure"), FakeTracer())
        assert "stubs" in seen
        assert "send_gmail" in seen["stubs"]

    async def test_a_graph_returning_no_response_leaves_the_answer_none(self) -> None:
        """An agent that produced nothing is a harness gap, not an empty answer,
        and the two must stay distinguishable."""
        graph = FakeGraph({"intent": "general"})
        subject = adapter(lambda **_: graph)
        tracer = FakeTracer()
        await subject(case(suite="nextrole.failure"), tracer)
        assert tracer.current_run.answer is None
        assert tracer.current_run.route == "general"

    async def test_a_non_mapping_result_does_not_raise(self) -> None:
        class Odd:
            async def ainvoke(self, state: Any, config: Any = None) -> str:
                return "not a mapping"

        subject = adapter(lambda **_: Odd())
        tracer = FakeTracer()
        await subject(case(suite="nextrole.failure"), tracer)
        assert tracer.current_run.answer is None


class TestSuiteSpecs:
    def test_failure_suite_has_a_lower_branch_floor(self) -> None:
        """The failure suite measures a pooled rate over injected faults, not a
        per-branch rate, so it does not need thirty per branch."""
        assert FAILURE_SUITE.min_per_branch < ROUTING_SUITE.min_per_branch
        assert FAILURE_SUITE.split == "test"

    def test_probe_graph_asserts_branches_exist(self) -> None:
        """A renamed branch must fail loudly, not score zero."""
        subject = adapter(lambda **_: object())
        with pytest.raises(Exception):  # noqa: B017 - langgraph absent or assert fails
            subject.probe_graph()
