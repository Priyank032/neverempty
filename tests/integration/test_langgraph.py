"""Acceptance row for ``integrations/langgraph``.

    fixture graph routes through the probe with zero branch executions; a
    renamed branch fails at import; handler captures usage from a fake LLM

The zero-branch-execution property is a safety requirement, not an
optimisation: NextRole's ``email_draft`` and ``followup`` branches sit next to
Gmail send, and a routing eval must never be able to email a real recruiter.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from neverempty import MemorySink, Tracer

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from examples.fixture_agent import BRANCHES, build_orchestrator, execution_log

from neverempty.integrations.langgraph import RouteProbeError, route_probe

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _clear_log() -> None:
    execution_log.clear()


def build() -> tuple[Tracer, MemorySink]:
    sink = MemorySink()
    return Tracer(sink=sink), sink


class TestZeroBranchExecutions:
    """The safety property. Every assertion here is about what did *not* run."""

    async def test_routing_never_executes_a_branch(self) -> None:
        probe = route_probe(build_orchestrator(), branch_nodes=BRANCHES)
        result = await probe.ainvoke({"query": "find me a python job in Pune"})

        assert result.route == "job_search"
        assert execution_log == ["safety_check", "classify_intent"]

    async def test_the_email_branch_is_never_reached(self) -> None:
        """The branch that sits next to Gmail send."""
        probe = route_probe(build_orchestrator(), branch_nodes=BRANCHES)
        result = await probe.ainvoke({"query": "write to the recruiter at Razorpay"})

        assert result.route == "email_draft"
        assert "email_draft" not in execution_log
        assert not any("SENT" in entry for entry in execution_log)

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("find me a python job in Pune", "job_search"),
            ("write to the recruiter at Razorpay", "email_draft"),
            ("any update on my application", "followup"),
            ("what does my resume say", "resume_query"),
            ("show me blog posts", "blog_search"),
            ("hello there", "general"),
            ("hmm", "clarify"),
        ],
    )
    async def test_every_branch_is_routable_without_executing(
        self, query: str, expected: str
    ) -> None:
        probe = route_probe(build_orchestrator(), branch_nodes=BRANCHES)
        result = await probe.ainvoke({"query": query})

        assert result.route == expected
        assert expected not in execution_log

    async def test_the_pre_branch_nodes_do_run_for_real(self) -> None:
        """safety_check and classify_intent are what is being measured."""
        probe = route_probe(build_orchestrator(), branch_nodes=BRANCHES)
        await probe.ainvoke({"query": "find me a job"})

        assert "safety_check" in execution_log
        assert "classify_intent" in execution_log

    async def test_concurrent_probes_do_not_share_a_thread(self) -> None:
        """Each probe run needs its own checkpointer thread, or two cases would
        resume each other's graph."""
        import asyncio

        probe = route_probe(build_orchestrator(), branch_nodes=BRANCHES)
        results = await asyncio.gather(
            probe.ainvoke({"query": "find me a job"}),
            probe.ainvoke({"query": "write to the recruiter"}),
            probe.ainvoke({"query": "hello there"}),
        )
        assert [r.route for r in results] == ["job_search", "email_draft", "general"]


class TestRenamedBranchFailsAtImport:
    def test_an_unknown_branch_name_is_rejected_when_the_probe_is_built(self) -> None:
        """A renamed branch must fail loudly, not quietly score zero."""
        with pytest.raises(RouteProbeError) as exc:
            route_probe(build_orchestrator(), branch_nodes=["job_search", "job_serch"])
        message = str(exc.value)
        assert "job_serch" in message
        assert "job_search" in message  # the available nodes are listed

    def test_the_error_names_every_missing_branch(self) -> None:
        with pytest.raises(RouteProbeError) as exc:
            route_probe(build_orchestrator(), branch_nodes=["nope_one", "nope_two"])
        message = str(exc.value)
        assert "nope_one" in message
        assert "nope_two" in message

    def test_an_empty_branch_list_is_rejected(self) -> None:
        """Probing with no interrupts would execute every branch."""
        with pytest.raises(RouteProbeError, match="at least one"):
            route_probe(build_orchestrator(), branch_nodes=[])

    def test_a_correct_branch_list_builds(self) -> None:
        assert route_probe(build_orchestrator(), branch_nodes=BRANCHES) is not None


class TestRouteSource:
    """The route comes from the edge the graph took, not the state field."""

    async def test_the_edge_is_authoritative_when_the_state_field_disagrees(self) -> None:
        """A state field and the actual edge can disagree, and the edge is what
        users experience."""
        probe = route_probe(build_orchestrator(lying_state=True), branch_nodes=BRANCHES)
        result = await probe.ainvoke({"query": "find me a python job in Pune"})

        assert result.route == "job_search"
        assert result.state_route == "general"
        assert result.disagreement is True

    async def test_agreement_is_not_flagged(self) -> None:
        probe = route_probe(build_orchestrator(), branch_nodes=BRANCHES)
        result = await probe.ainvoke({"query": "find me a job"})

        assert result.disagreement is False
        assert result.state_route == result.route

    async def test_a_disagreement_is_recorded_on_the_span(self) -> None:
        tracer, sink = build()
        probe = route_probe(build_orchestrator(lying_state=True), branch_nodes=BRANCHES)
        async with tracer.run(case_id="c1"):
            await probe.ainvoke(
                {"query": "find me a job"}, config={"callbacks": [tracer.langchain_handler()]}
            )

        probe_spans = [s for s in sink.traces[0].spans if "graph.next" in s.attributes]
        assert probe_spans, "no span carried graph.next"
        attributes = probe_spans[0].attributes
        assert attributes["graph.next"] == "job_search"
        assert attributes["graph.state_route"] == "general"
        assert attributes["graph.route_disagreement"] is True

    async def test_a_graph_that_ends_without_interrupting_reports_no_route(self) -> None:
        """Not-applicable, never a guess: the scorer must be able to tell."""
        probe = route_probe(build_orchestrator(block_everything=True), branch_nodes=BRANCHES)
        result = await probe.ainvoke({"query": "anything"})
        assert result.route is None
        assert result.reached_end is True


# On Python 3.10, langgraph does not propagate the callback context into a
# node's async task, so an LLM invoked *inside a node* fires no callback at all:
# not on_chat_model_start, not on_llm_end. Verified by isolating both variables —
# with langchain-core 1.6.6 held constant, the same model called directly fires
# its callbacks on 3.10 and inside a node does not, while on 3.12 both work.
#
# So the handler is not at fault, and its own 26 tests pass on 3.10. These two
# assert an upstream capability that does not exist there. Skipped rather than
# weakened, because the assertion is the right one everywhere it can hold, and
# deleting it would lose coverage on the four versions where it does.
_NODE_CALLBACKS_PROPAGATE = sys.version_info >= (3, 11)

needs_node_callbacks = pytest.mark.skipif(
    not _NODE_CALLBACKS_PROPAGATE,
    reason=(
        "langgraph does not propagate callbacks into a node's async task on "
        "Python 3.10, so an LLM called inside a node emits no callback for the "
        "handler to capture. The handler itself is covered by "
        "tests/integration/test_langchain_handler.py, which passes on 3.10."
    ),
)


class TestHandlerCapturesUsage:
    @needs_node_callbacks
    async def test_usage_from_a_fake_llm_reaches_the_trace(self) -> None:
        from examples.fixture_agent import build_llm_graph

        tracer, sink = build()
        async with tracer.run(case_id="c1"):
            await build_llm_graph().ainvoke(
                {"query": "hello"},
                config={"callbacks": [tracer.langchain_handler()]},
            )

        trace = sink.traces[0]
        assert trace.usage.input_tokens == 11
        assert trace.usage.output_tokens == 5
        assert trace.usage.usage_missing is False

    @needs_node_callbacks
    async def test_the_resolved_model_is_captured_separately(self) -> None:
        from examples.fixture_agent import build_llm_graph

        tracer, sink = build()
        async with tracer.run():
            await build_llm_graph().ainvoke(
                {"query": "hello"},
                config={"callbacks": [tracer.langchain_handler()]},
            )

        llm_spans = [s for s in sink.traces[0].spans if s.kind == "llm"]
        assert llm_spans
        assert llm_spans[0].attributes["gen_ai.response.model"] == "fake-model-2026-09-01"

    async def test_node_spans_are_emitted_for_each_graph_node(self) -> None:
        tracer, sink = build()
        probe = route_probe(build_orchestrator(), branch_nodes=BRANCHES)
        async with tracer.run():
            await probe.ainvoke(
                {"query": "find me a job"},
                config={"callbacks": [tracer.langchain_handler()]},
            )

        node_names = {s.name for s in sink.traces[0].spans if s.kind == "node"}
        assert "safety_check" in node_names
        assert "classify_intent" in node_names

    async def test_no_branch_node_span_is_emitted(self) -> None:
        """If a branch never ran, no span should claim it did."""
        tracer, sink = build()
        probe = route_probe(build_orchestrator(), branch_nodes=BRANCHES)
        async with tracer.run():
            await probe.ainvoke(
                {"query": "write to the recruiter"},
                config={"callbacks": [tracer.langchain_handler()]},
            )

        node_names = {s.name for s in sink.traces[0].spans if s.kind == "node"}
        assert "email_draft" not in node_names

    async def test_a_traced_tool_inside_a_node_parents_correctly(self) -> None:
        from examples.fixture_agent import build_tool_graph

        tracer, sink = build()
        async with tracer.run():
            await build_tool_graph().ainvoke(
                {"query": "python jobs"},
                config={"callbacks": [tracer.langchain_handler()]},
            )

        spans = {s.name: s for s in sink.traces[0].spans}
        assert "search_jobs" in spans
        tool_span = spans["search_jobs"]
        assert tool_span.kind == "tool"
        assert tool_span.parent_id is not None

    async def test_the_handler_works_without_a_tracer_run_active(self) -> None:
        """A handler attached in production outside a run must not raise."""
        tracer, sink = build()
        probe = route_probe(build_orchestrator(), branch_nodes=BRANCHES)
        result = await probe.ainvoke(
            {"query": "find me a job"},
            config={"callbacks": [tracer.langchain_handler()]},
        )
        assert result.route == "job_search"
        assert sink.traces == []

    async def test_a_handler_error_never_breaks_the_graph(self) -> None:
        """Telemetry must never be what fails a production request."""
        tracer, _ = build()
        handler = tracer.langchain_handler()
        probe = route_probe(build_orchestrator(), branch_nodes=BRANCHES)

        async with tracer.run():
            # Corrupt the handler's internal state the way a version skew might.
            handler._spans = None
            result = await probe.ainvoke(
                {"query": "find me a job"}, config={"callbacks": [handler]}
            )

        assert result.route == "job_search"


class TestMissingExtra:
    def test_importing_without_langgraph_gives_an_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith(("langgraph", "langchain_core")):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        from neverempty.integrations import langgraph as module

        monkeypatch.setattr(module, "_LANGGRAPH", None)
        with pytest.raises(ImportError, match=r"neverempty\[langgraph\]"):
            module.route_probe(object(), branch_nodes=["a"])


class TestSyncAndHelpers:
    def test_the_probe_works_synchronously(self) -> None:
        """A sync target must be able to route too."""
        probe = route_probe(build_orchestrator(), branch_nodes=BRANCHES)
        result = probe.invoke({"query": "find me a python job"})

        assert result.route == "job_search"
        assert "job_search" not in execution_log

    def test_next_node_is_an_alias_for_route(self) -> None:
        """The doc's adapter reads result.next_node."""
        probe = route_probe(build_orchestrator(), branch_nodes=BRANCHES)
        result = probe.invoke({"query": "find me a job"})
        assert result.next_node == result.route == "job_search"

    def test_assert_branches_exist_passes_on_a_correct_list(self) -> None:
        from neverempty.integrations.langgraph import assert_branches_exist

        assert_branches_exist(build_orchestrator(), BRANCHES)

    def test_assert_branches_exist_fails_on_a_renamed_branch(self) -> None:
        """This is the target adapter's import-time guard."""
        from neverempty.integrations.langgraph import assert_branches_exist

        with pytest.raises(RouteProbeError) as exc:
            assert_branches_exist(build_orchestrator(), ["job_search", "renamed"])
        assert "renamed" in str(exc.value)

    def test_a_graph_without_nodes_is_rejected_clearly(self) -> None:
        from neverempty.integrations.langgraph import assert_branches_exist

        with pytest.raises(RouteProbeError, match="uncompiled"):
            assert_branches_exist(object(), ["a"])

    def test_a_caller_supplied_thread_id_is_respected(self) -> None:
        """So a target can correlate a probe run with its own checkpointer."""
        probe = route_probe(build_orchestrator(), branch_nodes=BRANCHES)
        result = probe.invoke(
            {"query": "find me a job"}, config={"configurable": {"thread_id": "mine-1"}}
        )
        assert result.route == "job_search"

    def test_the_final_state_is_exposed_for_scorers(self) -> None:
        probe = route_probe(build_orchestrator(), branch_nodes=BRANCHES)
        result = probe.invoke({"query": "find me a job"})
        assert result.state["intent"] == "job_search"

    def test_a_compiled_graph_is_also_readable_for_the_branch_check(self) -> None:
        from neverempty.integrations.langgraph import assert_branches_exist

        assert_branches_exist(build_orchestrator().compile(), BRANCHES)
