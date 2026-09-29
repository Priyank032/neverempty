"""The scorers through the real runner, against the real fixture agent.

The unit tests build traces by hand, which proves the scorers read the contract.
This proves the contract is what the adapters actually write: a span shape the
scorers can read, produced by a LangGraph agent through the LangChain callback
handler, with no test-only plumbing in between.

It is also where the headline experiment runs for the first time end to end: a
fault is injected into a real tool, a real agent answers, and the misreport rate
comes out of a real report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neverempty import Case, Dataset, Runner, Tracer, scorers
from neverempty.tracer.sinks import MemorySink

pytest.importorskip("langgraph")

from examples import fixture_agent


def routing_case(case_id: str, query: str, label: str, **extra: Any) -> dict[str, Any]:
    expect: dict[str, Any] = {"route": {"label": label}}
    expect.update(extra.pop("expect", {}))
    return {
        "schema_version": 1,
        "id": case_id,
        "suite": "fixture.routing",
        "split": "dev",
        "input": {"messages": [{"role": "user", "content": query}]},
        "expect": expect,
        **extra,
    }


BRANCH_QUERIES: dict[str, str] = {
    # Queries the fixture classifier actually routes, rather than the branch
    # names themselves: a test that fed the label back in would prove only that
    # the label round-trips.
    "job_search": "show me remote python jobs in Pune",
    "email_draft": "can you write to the recruiter about this role",
    "blog_search": "find that blog post about system design interviews",
    "resume_query": "does my resume mention kubernetes",
    "followup": "it has been 10 days since I applied, any update",
    "general": "how should I prepare for a backend interview",
    "clarify": "help",
}


def write_dataset(path: Path, rows: list[dict[str, Any]]) -> Dataset:
    import json

    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    return Dataset.load(path)


@pytest.fixture(autouse=True)
def _clear_log() -> Any:
    fixture_agent.execution_log.clear()
    yield
    fixture_agent.execution_log.clear()


def orchestrator_target(**build: Any) -> Any:
    """A target that runs the fixture orchestrator under the tracer."""
    compiled = fixture_agent.build_orchestrator(**build).compile()

    async def target(case: Case, tracer: Tracer) -> None:
        assert case.input.messages is not None
        query = case.input.messages[-1].content
        run = tracer.current_run
        assert run is not None
        result = await compiled.ainvoke(
            {"query": query},
            config={"callbacks": [tracer.langchain_handler()]},
        )
        run.set_output(answer=result.get("response"), route=result.get("intent"))

    return target


class TestRoutingThroughTheRunner:
    async def test_every_branch_is_scored_from_a_real_trace(self, tmp_path: Path) -> None:
        """The route scorer reads node spans the LangChain handler wrote, with
        no probe: the second source in the fallback order, exercised for real."""
        dataset = write_dataset(
            tmp_path / "routing.jsonl",
            [
                routing_case(f"fx-route-{index:04d}", BRANCH_QUERIES[branch], branch)
                for index, branch in enumerate(fixture_agent.BRANCHES)
            ],
        )
        runner = Runner(
            target=orchestrator_target(),
            scorers=[scorers.route()],
            tracer=Tracer(sink=MemorySink()),
            stubs={"send_gmail": lambda **kwargs: "stubbed"},
        )
        report = await runner.run(dataset)

        assert report.status == "ok"
        assert report.complete is True
        assert report.unscored_ids() == []
        verdicts = [outcome.scores["route"] for outcome in report.outcomes]
        assert all(verdict.passed for verdict in verdicts)
        assert {verdict.detail["source"] for verdict in verdicts} == {"node_span"}

    async def test_a_wrong_label_fails_rather_than_erroring(self, tmp_path: Path) -> None:
        dataset = write_dataset(
            tmp_path / "routing.jsonl",
            [routing_case("fx-route-0001", BRANCH_QUERIES["job_search"], "email_draft")],
        )
        runner = Runner(
            target=orchestrator_target(),
            scorers=[scorers.route()],
            tracer=Tracer(sink=MemorySink()),
            stubs={"send_gmail": lambda **kwargs: "stubbed"},
        )
        report = await runner.run(dataset)

        verdict = report.outcomes[0].scores["route"]
        assert verdict.passed is False
        assert verdict.detail["predicted"] == "job_search"
        assert verdict.detail["expected"] == "email_draft"

    async def test_a_forbidden_tool_is_scored_from_the_real_span(self, tmp_path: Path) -> None:
        dataset = write_dataset(
            tmp_path / "routing.jsonl",
            [
                routing_case(
                    "fx-route-0001",
                    BRANCH_QUERIES["job_search"],
                    "job_search",
                    expect={"forbidden_tools": ["send_gmail"]},
                )
            ],
        )
        runner = Runner(
            target=orchestrator_target(),
            scorers=[scorers.route(), scorers.forbidden_tools()],
            tracer=Tracer(sink=MemorySink()),
            stubs={"send_gmail": lambda **kwargs: "stubbed"},
        )
        report = await runner.run(dataset)

        assert report.outcomes[0].scores["forbidden_tools"].passed is True

    async def test_a_case_with_no_route_expectation_is_not_applicable(self, tmp_path: Path) -> None:
        """Through the runner, "not applicable" has to land in the outcome as
        not-applicable, not as a failed score."""
        dataset = write_dataset(
            tmp_path / "routing.jsonl",
            [
                {
                    "schema_version": 1,
                    "id": "fx-tool-0001",
                    "suite": "fixture.routing",
                    "split": "dev",
                    "input": {
                        "messages": [{"role": "user", "content": BRANCH_QUERIES["job_search"]}]
                    },
                    "expect": {"forbidden_tools": ["send_gmail"]},
                }
            ],
        )
        runner = Runner(
            target=orchestrator_target(),
            scorers=[scorers.route(), scorers.forbidden_tools()],
            tracer=Tracer(sink=MemorySink()),
            stubs={"send_gmail": lambda **kwargs: "stubbed"},
        )
        report = await runner.run(dataset)

        outcome = report.outcomes[0]
        assert "route" in outcome.not_applicable
        assert "route" not in outcome.scores
        assert outcome.scores["forbidden_tools"].passed is True
        assert outcome.scored is True


class TestFaultInjectionEndToEnd:
    """The headline experiment, through a real tool and a real fault."""

    def fault_dataset(self, path: Path, *, answer_style: str) -> Dataset:
        return write_dataset(
            path,
            [
                {
                    "schema_version": 1,
                    "id": f"fx-fault-{answer_style}",
                    "suite": "fixture.failure",
                    "split": "dev",
                    "input": {"messages": [{"role": "user", "content": "python jobs?"}]},
                    "faults": [{"tool": "search_jobs", "kind": "timeout", "after_calls": 0}],
                    "expect": {
                        "forbidden_claims": [
                            {"id": "absence", "statement": "no matching jobs", "match": "contains"}
                        ]
                    },
                }
            ],
        )

    def tool_target(self, answer: str) -> Any:
        """Calls the real traced tool, then answers with the given wording."""

        async def target(case: Case, tracer: Tracer) -> None:
            run = tracer.current_run
            assert run is not None
            result = await fixture_agent.search_jobs(query="python")
            run.set_output(answer=answer, structured={"tool_status": result.status})

        return target

    async def test_an_absence_claim_under_a_real_fault_is_a_misreport(self, tmp_path: Path) -> None:
        """The fault is injected by the ``@tool`` wrapper through a context
        variable, with no change to the tool's own code. The scorer then reads
        the error span the wrapper wrote."""
        dataset = self.fault_dataset(tmp_path / "failure.jsonl", answer_style="bad")
        runner = Runner(
            target=self.tool_target("There are no matching jobs for you."),
            scorers=[scorers.failure_handling(), scorers.forbidden_claims()],
            tracer=Tracer(sink=MemorySink()),
        )
        report = await runner.run(dataset)

        outcome = report.outcomes[0]
        assert outcome.scores["failure_handling"].passed is False
        assert outcome.scores["failure_handling"].detail["outcome"] == "misreport"
        assert outcome.scores["failure_handling"].detail["fault_injected"] is True
        assert outcome.scores["forbidden_claims"].passed is False

    async def test_reporting_the_failure_passes(self, tmp_path: Path) -> None:
        dataset = self.fault_dataset(tmp_path / "failure.jsonl", answer_style="good")
        runner = Runner(
            target=self.tool_target("The job search tool failed, so I could not check."),
            scorers=[scorers.failure_handling(), scorers.forbidden_claims()],
            tracer=Tracer(sink=MemorySink()),
        )
        report = await runner.run(dataset)

        outcome = report.outcomes[0]
        assert outcome.scores["failure_handling"].passed is True
        assert outcome.scores["failure_handling"].detail["outcome"] == "reported"
        assert outcome.scores["forbidden_claims"].passed is True

    async def test_without_a_fault_the_scorer_is_not_applicable(self, tmp_path: Path) -> None:
        """The same case, same answer, no fault declared. Nothing failed, so
        there is nothing to measure — and the identical wording that was a
        misreport above is now correct, because the tool really did return rows."""
        dataset = write_dataset(
            tmp_path / "clean.jsonl",
            [
                {
                    "schema_version": 1,
                    "id": "fx-clean-0001",
                    "suite": "fixture.failure",
                    "split": "dev",
                    "input": {"messages": [{"role": "user", "content": "python jobs?"}]},
                    "expect": {
                        "forbidden_claims": [
                            {"id": "absence", "statement": "no matching jobs", "match": "contains"}
                        ]
                    },
                }
            ],
        )
        runner = Runner(
            target=self.tool_target("I found a Backend Engineer role in Pune."),
            scorers=[scorers.failure_handling(), scorers.forbidden_claims()],
            tracer=Tracer(sink=MemorySink()),
        )
        report = await runner.run(dataset)

        outcome = report.outcomes[0]
        assert "failure_handling" in outcome.not_applicable
        assert outcome.scores["forbidden_claims"].passed is True


class TestRepeatsAndCollapse:
    async def test_repeats_produce_one_outcome_per_repeat(self, tmp_path: Path) -> None:
        """Collapse happens in the metrics layer (M7). The runner keeps every
        repeat, so instability stays visible rather than being averaged away."""
        dataset = write_dataset(
            tmp_path / "routing.jsonl",
            [routing_case("fx-route-0001", BRANCH_QUERIES["job_search"], "job_search")],
        )
        runner = Runner(
            target=orchestrator_target(),
            scorers=[scorers.route()],
            tracer=Tracer(sink=MemorySink()),
            repeats=3,
            stubs={"send_gmail": lambda **kwargs: "stubbed"},
        )
        report = await runner.run(dataset)

        assert [outcome.repeat for outcome in report.outcomes] == [0, 1, 2]
        assert all(outcome.scores["route"].passed for outcome in report.outcomes)

    async def test_collapsing_the_real_repeats_gives_one_verdict(self, tmp_path: Path) -> None:
        dataset = write_dataset(
            tmp_path / "routing.jsonl",
            [routing_case("fx-route-0001", BRANCH_QUERIES["job_search"], "job_search")],
        )
        runner = Runner(
            target=orchestrator_target(),
            scorers=[scorers.route()],
            tracer=Tracer(sink=MemorySink()),
            repeats=3,
            stubs={"send_gmail": lambda **kwargs: "stubbed"},
        )
        report = await runner.run(dataset)

        collapsed = scorers.collapse(
            "route", [outcome.scores["route"] for outcome in report.outcomes]
        )
        assert collapsed is not None
        assert collapsed.passed is True
        assert collapsed.detail["repeats"] == 3
        assert collapsed.detail["unstable"] is False


class TestRouteProbeSource:
    async def test_the_probe_span_is_the_preferred_source(self, tmp_path: Path) -> None:
        """With a probe span present, the scorer takes ``graph.next`` — the first
        source in the fallback order. The probe writes that span itself, so this
        exercises the real integration rather than a hand-set attribute."""
        from neverempty.integrations.langgraph import route_probe

        probe = route_probe(fixture_agent.build_orchestrator(), branch_nodes=fixture_agent.BRANCHES)
        compiled = fixture_agent.build_orchestrator().compile()

        async def target(case: Case, tracer: Tracer) -> None:
            assert case.input.messages is not None
            query = case.input.messages[-1].content
            run = tracer.current_run
            assert run is not None
            await probe.ainvoke({"query": query})
            out = await compiled.ainvoke({"query": query})
            run.set_output(answer=out.get("response"), route=out.get("intent"))

        dataset = write_dataset(
            tmp_path / "routing.jsonl",
            [routing_case("fx-route-0001", BRANCH_QUERIES["job_search"], "job_search")],
        )
        runner = Runner(
            target=target,
            scorers=[scorers.route()],
            tracer=Tracer(sink=MemorySink()),
            stubs={"send_gmail": lambda **kwargs: "stubbed"},
        )
        report = await runner.run(dataset)

        verdict = report.outcomes[0].scores["route"]
        assert verdict.detail["source"] == "graph.next"
        assert verdict.passed is True

    async def test_the_probe_beats_the_node_spans_it_runs_beside(self, tmp_path: Path) -> None:
        """Both sources are present and they agree here, so the assertion that
        matters is which one the scorer reported reading."""
        from neverempty.integrations.langgraph import route_probe

        probe = route_probe(fixture_agent.build_orchestrator(), branch_nodes=fixture_agent.BRANCHES)
        compiled = fixture_agent.build_orchestrator().compile()

        async def target(case: Case, tracer: Tracer) -> None:
            assert case.input.messages is not None
            query = case.input.messages[-1].content
            run = tracer.current_run
            assert run is not None
            await probe.ainvoke({"query": query})
            out = await compiled.ainvoke(
                {"query": query}, config={"callbacks": [tracer.langchain_handler()]}
            )
            run.set_output(answer=out.get("response"), route=out.get("intent"))

        dataset = write_dataset(
            tmp_path / "routing.jsonl",
            [routing_case("fx-route-0001", BRANCH_QUERIES["job_search"], "job_search")],
        )
        runner = Runner(
            target=target,
            scorers=[scorers.route()],
            tracer=Tracer(sink=MemorySink()),
            stubs={"send_gmail": lambda **kwargs: "stubbed"},
        )
        report = await runner.run(dataset)

        verdict = report.outcomes[0].scores["route"]
        assert verdict.detail["source"] == "graph.next"
        assert verdict.detail["predicted"] == "job_search"
