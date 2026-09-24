"""M9 acceptance: the suites run, gate green on a no-op, red on a sabotaged prompt.

Run through the real NextRole adapter and suite specs rather than a bespoke test
harness, against the fixture agent standing in for the live graph. The point is
that the adapter, the coverage gate and the suite specs are the ones that ship.
"""

from __future__ import annotations

import pytest

from toolproof import Dataset, Runner, compare, gate, scorers
from toolproof.dataset.case import Case
from toolproof.evals.nextrole import BRANCHES, NextRoleAdapter
from toolproof.evals.suites import SuiteSpec, check_coverage
from toolproof.report.gate import GateConfig
from toolproof.report.report import Report
from toolproof.tracer.sinks import MemorySink
from toolproof.tracer.tracer import Tracer

pytestmark = pytest.mark.integration

FIXTURE_BRANCHES = ("job_search", "email_draft", "general")


def build_case(case_id: str, content: str, route: str, *, split: str = "test") -> Case:
    return Case.model_validate(
        {
            "id": case_id,
            "suite": "nextrole.routing",
            "split": split,
            "input": {"messages": [{"role": "user", "content": content}]},
            "expect": {"route": {"label": route}},
            "provenance": {"method": "human", "labeller": "priyank"},
        }
    )


QUERIES = {
    # Phrased for the fixture agent's keyword classifier, which is the stand-in
    # for the live router here. The adapter under test is the same either way.
    "job_search": ["find me a job", "backend job in Pune", "remote python job"],
    "email_draft": [
        "write to the recruiter",
        "draft a note to the recruiter",
        "write to the recruiter about the offer",
    ],
    "general": ["how does this work", "tell me more", "what can you do"],
}


@pytest.fixture
def dataset() -> Dataset:
    cases: list[Case] = []
    for branch, queries in QUERIES.items():
        for index, query in enumerate(queries):
            cases.append(build_case(f"nr-{branch}-{index:04d}", query, branch))
    return Dataset(cases)


@pytest.fixture
def spec() -> SuiteSpec:
    return SuiteSpec(
        name="nextrole.routing",
        branches=FIXTURE_BRANCHES,
        min_per_branch=3,
        split="test",
    )


async def run_suite(dataset: Dataset, *, sabotage: bool = False) -> Report:
    from examples import fixture_agent

    def state_from(case: Case) -> dict[str, object]:
        # The fixture graph's state is ``{"query": ...}``; the live NextRole
        # graph takes ``{"messages": [...]}``. That difference is exactly what
        # ``state_from`` exists for, so the adapter needs no agent-specific code.
        assert case.input.messages is not None
        return {"query": case.input.messages[-1].content}

    adapter = NextRoleAdapter(
        lambda **kwargs: fixture_agent.build_orchestrator(block_everything=sabotage, **kwargs),
        branches=FIXTURE_BRANCHES,
        state_from=state_from,
    )
    runner = Runner(
        target=adapter,
        scorers=[scorers.route()],
        tracer=Tracer(sink=MemorySink()),
        repeats=1,
        concurrency=4,
        seed=20260921,
        # The preflight refuses to start without these, even for a routing suite
        # that can never reach a branch. It does not reason about reachability,
        # which is the conservative and correct choice.
        stubs={
            name: (lambda **kwargs: "stubbed")
            for name in ("send_gmail", "save_application", "send_email")
        },
    )
    return await runner.run(dataset)


class TestCoverageGate:
    def test_a_labelled_balanced_split_is_ready(self, dataset: Dataset, spec: SuiteSpec) -> None:
        assert check_coverage(dataset.cases, spec).ok

    def test_the_real_eleven_branch_spec_refuses_this_small_set(self, dataset: Dataset) -> None:
        """The shipped spec wants 30 per branch across 11 branches. A nine-case
        set is not a suite, and the coverage check says so rather than running."""
        from toolproof.evals.nextrole import ROUTING_SUITE

        report = check_coverage(dataset.cases, ROUTING_SUITE)
        assert not report.ok
        shortfall = len(BRANCHES) - len(FIXTURE_BRANCHES)
        assert len(report.problems) >= shortfall

    def test_an_empty_split_refuses_rather_than_reporting_zero(self, spec: SuiteSpec) -> None:
        report = check_coverage([], spec)
        assert not report.ok
        assert "no cases" in " ".join(report.problems)


class TestSuiteRuns:
    async def test_the_routing_suite_runs_through_the_adapter(self, dataset: Dataset) -> None:
        report = await run_suite(dataset)
        assert report.complete
        assert report.counts.cases == 9
        assert report.counts.unscored == 0

    async def test_no_branch_executes_under_the_probe(self, dataset: Dataset) -> None:
        """A routing run reads the branch without entering it, so a 330-case
        suite cannot send an email even if a branch would have."""
        report = await run_suite(dataset)
        for trace in report.traces:
            executed = {
                span.attributes.get("graph.node") for span in trace.spans if span.kind == "node"
            }
            assert not executed & set(FIXTURE_BRANCHES), f"a branch executed: {executed}"

    async def test_no_answer_is_recorded_so_fact_scorers_stay_not_applicable(
        self, dataset: Dataset
    ) -> None:
        report = await run_suite(dataset)
        assert all(trace.final_output.answer is None for trace in report.traces)


class TestGateOnANoOp:
    async def test_comparing_a_run_against_itself_is_green(self, dataset: Dataset) -> None:
        report = await run_suite(dataset)
        result = gate(report, report, config=GateConfig())
        assert result.verdict == "pass"
        assert result.exit_code == 0

    async def test_two_runs_of_the_same_config_show_no_regression(self, dataset: Dataset) -> None:
        base = await run_suite(dataset)
        candidate = await run_suite(dataset)
        comparison = compare(base, candidate)
        assert not comparison.regressed


class TestLyingStateDoesNotMoveRouting:
    """The probe reads the edge, so a corrupted state field must not change it.

    ``lying_state`` makes ``classify_intent`` write an ``intent`` that disagrees
    with the branch the graph actually takes. Routing accuracy staying flat here
    is the M3 rule working: the edge is authoritative, and the disagreement is
    its own finding rather than a routing failure.
    """

    async def test_route_accuracy_is_unchanged_by_a_lying_state_field(
        self, dataset: Dataset
    ) -> None:
        from examples import fixture_agent

        def state_from(case: Case) -> dict[str, object]:
            assert case.input.messages is not None
            return {"query": case.input.messages[-1].content}

        async def run(*, lying: bool) -> Report:
            runner = Runner(
                target=NextRoleAdapter(
                    lambda **kwargs: fixture_agent.build_orchestrator(lying_state=lying, **kwargs),
                    branches=FIXTURE_BRANCHES,
                    state_from=state_from,
                ),
                scorers=[scorers.route()],
                tracer=Tracer(sink=MemorySink()),
                repeats=1,
                concurrency=4,
                seed=20260921,
                stubs={
                    name: (lambda **kwargs: "stubbed")
                    for name in ("send_gmail", "save_application", "send_email")
                },
            )
            return await runner.run(dataset)

        honest = await run(lying=False)
        lying = await run(lying=True)
        before = next(m for m in honest.metrics if m.name == "route")
        after = next(m for m in lying.metrics if m.name == "route")
        assert before.value == after.value == 1.0


class TestGateOnASabotagedAgent:
    """``block_everything`` is a real behaviour change: the safety check rejects,
    so the graph reaches finalize without entering any branch and the probe
    reports no route at all. That is a routing regression, not a doctored report.
    """

    async def test_the_sabotage_produces_no_route(self, dataset: Dataset) -> None:
        sabotaged = await run_suite(dataset, sabotage=True)
        assert all(trace.final_output.route is None for trace in sabotaged.traces)

    async def test_an_unroutable_case_is_unscored_not_failed(self, dataset: Dataset) -> None:
        """No route is read, so the scorer raises and the case is unscored. The
        run is incomplete, which the gate treats as a failure of the run rather
        than of the agent -- a stronger claim than 0% accuracy."""
        sabotaged = await run_suite(dataset, sabotage=True)
        assert sabotaged.counts.unscored == 9
        assert not sabotaged.complete

    async def test_the_gate_refuses_an_incomplete_candidate(self, dataset: Dataset) -> None:
        base = await run_suite(dataset)
        sabotaged = await run_suite(dataset, sabotage=True)
        result = gate(base, sabotaged, config=GateConfig())
        assert result.exit_code != 0
        assert result.verdict != "pass"
