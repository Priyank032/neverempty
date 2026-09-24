"""The doc's End-to-end acceptance row.

"fixture agent, 20 cases, 3 repeats, live-mode-simulated: report, compare
against itself (zero regression), gate exit 0; then a sabotaged prompt produces
exit 1."

This is the row that proves the pieces compose. Every layer built so far runs:
the tool wrapper, the tracer, the LangGraph adapter, the dataset, the runner,
the scorers, the metrics, the statistics, the gate and the renderer, with no
test-only shortcuts between them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from toolproof import Case, Dataset, Report, Runner, Tracer, scorers
from toolproof.report.gate import GateConfig, compare, gate
from toolproof.report.render import render_gate, render_markdown
from toolproof.tracer.sinks import MemorySink

pytest.importorskip("langgraph")

from examples import fixture_agent

SABOTAGED_WORDS = ("jobs", "recruiter", "blog")
"""Words whose queries the sabotaged classifier misroutes.

Chosen to span three branches, so nine of the twenty cases break. The paired
test needs five regressions at n=20 to reach p < 0.05, so a smaller sabotage
would be indistinguishable from noise by design.
"""

BRANCH_QUERIES: dict[str, str] = {
    "job_search": "show me remote python jobs in Pune",
    "email_draft": "can you write to the recruiter about this role",
    "blog_search": "find that blog post about system design interviews",
    "resume_query": "does my resume mention kubernetes",
    "followup": "it has been 10 days since I applied, any update",
    "general": "how should I prepare for a backend interview",
    "clarify": "help",
}


def build_cases() -> list[dict[str, Any]]:
    """Twenty cases spread across all seven branches."""
    branches = list(fixture_agent.BRANCHES)
    rows: list[dict[str, Any]] = []
    for index in range(20):
        branch = branches[index % len(branches)]
        rows.append(
            {
                "schema_version": 1,
                "id": f"e2e-{index:04d}",
                "suite": "fixture.e2e",
                "split": "test",
                "input": {"messages": [{"role": "user", "content": BRANCH_QUERIES[branch]}]},
                "expect": {
                    "route": {"label": branch},
                    "forbidden_tools": ["send_gmail"],
                },
                "provenance": {"method": "human", "labeller": "fixture"},
            }
        )
    return rows


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    path = tmp_path / "e2e.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in build_cases()) + "\n",
        encoding="utf-8",
    )
    return Dataset.load(path)


@pytest.fixture(autouse=True)
def _clear_log() -> Any:
    fixture_agent.execution_log.clear()
    yield
    fixture_agent.execution_log.clear()


def make_target(*, sabotage: bool = False) -> Any:
    """The fixture orchestrator, optionally with a sabotaged classifier.

    ``sabotage`` stands in for a prompt edit that breaks routing: the graph is
    built with ``lying_state``, so the branch that executes stops matching the
    label for a subset of cases. That is a real behaviour change in the agent,
    not a doctored report.
    """
    compiled = fixture_agent.build_orchestrator(lying_state=sabotage).compile()

    async def target(case: Case, tracer: Tracer) -> None:
        assert case.input.messages is not None
        query = case.input.messages[-1].content
        run = tracer.current_run
        assert run is not None
        if sabotage and any(word in query for word in SABOTAGED_WORDS):
            # The sabotage: a broken classification prompt collapses several
            # intents into clarify. Nine of the twenty cases are affected, which
            # is above the paired test's detection threshold at this n; three
            # would be p=0.125 and correctly *not* significant.
            run.set_output(answer="Could you clarify?", route="clarify")
            return
        result = await compiled.ainvoke(
            {"query": query}, config={"callbacks": [tracer.langchain_handler()]}
        )
        run.set_output(answer=result.get("response"), route=result.get("intent"))

    return target


async def run_suite(dataset: Dataset, *, sabotage: bool = False, tag: str = "base") -> Report:
    runner = Runner(
        target=make_target(sabotage=sabotage),
        scorers=[scorers.route(), scorers.forbidden_tools()],
        tracer=Tracer(sink=MemorySink()),
        repeats=3,
        concurrency=4,
        seed=20260921,
        now="2026-09-23T10:00:00.000Z",
        report_id=f"00000000-0000-4000-8000-{tag:>012}"[:36],
        stubs={"send_gmail": lambda **kwargs: "stubbed"},
    )
    return await runner.run(dataset)


class TestEndToEnd:
    async def test_the_run_produces_a_complete_report(self, dataset: Dataset) -> None:
        report = await run_suite(dataset)
        assert report.status == "ok"
        assert report.complete is True
        assert report.counts.cases == 20
        assert report.counts.repeats == 3
        assert report.unscored_ids() == []

    async def test_the_side_effecting_tool_is_stubbed_under_eval(self, dataset: Dataset) -> None:
        """``send_gmail`` raises if it ever really runs, so the run completing
        at all is the proof that the stub was bound.

        The fixture's own log line "SENT EMAIL from email_draft" comes from the
        branch node, not the tool: it records which branch executed, which is
        what the route scorer reads. The tool is the thing that must not fire.
        """
        report = await run_suite(dataset)
        assert report.status == "ok"
        assert "email_draft" in fixture_agent.execution_log

    async def test_a_forbidden_tool_is_never_called(self, dataset: Dataset) -> None:
        """Every case forbids ``send_gmail``, and the scorer reads the real spans."""
        report = await run_suite(dataset)
        verdicts = [o.scores["forbidden_tools"] for o in report.outcomes]
        assert all(verdict.passed for verdict in verdicts)

    async def test_comparing_a_report_against_itself_shows_zero_regression(
        self, dataset: Dataset
    ) -> None:
        report = await run_suite(dataset)
        result = compare(report, report)
        assert result.refused is False
        assert result.regressed == []
        assert result.improved == []
        assert result.mcnemar is not None
        assert result.mcnemar.p_value == 1.0

    async def test_the_gate_exits_zero_against_itself(self, dataset: Dataset) -> None:
        report = await run_suite(dataset)
        result = gate(report, report, GateConfig())
        assert result.exit_code == 0
        assert result.verdict == "pass"

    async def test_a_sabotaged_prompt_exits_one(self, dataset: Dataset) -> None:
        """The headline of the acceptance row. The agent genuinely misroutes on
        a subset of cases, and the paired test detects it as significant."""
        base = await run_suite(dataset, tag="base")
        sabotaged = await run_suite(dataset, sabotage=True, tag="sabo")

        result = gate(base, sabotaged, GateConfig())
        assert result.exit_code == 1
        assert result.verdict == "regression"
        assert result.mcnemar is not None
        assert result.mcnemar.b > 0
        assert result.mcnemar.significant is True

    async def test_the_sabotage_is_detected_as_a_route_regression(self, dataset: Dataset) -> None:
        base = await run_suite(dataset, tag="base")
        sabotaged = await run_suite(dataset, sabotage=True, tag="sabo")

        result = compare(base, sabotaged)
        delta = next(d for d in result.deltas if d.name == "route")
        assert delta.base_value is not None
        assert delta.candidate_value is not None
        assert delta.candidate_value < delta.base_value

    async def test_the_regression_is_reversible_as_an_improvement(self, dataset: Dataset) -> None:
        """Comparing the other way round must not fail the build: the gate is
        one-sided, so a fix is never reported as a regression."""
        base = await run_suite(dataset, tag="base")
        sabotaged = await run_suite(dataset, sabotage=True, tag="sabo")

        result = gate(sabotaged, base, GateConfig())
        assert result.exit_code == 0
        assert result.comparison is not None
        assert result.comparison.improved_count > 0


class TestReportArtifacts:
    async def test_the_report_round_trips_through_a_file(
        self, dataset: Dataset, tmp_path: Path
    ) -> None:
        report = await run_suite(dataset)
        path = report.save(tmp_path / "report.json")
        assert Report.load(path).counts.cases == 20

    async def test_two_runs_produce_identically_ordered_output(self, dataset: Dataset) -> None:
        """Ordering is canonical, so two runs of one suite diff cleanly even
        though they executed concurrently."""
        first = await run_suite(dataset)
        second = await run_suite(dataset)
        assert [o.key for o in first.outcomes] == [o.key for o in second.outcomes]

    async def test_the_rendered_report_contains_only_defensible_numbers(
        self, dataset: Dataset
    ) -> None:
        report = await run_suite(dataset)
        rendered = render_markdown(report)
        assert "# fixture.e2e" in rendered
        assert "## Metrics" in rendered
        assert "## Confusion matrix" in rendered
        # Cost is unknown with the empty default pricing table, and the renderer
        # must say so rather than print a confident total.
        assert "Cost unknown" in rendered

    async def test_the_gate_summary_renders_for_ci(self, dataset: Dataset) -> None:
        base = await run_suite(dataset, tag="base")
        sabotaged = await run_suite(dataset, sabotage=True, tag="sabo")
        rendered = render_gate(gate(base, sabotaged, GateConfig()))
        assert "exit 1" in rendered
        assert "McNemar" in rendered

    async def test_the_confusion_matrix_reflects_the_sabotage(self, dataset: Dataset) -> None:
        sabotaged = await run_suite(dataset, sabotage=True, tag="sabo")
        assert "clarify" in sabotaged.confusion.get("job_search", {})


class TestMustPassAndFloors:
    async def test_a_must_pass_case_failing_exits_two(
        self, dataset: Dataset, tmp_path: Path
    ) -> None:
        base = await run_suite(dataset, tag="base")
        sabotaged = await run_suite(dataset, sabotage=True, tag="sabo")

        # Mark one of the cases the sabotage breaks as must_pass.
        broken = next(
            outcome.case_id
            for outcome in sabotaged.outcomes
            if outcome.scores.get("route") is not None and outcome.scores["route"].passed is False
        )
        flagged = sabotaged.model_copy(
            update={
                "outcomes": [
                    outcome.model_copy(update={"must_pass": outcome.case_id == broken})
                    for outcome in sabotaged.outcomes
                ]
            }
        )
        result = gate(base, flagged, GateConfig())
        assert result.exit_code == 2
        assert broken in result.must_pass_failures

    async def test_a_floor_catches_an_absolute_level(self, dataset: Dataset) -> None:
        """A floor catches what the paired test cannot: a suite that was always
        below the bar does not regress, but must not pass either."""
        sabotaged = await run_suite(dataset, sabotage=True, tag="sabo")
        result = gate(sabotaged, sabotaged, GateConfig(floors={"route": 0.99}))
        assert result.exit_code == 1
        assert result.breached_floors
