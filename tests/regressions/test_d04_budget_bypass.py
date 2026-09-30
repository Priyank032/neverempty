"""D4: ``max_cost_usd`` does nothing when costs cannot be computed.

With a model absent from the pricing table, every cost is null, the budget
counted only the costs it knew, and a run spending 50M tokens under a $0.05 cap
finished with ``status="ok"`` and no warning that the cap was unenforceable.

``_Budget``'s own docstring already named this:

    An unknown cost is never counted as zero. With the empty pricing table
    every cost is null, so a budget that treated null as free would never
    fire -- which is exactly the missing-versus-zero bug, in the budget.

The class was right and the caller skipped it: ``if outcome.cost_usd is not
None`` meant an unknown cost reached the budget as nothing at all.

A budget that cannot be enforced is refused rather than ignored. Choosing to
run without a cap is fine; believing you have one that is not there is not.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from neverempty import Dataset, Report, Runner, Tracer, scorers
from neverempty.tracer.pricing import ModelPrice, Pricing
from neverempty.tracer.sinks import MemorySink

PRICED = "priced-model"
UNPRICED = "unpriced-model"


def _pricing() -> Pricing:
    return Pricing(
        version="test-2026-09-01",
        models={
            PRICED: ModelPrice(
                input_usd_per_mtok=3.0,
                output_usd_per_mtok=10.0,
                as_of="2026-09-01",
                source_url="https://example.invalid/pricing",
            )
        },
    )


def _dataset(tmp_path: Path, *, cases: int = 5) -> Dataset:
    lines = [
        json.dumps(
            {
                "schema_version": 1,
                "id": f"demo-{index:04d}",
                "suite": "demo.routing",
                "split": "dev",
                "input": {"messages": [{"role": "user", "content": "q"}]},
                "expect": {"route": {"label": "job_search"}},
            }
        )
        for index in range(cases)
    ]
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Dataset.load(path, split="dev")


def _run(
    tmp_path: Path,
    *,
    model: str,
    budget: float | None,
    cases: int = 5,
    concurrency: int = 1,
) -> Report:
    async def target(case: object, tracer: object) -> None:
        with tracer.span("llm", name="c") as span:  # type: ignore[attr-defined]
            span.record_usage(model=model, input_tokens=10_000_000, output_tokens=0)
        tracer.current_run.set_output(answer="a", route="job_search")  # type: ignore[attr-defined]

    runner = Runner(
        target=target,
        scorers=[scorers.route()],
        tracer=Tracer(sink=MemorySink(), pricing=_pricing()),
        seed=1,
        max_cost_usd=budget,
        # Serial by default here: the budget is checked before each case, so a
        # concurrency window wider than the suite starts every case before any
        # cost lands. That is a real property of a per-case check, not the
        # defect under test, and conflating them would test the wrong thing.
        concurrency=concurrency,
    )
    return asyncio.run(runner.run(_dataset(tmp_path, cases=cases)))


class TestAnUnenforceableBudgetStopsTheRun:
    def test_the_run_does_not_report_ok(self, tmp_path: Path) -> None:
        """The defect at its narrowest: 50M tokens under a $0.05 cap, ok."""
        report = _run(tmp_path, model=UNPRICED, budget=0.05)
        assert report.status != "ok"

    def test_it_aborts_on_the_budget(self, tmp_path: Path) -> None:
        report = _run(tmp_path, model=UNPRICED, budget=0.05)
        assert report.status == "aborted_budget"

    def test_the_run_is_not_complete(self, tmp_path: Path) -> None:
        report = _run(tmp_path, model=UNPRICED, budget=0.05)
        assert report.complete is False

    def test_the_unknown_costs_are_still_counted(self, tmp_path: Path) -> None:
        report = _run(tmp_path, model=UNPRICED, budget=0.05)
        assert report.costs.unknown_count > 0

    def test_the_rendered_report_explains_why(self, tmp_path: Path) -> None:
        """``aborted_budget`` beside a null total is otherwise baffling: it
        reads as "you overspent" when nothing was measured at all."""
        from neverempty import render_markdown

        rendered = render_markdown(_run(tmp_path, model=UNPRICED, budget=0.05))
        assert "could not be enforced" in rendered
        assert "unknown cost" in rendered


class TestNoBudgetMeansNoRefusal:
    """Running without a cap is a choice, and unknown costs are ordinary."""

    def test_an_unpriced_model_without_a_budget_runs_fine(self, tmp_path: Path) -> None:
        report = _run(tmp_path, model=UNPRICED, budget=None)
        assert report.status == "ok"
        assert report.complete is True

    def test_the_cost_is_still_null_with_a_reason(self, tmp_path: Path) -> None:
        report = _run(tmp_path, model=UNPRICED, budget=None)
        assert report.costs.total_usd is None
        assert report.costs.unknown_count == 5


class TestAPricedModelIsUnaffected:
    def test_a_budget_that_is_not_reached_lets_the_run_finish(self, tmp_path: Path) -> None:
        report = _run(tmp_path, model=PRICED, budget=1000.0)
        assert report.status == "ok"
        assert report.complete is True
        assert report.costs.total_usd == pytest.approx(5 * 10_000_000 * 3.0 / 1_000_000)

    def test_a_budget_that_is_exceeded_still_aborts(self, tmp_path: Path) -> None:
        report = _run(tmp_path, model=PRICED, budget=0.05)
        assert report.status == "aborted_budget"

    def test_the_cap_holds_under_concurrency_once_a_cost_has_landed(self, tmp_path: Path) -> None:
        """With more cases than the concurrency window, the cap still fires --
        after at most one window's worth of overspend, which is inherent to
        checking a budget before each case rather than mid-flight."""
        report = _run(tmp_path, model=PRICED, budget=0.05, cases=20, concurrency=4)
        assert report.status == "aborted_budget"
        assert report.counts.scored <= 4
