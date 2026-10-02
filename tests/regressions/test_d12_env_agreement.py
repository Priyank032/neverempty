"""D12: the trace's env contradicts the report's env for the same run.

A run at concurrency 4, seed 1, injecting ``lookup:timeout`` wrote traces
saying ``concurrency: 1, seed: null, fault_profile: null`` while its report
said ``4, 1, lookup:timeout``. Two artifacts of one run disagreeing about how
it was configured.

It matters more for the trace than the report: Trace v1 is the cross-language
contract, so a Node reader asking "was this run concurrent?" or "did this run
inject faults?" got the wrong answer. The fault profile is the sharper case --
the doc records it so a misreport-as-empty number can never be read as coming
from a run that injected nothing (runner.py), and a trace claiming ``null``
undoes exactly that protection.

The runner knows all three. It simply was not telling the tracer.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from neverempty import Dataset, Runner, Tracer, scorers, tool
from neverempty.core.trace import Trace
from neverempty.report.report import Report
from neverempty.tracer.sinks import MemorySink


def _dataset(tmp_path: Path, *, faults: bool) -> Dataset:
    case: dict[str, object] = {
        "schema_version": 1,
        "id": "demo-0001",
        "suite": "demo.routing",
        "split": "dev",
        "input": {"messages": [{"role": "user", "content": "q"}]},
        "expect": {"route": {"label": "job_search"}},
    }
    if faults:
        case["faults"] = [{"tool": "lookup", "kind": "timeout", "after_calls": 0}]
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    return Dataset.load(path, split="dev")


def _run(
    tmp_path: Path, *, concurrency: int = 4, seed: int | None = 1, faults: bool = True
) -> tuple[Trace, Report]:
    sink = MemorySink()

    async def agent(case: object, tracer: object) -> None:
        await lookup("Pune")
        tracer.current_run.set_output(answer="a", route="job_search")  # type: ignore[attr-defined]

    report = asyncio.run(
        Runner(
            target=agent,
            scorers=[scorers.route()],
            tracer=Tracer(sink=sink),
            concurrency=concurrency,
            seed=seed,
        ).run(_dataset(tmp_path, faults=faults))
    )
    return sink.traces[0], report


@tool(never_empty=True)
async def lookup(city: str) -> list[dict[str, str]]:
    """Wrapped, because preflight now refuses a fault declared on a function
    that is not: an unfirable fault would make this fixture measure nothing."""
    return [{"title": "Backend Engineer"}]


def _env(obj: Trace | Report) -> dict[str, object]:
    parsed: dict[str, object] = json.loads(obj.env.model_dump_json())
    return parsed


class TestTheTraceAgreesWithTheReport:
    def test_concurrency_matches(self, tmp_path: Path) -> None:
        trace, report = _run(tmp_path)
        assert _env(trace)["concurrency"] == _env(report)["concurrency"] == 4

    def test_the_seed_matches(self, tmp_path: Path) -> None:
        trace, report = _run(tmp_path)
        assert _env(trace)["seed"] == _env(report)["seed"] == 1

    def test_the_fault_profile_matches(self, tmp_path: Path) -> None:
        """The sharpest case: a trace claiming no faults were injected would
        let a misreport-as-empty number be read as coming from a clean run."""
        trace, report = _run(tmp_path)
        assert _env(trace)["fault_profile"] == _env(report)["fault_profile"]
        assert _env(trace)["fault_profile"] == "lookup:timeout"

    def test_a_run_without_faults_says_so_on_both(self, tmp_path: Path) -> None:
        trace, report = _run(tmp_path, faults=False)
        assert _env(trace)["fault_profile"] is None
        assert _env(report)["fault_profile"] is None

    def test_serial_runs_record_concurrency_one(self, tmp_path: Path) -> None:
        trace, report = _run(tmp_path, concurrency=1)
        assert _env(trace)["concurrency"] == _env(report)["concurrency"] == 1

    def test_an_unseeded_run_records_null_on_both(self, tmp_path: Path) -> None:
        trace, report = _run(tmp_path, seed=None)
        assert _env(trace)["seed"] is None
        assert _env(report)["seed"] is None


class TestATracerUsedOutsideARunnerIsUnaffected:
    """The tracer has no runner to learn these from, and must not invent them."""

    async def test_a_standalone_trace_keeps_its_defaults(self) -> None:
        sink = MemorySink()
        tracer = Tracer(sink=sink)
        async with tracer.run():
            pass
        env = json.loads(sink.traces[0].env.model_dump_json())
        assert env["concurrency"] == 1
        assert env["fault_profile"] is None
