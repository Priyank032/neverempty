"""D20: a declared fault that never fires is silent.

``faults`` are applied by the ``@tool`` wrapper. A case declaring a fault on a
function that is not wrapped runs normally, scores normally, and reports
``misreport_as_empty`` over cases where nothing was ever injected -- a rate
computed from zero opportunities, presented as a measurement.

That is this library's own headline bug, inside its own fault injector, and it
is the worst place for it: the number it corrupts is the one the whole project
exists to produce. Doc row 993 names exactly this trap ("publishing a
misreport-as-empty rate without injecting faults ... the exact
failure-looks-like-empty bug, in your own eval tool") and the protection was
never built.

Found by the agent writing the NextRole labels, which correctly refused to
write 33 fault cases against an unwrapped agent rather than produce a suite
that measures nothing while claiming to.

``core/faults.py``'s own docstring also said faults "require no change to agent
code", which is false -- they require the ``@tool`` wrapper. Corrected.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from neverempty import Dataset, PreflightError, Report, Runner, Tracer, scorers, tool
from neverempty.tracer.sinks import MemorySink

UNWRAPPED = "never_wrapped_search"


@tool(empty_when=lambda rows: len(rows) == 0)
async def wrapped_search(city: str) -> list[dict[str, str]]:
    return [{"title": "Backend Engineer"}]


async def plain_search(city: str) -> list[dict[str, str]]:
    """The shape a real agent has: a plain method, no ``@tool``."""
    return [{"title": "Backend Engineer"}]


def _dataset(tmp_path: Path, *, fault_on: str | None) -> Dataset:
    case: dict[str, Any] = {
        "schema_version": 1,
        "id": "demo-0001",
        "suite": "demo.failure",
        "split": "dev",
        "input": {"messages": [{"role": "user", "content": "any jobs?"}]},
        "expect": {"route": {"label": "job_search"}},
    }
    if fault_on is not None:
        case["faults"] = [{"tool": fault_on, "kind": "timeout", "after_calls": 0}]
    path = tmp_path / "f.jsonl"
    path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    return Dataset.load(path, split="dev")


def _run(dataset: Dataset, target: Any) -> Report:
    runner = Runner(
        target=target,
        scorers=[scorers.route()],
        tracer=Tracer(sink=MemorySink()),
        seed=1,
    )
    return asyncio.run(runner.run(dataset))


async def _unwrapped_target(case: Any, tracer: Any) -> None:
    await plain_search("Pune")
    tracer.current_run.set_output(answer="a", route="job_search")


class TestADeclaredFaultThatCannotFireIsRefused:
    def test_the_run_does_not_start(self, tmp_path: Path) -> None:
        """The defect at its narrowest: a suite that measures nothing while
        reporting a misreport rate over it."""
        with pytest.raises(PreflightError):
            _run(_dataset(tmp_path, fault_on=UNWRAPPED), _unwrapped_target)

    def test_the_refusal_names_the_tool(self, tmp_path: Path) -> None:
        with pytest.raises(PreflightError, match=UNWRAPPED):
            _run(_dataset(tmp_path, fault_on=UNWRAPPED), _unwrapped_target)

    def test_the_refusal_says_what_to_do(self, tmp_path: Path) -> None:
        with pytest.raises(PreflightError, match="@tool"):
            _run(_dataset(tmp_path, fault_on=UNWRAPPED), _unwrapped_target)

    def test_it_lists_the_tools_that_are_wrapped(self, tmp_path: Path) -> None:
        """A typo in a tool name is the likeliest cause, so the message says
        what was available."""
        with pytest.raises(PreflightError, match="wrapped_search"):
            _run(_dataset(tmp_path, fault_on=UNWRAPPED), _unwrapped_target)


class TestAWrappedToolRunsNormally:
    async def _target(self, case: Any, tracer: Any) -> None:
        result = await wrapped_search(city="Pune")
        answer = (
            "The search failed, so I could not check."
            if result.status == "error"
            else "Found jobs."
        )
        tracer.current_run.set_output(answer=answer, route="job_search")

    def test_a_fault_on_a_wrapped_tool_is_accepted(self, tmp_path: Path) -> None:
        report = _run(_dataset(tmp_path, fault_on="wrapped_search"), self._target)
        assert report.complete is True

    def test_the_fault_actually_fired(self, tmp_path: Path) -> None:
        """The point of the whole check: the tool really did fail."""
        seen: list[str] = []

        async def target(case: Any, tracer: Any) -> None:
            result = await wrapped_search(city="Pune")
            seen.append(result.status)
            tracer.current_run.set_output(answer="The search failed.", route="job_search")

        _run(_dataset(tmp_path, fault_on="wrapped_search"), target)
        assert seen == ["error"]


class TestASuiteWithoutFaultsIsUnaffected:
    def test_no_declared_fault_means_no_check(self, tmp_path: Path) -> None:
        report = _run(_dataset(tmp_path, fault_on=None), _unwrapped_target)
        assert report.complete is True
