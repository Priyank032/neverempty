"""The fault hook inside the tool wrapper (wrapper rule 4).

    Fault injection is checked before the function body runs, keyed on
    ``(tool_name, call_index)`` from a context variable. An injected fault sets
    ``tool.fault_injected=true`` and never calls the real dependency.

The dataset-facing ``faults.inject`` API and the failure_handling scorer are
M6; this covers only the hook the wrapper exposes.
"""

from __future__ import annotations

import asyncio

import pytest

from neverempty import Empty, Err, Ok, tool
from neverempty.core.faults import FaultSpec, fault_scope


class TestInjectedFaultsPreventTheRealCall:
    async def test_timeout_fault_skips_the_function_body(self) -> None:
        called = False

        @tool(never_empty=True)
        async def search_jobs() -> str:
            nonlocal called
            called = True
            return "real data"

        with fault_scope([FaultSpec(tool="search_jobs", kind="timeout")]):
            result = await search_jobs()

        assert isinstance(result, Err)
        assert result.kind == "timeout"
        assert called is False, "the real dependency was contacted under fault injection"

    def test_sync_tools_honour_injected_faults(self) -> None:
        called = False

        @tool(never_empty=True)
        def search_jobs() -> str:
            nonlocal called
            called = True
            return "real data"

        with fault_scope([FaultSpec(tool="search_jobs", kind="upstream")]):
            result = search_jobs()

        assert isinstance(result, Err)
        assert called is False

    @pytest.mark.parametrize(
        ("kind", "expected_kind", "retryable"),
        [
            ("timeout", "timeout", True),
            ("upstream", "upstream", True),
            ("rate_limit", "rate_limit", True),
        ],
    )
    async def test_error_fault_kinds_map_to_err(
        self, kind: str, expected_kind: str, retryable: bool
    ) -> None:
        @tool(never_empty=True)
        async def t() -> str:
            return "real"

        with fault_scope([FaultSpec(tool="t", kind=kind)]):  # type: ignore[arg-type]
            result = await t()

        assert isinstance(result, Err)
        assert result.kind == expected_kind
        assert result.retryable is retryable

    async def test_empty_fault_yields_empty_not_error(self) -> None:
        """A genuine-empty fault is how the false-alarm rate gets measured."""

        @tool(never_empty=True)
        async def t() -> str:
            return "real"

        with fault_scope([FaultSpec(tool="t", kind="empty")]):
            result = await t()

        assert isinstance(result, Empty)

    async def test_truncated_fault_runs_the_tool_and_flags_the_result(self) -> None:
        """Truncation is a property of a real result, so the body does run."""
        called = False

        @tool(never_empty=True)
        async def t() -> list[int]:
            nonlocal called
            called = True
            return [1, 2, 3]

        with fault_scope([FaultSpec(tool="t", kind="truncated")]):
            result = await t()

        assert called is True
        assert isinstance(result, Ok)
        assert result.truncated is True


class TestAfterCalls:
    async def test_after_calls_zero_hits_the_first_call(self) -> None:
        @tool(never_empty=True)
        async def t() -> str:
            return "real"

        with fault_scope([FaultSpec(tool="t", kind="timeout", after_calls=0)]):
            assert isinstance(await t(), Err)

    async def test_after_calls_two_lets_the_first_two_calls_through(self) -> None:
        @tool(never_empty=True)
        async def t() -> str:
            return "real"

        with fault_scope([FaultSpec(tool="t", kind="timeout", after_calls=2)]):
            first = await t()
            second = await t()
            third = await t()

        assert isinstance(first, Ok)
        assert isinstance(second, Ok)
        assert isinstance(third, Err)

    async def test_the_fault_persists_on_later_calls(self) -> None:
        @tool(never_empty=True)
        async def t() -> str:
            return "real"

        with fault_scope([FaultSpec(tool="t", kind="timeout", after_calls=1)]):
            await t()
            assert isinstance(await t(), Err)
            assert isinstance(await t(), Err)

    async def test_call_counts_are_tracked_per_tool(self) -> None:
        @tool(never_empty=True)
        async def a() -> str:
            return "a"

        @tool(never_empty=True)
        async def b() -> str:
            return "b"

        with fault_scope([FaultSpec(tool="b", kind="timeout", after_calls=1)]):
            await a()
            await a()
            assert isinstance(await b(), Ok)
            assert isinstance(await b(), Err)


class TestFaultScoping:
    async def test_other_tools_are_unaffected(self) -> None:
        @tool(never_empty=True)
        async def target() -> str:
            return "t"

        @tool(never_empty=True)
        async def bystander() -> str:
            return "b"

        with fault_scope([FaultSpec(tool="target", kind="timeout")]):
            assert isinstance(await target(), Err)
            assert isinstance(await bystander(), Ok)

    async def test_faults_do_not_leak_outside_the_scope(self) -> None:
        @tool(never_empty=True)
        async def t() -> str:
            return "real"

        with fault_scope([FaultSpec(tool="t", kind="timeout")]):
            assert isinstance(await t(), Err)
        assert isinstance(await t(), Ok)

    async def test_call_counters_reset_between_scopes(self) -> None:
        @tool(never_empty=True)
        async def t() -> str:
            return "real"

        spec = [FaultSpec(tool="t", kind="timeout", after_calls=1)]
        for _ in range(2):
            with fault_scope(spec):
                assert isinstance(await t(), Ok)
                assert isinstance(await t(), Err)

    async def test_concurrent_scopes_do_not_cross_contaminate(self) -> None:
        """contextvars, so two eval cases in flight keep separate fault state."""

        @tool(never_empty=True)
        async def t() -> str:
            await asyncio.sleep(0)
            return "real"

        async def faulted() -> object:
            with fault_scope([FaultSpec(tool="t", kind="timeout")]):
                await asyncio.sleep(0.01)
                return await t()

        async def clean() -> object:
            await asyncio.sleep(0.005)
            return await t()

        faulted_result, clean_result = await asyncio.gather(faulted(), clean())
        assert isinstance(faulted_result, Err)
        assert isinstance(clean_result, Ok)

    async def test_the_fault_is_recorded_on_the_result_for_the_trace(self) -> None:
        """M2 copies this onto the span as ``tool.fault_injected``."""

        @tool(never_empty=True)
        async def t() -> str:
            return "real"

        with fault_scope([FaultSpec(tool="t", kind="timeout")]):
            result = await t()

        assert isinstance(result, Err)
        assert result.fault_injected is True

    async def test_an_uninjected_result_is_not_flagged(self) -> None:
        @tool(never_empty=True)
        async def t() -> str:
            return "real"

        result = await t()
        assert isinstance(result, Ok)
        assert result.fault_injected is False

    async def test_a_faulted_tool_matches_on_the_declared_name(self) -> None:
        @tool(name="search", never_empty=True)
        async def search_jobs() -> str:
            return "real"

        with fault_scope([FaultSpec(tool="search", kind="timeout")]):
            assert isinstance(await search_jobs(), Err)
