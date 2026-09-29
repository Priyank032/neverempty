"""Acceptance row for ``core/tool``: under 1 ms overhead.

    Overhead budget: under 1 ms per call excluding the function itself,
    asserted by a benchmark test.

Timing tests are environment sensitive. This one measures the median of many
calls against an undecorated baseline, so a loaded CI runner slows both sides
together. The budget is 1 ms, and the wrapper should come in far under it.
"""

from __future__ import annotations

import asyncio
import statistics
import time

import pytest

from neverempty import tool

CALLS = 2000
BUDGET_S = 1e-3


def _median_overhead(decorated: object, bare: object, calls: int) -> float:
    """Median per-call wall time of the wrapper minus the bare function."""
    decorated_samples: list[float] = []
    bare_samples: list[float] = []
    for _ in range(calls):
        start = time.perf_counter()
        bare()  # type: ignore[operator]
        bare_samples.append(time.perf_counter() - start)

        start = time.perf_counter()
        decorated()  # type: ignore[operator]
        decorated_samples.append(time.perf_counter() - start)
    return statistics.median(decorated_samples) - statistics.median(bare_samples)


@pytest.mark.benchmark
def test_sync_wrapper_overhead_is_under_1ms() -> None:
    def bare(x: int = 1) -> int:
        return x + 1

    @tool(never_empty=True)
    def decorated(x: int = 1) -> int:
        return x + 1

    overhead = _median_overhead(decorated, bare, CALLS)
    assert overhead < BUDGET_S, f"{overhead * 1e6:.1f}us per call exceeds the 1ms budget"


@pytest.mark.benchmark
def test_async_wrapper_overhead_is_under_1ms() -> None:
    async def bare(x: int = 1) -> int:
        return x + 1

    @tool(never_empty=True)
    async def decorated(x: int = 1) -> int:
        return x + 1

    async def measure() -> float:
        decorated_samples: list[float] = []
        bare_samples: list[float] = []
        for _ in range(CALLS):
            start = time.perf_counter()
            await bare()
            bare_samples.append(time.perf_counter() - start)

            start = time.perf_counter()
            await decorated()
            decorated_samples.append(time.perf_counter() - start)
        return statistics.median(decorated_samples) - statistics.median(bare_samples)

    overhead = asyncio.run(measure())
    assert overhead < BUDGET_S, f"{overhead * 1e6:.1f}us per call exceeds the 1ms budget"


@pytest.mark.benchmark
def test_predicates_do_not_blow_the_budget() -> None:
    rows = list(range(100))

    def bare() -> list[int]:
        return rows

    @tool(
        empty_when=lambda r: len(r) == 0,
        truncated_when=lambda r: len(r) >= 100,
    )
    def decorated() -> list[int]:
        return rows

    overhead = _median_overhead(decorated, bare, CALLS)
    assert overhead < BUDGET_S, f"{overhead * 1e6:.1f}us per call exceeds the 1ms budget"


@pytest.mark.benchmark
def test_no_network_io_on_the_hot_path() -> None:
    """The wrapper must not touch a socket, even with a fault scope active."""
    import socket

    from neverempty.core.faults import FaultSpec, fault_scope

    original = socket.socket

    class Tripwire(socket.socket):
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("the tool wrapper opened a socket")

    @tool(never_empty=True)
    def t() -> int:
        return 1

    socket.socket = Tripwire  # type: ignore[misc]
    try:
        t()
        with fault_scope([FaultSpec(tool="t", kind="timeout")]):
            t()
    finally:
        socket.socket = original  # type: ignore[misc]
