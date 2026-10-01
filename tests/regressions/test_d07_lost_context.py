"""D7: a tool run via ``loop.run_in_executor`` vanishes from the trace.

The tool still runs and still returns the right result, but no span is
recorded, so tool-selection scoring treats the tool as never called -- a
missing measurement that looks exactly like a correct one.

The cause is stdlib behaviour the wrapper cannot change: ``asyncio.to_thread``
copies the caller's context into the worker thread, and
``loop.run_in_executor`` does not. Without the context the ``ContextVar``
holding the current run reads ``None``.

That is indistinguishable, locally, from a tool called outside any trace --
which doc rule 7 says must work and record nothing. So the wrapper cannot tell
them apart on its own. The tracer can: it knows whether a run is open anywhere
in the process. A tool that finds no run while a run is open has lost its
context, and that is worth one warning rather than silence.

The fix is a warning, not a capture: the span genuinely cannot be attached
from a thread with no context, and inventing a parent would put the tool under
the wrong node. ``asyncio.to_thread`` remains the supported path and is
asserted here so the recommendation cannot rot.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from neverempty import Tracer, tool
from neverempty.tracer.sinks import MemorySink


@tool(never_empty=True)
def blocking_tool() -> dict[str, Any]:
    return {"rows": [1, 2, 3]}


@pytest.fixture(autouse=True)
def _forget_previous_warnings() -> Any:
    """The once-per-tool guard is process-wide, which is what production wants
    and what makes these tests order-dependent. Cleared between them."""
    from neverempty.core import tracing

    tracing._warned_tools.clear()
    yield
    tracing._warned_tools.clear()


class TestToThreadStillWorks:
    """The supported path, pinned so the documented recommendation stays true."""

    async def test_the_span_is_recorded(self) -> None:
        sink = MemorySink()
        tracer = Tracer(sink=sink)
        async with tracer.run():
            await asyncio.to_thread(blocking_tool)
        assert len(sink.traces[0].spans) == 1
        assert sink.traces[0].spans[0].name == "blocking_tool"

    async def test_no_warning_is_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        sink = MemorySink()
        tracer = Tracer(sink=sink)
        with caplog.at_level(logging.WARNING, logger="neverempty"):
            async with tracer.run():
                await asyncio.to_thread(blocking_tool)
        assert not caplog.records


class TestRunInExecutorWarnsInsteadOfVanishing:
    async def test_a_warning_is_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        sink = MemorySink()
        tracer = Tracer(sink=sink)
        with caplog.at_level(logging.WARNING, logger="neverempty"):
            async with tracer.run():
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, blocking_tool)
        assert caplog.records

    async def test_the_warning_names_the_tool_and_the_remedy(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        sink = MemorySink()
        tracer = Tracer(sink=sink)
        with caplog.at_level(logging.WARNING, logger="neverempty"):
            async with tracer.run():
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, blocking_tool)
        message = caplog.records[0].getMessage()
        assert "blocking_tool" in message
        assert "to_thread" in message

    async def test_the_tool_still_returns_its_result(self) -> None:
        """Warning, not breaking. The agent keeps working."""
        sink = MemorySink()
        tracer = Tracer(sink=sink)
        async with tracer.run():
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, blocking_tool)
        assert result.status == "ok"
        assert result.value == {"rows": [1, 2, 3]}

    async def test_it_warns_once_per_tool_not_once_per_call(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A tool called in a loop must not produce a thousand warnings."""
        sink = MemorySink()
        tracer = Tracer(sink=sink)
        with caplog.at_level(logging.WARNING, logger="neverempty"):
            async with tracer.run():
                loop = asyncio.get_running_loop()
                for _ in range(5):
                    await loop.run_in_executor(None, blocking_tool)
        assert len(caplog.records) == 1


class TestAToolOutsideAnyRunIsStillSilent:
    """Doc rule 7: outside a tracer context the wrapper still works and simply
    does not record. That is a supported use, not a lost context."""

    async def test_no_warning_when_no_run_is_open(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="neverempty"):
            result = blocking_tool()
        assert not caplog.records
        assert result.status == "ok"

    async def test_no_warning_in_a_thread_when_no_run_is_open(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="neverempty"):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, blocking_tool)
        assert not caplog.records


class TestTheDocumentedEscapeHatchWorks:
    """``docs/getting-started.md`` offers an explicit context copy for callers
    who must use an executor. It has to actually record the span."""

    async def test_copying_the_context_records_the_span(self) -> None:
        import contextvars
        import functools

        sink = MemorySink()
        tracer = Tracer(sink=sink)
        async with tracer.run():
            context = contextvars.copy_context()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, functools.partial(context.run, blocking_tool))
        assert len(sink.traces[0].spans) == 1

    async def test_it_emits_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import contextvars
        import functools

        sink = MemorySink()
        tracer = Tracer(sink=sink)
        with caplog.at_level(logging.WARNING, logger="neverempty"):
            async with tracer.run():
                context = contextvars.copy_context()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, functools.partial(context.run, blocking_tool))
        assert not caplog.records

    def test_the_docs_recommend_to_thread(self) -> None:
        from pathlib import Path

        text = (Path(__file__).resolve().parents[2] / "docs" / "getting-started.md").read_text(
            encoding="utf-8"
        )
        assert "asyncio.to_thread" in text
        assert "run_in_executor" in text
