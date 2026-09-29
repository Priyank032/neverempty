"""Recording stubs for side-effecting tools.

The preflight that depends on these is a safety check: an eval run must never be
able to email a real recruiter. So the stub path gets the same scrutiny as the
real one, because it is what actually executes under eval.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from neverempty import Empty, Err, MemorySink, Ok, Tracer, stub_scope, tool
from neverempty.core.stubs import (
    clear_side_effect_registry,
    find_stub,
    register_side_effect,
    registered_side_effect_tools,
)


class TestRegistry:
    def test_a_side_effect_tool_registers_at_decoration_time(self) -> None:
        """The preflight needs the full set before any case runs."""

        @tool(never_empty=True, side_effect=True)
        async def send_gmail(to: str) -> str:
            return "sent"

        assert "send_gmail" in registered_side_effect_tools()

    def test_an_ordinary_tool_does_not_register(self) -> None:
        @tool(never_empty=True)
        async def search_jobs() -> str:
            return "results"

        assert "search_jobs" not in registered_side_effect_tools()

    def test_the_declared_name_is_what_registers(self) -> None:
        @tool(name="gmail_send", never_empty=True, side_effect=True)
        async def send(to: str) -> str:
            return "sent"

        assert "gmail_send" in registered_side_effect_tools()

    def test_clearing_empties_the_registry(self) -> None:
        register_side_effect("x")
        clear_side_effect_registry()
        assert registered_side_effect_tools() == frozenset()


class TestStubScope:
    def test_no_stub_is_found_outside_a_scope(self) -> None:
        assert find_stub("send_gmail") is None

    def test_a_stub_is_found_inside_its_scope(self) -> None:
        with stub_scope({"send_gmail": lambda **kw: "stubbed"}):
            assert find_stub("send_gmail") is not None

    def test_stubs_do_not_leak_outside_the_scope(self) -> None:
        with stub_scope({"send_gmail": lambda **kw: "stubbed"}):
            pass
        assert find_stub("send_gmail") is None

    async def test_concurrent_scopes_do_not_cross_contaminate(self) -> None:
        """Two eval cases in flight must not see each other's stubs."""

        @tool(never_empty=True, side_effect=True)
        async def send_gmail(to: str) -> str:
            return "REAL SEND"

        async def stubbed() -> Any:
            with stub_scope({"send_gmail": lambda **kw: "stubbed"}):
                await asyncio.sleep(0.01)
                return await send_gmail(to="a@b.com")

        async def unstubbed() -> Any:
            await asyncio.sleep(0.005)
            return await send_gmail(to="a@b.com")

        first, second = await asyncio.gather(stubbed(), unstubbed())
        assert isinstance(first, Ok)
        assert first.value == "stubbed"
        assert isinstance(second, Ok)
        assert second.value == "REAL SEND"


class TestStubbedCalls:
    async def test_an_async_stub_replaces_the_body(self) -> None:
        called = False

        @tool(never_empty=True, side_effect=True)
        async def send_gmail(to: str) -> str:
            nonlocal called
            called = True
            return "REAL SEND"

        async def stub(**kwargs: Any) -> str:
            return "stubbed"

        with stub_scope({"send_gmail": stub}):
            result = await send_gmail(to="a@b.com")

        assert isinstance(result, Ok)
        assert result.value == "stubbed"
        assert called is False

    def test_a_sync_stub_replaces_a_sync_body(self) -> None:
        called = False

        @tool(never_empty=True, side_effect=True)
        def save_application(row: str) -> str:
            nonlocal called
            called = True
            return "REAL WRITE"

        with stub_scope({"save_application": lambda **kw: "stubbed"}):
            result = save_application(row="x")

        assert isinstance(result, Ok)
        assert result.value == "stubbed"
        assert called is False

    async def test_a_sync_stub_works_for_an_async_tool(self) -> None:
        """Recording stubs are usually trivial; requiring async would be noise."""

        @tool(never_empty=True, side_effect=True)
        async def send_gmail(to: str) -> str:
            return "REAL SEND"

        with stub_scope({"send_gmail": lambda **kw: "stubbed"}):
            result = await send_gmail(to="a@b.com")

        assert isinstance(result, Ok)
        assert result.value == "stubbed"

    async def test_a_stub_receives_the_real_arguments(self) -> None:
        seen: dict[str, Any] = {}

        @tool(never_empty=True, side_effect=True)
        async def send_gmail(to: str, body: str) -> str:
            return "REAL SEND"

        def stub(**kwargs: Any) -> str:
            seen.update(kwargs)
            return "stubbed"

        with stub_scope({"send_gmail": stub}):
            await send_gmail(to="a@b.com", body="hello")

        assert seen == {"to": "a@b.com", "body": "hello"}

    async def test_a_stub_result_goes_through_the_same_interpretation(self) -> None:
        """A stub returning nothing still means what the tool declared."""

        @tool(empty_when=lambda rows: len(rows) == 0, side_effect=True)
        async def fetch_rows() -> list[int]:
            return [1, 2, 3]

        with stub_scope({"fetch_rows": lambda **kw: []}):
            result = await fetch_rows()

        assert isinstance(result, Empty)

    async def test_a_raising_stub_becomes_an_err(self) -> None:
        @tool(never_empty=True, side_effect=True)
        async def send_gmail(to: str) -> str:
            return "REAL SEND"

        def stub(**kwargs: Any) -> str:
            raise ConnectionError("stub simulating an outage")

        with stub_scope({"send_gmail": stub}):
            result = await send_gmail(to="a@b.com")

        assert isinstance(result, Err)
        assert result.kind == "upstream"

    def test_a_raising_sync_stub_becomes_an_err(self) -> None:
        @tool(never_empty=True, side_effect=True)
        def save(row: str) -> str:
            return "REAL WRITE"

        def stub(**kwargs: Any) -> str:
            raise ValueError("bad row")

        with stub_scope({"save": stub}):
            result = save(row="x")

        assert isinstance(result, Err)
        assert result.kind == "validation"

    async def test_cancellation_inside_a_stub_propagates(self) -> None:
        @tool(never_empty=True, side_effect=True)
        async def send_gmail(to: str) -> str:
            return "REAL SEND"

        def stub(**kwargs: Any) -> str:
            raise asyncio.CancelledError

        with stub_scope({"send_gmail": stub}), pytest.raises(asyncio.CancelledError):
            await send_gmail(to="a@b.com")

    def test_cancellation_inside_a_sync_stub_propagates(self) -> None:
        @tool(never_empty=True, side_effect=True)
        def save(row: str) -> str:
            return "REAL WRITE"

        def stub(**kwargs: Any) -> str:
            raise KeyboardInterrupt

        with stub_scope({"save": stub}), pytest.raises(KeyboardInterrupt):
            save(row="x")

    async def test_an_unrelated_tool_is_unaffected(self) -> None:
        @tool(never_empty=True)
        async def search_jobs() -> str:
            return "real results"

        with stub_scope({"send_gmail": lambda **kw: "stubbed"}):
            result = await search_jobs()

        assert isinstance(result, Ok)
        assert result.value == "real results"


class TestStubTracing:
    """A stubbed call is still a tool call, so it is traced the same way."""

    async def test_a_stubbed_call_gets_a_span(self) -> None:
        sink = MemorySink()
        tracer = Tracer(sink=sink)

        @tool(never_empty=True, side_effect=True)
        async def send_gmail(to: str) -> str:
            return "REAL SEND"

        async with tracer.run():
            with stub_scope({"send_gmail": lambda **kw: "stubbed"}):
                await send_gmail(to="a@b.com")

        span = sink.traces[0].spans[0]
        assert span.name == "send_gmail"
        assert span.kind == "tool"
        assert span.attributes["tool.status"] == "ok"

    async def test_the_span_records_that_it_was_stubbed(self) -> None:
        """Otherwise a report could not tell a stubbed run from a real one."""
        sink = MemorySink()
        tracer = Tracer(sink=sink)

        @tool(never_empty=True, side_effect=True)
        async def send_gmail(to: str) -> str:
            return "REAL SEND"

        async with tracer.run():
            with stub_scope({"send_gmail": lambda **kw: "stubbed"}):
                await send_gmail(to="a@b.com")

        assert sink.traces[0].spans[0].attributes["tool.stubbed"] is True

    def test_a_sync_stubbed_call_is_traced(self) -> None:
        sink = MemorySink()
        tracer = Tracer(sink=sink)

        @tool(never_empty=True, side_effect=True)
        def save(row: str) -> str:
            return "REAL WRITE"

        with tracer.run(), stub_scope({"save": lambda **kw: "stubbed"}):
            save(row="x")

        assert sink.traces[0].spans[0].attributes["tool.stubbed"] is True

    async def test_a_failing_stub_is_traced_as_an_error(self) -> None:
        sink = MemorySink()
        tracer = Tracer(sink=sink)

        @tool(never_empty=True, side_effect=True)
        async def send_gmail(to: str) -> str:
            return "REAL SEND"

        def stub(**kwargs: Any) -> str:
            raise ConnectionError("down")

        async with tracer.run():
            with stub_scope({"send_gmail": stub}):
                await send_gmail(to="a@b.com")

        span = sink.traces[0].spans[0]
        assert span.status == "error"
        assert span.attributes["tool.error_kind"] == "upstream"

    async def test_stub_arguments_are_redacted(self) -> None:
        from neverempty import redact

        sink = MemorySink()
        tracer = Tracer(sink=sink, redact=redact.keys("email"))

        @tool(never_empty=True, side_effect=True)
        async def send_gmail(email: str) -> str:
            return "REAL SEND"

        async with tracer.run():
            with stub_scope({"send_gmail": lambda **kw: "stubbed"}):
                await send_gmail(email="real@person.com")

        assert "real@person.com" not in sink.traces[0].model_dump_json()
