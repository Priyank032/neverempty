"""Handler paths that a normal graph run does not reach.

Mostly error and version-skew paths. They matter more than the happy path:
telemetry must never be the thing that fails an agent, so every callback has to
survive a payload shape it does not recognise.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from toolproof import MemorySink, Tracer

pytest.importorskip("langchain_core")

pytestmark = pytest.mark.integration


def build() -> tuple[Tracer, MemorySink, Any]:
    sink = MemorySink()
    tracer = Tracer(sink=sink)
    return tracer, sink, tracer.langchain_handler()


class FakeMessage:
    def __init__(
        self,
        usage: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.usage_metadata = usage
        self.response_metadata = metadata or {}


class FakeGeneration:
    def __init__(
        self, message: FakeMessage | None = None, info: dict[str, Any] | None = None
    ) -> None:
        if message is not None:
            self.message = message
        self.generation_info = info


class FakeResult:
    def __init__(self, generations: Any = None, llm_output: dict[str, Any] | None = None) -> None:
        self.generations = generations if generations is not None else []
        self.llm_output = llm_output


class TestErrorPaths:
    async def test_a_chain_error_closes_the_span_as_error(self) -> None:
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_chain_start({"name": "classify_intent"}, {}, run_id=run_id)
            handler.on_chain_error(RuntimeError("boom"), run_id=run_id)

        span = sink.traces[0].spans[0]
        assert span.status == "error"
        assert span.attributes["error.kind"] == "RuntimeError"

    async def test_an_llm_error_closes_the_span_as_error(self) -> None:
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_llm_start({"id": ["FakeChat"]}, ["prompt"], run_id=run_id)
            handler.on_llm_error(RuntimeError("rate limited"), run_id=run_id)

        assert sink.traces[0].spans[0].status == "error"

    async def test_a_tool_error_closes_the_span_as_error(self) -> None:
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_tool_start({"name": "search_jobs"}, "{}", run_id=run_id)
            handler.on_tool_error(RuntimeError("down"), run_id=run_id)

        assert sink.traces[0].spans[0].status == "error"

    async def test_an_end_event_for_an_unknown_run_is_ignored(self) -> None:
        """Events can arrive for a run that started before the tracer did."""
        tracer, sink, handler = build()
        async with tracer.run():
            handler.on_chain_end({}, run_id=uuid4())
            handler.on_llm_end(FakeResult(), run_id=uuid4())
            handler.on_tool_end(None, run_id=uuid4())

        assert sink.traces[0].spans == []

    async def test_a_malformed_llm_result_does_not_raise(self) -> None:
        """A version skew must cost a span, not the request."""
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_llm_start({"id": ["FakeChat"]}, ["p"], run_id=run_id)
            handler.on_llm_end(object(), run_id=run_id)

        assert len(sink.traces[0].spans) == 1

    async def test_cancellation_inside_a_callback_propagates(self) -> None:
        """The guard swallows ordinary errors, never a cancellation."""
        import asyncio

        tracer, _, handler = build()

        class Exploding:
            @property
            def generations(self) -> Any:
                raise asyncio.CancelledError

        run_id = uuid4()

        async def go() -> None:
            async with tracer.run():
                handler.on_llm_start({"id": ["FakeChat"]}, ["p"], run_id=run_id)
                handler.on_llm_end(Exploding(), run_id=run_id)

        with pytest.raises(asyncio.CancelledError):
            await go()


class TestNodeNameFiltering:
    @pytest.mark.parametrize(
        "name",
        ["LangGraph", "RunnableSequence", "ChannelWrite", "__start__", "_internal"],
    )
    async def test_plumbing_chains_get_no_span(self, name: str) -> None:
        """A span per channel writer would bury the real trace."""
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_chain_start({"name": name}, {}, run_id=run_id)
            handler.on_chain_end({}, run_id=run_id)

        assert sink.traces[0].spans == []

    async def test_an_unnamed_chain_gets_no_span(self) -> None:
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_chain_start(None, {}, run_id=run_id)
            handler.on_chain_end({}, run_id=run_id)

        assert sink.traces[0].spans == []

    async def test_a_real_node_gets_a_span(self) -> None:
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_chain_start({"name": "safety_check"}, {}, run_id=run_id)
            handler.on_chain_end({}, run_id=run_id)

        span = sink.traces[0].spans[0]
        assert span.name == "safety_check"
        assert span.attributes["graph.node"] == "safety_check"


class TestUsageExtraction:
    async def test_usage_from_llm_output_token_usage(self) -> None:
        """The older OpenAI-style shape."""
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_llm_start({"id": ["ChatOpenAI"]}, ["p"], run_id=run_id)
            handler.on_llm_end(
                FakeResult(
                    llm_output={
                        "token_usage": {"prompt_tokens": 42, "completion_tokens": 7},
                        "model_name": "gpt-4o-2024-08-06",
                    }
                ),
                run_id=run_id,
            )

        trace = sink.traces[0]
        assert trace.usage.input_tokens == 42
        assert trace.usage.output_tokens == 7
        assert trace.spans[0].attributes["gen_ai.response.model"] == "gpt-4o-2024-08-06"

    async def test_a_finish_reason_from_generation_info(self) -> None:
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_llm_start({"id": ["ChatOpenAI"]}, ["p"], run_id=run_id)
            handler.on_llm_end(
                FakeResult(generations=[[FakeGeneration(info={"finish_reason": "length"})]]),
                run_id=run_id,
            )

        assert sink.traces[0].spans[0].attributes["gen_ai.response.finish_reason"] == "length"

    async def test_cached_read_tokens_are_captured(self) -> None:
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_llm_start({"id": ["ChatOpenAI"]}, ["p"], run_id=run_id)
            handler.on_llm_end(
                FakeResult(
                    generations=[
                        [
                            FakeGeneration(
                                FakeMessage(
                                    usage={
                                        "input_tokens": 100,
                                        "output_tokens": 10,
                                        "input_token_details": {"cache_read": 60},
                                    }
                                )
                            )
                        ]
                    ]
                ),
                run_id=run_id,
            )

        assert sink.traces[0].usage.cached_input_tokens == 60

    async def test_no_usage_reported_records_null_not_zero(self) -> None:
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_llm_start({"id": ["ChatOpenAI"]}, ["p"], run_id=run_id)
            handler.on_llm_end(FakeResult(), run_id=run_id)

        usage = sink.traces[0].usage
        assert usage.input_tokens is None
        assert usage.usage_missing is True

    async def test_invocation_params_supply_the_requested_model(self) -> None:
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_chat_model_start(
                {"id": ["ChatOpenAI"]},
                [[]],
                run_id=run_id,
                invocation_params={"model": "gpt-4o", "temperature": 0.0, "seed": 7},
            )
            handler.on_llm_end(FakeResult(), run_id=run_id)

        attributes = sink.traces[0].spans[0].attributes
        assert attributes["gen_ai.request.model"] == "gpt-4o"
        assert attributes["gen_ai.request.temperature"] == 0.0
        assert attributes["gen_ai.request.seed"] == 7

    async def test_the_class_name_is_the_fallback_span_name(self) -> None:
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_llm_start({"id": ["langchain", "ChatAnthropic"]}, ["p"], run_id=run_id)
            handler.on_llm_end(FakeResult(), run_id=run_id)

        assert sink.traces[0].spans[0].name == "ChatAnthropic"


class TestToolStatus:
    @pytest.mark.parametrize("status", ["ok", "empty", "error"])
    async def test_a_tool_result_status_reaches_the_span(self, status: str) -> None:
        tracer, sink, handler = build()
        run_id = uuid4()

        class Output:
            def __init__(self, value: str) -> None:
                self.status = value

        async with tracer.run():
            handler.on_tool_start({"name": "search_jobs"}, "{}", run_id=run_id)
            handler.on_tool_end(Output(status), run_id=run_id)

        assert sink.traces[0].spans[0].status == status

    async def test_a_plain_string_output_leaves_the_status_ok(self) -> None:
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_tool_start({"name": "t"}, "{}", run_id=run_id)
            handler.on_tool_end("just text", run_id=run_id)

        assert sink.traces[0].spans[0].status == "ok"

    async def test_an_unnamed_tool_falls_back_to_a_generic_name(self) -> None:
        tracer, sink, handler = build()
        run_id = uuid4()
        async with tracer.run():
            handler.on_tool_start(None, "{}", run_id=run_id)
            handler.on_tool_end(None, run_id=run_id)

        assert sink.traces[0].spans[0].name == "tool"


class TestOutsideARun:
    async def test_every_callback_is_safe_with_no_run_active(self) -> None:
        _, sink, handler = build()
        run_id = uuid4()
        handler.on_chain_start({"name": "n"}, {}, run_id=run_id)
        handler.on_chain_end({}, run_id=run_id)
        handler.on_llm_start({"id": ["X"]}, ["p"], run_id=run_id)
        handler.on_llm_end(FakeResult(), run_id=run_id)
        handler.on_tool_start({"name": "t"}, "{}", run_id=run_id)
        handler.on_tool_end(None, run_id=run_id)
        assert sink.traces == []


class TestSkippedWhenUnsampled:
    async def test_no_spans_are_recorded_at_sample_rate_zero(self) -> None:
        sink = MemorySink()
        tracer = Tracer(sink=sink, sample_rate=0.0)
        handler = tracer.langchain_handler()
        run_id = uuid4()
        async with tracer.run():
            handler.on_chain_start({"name": "safety_check"}, {}, run_id=run_id)
            handler.on_chain_end({}, run_id=run_id)

        assert sink.traces == []
