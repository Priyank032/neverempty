"""Usage capture from provider SDKs.

    OpenAI SDK direct: tracer.instrument_openai(client) captures usage,
    resolved model, latency, finish reason
    Bedrock: tracer.instrument_bedrock(client) captures usage from converse

Neither SDK is imported: instrumentation is duck-typed against the response
shapes, so core stays at one dependency and these tests need no extra. That is
also the honest test — a fake with the real response shape exercises the same
code path a live client would.
"""

from __future__ import annotations

from typing import Any

import pytest

from neverempty import MemorySink, Tracer


def build() -> tuple[Tracer, MemorySink]:
    sink = MemorySink()
    return Tracer(sink=sink), sink


class FakeUsage:
    """Matches ``openai.types.CompletionUsage``."""

    def __init__(
        self, prompt: int | None = 11, completion: int | None = 5, cached: int | None = None
    ) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        if cached is not None:
            self.prompt_tokens_details = type("Details", (), {"cached_tokens": cached})()


class FakeChoice:
    def __init__(self, finish_reason: str = "stop") -> None:
        self.finish_reason = finish_reason


class FakeCompletion:
    """Matches the shape of ``client.chat.completions.create(...)``."""

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        usage: FakeUsage | None = None,
        finish_reason: str = "stop",
    ) -> None:
        self.model = model
        self.usage = usage if usage is not None else FakeUsage()
        self.choices = [FakeChoice(finish_reason)]


class FakeCompletions:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class FakeOpenAI:
    def __init__(self, response: Any = None) -> None:
        self.chat = type(
            "Chat", (), {"completions": FakeCompletions(response or FakeCompletion())}
        )()


class FakeBedrock:
    """Matches ``boto3.client("bedrock-runtime").converse(...)``."""

    def __init__(self, response: Any = None) -> None:
        self._response: Any = response or {
            "usage": {"inputTokens": 812, "outputTokens": 240},
            "stopReason": "end_turn",
            "output": {"message": {"content": [{"text": "hello"}]}},
        }
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class TestOpenAICapture:
    async def test_usage_reaches_the_trace(self) -> None:
        tracer, sink = build()
        client = FakeOpenAI()
        tracer.instrument_openai(client)

        async with tracer.run():
            client.chat.completions.create(model="gpt-4o", messages=[])

        trace = sink.traces[0]
        assert trace.usage.input_tokens == 11
        assert trace.usage.output_tokens == 5
        assert trace.usage.usage_missing is False

    async def test_the_requested_and_resolved_models_are_both_recorded(self) -> None:
        """Model drift is only detectable if both are kept."""
        tracer, sink = build()
        client = FakeOpenAI(FakeCompletion(model="gpt-4o-2024-08-06"))
        tracer.instrument_openai(client)

        async with tracer.run():
            client.chat.completions.create(model="gpt-4o", messages=[])

        attributes = sink.traces[0].spans[0].attributes
        assert attributes["gen_ai.request.model"] == "gpt-4o"
        assert attributes["gen_ai.response.model"] == "gpt-4o-2024-08-06"

    async def test_the_finish_reason_is_recorded(self) -> None:
        tracer, sink = build()
        client = FakeOpenAI(FakeCompletion(finish_reason="length"))
        tracer.instrument_openai(client)

        async with tracer.run():
            client.chat.completions.create(model="gpt-4o", messages=[])

        assert sink.traces[0].spans[0].attributes["gen_ai.response.finish_reason"] == "length"

    async def test_request_parameters_are_recorded(self) -> None:
        tracer, sink = build()
        client = FakeOpenAI()
        tracer.instrument_openai(client)

        async with tracer.run():
            client.chat.completions.create(model="gpt-4o", messages=[], temperature=0.0, seed=42)

        attributes = sink.traces[0].spans[0].attributes
        assert attributes["gen_ai.request.temperature"] == 0.0
        assert attributes["gen_ai.request.seed"] == 42

    async def test_cached_input_tokens_are_captured_when_present(self) -> None:
        tracer, sink = build()
        client = FakeOpenAI(FakeCompletion(usage=FakeUsage(cached=8)))
        tracer.instrument_openai(client)

        async with tracer.run():
            client.chat.completions.create(model="gpt-4o", messages=[])

        assert sink.traces[0].usage.cached_input_tokens == 8

    async def test_missing_usage_is_null_not_zero(self) -> None:
        """A provider that returns no usage must not look like a free call."""
        tracer, sink = build()
        completion = FakeCompletion()
        completion.usage = None  # type: ignore[assignment]
        client = FakeOpenAI(completion)
        tracer.instrument_openai(client)

        async with tracer.run():
            client.chat.completions.create(model="gpt-4o", messages=[])

        usage = sink.traces[0].usage
        assert usage.input_tokens is None
        assert usage.usage_missing is True
        assert sink.traces[0].cost.usd is None

    async def test_the_span_records_latency(self) -> None:
        tracer, sink = build()
        client = FakeOpenAI()
        tracer.instrument_openai(client)

        async with tracer.run():
            client.chat.completions.create(model="gpt-4o", messages=[])

        span = sink.traces[0].spans[0]
        assert span.kind == "llm"
        assert span.duration_ns >= 0

    async def test_a_provider_error_marks_the_span_and_propagates(self) -> None:
        tracer, sink = build()
        client = FakeOpenAI(RuntimeError("rate limited"))
        tracer.instrument_openai(client)

        with pytest.raises(RuntimeError, match="rate limited"):
            async with tracer.run():
                client.chat.completions.create(model="gpt-4o", messages=[])

        assert sink.traces[0].spans[0].status == "error"

    async def test_no_message_content_is_ever_recorded(self) -> None:
        """Prompts carry user data; usage capture is not a prompt logger."""
        tracer, sink = build()
        client = FakeOpenAI()
        tracer.instrument_openai(client)

        async with tracer.run():
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "my aadhaar is 1234 5678 9012"}],
            )

        serialized = sink.traces[0].model_dump_json()
        assert "aadhaar" not in serialized
        assert "1234" not in serialized

    def test_it_works_outside_a_run(self) -> None:
        tracer, sink = build()
        client = FakeOpenAI()
        tracer.instrument_openai(client)
        assert client.chat.completions.create(model="gpt-4o", messages=[]) is not None
        assert sink.traces == []

    def test_instrumenting_twice_does_not_double_count(self) -> None:
        tracer, sink = build()
        client = FakeOpenAI()
        tracer.instrument_openai(client)
        tracer.instrument_openai(client)

        with tracer.run():
            client.chat.completions.create(model="gpt-4o", messages=[])

        assert len(sink.traces[0].spans) == 1
        assert sink.traces[0].usage.input_tokens == 11

    def test_the_original_response_is_returned_unchanged(self) -> None:
        tracer, _ = build()
        expected = FakeCompletion()
        client = FakeOpenAI(expected)
        tracer.instrument_openai(client)
        with tracer.run():
            assert client.chat.completions.create(model="gpt-4o", messages=[]) is expected

    def test_a_client_of_the_wrong_shape_is_rejected_clearly(self) -> None:
        tracer, _ = build()
        with pytest.raises(TypeError, match=r"chat\.completions\.create"):
            tracer.instrument_openai(object())


class TestBedrockCapture:
    async def test_usage_from_converse_reaches_the_trace(self) -> None:
        tracer, sink = build()
        client = FakeBedrock()
        tracer.instrument_bedrock(client)

        async with tracer.run():
            client.converse(modelId="anthropic.claude-3-5-sonnet", messages=[])

        trace = sink.traces[0]
        assert trace.usage.input_tokens == 812
        assert trace.usage.output_tokens == 240

    async def test_the_model_id_is_recorded(self) -> None:
        tracer, sink = build()
        client = FakeBedrock()
        tracer.instrument_bedrock(client)

        async with tracer.run():
            client.converse(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0", messages=[])

        attributes = sink.traces[0].spans[0].attributes
        assert attributes["gen_ai.request.model"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"

    async def test_the_stop_reason_is_recorded_as_the_finish_reason(self) -> None:
        tracer, sink = build()
        client = FakeBedrock()
        tracer.instrument_bedrock(client)

        async with tracer.run():
            client.converse(modelId="m", messages=[])

        assert sink.traces[0].spans[0].attributes["gen_ai.response.finish_reason"] == "end_turn"

    async def test_missing_usage_is_null_not_zero(self) -> None:
        tracer, sink = build()
        client = FakeBedrock({"stopReason": "end_turn"})
        tracer.instrument_bedrock(client)

        async with tracer.run():
            client.converse(modelId="m", messages=[])

        assert sink.traces[0].usage.input_tokens is None
        assert sink.traces[0].usage.usage_missing is True

    async def test_cache_read_tokens_are_captured(self) -> None:
        tracer, sink = build()
        client = FakeBedrock(
            {
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 20,
                    "cacheReadInputTokens": 40,
                },
                "stopReason": "end_turn",
            }
        )
        tracer.instrument_bedrock(client)

        async with tracer.run():
            client.converse(modelId="m", messages=[])

        assert sink.traces[0].usage.cached_input_tokens == 40

    async def test_no_message_content_is_recorded(self) -> None:
        tracer, sink = build()
        client = FakeBedrock()
        tracer.instrument_bedrock(client)

        async with tracer.run():
            client.converse(
                modelId="m",
                messages=[{"role": "user", "content": [{"text": "my pan is ABCDE1234F"}]}],
            )

        assert "ABCDE1234F" not in sink.traces[0].model_dump_json()

    async def test_an_error_marks_the_span_and_propagates(self) -> None:
        tracer, sink = build()
        client = FakeBedrock()
        client._response = RuntimeError("throttled")
        tracer.instrument_bedrock(client)

        with pytest.raises(RuntimeError, match="throttled"):
            async with tracer.run():
                client.converse(modelId="m", messages=[])

        assert sink.traces[0].spans[0].status == "error"

    def test_a_client_without_converse_is_rejected_clearly(self) -> None:
        tracer, _ = build()
        with pytest.raises(TypeError, match="converse"):
            tracer.instrument_bedrock(object())

    def test_the_response_is_returned_unchanged(self) -> None:
        tracer, _ = build()
        client = FakeBedrock()
        expected = client._response
        tracer.instrument_bedrock(client)
        with tracer.run():
            assert client.converse(modelId="m", messages=[]) is expected


class TestResolvedModelsCollectIntoEnv:
    async def test_several_providers_in_one_trace_are_all_recorded(self) -> None:
        tracer, sink = build()
        openai_client = FakeOpenAI(FakeCompletion(model="gpt-4o-2024-08-06"))
        bedrock_client = FakeBedrock()
        tracer.instrument_openai(openai_client)
        tracer.instrument_bedrock(bedrock_client)

        async with tracer.run():
            openai_client.chat.completions.create(model="gpt-4o", messages=[])
            bedrock_client.converse(modelId="anthropic.claude-3-5-sonnet", messages=[])

        models = sink.traces[0].env.resolved_models
        assert "gpt-4o-2024-08-06" in models
        assert "anthropic.claude-3-5-sonnet" in models

    async def test_usage_is_summed_across_providers(self) -> None:
        tracer, sink = build()
        openai_client = FakeOpenAI()
        bedrock_client = FakeBedrock()
        tracer.instrument_openai(openai_client)
        tracer.instrument_bedrock(bedrock_client)

        async with tracer.run():
            openai_client.chat.completions.create(model="gpt-4o", messages=[])
            bedrock_client.converse(modelId="m", messages=[])

        assert sink.traces[0].usage.input_tokens == 11 + 812
