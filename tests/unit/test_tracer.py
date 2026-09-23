"""Acceptance row for ``tracer``.

500 concurrent runs with zero cross-attached spans; nested tool inside node
inside run gives correct parents; redaction applied before sink with a test
asserting no email or token in serialized output; sampling at 0.0 records
nothing and breaks nothing; sink failure never raises into the agent
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from toolproof import (
    Empty,
    Err,
    JsonlSink,
    MemorySink,
    Ok,
    Pricing,
    Tracer,
    redact,
    tool,
)


def build(**kwargs: Any) -> tuple[Tracer, MemorySink]:
    sink = MemorySink()
    kwargs.setdefault("sink", sink)
    return Tracer(**kwargs), sink


class TestRunLifecycle:
    async def test_a_run_produces_one_trace_in_the_sink(self) -> None:
        tracer, sink = build()
        async with tracer.run(case_id="c1") as run:
            run.set_output(answer="hello", route="job_search")
        assert len(sink.traces) == 1
        trace = sink.traces[0]
        assert trace.case_id == "c1"
        assert trace.final_output.answer == "hello"
        assert trace.final_output.route == "job_search"

    async def test_the_trace_is_flushed_and_readable_after_the_block(self) -> None:
        tracer, _ = build()
        async with tracer.run(case_id="c1") as run:
            pass
        assert run.trace is not None
        assert run.trace.status == "ok"

    async def test_duration_comes_from_a_monotonic_clock(self) -> None:
        tracer, sink = build()
        async with tracer.run():
            await asyncio.sleep(0.02)
        assert sink.traces[0].duration_ms >= 15

    async def test_tags_reach_the_trace(self) -> None:
        tracer, sink = build()
        async with tracer.run(tags={"suite": "nextrole.routing"}):
            pass
        assert sink.traces[0].tags["suite"] == "nextrole.routing"

    async def test_repeat_is_recorded(self) -> None:
        tracer, sink = build()
        async with tracer.run(case_id="c1", repeat=2):
            pass
        assert sink.traces[0].repeat == 2

    async def test_an_exception_in_the_body_marks_the_trace_and_propagates(self) -> None:
        tracer, sink = build()
        with pytest.raises(RuntimeError):
            async with tracer.run():
                raise RuntimeError("target blew up")
        assert sink.traces[0].status == "target_error"
        assert sink.traces[0].error is not None
        assert sink.traces[0].error.kind == "RuntimeError"

    async def test_cancellation_propagates_and_is_not_recorded_as_an_error(self) -> None:
        """A budget abort is not an agent failure."""
        tracer, sink = build()
        with pytest.raises(asyncio.CancelledError):
            async with tracer.run():
                raise asyncio.CancelledError
        assert sink.traces[0].status == "budget_abort"

    async def test_a_traceback_never_reaches_the_trace(self) -> None:
        tracer, sink = build()
        with pytest.raises(ValueError):
            async with tracer.run():
                raise ValueError("boom at /home/me/secret.py")
        serialized = sink.traces[0].model_dump_json()
        assert "Traceback" not in serialized


class TestSpanParenting:
    async def test_nested_tool_inside_node_inside_run_gives_correct_parents(self) -> None:
        tracer, sink = build()

        @tool(never_empty=True)
        async def search_jobs() -> list[int]:
            return [1]

        async with tracer.run(case_id="c1"):
            with tracer.span("node", name="classify_intent"):
                await search_jobs()

        spans = {s.name: s for s in sink.traces[0].spans}
        node = spans["classify_intent"]
        tool_span = spans["search_jobs"]
        assert node.parent_id is None
        assert tool_span.parent_id == node.span_id
        assert node.kind == "node"
        assert tool_span.kind == "tool"

    async def test_span_ids_are_monotonic_within_a_trace(self) -> None:
        tracer, sink = build()
        async with tracer.run():
            for i in range(3):
                with tracer.span("custom", name=f"s{i}"):
                    pass
        ids = [s.span_id for s in sink.traces[0].spans]
        assert ids == sorted(ids)
        assert ids[0] == "s0001"

    async def test_sibling_spans_share_a_parent(self) -> None:
        tracer, sink = build()
        async with tracer.run():
            with tracer.span("node", name="parent"):
                with tracer.span("custom", name="a"):
                    pass
                with tracer.span("custom", name="b"):
                    pass

        spans = {s.name: s for s in sink.traces[0].spans}
        assert spans["a"].parent_id == spans["parent"].span_id
        assert spans["b"].parent_id == spans["parent"].span_id

    async def test_deep_nesting_keeps_a_single_chain(self) -> None:
        tracer, sink = build()
        async with tracer.run():
            with (
                tracer.span("node", name="l1"),
                tracer.span("node", name="l2"),
                tracer.span("custom", name="l3"),
            ):
                pass

        spans = {s.name: s for s in sink.traces[0].spans}
        assert spans["l1"].parent_id is None
        assert spans["l2"].parent_id == spans["l1"].span_id
        assert spans["l3"].parent_id == spans["l2"].span_id

    async def test_a_raising_span_is_recorded_as_error_and_propagates(self) -> None:
        tracer, sink = build()

        async def boom() -> None:
            # Spans are synchronous context managers by design: nothing in
            # opening or closing one awaits.
            async with tracer.run():
                with tracer.span("custom", name="bad"):
                    raise ValueError("x")

        with pytest.raises(ValueError):
            await boom()

        span = next(s for s in sink.traces[0].spans if s.name == "bad")
        assert span.status == "error"


class TestToolSpanAttributes:
    """The doc fixes these attribute names; scorers key on them."""

    async def test_an_ok_tool_records_its_documented_attributes(self) -> None:
        tracer, sink = build()

        @tool(never_empty=True, truncated_when=lambda rows: len(rows) >= 2)
        async def search_jobs(city: str) -> list[int]:
            return [1, 2]

        async with tracer.run():
            await search_jobs(city="Pune")

        attrs = sink.traces[0].spans[0].attributes
        assert attrs["tool.name"] == "search_jobs"
        assert attrs["tool.status"] == "ok"
        assert attrs["tool.truncated"] is True
        assert attrs["tool.fault_injected"] is False
        assert json.loads(str(attrs["tool.args"]))["city"] == "Pune"

    async def test_an_empty_tool_records_status_empty(self) -> None:
        tracer, sink = build()

        @tool(empty_when=lambda rows: len(rows) == 0)
        async def t() -> list[int]:
            return []

        async with tracer.run():
            assert isinstance(await t(), Empty)

        span = sink.traces[0].spans[0]
        assert span.status == "empty"
        assert span.attributes["tool.status"] == "empty"

    async def test_a_failing_tool_records_kind_and_retryability(self) -> None:
        tracer, sink = build()

        @tool(never_empty=True)
        async def t() -> str:
            raise ConnectionError("down")

        async with tracer.run():
            assert isinstance(await t(), Err)

        attrs = sink.traces[0].spans[0].attributes
        assert attrs["tool.status"] == "error"
        assert attrs["tool.error_kind"] == "upstream"
        assert attrs["tool.retryable"] is True

    async def test_an_injected_fault_is_flagged_on_the_span(self) -> None:
        from toolproof import FaultSpec, fault_scope

        tracer, sink = build()

        @tool(never_empty=True)
        async def search_jobs() -> str:
            return "real"

        async with tracer.run():
            with fault_scope([FaultSpec(tool="search_jobs", kind="timeout")]):
                await search_jobs()

        attrs = sink.traces[0].spans[0].attributes
        assert attrs["tool.fault_injected"] is True
        assert attrs["tool.error_kind"] == "timeout"

    async def test_each_retry_attempt_gets_its_own_span(self) -> None:
        tracer, sink = build()
        attempts = 0

        @tool(never_empty=True, retries=2, retry_backoff_s=0.0)
        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            raise ConnectionError("down")

        async with tracer.run():
            await flaky()

        spans = sink.traces[0].spans
        assert len(spans) == 3
        assert [s.attributes["tool.attempt"] for s in spans] == [0, 1, 2]

    async def test_tool_args_are_capped_at_8kb(self) -> None:
        tracer, sink = build()

        @tool(never_empty=True)
        async def t(blob: str) -> int:
            return 1

        async with tracer.run():
            await t(blob="x" * 20_000)

        args = str(sink.traces[0].spans[0].attributes["tool.args"])
        assert len(args.encode()) <= 8192

    async def test_a_sync_tool_is_traced_too(self) -> None:
        tracer, sink = build()

        @tool(never_empty=True)
        def t() -> int:
            return 1

        async with tracer.run():
            t()

        assert sink.traces[0].spans[0].name == "t"


class TestLlmSpans:
    async def test_record_usage_writes_the_gen_ai_attributes(self) -> None:
        tracer, sink = build()
        async with tracer.run():
            with tracer.span("llm", name="rerank") as span:
                span.record_usage(
                    model="gpt-4o-2024-08-06",
                    input_tokens=812,
                    output_tokens=240,
                )

        attrs = sink.traces[0].spans[0].attributes
        assert attrs["gen_ai.request.model"] == "gpt-4o-2024-08-06"
        assert attrs["gen_ai.usage.input_tokens"] == 812
        assert attrs["gen_ai.usage.output_tokens"] == 240

    async def test_the_resolved_model_is_recorded_separately_from_the_request(self) -> None:
        """Model drift is only detectable if the response model is kept."""
        tracer, sink = build()
        async with tracer.run():
            with tracer.span("llm", name="rerank") as span:
                span.record_usage(
                    model="gpt-4o",
                    resolved_model="gpt-4o-2024-08-06",
                    input_tokens=1,
                    output_tokens=1,
                )

        attrs = sink.traces[0].spans[0].attributes
        assert attrs["gen_ai.request.model"] == "gpt-4o"
        assert attrs["gen_ai.response.model"] == "gpt-4o-2024-08-06"

    async def test_trace_usage_is_summed_across_llm_spans(self) -> None:
        tracer, sink = build()
        async with tracer.run():
            for _ in range(2):
                with tracer.span("llm", name="call") as span:
                    span.record_usage(model="m", input_tokens=100, output_tokens=50)

        usage = sink.traces[0].usage
        assert usage.input_tokens == 200
        assert usage.output_tokens == 100
        assert usage.usage_missing is False

    async def test_usage_is_null_with_a_flag_when_no_llm_span_reported_any(self) -> None:
        """Never a local tokenizer estimate, and never a silent zero."""
        tracer, sink = build()
        async with tracer.run():
            with tracer.span("llm", name="call"):
                pass

        usage = sink.traces[0].usage
        assert usage.input_tokens is None
        assert usage.usage_missing is True

    async def test_a_partially_reported_usage_keeps_the_known_half(self) -> None:
        tracer, sink = build()
        async with tracer.run():
            with tracer.span("llm", name="call") as span:
                span.record_usage(model="m", input_tokens=10, output_tokens=None)

        usage = sink.traces[0].usage
        assert usage.input_tokens == 10
        assert usage.output_tokens is None

    async def test_optional_request_parameters_are_recorded_when_given(self) -> None:
        tracer, sink = build()
        async with tracer.run():
            with tracer.span("llm", name="c") as span:
                span.record_usage(
                    model="m",
                    input_tokens=1,
                    output_tokens=1,
                    temperature=0.0,
                    seed=42,
                    finish_reason="stop",
                )

        attrs = sink.traces[0].spans[0].attributes
        assert attrs["gen_ai.request.temperature"] == 0.0
        assert attrs["gen_ai.request.seed"] == 42
        assert attrs["gen_ai.response.finish_reason"] == "stop"

    async def test_resolved_models_are_collected_into_env(self) -> None:
        tracer, sink = build()
        async with tracer.run():
            with tracer.span("llm", name="a") as span:
                span.record_usage(model="gpt-4o", resolved_model="gpt-4o-2024-08-06")
            with tracer.span("llm", name="b") as span:
                span.record_usage(model="gpt-4o", resolved_model="gpt-4o-2024-08-06")

        assert sink.traces[0].env.resolved_models == ["gpt-4o-2024-08-06"]


class TestConcurrency:
    async def test_500_concurrent_runs_have_zero_cross_attached_spans(self) -> None:
        """The property that makes contextvars the right mechanism."""
        tracer, sink = build()

        @tool(never_empty=True)
        async def work(marker: int) -> int:
            await asyncio.sleep(0)
            return marker

        async def one(index: int) -> None:
            async with tracer.run(case_id=f"c{index}") as run:
                with tracer.span("node", name=f"node-{index}"):
                    await work(marker=index)
                await asyncio.sleep(0)
                run.set_output(answer=str(index))

        await asyncio.gather(*(one(i) for i in range(500)))

        assert len(sink.traces) == 500
        for trace in sink.traces:
            index = trace.case_id.removeprefix("c") if trace.case_id else ""
            names = {s.name for s in trace.spans}
            assert names == {f"node-{index}", "work"}, trace.case_id
            assert trace.final_output.answer == index
            args = json.loads(str(trace.spans[1].attributes["tool.args"]))
            assert args["marker"] == int(index)

    async def test_concurrent_spans_inside_one_run_keep_their_own_parents(self) -> None:
        tracer, sink = build()

        async def branch(name: str) -> None:
            with tracer.span("node", name=name):
                await asyncio.sleep(0.005)
                with tracer.span("custom", name=f"{name}-child"):
                    await asyncio.sleep(0)

        async with tracer.run():
            await asyncio.gather(branch("a"), branch("b"))

        spans = {s.name: s for s in sink.traces[0].spans}
        assert spans["a-child"].parent_id == spans["a"].span_id
        assert spans["b-child"].parent_id == spans["b"].span_id

    async def test_a_tool_called_outside_any_run_records_nothing(self) -> None:
        _tracer, sink = build()

        @tool(never_empty=True)
        async def t() -> int:
            return 1

        assert isinstance(await t(), Ok)
        assert sink.traces == []


class TestRedaction:
    async def test_redacted_keys_never_reach_the_sink(self) -> None:
        tracer, sink = build(redact=redact.keys("email", "phone", "token", "api_key", "aadhaar"))

        @tool(never_empty=True)
        async def send(email: str, token: str, city: str) -> int:
            return 1

        async with tracer.run():
            await send(
                email="priyank@example.com",
                token="sk-live-abcdef123456",
                city="Pune",
            )

        serialized = sink.traces[0].model_dump_json()
        assert "priyank@example.com" not in serialized
        assert "sk-live-abcdef123456" not in serialized
        assert "Pune" in serialized

    async def test_redaction_reaches_nested_argument_structures(self) -> None:
        tracer, sink = build(redact=redact.keys("email"))

        @tool(never_empty=True)
        async def t(profile: dict[str, Any]) -> int:
            return 1

        async with tracer.run():
            await t(profile={"contact": {"email": "a@b.com"}, "city": "Pune"})

        serialized = sink.traces[0].model_dump_json()
        assert "a@b.com" not in serialized
        assert "Pune" in serialized

    async def test_redaction_reaches_values_inside_lists(self) -> None:
        tracer, sink = build(redact=redact.keys("email"))

        @tool(never_empty=True)
        async def t(people: list[dict[str, str]]) -> int:
            return 1

        async with tracer.run():
            await t(people=[{"email": "a@b.com"}, {"email": "c@d.com"}])

        serialized = sink.traces[0].model_dump_json()
        assert "a@b.com" not in serialized
        assert "c@d.com" not in serialized

    async def test_key_matching_is_case_insensitive(self) -> None:
        tracer, sink = build(redact=redact.keys("email"))

        @tool(never_empty=True)
        async def t(userEmail: str, EMAIL: str) -> int:  # noqa: N803
            return 1

        async with tracer.run():
            await t(userEmail="a@b.com", EMAIL="c@d.com")

        serialized = sink.traces[0].model_dump_json()
        assert "a@b.com" not in serialized
        assert "c@d.com" not in serialized

    async def test_the_placeholder_is_stable_for_equal_values(self) -> None:
        """Two spans with the same redacted value show the same suffix, so a
        cache-key bug is debuggable without revealing the value."""
        tracer, sink = build(redact=redact.keys("email"))

        @tool(never_empty=True)
        async def t(email: str) -> int:
            return 1

        async with tracer.run():
            await t(email="same@example.com")
            await t(email="same@example.com")
            await t(email="other@example.com")

        args = [json.loads(str(s.attributes["tool.args"]))["email"] for s in sink.traces[0].spans]
        assert args[0] == args[1]
        assert args[0] != args[2]
        assert "same@example.com" not in args[0]

    async def test_the_output_answer_is_redacted_too(self) -> None:
        tracer, sink = build(redact=redact.keys("email"))
        async with tracer.run() as run:
            run.set_output(structured={"email": "a@b.com", "route": "x"})

        assert "a@b.com" not in sink.traces[0].model_dump_json()

    async def test_no_redaction_configured_leaves_values_intact(self) -> None:
        tracer, sink = build()

        @tool(never_empty=True)
        async def t(city: str) -> int:
            return 1

        async with tracer.run():
            await t(city="Pune")

        assert "Pune" in sink.traces[0].model_dump_json()

    def test_redaction_runs_before_the_sink_not_after(self) -> None:
        """A sink must never see an unredacted value, even briefly."""
        seen: list[str] = []

        class Spy(MemorySink):
            def write(self, trace: Any) -> None:
                seen.append(trace.model_dump_json())
                super().write(trace)

        tracer = Tracer(sink=Spy(), redact=redact.keys("email"))

        @tool(never_empty=True)
        def t(email: str) -> int:
            return 1

        async def go() -> None:
            async with tracer.run():
                t(email="leak@example.com")

        asyncio.run(go())
        assert seen
        assert all("leak@example.com" not in payload for payload in seen)


class TestSampling:
    async def test_sampling_at_zero_records_nothing_and_breaks_nothing(self) -> None:
        tracer, sink = build(sample_rate=0.0)

        @tool(never_empty=True)
        async def t() -> int:
            return 1

        async with tracer.run(case_id="c1") as run:
            with tracer.span("node", name="n"):
                assert isinstance(await t(), Ok)
            run.set_output(answer="still works")

        assert sink.traces == []

    async def test_sampling_at_one_records_everything(self) -> None:
        tracer, sink = build(sample_rate=1.0)
        async with tracer.run():
            pass
        assert len(sink.traces) == 1

    async def test_an_unsampled_run_still_exposes_a_usable_run_object(self) -> None:
        tracer, _ = build(sample_rate=0.0)
        async with tracer.run() as run:
            run.set_output(answer="a")
        assert run.sampled is False

    async def test_sampling_is_deterministic_under_a_seed(self) -> None:
        tracer_a, sink_a = build(sample_rate=0.5, seed=7)
        tracer_b, sink_b = build(sample_rate=0.5, seed=7)
        for tracer in (tracer_a, tracer_b):
            for i in range(50):
                async with tracer.run(case_id=f"c{i}"):
                    pass
        assert [t.case_id for t in sink_a.traces] == [t.case_id for t in sink_b.traces]

    def test_an_out_of_range_sample_rate_is_rejected(self) -> None:
        for bad in (-0.1, 1.1):
            with pytest.raises(ValueError, match="sample_rate"):
                Tracer(sink=MemorySink(), sample_rate=bad)


class TestSinkFailures:
    async def test_a_failing_sink_never_raises_into_the_agent(self) -> None:
        class Broken(MemorySink):
            def write(self, trace: Any) -> None:
                raise OSError("disk full")

        tracer = Tracer(sink=Broken())
        async with tracer.run() as run:
            run.set_output(answer="agent kept working")
        assert run.trace is not None

    async def test_a_dropped_trace_is_counted(self) -> None:
        class Broken(MemorySink):
            def write(self, trace: Any) -> None:
                raise OSError("disk full")

        tracer = Tracer(sink=Broken())
        for _ in range(3):
            async with tracer.run():
                pass
        assert tracer.dropped_traces == 3

    async def test_the_last_sink_error_is_retained_for_diagnosis(self) -> None:
        class Broken(MemorySink):
            def write(self, trace: Any) -> None:
                raise OSError("disk full")

        tracer = Tracer(sink=Broken())
        async with tracer.run():
            pass
        assert isinstance(tracer.last_sink_error, OSError)

    async def test_a_sink_failure_is_logged_at_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Silent data loss is the failure mode this library exists to prevent."""

        class Broken(MemorySink):
            def write(self, trace: Any) -> None:
                raise OSError("disk full")

        tracer = Tracer(sink=Broken())
        with caplog.at_level(logging.ERROR, logger="toolproof.tracer"):
            async with tracer.run():
                pass
        assert any("disk full" in record.getMessage() for record in caplog.records)

    async def test_a_healthy_sink_reports_no_drops(self) -> None:
        tracer, _ = build()
        async with tracer.run():
            pass
        assert tracer.dropped_traces == 0
        assert tracer.last_sink_error is None


class TestJsonlSink:
    async def test_it_writes_one_json_object_per_line(self, tmp_path: Path) -> None:
        tracer = Tracer(sink=JsonlSink(tmp_path))
        for i in range(3):
            async with tracer.run(case_id=f"c{i}"):
                pass

        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        assert [json.loads(line)["case_id"] for line in lines] == ["c0", "c1", "c2"]

    async def test_every_line_validates_against_the_trace_model(self, tmp_path: Path) -> None:
        from toolproof import Trace

        tracer = Tracer(sink=JsonlSink(tmp_path))

        @tool(never_empty=True)
        async def t(city: str) -> list[int]:
            return [1]

        async with tracer.run(case_id="c1") as run:
            with tracer.span("node", name="n"):
                await t(city="Pune")
            run.set_output(answer="a", route="job_search")

        line = next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8").strip()
        assert Trace.model_validate_json(line).case_id == "c1"

    async def test_it_creates_a_missing_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "traces"
        tracer = Tracer(sink=JsonlSink(target))
        async with tracer.run():
            pass
        assert list(target.glob("*.jsonl"))

    async def test_a_file_path_is_used_directly(self, tmp_path: Path) -> None:
        target = tmp_path / "out.jsonl"
        tracer = Tracer(sink=JsonlSink(target))
        async with tracer.run():
            pass
        assert target.is_file()

    async def test_no_secret_appears_in_the_written_file(self, tmp_path: Path) -> None:
        tracer = Tracer(sink=JsonlSink(tmp_path), redact=redact.keys("api_key"))

        @tool(never_empty=True)
        async def t(api_key: str) -> int:
            return 1

        async with tracer.run():
            await t(api_key="sk-live-secret")

        text = next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8")
        assert "sk-live-secret" not in text


class TestEnvBlock:
    async def test_the_toolproof_version_is_recorded(self) -> None:
        from toolproof import __version__

        tracer, sink = build()
        async with tracer.run():
            pass
        assert sink.traces[0].env.toolproof_version == __version__

    async def test_the_python_version_is_recorded(self) -> None:
        import sys

        tracer, sink = build()
        async with tracer.run():
            pass
        expected = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert sink.traces[0].env.python_version.startswith(expected)

    async def test_supplied_env_fields_override_the_defaults(self) -> None:
        tracer, sink = build(
            env_overrides={
                "target_git_sha": "b" * 40,
                "target_dirty": True,
                "concurrency": 8,
                "seed": 99,
                "mode": "replay",
            }
        )
        async with tracer.run():
            pass
        env = sink.traces[0].env
        assert env.target_git_sha == "b" * 40
        assert env.target_dirty is True
        assert env.mode == "replay"
        assert env.seed == 99

    async def test_prompt_hashes_are_carried_through(self) -> None:
        tracer, sink = build(env_overrides={"prompt_hashes": {"INTENT": "c" * 64}})
        async with tracer.run():
            pass
        assert sink.traces[0].env.prompt_hashes["INTENT"] == "c" * 64

    async def test_the_pricing_version_in_env_matches_the_pricing_table(self) -> None:
        pricing = Pricing.default()
        tracer, sink = build(pricing=pricing)
        async with tracer.run():
            pass
        assert sink.traces[0].env.pricing_version == pricing.version
        assert sink.traces[0].cost.pricing_version == pricing.version


class TestOverhead:
    @pytest.mark.benchmark
    async def test_a_span_costs_under_1ms(self) -> None:
        import statistics
        import time

        tracer, _ = build()
        samples: list[float] = []
        async with tracer.run():
            for _ in range(2000):
                start = time.perf_counter()
                with tracer.span("custom", name="s"):
                    pass
                samples.append(time.perf_counter() - start)

        median = statistics.median(samples)
        assert median < 1e-3, f"{median * 1e6:.1f}us per span exceeds the 1ms budget"

    @pytest.mark.benchmark
    async def test_no_network_io_on_the_hot_path(self) -> None:
        import socket

        original = socket.socket

        class Tripwire(socket.socket):
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise AssertionError("the tracer opened a socket")

        tracer, _ = build()
        socket.socket = Tripwire  # type: ignore[misc]
        try:
            async with tracer.run():
                with tracer.span("custom", name="s"):
                    pass
        finally:
            socket.socket = original  # type: ignore[misc]


class TestTracedFaultsAndCancellation:
    async def test_an_injected_empty_fault_produces_an_empty_span(self) -> None:
        from toolproof import FaultSpec, fault_scope

        tracer, sink = build()

        @tool(never_empty=True)
        async def search_jobs(city: str) -> str:
            return "real"

        async with tracer.run():
            with fault_scope([FaultSpec(tool="search_jobs", kind="empty")]):
                await search_jobs(city="Pune")

        span = sink.traces[0].spans[0]
        assert span.status == "empty"
        assert span.attributes["tool.fault_injected"] is True

    async def test_a_cancelled_tool_still_closes_its_span(self) -> None:
        """An open span would corrupt every later parent id in the trace."""
        tracer, sink = build()

        @tool(never_empty=True)
        async def t() -> str:
            raise asyncio.CancelledError

        async def go() -> None:
            async with tracer.run():
                await t()

        with pytest.raises(asyncio.CancelledError):
            await go()

        assert len(sink.traces[0].spans) == 1

    def test_a_cancelled_sync_tool_still_closes_its_span(self) -> None:
        tracer, sink = build()

        @tool(never_empty=True)
        def t() -> str:
            raise KeyboardInterrupt

        def go() -> None:
            with tracer.run():
                t()

        with pytest.raises(KeyboardInterrupt):
            go()

        assert len(sink.traces[0].spans) == 1

    async def test_an_ambiguous_empty_closes_its_span_before_raising(self) -> None:
        from toolproof import AmbiguousEmptyError

        tracer, sink = build()

        @tool()
        async def t() -> list[int]:
            return []

        async def go() -> None:
            async with tracer.run():
                await t()

        with pytest.raises(AmbiguousEmptyError):
            await go()

        assert len(sink.traces[0].spans) == 1

    def test_a_sync_tool_overrunning_its_timeout_is_traced(self) -> None:
        import time as _time

        tracer, sink = build()

        @tool(never_empty=True, timeout_s=0.001)
        def slow() -> str:
            _time.sleep(0.05)
            return "late"

        with tracer.run():
            slow()

        assert sink.traces[0].spans[0].attributes["tool.error_kind"] == "timeout"

    async def test_a_retrying_tool_records_every_attempt_in_order(self) -> None:
        tracer, sink = build()
        attempts = 0

        @tool(never_empty=True, retries=1, retry_backoff_s=0.0)
        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("down")
            return "recovered"

        async with tracer.run():
            await flaky()

        statuses = [s.attributes["tool.status"] for s in sink.traces[0].spans]
        assert statuses == ["error", "ok"]


class TestUnsampledAndDetached:
    async def test_span_helpers_are_callable_on_an_unsampled_run(self) -> None:
        """Adapters call these unconditionally; they must accept everything."""
        tracer, sink = build(sample_rate=0.0)
        async with tracer.run():
            with tracer.span("llm", name="c") as span:
                span.set("k", "v")
                span.update({"a": "b"})
                span.record_usage(model="m", input_tokens=1, output_tokens=1)
                span.record_tool_result(Ok(value=1))
        assert sink.traces == []

    async def test_a_span_outside_any_run_records_nothing_and_does_not_raise(self) -> None:
        tracer, sink = build()
        with tracer.span("custom", name="orphan") as span:
            span.set("k", "v")
        assert sink.traces == []

    async def test_current_run_is_none_outside_a_run(self) -> None:
        tracer, _ = build()
        assert tracer.current_run is None

    async def test_current_run_exposes_the_active_trace_id(self) -> None:
        tracer, _ = build()
        async with tracer.run() as run:
            current = tracer.current_run
            assert current is not None
            assert current.trace_id == run.trace_id

    async def test_set_tag_reaches_the_trace(self) -> None:
        tracer, sink = build()
        async with tracer.run() as run:
            run.set_tag("lang", "hinglish")
        assert sink.traces[0].tags["lang"] == "hinglish"

    async def test_cached_input_tokens_are_summed(self) -> None:
        tracer, sink = build()
        async with tracer.run():
            with tracer.span("llm", name="c") as span:
                span.record_usage(
                    model="m", input_tokens=10, output_tokens=2, cached_input_tokens=5
                )
        assert sink.traces[0].usage.cached_input_tokens == 5

    async def test_span_update_writes_several_attributes(self) -> None:
        tracer, sink = build()
        async with tracer.run():
            with tracer.span("custom", name="c") as span:
                span.update({"a": "1", "b": "2"})
        attrs = sink.traces[0].spans[0].attributes
        assert attrs["a"] == "1"
        assert attrs["b"] == "2"

    def test_a_tracer_can_run_synchronously(self) -> None:
        """The doc shows `async with`, but a sync agent must be traceable."""
        tracer, sink = build()
        with tracer.run(case_id="c1") as run:
            with tracer.span("node", name="n"):
                pass
            run.set_output(answer="sync")
        assert sink.traces[0].final_output.answer == "sync"


class TestArgumentBinding:
    async def test_positional_arguments_are_named_in_the_trace(self) -> None:
        tracer, sink = build()

        @tool(never_empty=True)
        async def t(city: str, radius: int) -> int:
            return 1

        async with tracer.run():
            await t("Pune", 10)

        args = json.loads(str(sink.traces[0].spans[0].attributes["tool.args"]))
        assert args == {"city": "Pune", "radius": 10}

    async def test_defaults_are_recorded_so_a_replay_is_reproducible(self) -> None:
        tracer, sink = build()

        @tool(never_empty=True)
        async def t(city: str, radius: int = 25) -> int:
            return 1

        async with tracer.run():
            await t(city="Pune")

        args = json.loads(str(sink.traces[0].spans[0].attributes["tool.args"]))
        assert args["radius"] == 25

    def test_unbindable_arguments_fall_back_to_positional_names(self) -> None:
        from toolproof.core.tracing import bind_arguments

        def f(a: int) -> int:
            return a

        # Deliberately wrong arity: the helper must not be what breaks the call.
        assert bind_arguments(f, (1, 2, 3), {"z": 4}) == {
            "arg0": 1,
            "arg1": 2,
            "arg2": 3,
            "z": 4,
        }
