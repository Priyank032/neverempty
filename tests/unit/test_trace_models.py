"""Trace v1 and Span: the cross-language contract, field by field.

Strictness rule from the doc: unknown keys in a *case* are a validation error,
while unknown keys in a *trace* are preserved, because other languages and
later versions write them.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import pytest
from pydantic import ValidationError

from neverempty import Span, Trace
from neverempty.core.trace import (
    SCHEMA_VERSION,
    Cost,
    Env,
    FinalOutput,
    TraceError,
    Usage,
)

RFC3339_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


def make_env(**overrides: Any) -> Env:
    base: dict[str, Any] = {
        "neverempty_version": "0.0.1",
        "target_git_sha": "a" * 40,
        "target_dirty": False,
        "prompt_hashes": {"INTENT_PROMPT": "b" * 64},
        "resolved_models": ["gpt-4o-2024-08-06"],
        "pricing_version": "empty-2026-09-23",
        "python_version": "3.12.10",
        "concurrency": 8,
        "seed": 20260921,
        "mode": "live",
        "fault_profile": None,
    }
    base.update(overrides)
    return Env(**base)


def make_trace(**overrides: Any) -> Trace:
    base: dict[str, Any] = {
        "trace_id": str(uuid.uuid4()),
        "repeat": 0,
        "started_at": "2026-09-23T10:00:00Z",
        "duration_ms": 1234,
        "spans": [],
        "final_output": FinalOutput(),
        "usage": Usage(),
        "cost": Cost(pricing_version="empty-2026-09-23"),
        "status": "ok",
        "env": make_env(),
    }
    base.update(overrides)
    return Trace(**base)


class TestSchemaVersion:
    def test_defaults_to_one(self) -> None:
        assert make_trace().schema_version == 1
        assert SCHEMA_VERSION == 1

    def test_version_one_is_accepted(self) -> None:
        assert make_trace(schema_version=1).schema_version == 1

    def test_a_higher_version_is_rejected_with_a_clear_error(self) -> None:
        """A reader must refuse a trace it cannot fully understand."""
        with pytest.raises(ValidationError) as exc:
            make_trace(schema_version=2)
        message = str(exc.value)
        assert "schema_version" in message
        assert "2" in message

    def test_version_zero_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_trace(schema_version=0)


class TestTraceIdentity:
    def test_trace_id_must_be_a_uuid(self) -> None:
        with pytest.raises(ValidationError, match="trace_id"):
            make_trace(trace_id="not-a-uuid")

    def test_a_uuid4_is_accepted(self) -> None:
        trace_id = str(uuid.uuid4())
        assert make_trace(trace_id=trace_id).trace_id == trace_id

    def test_case_id_is_null_for_production_traces(self) -> None:
        assert make_trace().case_id is None

    def test_repeat_is_zero_based_and_non_negative(self) -> None:
        assert make_trace(repeat=0).repeat == 0
        with pytest.raises(ValidationError, match="repeat"):
            make_trace(repeat=-1)

    def test_suite_is_optional(self) -> None:
        assert make_trace().suite is None
        assert make_trace(suite="nextrole.routing").suite == "nextrole.routing"


class TestTimestamps:
    def test_started_at_must_be_rfc3339_utc_with_z(self) -> None:
        assert RFC3339_Z.match(make_trace().started_at)

    @pytest.mark.parametrize(
        "bad",
        [
            "2026-09-23T10:00:00+05:30",  # not UTC
            "2026-09-23 10:00:00Z",  # no T
            "2026-09-23T10:00:00",  # no Z
            "23-09-2026T10:00:00Z",  # wrong order
            "not-a-date",
        ],
    )
    def test_non_conforming_timestamps_are_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="started_at"):
            make_trace(started_at=bad)

    def test_duration_ms_is_non_negative(self) -> None:
        assert make_trace(duration_ms=0).duration_ms == 0
        with pytest.raises(ValidationError, match="duration_ms"):
            make_trace(duration_ms=-1)


class TestUsageAndCost:
    def test_usage_fields_default_to_null_never_zero(self) -> None:
        """Missing must never look like zero: 0 tokens is a measurement."""
        usage = Usage()
        assert usage.input_tokens is None
        assert usage.output_tokens is None
        assert usage.cached_input_tokens is None

    def test_usage_missing_flag_is_set_when_no_counts_are_present(self) -> None:
        assert Usage().usage_missing is True

    def test_usage_missing_is_false_once_counts_arrive(self) -> None:
        assert Usage(input_tokens=10, output_tokens=2).usage_missing is False

    def test_a_genuine_zero_token_count_is_not_missing(self) -> None:
        usage = Usage(input_tokens=0, output_tokens=0)
        assert usage.usage_missing is False
        assert usage.input_tokens == 0

    def test_cost_defaults_to_null_with_a_reason(self) -> None:
        cost = Cost(pricing_version="empty-2026-09-23")
        assert cost.usd is None
        assert cost.unknown_reason is not None

    def test_a_null_cost_must_carry_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="unknown_reason"):
            Cost(pricing_version="v1", usd=None, unknown_reason=None)

    def test_a_known_cost_must_not_carry_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="unknown_reason"):
            Cost(pricing_version="v1", usd=0.01, unknown_reason="model_not_in_pricing_table")

    def test_a_zero_cost_is_allowed_when_genuinely_measured(self) -> None:
        cost = Cost(pricing_version="v1", usd=0.0)
        assert cost.usd == 0.0
        assert cost.unknown_reason is None

    def test_cost_is_never_negative(self) -> None:
        with pytest.raises(ValidationError, match="usd"):
            Cost(pricing_version="v1", usd=-0.01)

    def test_pricing_version_is_required(self) -> None:
        with pytest.raises(ValidationError, match="pricing_version"):
            Cost()  # type: ignore[call-arg]


class TestStatusAndError:
    @pytest.mark.parametrize("status", ["ok", "target_error", "timeout", "budget_abort"])
    def test_every_documented_status_is_accepted(self, status: str) -> None:
        assert make_trace(status=status).status == status

    def test_an_undocumented_status_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="status"):
            make_trace(status="failed")

    def test_error_is_null_on_a_successful_trace(self) -> None:
        assert make_trace().error is None

    def test_error_message_is_truncated_at_2kb(self) -> None:
        error = TraceError(kind="target_error", message="x" * 5000)
        assert len(error.message.encode()) <= 2048

    def test_a_short_error_message_is_untouched(self) -> None:
        assert TraceError(kind="timeout", message="took too long").message == "took too long"

    def test_truncation_is_marked_so_it_is_not_mistaken_for_the_whole_message(self) -> None:
        assert TraceError(kind="target_error", message="x" * 5000).message.endswith("truncated]")


class TestFinalOutput:
    def test_all_three_fields_default_to_null(self) -> None:
        out = FinalOutput()
        assert out.answer is None
        assert out.route is None
        assert out.structured is None

    def test_an_empty_answer_string_is_distinct_from_null(self) -> None:
        """A model that replied with nothing is not a model that never replied."""
        assert FinalOutput(answer="").answer == ""
        assert FinalOutput(answer="").answer is not None


class TestSpan:
    def make_span(self, **overrides: Any) -> Span:
        base: dict[str, Any] = {
            "span_id": "s0001",
            "parent_id": None,
            "kind": "tool",
            "name": "search_jobs",
            "start_ns": 1_000,
            "end_ns": 2_000,
            "status": "ok",
            "attributes": {},
        }
        base.update(overrides)
        return Span(**base)

    @pytest.mark.parametrize("span_id", ["s0001", "s0042", "s9999"])
    def test_span_id_follows_the_documented_shape(self, span_id: str) -> None:
        assert self.make_span(span_id=span_id).span_id == span_id

    @pytest.mark.parametrize("bad", ["1", "span-1", "S0001", "s1", ""])
    def test_a_malformed_span_id_is_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="span_id"):
            self.make_span(span_id=bad)

    def test_parent_id_is_null_for_a_root_span(self) -> None:
        assert self.make_span(parent_id=None).parent_id is None

    @pytest.mark.parametrize("kind", ["llm", "tool", "node", "judge", "custom"])
    def test_every_documented_kind_is_accepted(self, kind: str) -> None:
        assert self.make_span(kind=kind).kind == kind

    def test_an_undocumented_kind_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="kind"):
            self.make_span(kind="retriever")

    @pytest.mark.parametrize("status", ["ok", "empty", "error"])
    def test_tool_span_statuses(self, status: str) -> None:
        assert self.make_span(kind="tool", status=status).status == status

    def test_a_non_tool_span_cannot_be_empty(self) -> None:
        """`empty` is a tool concept; an LLM call has no empty outcome."""
        with pytest.raises(ValidationError, match="empty"):
            self.make_span(kind="llm", status="empty")

    def test_end_must_not_precede_start(self) -> None:
        with pytest.raises(ValidationError, match="end_ns"):
            self.make_span(start_ns=2_000, end_ns=1_000)

    def test_a_zero_duration_span_is_allowed(self) -> None:
        assert self.make_span(start_ns=1_000, end_ns=1_000).duration_ns == 0

    def test_duration_is_derived_from_the_monotonic_bounds(self) -> None:
        assert self.make_span(start_ns=1_000, end_ns=3_500).duration_ns == 2_500

    def test_unknown_attributes_are_preserved_for_forward_compatibility(self) -> None:
        """Another language or a later version may write keys we do not know."""
        span = self.make_span(attributes={"tool.name": "x", "future.key": "keep me"})
        assert span.attributes["future.key"] == "keep me"
        assert "future.key" in span.model_dump_json()

    def test_attributes_are_a_flat_map(self) -> None:
        with pytest.raises(ValidationError, match="attributes"):
            self.make_span(attributes={"nested": {"a": 1}})

    def test_unknown_top_level_span_keys_are_preserved(self) -> None:
        payload: dict[str, Any] = {
            "span_id": "s0001",
            "parent_id": None,
            "kind": "tool",
            "name": "t",
            "start_ns": 1,
            "end_ns": 2,
            "status": "ok",
            "attributes": {},
            "written_by_a_newer_writer": True,
        }
        span = Span.model_validate(payload)
        assert json.loads(span.model_dump_json())["written_by_a_newer_writer"] is True


class TestSpanOrdering:
    def test_spans_are_ordered_by_start_ns_on_write(self) -> None:
        spans = [
            Span(
                span_id=f"s000{i}",
                parent_id=None,
                kind="custom",
                name=f"n{i}",
                start_ns=start,
                end_ns=start + 10,
                status="ok",
                attributes={},
            )
            for i, start in enumerate([300, 100, 200], start=1)
        ]
        trace = make_trace(spans=spans)
        assert [s.start_ns for s in trace.spans] == [100, 200, 300]

    def test_ordering_is_stable_across_reserialisation(self) -> None:
        trace = make_trace(
            spans=[
                Span(
                    span_id="s0002",
                    parent_id=None,
                    kind="custom",
                    name="b",
                    start_ns=200,
                    end_ns=210,
                    status="ok",
                    attributes={},
                ),
                Span(
                    span_id="s0001",
                    parent_id=None,
                    kind="custom",
                    name="a",
                    start_ns=100,
                    end_ns=110,
                    status="ok",
                    attributes={},
                ),
            ]
        )
        once = trace.model_dump_json()
        twice = Trace.model_validate_json(once).model_dump_json()
        assert once == twice


class TestEnv:
    def test_every_documented_reproducibility_field_is_present(self) -> None:
        env = make_env()
        for field in (
            "neverempty_version",
            "target_git_sha",
            "target_dirty",
            "prompt_hashes",
            "resolved_models",
            "pricing_version",
            "python_version",
            "concurrency",
            "seed",
            "mode",
            "fault_profile",
        ):
            assert hasattr(env, field), field

    @pytest.mark.parametrize("mode", ["live", "replay"])
    def test_mode_accepts_both_documented_values(self, mode: str) -> None:
        assert make_env(mode=mode).mode == mode

    def test_an_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="mode"):
            make_env(mode="dry-run")

    def test_prompt_hashes_must_be_sha256_hex(self) -> None:
        with pytest.raises(ValidationError, match="prompt_hashes"):
            make_env(prompt_hashes={"P": "tooshort"})

    def test_a_dirty_target_is_recorded(self) -> None:
        assert make_env(target_dirty=True).target_dirty is True


class TestTraceRoundTrip:
    def test_a_full_trace_round_trips_byte_identically(self) -> None:
        trace = make_trace(
            case_id="nr-route-017",
            suite="nextrole.routing",
            spans=[
                Span(
                    span_id="s0001",
                    parent_id=None,
                    kind="tool",
                    name="search_jobs",
                    start_ns=100,
                    end_ns=500,
                    status="error",
                    attributes={
                        "tool.name": "search_jobs",
                        "tool.status": "error",
                        "tool.error_kind": "timeout",
                        "tool.fault_injected": True,
                    },
                )
            ],
            final_output=FinalOutput(answer="a", route="job_search"),
            usage=Usage(input_tokens=812, output_tokens=240),
            tags={"suite": "nextrole.routing"},
        )
        once = trace.model_dump_json()
        assert Trace.model_validate_json(once).model_dump_json() == once

    def test_tags_are_a_flat_string_map(self) -> None:
        assert make_trace(tags={"a": "b"}).tags == {"a": "b"}
        with pytest.raises(ValidationError, match="tags"):
            make_trace(tags={"a": 1})
