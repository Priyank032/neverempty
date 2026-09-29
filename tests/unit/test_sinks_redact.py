"""Sink and redactor behaviour not reached through the Tracer."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from neverempty import JsonlSink, MemorySink, MultiSink, NullSink, Trace, redact
from neverempty.core.trace import Cost, Env, FinalOutput, Usage


def a_trace() -> Trace:
    return Trace(
        trace_id=str(uuid.uuid4()),
        started_at="2026-09-23T10:00:00Z",
        duration_ms=1,
        final_output=FinalOutput(),
        usage=Usage(),
        cost=Cost(pricing_version="v1"),
        env=Env(neverempty_version="0.0.1", pricing_version="v1", python_version="3.12.0"),
    )


class TestMemorySink:
    def test_it_collects_traces(self) -> None:
        sink = MemorySink()
        sink.write(a_trace())
        assert len(sink.traces) == 1

    def test_clear_empties_it(self) -> None:
        sink = MemorySink()
        sink.write(a_trace())
        sink.clear()
        assert sink.traces == []

    def test_close_is_a_no_op_and_repeatable(self) -> None:
        sink = MemorySink()
        sink.close()
        sink.close()


class TestNullSink:
    def test_it_discards_without_error(self) -> None:
        sink = NullSink()
        sink.write(a_trace())
        sink.close()


class TestJsonlSinkDirect:
    def test_fsync_mode_still_writes_one_line(self, tmp_path: Path) -> None:
        sink = JsonlSink(tmp_path, fsync=True)
        sink.write(a_trace())
        assert len(sink.path.read_text(encoding="utf-8").strip().splitlines()) == 1

    def test_a_custom_filename_is_honoured(self, tmp_path: Path) -> None:
        sink = JsonlSink(tmp_path, filename="mine.jsonl")
        sink.write(a_trace())
        assert (tmp_path / "mine.jsonl").is_file()

    def test_it_appends_rather_than_truncating(self, tmp_path: Path) -> None:
        JsonlSink(tmp_path).write(a_trace())
        JsonlSink(tmp_path).write(a_trace())
        written = next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8")
        assert len(written.strip().splitlines()) == 2

    def test_it_works_as_a_context_manager(self, tmp_path: Path) -> None:
        with JsonlSink(tmp_path) as sink:
            sink.write(a_trace())
        assert sink.path.is_file()

    def test_close_is_repeatable(self, tmp_path: Path) -> None:
        sink = JsonlSink(tmp_path)
        sink.close()
        sink.close()


class TestMultiSink:
    def test_it_writes_to_every_sink(self) -> None:
        a, b = MemorySink(), MemorySink()
        MultiSink(a, b).write(a_trace())
        assert len(a.traces) == len(b.traces) == 1

    def test_one_failing_sink_does_not_stop_the_others(self) -> None:
        """A broken exporter must not cost you the local copy."""

        class Broken(MemorySink):
            def write(self, trace: Trace) -> None:
                raise OSError("nope")

        healthy = MemorySink()
        with pytest.raises(OSError, match="nope"):
            MultiSink(Broken(), healthy).write(a_trace())
        assert len(healthy.traces) == 1

    def test_close_tolerates_a_failing_sink(self) -> None:
        class Broken(MemorySink):
            def close(self) -> None:
                raise OSError("nope")

        MultiSink(Broken(), MemorySink()).close()

    def test_an_empty_multisink_is_harmless(self) -> None:
        MultiSink().write(a_trace())


class TestRedactors:
    def test_defaults_cover_the_documented_key_list(self) -> None:
        redactor = redact.defaults()
        cleaned = redactor({"email": "a@b.com", "aadhaar": "1234", "city": "Pune"})
        assert cleaned["email"].startswith(redact.PLACEHOLDER_PREFIX)
        assert cleaned["aadhaar"].startswith(redact.PLACEHOLDER_PREFIX)
        assert cleaned["city"] == "Pune"

    def test_nothing_is_an_explicit_no_op(self) -> None:
        assert redact.nothing()({"email": "a@b.com"}) == {"email": "a@b.com"}

    def test_extend_defaults_adds_to_the_builtin_list(self) -> None:
        redactor = redact.keys("persona_ref", extend_defaults=True)
        cleaned = redactor({"persona_ref": "x", "email": "a@b.com", "city": "Pune"})
        assert cleaned["persona_ref"].startswith(redact.PLACEHOLDER_PREFIX)
        assert cleaned["email"].startswith(redact.PLACEHOLDER_PREFIX)
        assert cleaned["city"] == "Pune"

    def test_combine_applies_several_redactors(self) -> None:
        redactor = redact.combine(redact.keys("email"), redact.keys("phone"))
        cleaned = redactor({"email": "a@b.com", "phone": "555", "city": "Pune"})
        assert cleaned["email"].startswith(redact.PLACEHOLDER_PREFIX)
        assert cleaned["phone"].startswith(redact.PLACEHOLDER_PREFIX)
        assert cleaned["city"] == "Pune"

    def test_a_placeholder_is_stable_and_hides_the_value(self) -> None:
        first = redact.placeholder("a@b.com")
        assert first == redact.placeholder("a@b.com")
        assert first != redact.placeholder("c@d.com")
        assert "a@b.com" not in first

    def test_tuples_are_walked_like_lists(self) -> None:
        cleaned = redact.keys("email")({"people": ({"email": "a@b.com"},)})
        assert "a@b.com" not in str(cleaned)

    def test_scalars_pass_through_untouched(self) -> None:
        redactor = redact.keys("email")
        for value in (1, "plain", None, True, 2.5):
            assert redactor(value) == value

    def test_very_deep_nesting_is_capped_rather_than_recursing_forever(self) -> None:
        payload: Any = {"email": "a@b.com"}
        for _ in range(40):
            payload = {"nested": payload}
        cleaned = str(redact.keys("email")(payload))
        assert "depth capped" in cleaned
        assert "a@b.com" not in cleaned

    def test_a_non_string_key_is_left_alone(self) -> None:
        assert redact.keys("email")({1: "value"}) == {1: "value"}


class TestAssertClean:
    def test_it_passes_when_nothing_leaked(self) -> None:
        redact.assert_clean('{"city": "Pune"}', ["a@b.com"])

    def test_it_raises_when_a_secret_is_present(self) -> None:
        with pytest.raises(AssertionError, match="unredacted"):
            redact.assert_clean('{"email": "a@b.com"}', ["a@b.com"])

    def test_an_empty_secret_is_ignored(self) -> None:
        """Otherwise every payload would 'contain' the empty string."""
        redact.assert_clean('{"city": "Pune"}', ["", None])  # type: ignore[list-item]
