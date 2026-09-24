"""Importing Node-exported traces.

M10's acceptance row: "Node traces validate, cache bypass asserted, LLM error
excluded from consistency". The first two are here; the third is in the
consistency scorer's own tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from toolproof.evals.importer import ImportReport, import_export, load_traces


def trace_row(
    case_id: str = "yk-p07-en",
    *,
    llm_status: str = "ok",
    cache_bypassed: bool = True,
    schema_version: int = 1,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "trace_id": "11111111-1111-4111-8111-111111111111",
        "case_id": case_id,
        "started_at": "2026-09-25T10:00:00.000Z",
        "duration_ms": 12,
        "spans": [
            {
                "span_id": "s0001",
                "kind": "llm",
                "name": "rerank",
                "start_ns": 1000,
                "end_ns": 2000,
                "status": llm_status,
                "attributes": {
                    "gen_ai.response.model": "gpt-4o-mini-2024-07-18",
                    "gen_ai.usage.input_tokens": 812,
                    "gen_ai.usage.output_tokens": 240,
                    "cache.bypassed": cache_bypassed,
                },
            }
        ],
        "final_output": {"structured": {"items": []}},
        "usage": {"input_tokens": 812, "output_tokens": 240},
        "cost": {"pricing_version": "openai-2026-09-01"},
        "status": "target_error" if llm_status == "error" else "ok",
        "env": {
            "toolproof_version": "node-export",
            "pricing_version": "unpriced-node-export",
            "python_version": "n/a (node export)",
            "cache_bypassed": cache_bypassed,
        },
    }


def write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def case_row(case_id: str = "yk-p07-en") -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": case_id,
        "suite": "yojanakhoj.consistency",
        "split": "test",
        "input": {"payload": {"lang": "en"}},
        "expect": {"items": [{"item_id": "pm-kisan", "rule_result": True}]},
        "provenance": {"method": "generated_from_rules", "source_commit": "a" * 40},
    }


class TestNodeTracesValidate:
    def test_a_well_formed_export_loads(self, tmp_path: Path) -> None:
        path = write(tmp_path / "traces.jsonl", [trace_row()])
        traces, problems = load_traces(path)
        assert problems == []
        assert len(traces) == 1
        assert traces[0].case_id == "yk-p07-en"

    def test_usage_and_resolved_model_survive_the_boundary(self, tmp_path: Path) -> None:
        """Rule 3 of the export contract: usage comes from the SDK response."""
        path = write(tmp_path / "traces.jsonl", [trace_row()])
        traces, _ = load_traces(path)
        span = traces[0].spans[0]
        assert span.attributes["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
        assert span.attributes["gen_ai.usage.input_tokens"] == 812

    def test_a_malformed_line_names_its_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.jsonl"
        path.write_text(json.dumps(trace_row()) + "\nnot json\n", encoding="utf-8")
        traces, problems = load_traces(path)
        assert len(traces) == 1
        assert any(":2:" in problem for problem in problems)

    def test_every_bad_line_is_reported_not_just_the_first(self, tmp_path: Path) -> None:
        """A malformed export usually repeats one mistake; one pass should show
        the whole shape of it."""
        path = tmp_path / "traces.jsonl"
        path.write_text("bad\nalso bad\nstill bad\n", encoding="utf-8")
        _, problems = load_traces(path)
        assert len(problems) == 3

    def test_a_trace_level_status_outside_the_enum_is_rejected(self, tmp_path: Path) -> None:
        """Trace v1 allows ok / target_error / timeout / budget_abort. A bare
        'error' is not a member, and an exporter emitting it would have every
        failed trace refused at import."""
        row = trace_row()
        row["status"] = "error"
        path = write(tmp_path / "traces.jsonl", [row])
        traces, problems = load_traces(path)
        assert traces == []
        assert any("status" in problem for problem in problems)

    def test_a_schema_version_from_the_future_is_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path / "traces.jsonl", [trace_row(schema_version=99)])
        traces, problems = load_traces(path)
        assert traces == []
        assert problems

    def test_a_missing_file_is_a_problem_not_a_crash(self, tmp_path: Path) -> None:
        traces, problems = load_traces(tmp_path / "absent.jsonl")
        assert traces == []
        assert "no such file" in problems[0]

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.jsonl"
        path.write_text("\n", encoding="utf-8")
        _, problems = load_traces(path)
        assert any("no traces" in problem for problem in problems)


class TestCacheBypassAsserted:
    def test_an_export_recording_the_bypass_is_accepted(self, tmp_path: Path) -> None:
        path = write(tmp_path / "traces.jsonl", [trace_row(cache_bypassed=True)])
        assert import_export(path).ok

    def test_an_export_without_the_bypass_is_refused(self, tmp_path: Path) -> None:
        """A cached run serves one persona another's explanation within an age
        and income bucket, which contaminates the rate being measured."""
        path = write(tmp_path / "traces.jsonl", [trace_row(cache_bypassed=False)])
        report = import_export(path)
        assert not report.ok
        assert any("cache bypass" in problem for problem in report.problems)

    def test_the_refusal_says_how_to_fix_it(self, tmp_path: Path) -> None:
        path = write(tmp_path / "traces.jsonl", [trace_row(cache_bypassed=False)])
        report = import_export(path)
        assert any("--no-cache" in problem for problem in report.problems)

    def test_the_check_can_be_disabled_explicitly(self, tmp_path: Path) -> None:
        path = write(tmp_path / "traces.jsonl", [trace_row(cache_bypassed=False)])
        assert import_export(path, require_cache_bypass=False).ok


class TestLlmErrorRate:
    def test_a_clean_export_reports_a_zero_rate(self, tmp_path: Path) -> None:
        path = write(tmp_path / "traces.jsonl", [trace_row() for _ in range(10)])
        report = import_export(path)
        assert report.llm_error_rate == 0.0
        assert report.ok

    def test_a_rate_above_the_ceiling_refuses(self, tmp_path: Path) -> None:
        rows = [trace_row(case_id=f"yk-{i}") for i in range(8)]
        rows += [trace_row(case_id=f"yk-e{i}", llm_status="error") for i in range(2)]
        path = write(tmp_path / "traces.jsonl", rows)
        report = import_export(path)
        assert not report.ok
        assert any("error rate" in problem for problem in report.problems)

    def test_the_message_separates_export_failure_from_agent_failure(self, tmp_path: Path) -> None:
        rows = [trace_row(case_id=f"yk-e{i}", llm_status="error") for i in range(5)]
        path = write(tmp_path / "traces.jsonl", rows)
        report = import_export(path)
        assert any("not of" in problem and "agent" in problem for problem in report.problems)

    def test_a_rate_at_the_ceiling_is_allowed(self, tmp_path: Path) -> None:
        rows = [trace_row(case_id=f"yk-{i}") for i in range(9)]
        rows += [trace_row(case_id="yk-e0", llm_status="error")]
        path = write(tmp_path / "traces.jsonl", rows)
        assert import_export(path).ok

    def test_an_empty_import_has_no_rate_rather_than_zero(self) -> None:
        report = ImportReport()
        assert report.llm_error_rate is None


class TestPairing:
    def test_traces_and_cases_pair_on_case_id(self, tmp_path: Path) -> None:
        traces = write(tmp_path / "traces.jsonl", [trace_row()])
        cases = write(tmp_path / "cases.jsonl", [case_row()])
        report = import_export(traces, cases_path=cases)
        assert report.ok
        assert len(report.cases) == 1

    def test_a_trace_with_no_case_is_a_problem(self, tmp_path: Path) -> None:
        traces = write(tmp_path / "traces.jsonl", [trace_row(case_id="yk-orphan")])
        cases = write(tmp_path / "cases.jsonl", [case_row()])
        report = import_export(traces, cases_path=cases)
        assert any("not exported" in problem for problem in report.problems)

    def test_a_case_with_no_trace_is_a_problem(self, tmp_path: Path) -> None:
        traces = write(tmp_path / "traces.jsonl", [trace_row()])
        cases = write(tmp_path / "cases.jsonl", [case_row(), case_row("yk-unrun")])
        report = import_export(traces, cases_path=cases)
        assert any("no trace" in problem for problem in report.problems)


class TestATraceWithoutACaseId:
    """``case_id`` is optional on a Trace, so nothing may assume it is set."""

    def test_it_is_named_by_its_trace_id_in_a_problem(self, tmp_path: Path) -> None:
        row = trace_row(cache_bypassed=False)
        del row["case_id"]
        path = write(tmp_path / "traces.jsonl", [row])
        report = import_export(path)
        assert not report.ok
        assert any("<trace " in problem for problem in report.problems)

    def test_it_does_not_pair_against_any_case(self, tmp_path: Path) -> None:
        row = trace_row()
        del row["case_id"]
        traces = write(tmp_path / "traces.jsonl", [row])
        cases = write(tmp_path / "cases.jsonl", [case_row()])
        report = import_export(traces, cases_path=cases)
        assert any("no trace" in problem for problem in report.problems)


class TestRender:
    def test_reports_counts_and_the_rate(self, tmp_path: Path) -> None:
        path = write(tmp_path / "traces.jsonl", [trace_row()])
        text = import_export(path).render()
        assert "1 trace(s)" in text
        assert "LLM error rate" in text

    def test_an_empty_import_says_not_measured(self) -> None:
        assert "not measured" in ImportReport().render()
