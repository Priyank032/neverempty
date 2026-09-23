"""The committed JSON Schemas are the cross-language contract.

    JSON Schema for Trace v1 is generated from the pydantic models and
    published in the repo at ``schemas/trace.v1.json``. The Node exporter
    validates against it in its own test.

CI regenerates and diffs, so these tests guard the generator itself: that the
committed file is current, and that it says what another language needs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from toolproof.schemas import SCHEMA_FILES, generate_schemas, write_schemas

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "schemas"


@pytest.fixture(scope="module")
def committed_trace_schema() -> dict[str, Any]:
    path = SCHEMA_DIR / "trace.v1.json"
    assert path.is_file(), f"{path} is not committed"
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


class TestCommittedFilesAreCurrent:
    def test_the_committed_schemas_match_the_models(self) -> None:
        """The same check CI runs. Regenerate and commit when a model changes."""
        for name, generated in generate_schemas().items():
            path = SCHEMA_DIR / name
            assert path.is_file(), f"{name} is not committed"
            committed = json.loads(path.read_text(encoding="utf-8"))
            assert committed == generated, (
                f"{name} is stale. Run: python scripts/gen_schemas.py --out schemas"
            )

    def test_every_declared_file_is_generated(self) -> None:
        assert set(generate_schemas()) == set(SCHEMA_FILES)

    def test_writing_is_idempotent(self, tmp_path: Path) -> None:
        write_schemas(tmp_path)
        first = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
        write_schemas(tmp_path)
        second = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
        assert first == second

    def test_the_files_are_written_with_a_trailing_newline(self, tmp_path: Path) -> None:
        """So a diff against the committed file is not newline noise."""
        write_schemas(tmp_path)
        for path in tmp_path.glob("*.json"):
            assert path.read_text(encoding="utf-8").endswith("\n")


class TestTraceSchemaContract:
    def test_it_declares_a_json_schema_dialect(self, committed_trace_schema: Any) -> None:
        assert "$schema" in committed_trace_schema

    def test_it_is_titled_and_versioned(self, committed_trace_schema: Any) -> None:
        assert committed_trace_schema["title"]
        assert "trace" in committed_trace_schema["title"].lower()

    def test_every_required_trace_field_is_marked_required(
        self, committed_trace_schema: Any
    ) -> None:
        required = set(committed_trace_schema["required"])
        for field in (
            "schema_version",
            "trace_id",
            "repeat",
            "started_at",
            "duration_ms",
            "spans",
            "final_output",
            "usage",
            "cost",
            "status",
            "env",
        ):
            assert field in required, field

    def test_optional_trace_fields_are_not_required(self, committed_trace_schema: Any) -> None:
        required = set(committed_trace_schema["required"])
        for field in ("case_id", "suite", "error", "tags"):
            assert field not in required, field

    def test_the_status_enum_is_published(self, committed_trace_schema: Any) -> None:
        text = json.dumps(committed_trace_schema)
        for status in ("ok", "target_error", "timeout", "budget_abort"):
            assert f'"{status}"' in text, status

    def test_the_span_kind_enum_is_published(self, committed_trace_schema: Any) -> None:
        text = json.dumps(committed_trace_schema)
        for kind in ("llm", "tool", "node", "judge", "custom"):
            assert f'"{kind}"' in text, kind

    def test_the_span_definition_is_reachable_for_a_node_validator(
        self, committed_trace_schema: Any
    ) -> None:
        definitions = committed_trace_schema.get("$defs", {})
        assert any("Span" in key for key in definitions), sorted(definitions)

    def test_unknown_span_attributes_are_permitted_by_the_schema(
        self, committed_trace_schema: Any
    ) -> None:
        """Forward compatibility: other languages write keys we do not know."""
        span = next(v for k, v in committed_trace_schema["$defs"].items() if "Span" in k)
        assert span.get("additionalProperties") is not False


class TestCaseSchemaPlaceholder:
    def test_the_case_schema_is_not_yet_generated(self) -> None:
        """Case v1 lands in M4; nothing must be committed for it before then."""
        assert "case.v1.json" not in SCHEMA_FILES
        assert not (SCHEMA_DIR / "case.v1.json").exists()


class TestGeneratedSchemaValidatesRealTraces:
    def test_a_real_trace_validates_against_the_committed_schema(
        self, committed_trace_schema: Any
    ) -> None:
        """Skipped unless jsonschema is installed; CI installs it as a dev dep."""
        jsonschema = pytest.importorskip("jsonschema")

        import asyncio

        from toolproof import MemorySink, Tracer, tool

        @tool(never_empty=True)
        async def search_jobs(city: str) -> list[int]:
            return [1]

        sink = MemorySink()
        tracer = Tracer(sink=sink)

        async def go() -> None:
            async with tracer.run(case_id="c1", suite="nextrole.routing") as run:
                with tracer.span("node", name="classify_intent"):
                    await search_jobs(city="Pune")
                with tracer.span("llm", name="rerank") as span:
                    span.record_usage(
                        model="gpt-4o", resolved_model="gpt-4o-x", input_tokens=8, output_tokens=2
                    )
                run.set_output(answer="a", route="job_search")

        asyncio.run(go())
        payload = json.loads(sink.traces[0].model_dump_json())
        jsonschema.validate(payload, committed_trace_schema)

    def test_a_trace_missing_a_required_field_is_rejected_by_the_schema(
        self, committed_trace_schema: Any
    ) -> None:
        jsonschema = pytest.importorskip("jsonschema")

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"schema_version": 1}, committed_trace_schema)
