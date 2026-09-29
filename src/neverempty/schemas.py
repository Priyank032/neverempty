"""JSON Schema generation for the cross-language contracts.

The schemas are generated from the pydantic models and committed. CI
regenerates them and fails on any diff, so a model change that is not reflected
in the committed schema cannot merge: a Node exporter validating against a
stale contract would pass its own tests and produce traces Python rejects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neverempty.core.trace import SCHEMA_VERSION, Trace
from neverempty.dataset.case import CASE_SCHEMA_VERSION, Case

SCHEMA_FILES: tuple[str, ...] = ("trace.v1.json", "case.v1.json")
"""Filenames this module owns, generated from the models and committed."""


TRACE_REQUIRED: tuple[str, ...] = (
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
)
"""Fields a conforming writer must emit, per the Trace v1 table.

Pydantic omits any field with a default from ``required``, which is right for
a Python constructor and wrong for a cross-language contract: a Node exporter
that leaves out ``status`` or ``spans`` has written an incomplete trace, and
the schema is the only place that can say so.
"""


def _normalise(node: Any) -> Any:
    """Remove formatting that varies between pydantic versions.

    pydantic 2.9 and 2.13 emit the same model differently: an open object gains
    an explicit ``additionalProperties: true``, and a numeric bound is spelled
    ``0`` or ``0.0``. The committed schema is diffed by CI, so a patch-level
    dependency bump must not look like a contract change. Only redundant
    spellings are normalised; nothing that constrains a document is touched.
    """
    if isinstance(node, dict):
        cleaned = {key: _normalise(value) for key, value in node.items()}
        if cleaned.get("additionalProperties") is True:
            del cleaned["additionalProperties"]
        return cleaned
    if isinstance(node, list):
        return [_normalise(item) for item in node]
    if isinstance(node, float) and node.is_integer():
        # pydantic emits a bound as 0 or 0.0 depending on the version. The two
        # are the same constraint, so they must not diff.
        return int(node)
    return node


def _trace_schema() -> dict[str, Any]:
    schema: dict[str, Any] = _normalise(Trace.model_json_schema(mode="serialization"))
    schema["required"] = sorted(set(schema.get("required", [])) | set(TRACE_REQUIRED))
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = f"neverempty Trace v{SCHEMA_VERSION}"
    schema["description"] = (
        "One agent run, for one case and one repeat. Unknown keys are preserved "
        "on read: other languages and later versions write fields this reader "
        "does not know. A schema_version above the reader's is rejected."
    )
    schema["$id"] = "https://github.com/priyank-agrawal/neverempty/blob/main/schemas/trace.v1.json"
    return schema


def _case_schema() -> dict[str, Any]:
    schema: dict[str, Any] = _normalise(Case.model_json_schema(mode="validation"))
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = f"neverempty Case v{CASE_SCHEMA_VERSION}"
    schema["description"] = (
        "One dataset line. Unknown keys are rejected: a typo in eval data is a "
        "silent killer, because a misspelled expectation is an absent one, and an "
        "absent expectation reports not-applicable rather than failing."
    )
    schema["$id"] = "https://github.com/priyank-agrawal/neverempty/blob/main/schemas/case.v1.json"
    return schema


def generate_schemas() -> dict[str, dict[str, Any]]:
    """Every schema this repo publishes, keyed by filename."""
    return {"trace.v1.json": _trace_schema(), "case.v1.json": _case_schema()}


def render(schema: dict[str, Any]) -> str:
    """Serialize one schema the way it is committed.

    Sorted keys and a trailing newline, so a regeneration diff shows real
    changes rather than key-ordering or newline noise.
    """
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_schemas(directory: str | Path) -> list[Path]:
    """Write every schema into ``directory``. Returns the paths written."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, schema in generate_schemas().items():
        path = target / name
        path.write_text(render(schema), encoding="utf-8", newline="\n")
        written.append(path)
    return written


__all__ = ["SCHEMA_FILES", "TRACE_REQUIRED", "generate_schemas", "render", "write_schemas"]
