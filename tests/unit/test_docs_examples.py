"""Every dataset example in the docs must load against the real models.

The docs previously showed a ``to_model()`` return type that was wrong and no
``expect.tool_calls`` schema at all, so the first dataset a reader wrote could
not load. Documentation that cannot be executed drifts silently, which is the
same class of failure this library exists to catch, so the examples are
checked here rather than trusted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, get_args

import pytest

from neverempty import Dataset

DOCS = Path(__file__).resolve().parents[2] / "docs"
LABELS = DOCS / "writing-labels.md"

# Every ``expect`` block in the reference, keyed by the heading above it.
EXPECT_EXAMPLES: dict[str, dict[str, Any]] = {
    "route": {"route": {"label": "job_search", "acceptable": ["general"]}},
    "tool_calls.first": {"tool_calls": {"mode": "first", "calls": [{"tool": "db_lookup"}]}},
    "tool_calls.set": {
        "tool_calls": {"mode": "set", "calls": [{"tool": "search_jobs"}, {"tool": "rerank"}]}
    },
    "tool_calls.sequence": {
        "tool_calls": {"mode": "sequence", "calls": [{"tool": "hard_filter"}, {"tool": "rerank"}]}
    },
    "args.exact": {
        "tool_calls": {
            "mode": "first",
            "calls": [
                {"tool": "search_jobs", "args": {"city": {"match": "exact", "value": "Pune"}}}
            ],
        }
    },
    "args.every_match_mode": {
        "tool_calls": {
            "mode": "first",
            "calls": [
                {
                    "tool": "t",
                    "args": {
                        "a": {"match": "exact", "value": "x"},
                        "b": {"match": "normalized", "value": "Pune "},
                        "c": {"match": "set", "value": ["x", "y"]},
                        "d": {"match": "numeric", "value": 10, "tol": 0.5},
                        "e": {"match": "regex", "value": "^P.*e$"},
                        "f": {"match": "present"},
                        "g": {"match": "date", "value": "2026-10-01"},
                    },
                }
            ],
        }
    },
    "facts": {
        "facts": [
            {"id": "f1", "statement": "10 lakh", "match": "contains"},
            {"id": "f2", "statement": "^Rs ?[0-9]+$", "match": "regex"},
            {
                "id": "f3",
                "statement": "the scheme covers farmers",
                "match": "judge",
                "evidence_key": "scheme_text",
            },
        ]
    },
    "items": {
        "items": [
            {
                "item_id": "PMKSY",
                "rule_result": None,
                "rule_trace": [
                    {"criterion": "land_holding", "result": None, "missing_field": "land_area"}
                ],
            }
        ]
    },
}


def _case(expect: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": "doc-0001",
        "suite": "demo.routing",
        "split": "dev",
        "input": {"messages": [{"role": "user", "content": "jobs in Pune"}]},
        "expect": expect,
    }


class TestTheExpectReferenceLoads:
    @pytest.mark.parametrize("name", sorted(EXPECT_EXAMPLES))
    def test_each_documented_expect_block_loads(self, name: str, tmp_path: Path) -> None:
        path = tmp_path / "doc.jsonl"
        path.write_text(json.dumps(_case(EXPECT_EXAMPLES[name])) + "\n", encoding="utf-8")
        assert len(Dataset.load(path, split="dev")) == 1

    def test_tol_outside_numeric_is_still_refused(self) -> None:
        """The reference says so, so it has to stay true."""
        from neverempty.dataset.case import ArgExpectation

        with pytest.raises(ValueError, match="numeric"):
            ArgExpectation(match="exact", value="x", tol=0.5)

    def test_acceptable_containing_the_label_is_still_refused(self) -> None:
        from neverempty.dataset.case import RouteExpectation

        with pytest.raises(ValueError, match="acceptable"):
            RouteExpectation(label="job_search", acceptable=["job_search"])


class TestTheDocsDescribeTheRealApi:
    def test_every_match_mode_in_the_table_is_a_real_one(self) -> None:
        """A mode documented but not implemented would silently never fire."""
        from neverempty.dataset.case import ArgMatch

        # ``- {"match"}`` drops the table's own header row.
        rows = re.findall(r"^\| `(\w+)` \| ", _match_table(), flags=re.M)
        assert set(rows) - {"match"} == set(get_args(ArgMatch))

    def test_every_tool_call_mode_in_the_table_is_a_real_one(self) -> None:
        from neverempty.dataset.case import ToolCallMode

        text = LABELS.read_text(encoding="utf-8")
        for mode in get_args(ToolCallMode):
            assert f"| `{mode}` |" in text

    def test_to_model_is_documented_as_returning_a_string(self) -> None:
        """It returns JSON text; the README showed it as a dict, and a reader
        who believed that would double-encode it with ``json.dumps``."""
        readme = (DOCS.parent / "README.md").read_text(encoding="utf-8")
        assert "returns the **JSON string**" in readme

    def test_the_readme_does_not_promise_a_working_pip_install(self) -> None:
        """Until it is published, the documented install command must not be
        one that fails."""
        readme = (DOCS.parent / "README.md").read_text(encoding="utf-8")
        assert "Not on PyPI yet" in readme


def _match_table() -> str:
    """The `match` table only, so the tool-call modes do not bleed into it."""
    text = LABELS.read_text(encoding="utf-8")
    start = text.index("| `match` | Compares |")
    return text[start : text.index("\n\n", start)]


class TestTheDocumentedDetectionFloorIsReal:
    """The README now publishes a specific claim about the gate's sensitivity,
    so it is pinned to the implementation rather than left as prose.

    This is the trap most likely to make a user trust a number they should
    not: believing they have a regression gate on 50 cases when the paired
    test alone cannot fire below 5 flips.
    """

    def test_four_clean_regressions_do_not_reach_significance(self) -> None:
        from neverempty.metrics.stats import mcnemar_exact

        assert mcnemar_exact(b=4, c=0).p_value == pytest.approx(0.0625)
        assert mcnemar_exact(b=4, c=0).p_value >= 0.05

    def test_five_clean_regressions_do(self) -> None:
        from neverempty.metrics.stats import mcnemar_exact

        assert mcnemar_exact(b=5, c=0).p_value == pytest.approx(0.03125)
        assert mcnemar_exact(b=5, c=0).p_value < 0.05

    def test_the_readme_states_the_five_flip_floor(self) -> None:
        readme = (DOCS.parent / "README.md").read_text(encoding="utf-8")
        assert "5 clean pass-to-fail flips" in readme
        assert "floors" in readme

    def test_a_floor_catches_what_the_paired_test_cannot(self) -> None:
        """The README's remedy has to work, so it is exercised here."""
        from neverempty.report.gate import GateConfig

        config = GateConfig(floors={"route": 0.95})
        assert config.floors == {"route": 0.95}
        # And the default really is empty, which is why the warning is needed.
        assert GateConfig().floors == {}


class TestTheDocumentedSizeCapWorks:
    """There is no default payload cap, which the docs now say outright.

    The doc's eight wrapper rules do not define one and ``truncated`` is
    author-declared, so inventing a default byte threshold would be choosing a
    number the specification does not set. The documented workaround is
    exercised here so the advice cannot rot.
    """

    async def test_a_size_predicate_flags_a_large_result(self) -> None:
        import json as _json

        from neverempty import tool

        cap = 8_000

        @tool(
            never_empty=True,
            truncated_when=lambda rows: len(_json.dumps(rows, default=str)) > cap,
        )
        async def big() -> list[dict[str, object]]:
            return [{"i": i, "pad": "x" * 200} for i in range(5000)]

        rendered = _json.loads((await big()).to_model())
        assert rendered["truncated"] is True
        assert "note" in rendered
        assert "incomplete" in rendered["note"]

    async def test_a_small_result_is_not_flagged(self) -> None:
        import json as _json

        from neverempty import tool

        @tool(never_empty=True, truncated_when=lambda rows: len(rows) >= 100)
        async def small() -> list[int]:
            return [1, 2, 3]

        rendered = _json.loads((await small()).to_model())
        assert rendered["truncated"] is False
        assert "note" not in rendered

    def test_the_docs_say_there_is_no_default_cap(self) -> None:
        text = (DOCS / "getting-started.md").read_text(encoding="utf-8")
        assert "no default size cap" in text
