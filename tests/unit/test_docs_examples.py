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


class TestTheCaseFormatRulesAreDocumented:
    """Four loader rules that reject a first dataset. Each error message is
    good; none of them was findable before you hit it."""

    def test_the_documented_minimal_case_loads(self, tmp_path: Path) -> None:
        case = {
            "schema_version": 1,
            "id": "nr-route-0001",
            "suite": "nextrole.routing",
            "split": "dev",
            "input": {"messages": [{"role": "user", "content": "any backend python jobs?"}]},
            "expect": {"route": {"label": "job_search"}},
        }
        path = tmp_path / "doc.jsonl"
        path.write_text(json.dumps(case) + "\n", encoding="utf-8")
        assert len(Dataset.load(path, split="dev")) == 1

    @pytest.mark.parametrize(
        "fragment",
        [
            "^[a-z0-9][a-z0-9._-]{2,63}$",
            "Dotted lowercase",
            "Exactly one of `messages` or `payload`",
            "At least one expectation",
        ],
    )
    def test_each_rule_appears_in_the_docs(self, fragment: str) -> None:
        text = (DOCS / "writing-labels.md").read_text(encoding="utf-8")
        assert fragment in text

    def test_the_documented_id_pattern_is_the_real_one(self) -> None:
        """A pattern that drifts from the loader is worse than none."""
        from pydantic import ValidationError

        from neverempty.dataset.case import Case

        base = {
            "schema_version": 1,
            "suite": "demo.routing",
            "split": "dev",
            "input": {"messages": [{"role": "user", "content": "q"}]},
            "expect": {"route": {"label": "job_search"}},
        }
        Case.model_validate({**base, "id": "abc"})
        with pytest.raises(ValidationError):
            Case.model_validate({**base, "id": "c1"})
        with pytest.raises(ValidationError):
            Case.model_validate({**base, "id": "UPPER-case"})


class TestTheTracingAndPricingDocsRun:
    """The reviewer found cost always null with no pointer to the fix."""

    async def test_the_documented_tracing_example_works(self, tmp_path: Path) -> None:
        from neverempty import Tracer, tool
        from neverempty.tracer.pricing import ModelPrice, Pricing
        from neverempty.tracer.sinks import JsonlSink

        pricing = Pricing(
            version="openai-2026-09-01",
            models={
                "gpt-4o-2024-08-06": ModelPrice(
                    input_usd_per_mtok=2.50,
                    output_usd_per_mtok=10.00,
                    cached_input_usd_per_mtok=1.25,
                    as_of="2026-09-01",
                    source_url="https://openai.com/api/pricing/",
                )
            },
        )

        @tool(empty_when=lambda rows: len(rows) == 0)
        async def search_jobs(city: str) -> list[dict[str, str]]:
            return [{"title": "Backend Engineer"}]

        tracer = Tracer(sink=JsonlSink(tmp_path / "t.jsonl"), pricing=pricing)
        async with tracer.run(case_id="req-42") as run:
            await search_jobs(city="Pune")
            with tracer.span("llm", name="rerank") as span:
                span.record_usage(
                    model="gpt-4o",
                    resolved_model="gpt-4o-2024-08-06",
                    input_tokens=812,
                    output_tokens=240,
                )
            run.set_output(answer="Found 3 jobs.", route="job_search")

        trace = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8").strip())
        assert trace["cost"]["usd"] == pytest.approx(812 * 2.50 / 1e6 + 240 * 10.0 / 1e6)
        assert len(trace["spans"]) == 2

    def test_the_docs_say_cost_is_null_without_a_pricing_table(self) -> None:
        text = (DOCS / "getting-started.md").read_text(encoding="utf-8")
        assert "model_not_in_pricing_table" in text
        assert "record_usage" in text

    def test_the_docs_state_the_cached_token_convention(self) -> None:
        """The convention the doc never fixed, and the cause of a 3.9x
        overcharge until it was settled."""
        text = (DOCS / "getting-started.md").read_text(encoding="utf-8")
        assert "subset" in text


class TestTheJudgeDocsAreAccurate:
    """The reviewer could not implement a JudgeModel: the required output
    format was documented nowhere, and every bare label returned judge_error."""

    async def _label(self, reply: str) -> str:
        from neverempty.judge.judge import Claim, ClaimJudge

        class Scripted:
            async def complete(self, *, system: str, user: str, temperature: float) -> str:
                return reply

        judge = ClaimJudge(model=Scripted(), model_id="claude-sonnet-5", agent_family="openai")
        verdicts = await judge.verify(claims=[Claim(id="c1", text="x")], evidence={"e": "y"})
        return verdicts[0].label

    async def test_a_bare_label_is_a_judge_error(self) -> None:
        """The documented warning, which cost the reviewer the judge entirely."""
        assert await self._label("supported") == "judge_error"

    async def test_the_documented_json_shape_works(self) -> None:
        assert await self._label('{"label": "supported", "rationale": "ok"}') == "supported"

    async def test_a_fenced_block_is_recovered(self) -> None:
        reply = '```json\n{"label": "supported", "rationale": "ok"}\n```'
        assert await self._label(reply) == "supported"

    async def test_a_preamble_is_recovered(self) -> None:
        reply = 'Here is the JSON: {"label": "contradicted", "rationale": "no"}'
        assert await self._label(reply) == "contradicted"

    async def test_an_unknown_label_is_refused(self) -> None:
        """Lenient about the wrapper, strict about the content."""
        assert await self._label('{"label": "maybe", "rationale": "x"}') == "judge_error"

    def test_the_documented_wiring_constructs(self) -> None:
        """``agent_family`` is required; a snippet omitting it fails for a reader."""
        import tempfile

        from neverempty import scorers
        from neverempty.judge.judge import ClaimJudge

        class Scripted:
            async def complete(self, *, system: str, user: str, temperature: float) -> str:
                return '{"label": "supported", "rationale": "ok"}'

        judge = ClaimJudge(
            model=Scripted(),
            model_id="claude-sonnet-5",
            agent_family="openai",
            cache_dir=tempfile.mkdtemp(),
        )
        assert scorers.facts(judge=judge) is not None

    def test_a_judge_sharing_the_agents_family_is_refused(self) -> None:
        from neverempty.judge.judge import ClaimJudge

        class Scripted:
            async def complete(self, *, system: str, user: str, temperature: float) -> str:
                return "{}"

        with pytest.raises(Exception, match="family"):
            ClaimJudge(model=Scripted(), model_id="claude-sonnet-5", agent_family="anthropic")

    def test_the_docs_state_the_required_format(self) -> None:
        text = (DOCS / "writing-labels.md").read_text(encoding="utf-8")
        assert '{"label": "supported", "rationale": "<= 200 characters"}' in text
        assert 'A bare `"supported"` is **not** accepted' in text

    def test_the_docs_separate_fact_from_claim(self) -> None:
        text = (DOCS / "writing-labels.md").read_text(encoding="utf-8")
        assert "`(id, statement, match, evidence_key)`" in text
        assert "`(id, text)`" in text

    def test_the_docs_distinguish_three_labels_from_four_outcomes(self) -> None:
        text = (DOCS / "writing-labels.md").read_text(encoding="utf-8")
        assert "Three labels, four outcomes" in text

    def test_that_distinction_matches_the_types(self) -> None:
        from neverempty.judge.judge import JudgeLabel, VerdictLabel

        assert len(get_args(JudgeLabel)) == 3
        assert len(get_args(VerdictLabel)) == 4
        assert "judge_error" not in get_args(JudgeLabel)
