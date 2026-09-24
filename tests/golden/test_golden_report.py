"""Golden report: fixed traces in, byte-identical JSON out.

Acceptance row: "golden report JSON byte-identical from fixed traces".

The point is not that the bytes match today. It is that any change to the report
shape, the metric computation, the collapse rules, the statistics or the key
ordering shows up as a diff in a committed file that a reviewer has to approve.
A report format that drifts silently makes every committed report incomparable
with the one before it.

Determinism comes from the runner's own parameters (``now``, ``report_id``,
``seed``), not from monkeypatching, so a user can regenerate this file with the
public API exactly as CI does.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from toolproof import Case, Dataset, Runner, Tracer, scorers
from toolproof.tracer.sinks import MemorySink

GOLDEN = Path(__file__).parent / "report.golden.json"

FIXED_NOW = "2026-09-23T10:00:00.000Z"
FIXED_REPORT_ID = "00000000-0000-4000-8000-000000000000"
FIXED_SEED = 20260921

CASES: list[dict[str, object]] = [
    # A mix chosen so every reported block is exercised: passing and failing
    # routes, an acceptable-route case, a forbidden tool, a fact list and a
    # case whose route expectation is absent so a scorer reports not-applicable.
    {
        "id": "g-0001",
        "route": "job_search",
        "predicted": "job_search",
        "answer": "I found 3 backend roles in Pune.",
    },
    {
        "id": "g-0002",
        "route": "job_search",
        "predicted": "clarify",
        "answer": "Could you tell me which city?",
    },
    {
        "id": "g-0003",
        "route": "followup",
        "acceptable": ["email_draft"],
        "predicted": "email_draft",
        "answer": "I drafted a follow-up note.",
    },
    {
        "id": "g-0004",
        "route": "email_draft",
        "predicted": "email_draft",
        "answer": "Here is a draft to the recruiter.",
        "forbidden_tools": ["send_gmail"],
    },
    {
        "id": "g-0005",
        "route": "resume_query",
        "predicted": "resume_query",
        "answer": "Your resume mentions Kubernetes and Python.",
        "facts": [
            {"id": "f1", "statement": "Kubernetes", "match": "contains"},
            {"id": "f2", "statement": "Terraform", "match": "contains"},
        ],
    },
    {
        "id": "g-0006",
        "route": "blog_search",
        "predicted": "blog_search",
        "answer": "Found two posts on system design.",
    },
    {
        "id": "g-0007",
        "route": "general",
        "predicted": "general",
        "answer": "Here is how to prepare.",
    },
    {
        "id": "g-0008",
        "route": "clarify",
        "predicted": "clarify",
        "answer": "What would you like help with?",
    },
    {
        "id": "g-0009",
        "route": "job_search",
        "predicted": "job_search",
        "answer": "Three roles match.",
    },
    {
        "id": "g-0010",
        "route": "job_search",
        "predicted": "general",
        "answer": "Here is some general advice.",
    },
    {
        "id": "g-0011",
        "route": "followup",
        "predicted": "followup",
        "answer": "Sent a follow-up reminder.",
    },
    {
        "id": "g-0012",
        "route": "followup",
        "predicted": "followup",
        "answer": "Following up now.",
    },
]


def dataset_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in CASES:
        expect: dict[str, object] = {"route": {"label": spec["route"]}}
        if "acceptable" in spec:
            expect["route"] = {"label": spec["route"], "acceptable": spec["acceptable"]}
        if "forbidden_tools" in spec:
            expect["forbidden_tools"] = spec["forbidden_tools"]
        if "facts" in spec:
            expect["facts"] = spec["facts"]
        rows.append(
            {
                "schema_version": 1,
                "id": spec["id"],
                "suite": "golden.suite",
                "split": "dev",
                "input": {"messages": [{"role": "user", "content": str(spec["id"])}]},
                "expect": expect,
            }
        )
    return rows


def build_dataset(tmp_path: Path) -> Dataset:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in dataset_rows()) + "\n",
        encoding="utf-8",
    )
    return Dataset.load(path)


def golden_target() -> Callable[[Case, Tracer], Awaitable[None]]:
    """A fixed target: no model, no clock, no randomness.

    Each case's predicted route and answer come from the table above, so the
    report depends only on code under test.
    """
    answers = {str(spec["id"]): spec for spec in CASES}

    async def target(case: Case, tracer: Tracer) -> None:
        spec = answers[case.id]
        run = tracer.current_run
        assert run is not None
        run.set_output(answer=str(spec["answer"]), route=str(spec["predicted"]))

    return target


def generate(tmp_path: Path) -> str:
    """Produce the report JSON, with every volatile field pinned."""
    dataset = build_dataset(tmp_path)
    runner = Runner(
        target=golden_target(),
        scorers=[
            scorers.route(),
            scorers.forbidden_tools(),
            scorers.facts(),
            scorers.arguments(),
        ],
        tracer=Tracer(sink=MemorySink()),
        seed=FIXED_SEED,
        concurrency=4,
        now=FIXED_NOW,
        report_id=FIXED_REPORT_ID,
        suite_version=1,
        config_hash="0" * 64,
        env_overrides={
            "toolproof_version": "0.0.1-golden",
            "python_version": "3.12.0",
            "target_git_sha": "abc1234",
        },
    )
    import asyncio

    report = asyncio.run(runner.run(dataset))
    # Latency is wall-clock and cannot be pinned by a seed, so it is zeroed
    # here rather than excluded: the golden then still asserts that the block
    # exists with the right shape and concurrency.
    pinned = report.model_copy(
        update={
            "latency": report.latency.model_copy(
                update={
                    "samples_ms": [0] * len(report.latency.samples_ms),
                    "p50_ms": 0,
                    "p95_ms": 0,
                }
            ),
            "outcomes": [
                outcome.model_copy(update={"trace_id": None, "duration_ms": 0})
                for outcome in report.outcomes
            ],
        }
    )
    return pinned.to_json()


class TestGoldenReport:
    def test_the_report_matches_the_committed_golden_file(self, tmp_path: Path) -> None:
        """Byte-for-byte. Regenerate with UPDATE_GOLDEN=1 and review the diff:
        a change here is a change to the published report contract."""
        produced = generate(tmp_path)

        if os.environ.get("UPDATE_GOLDEN") == "1":  # pragma: no cover - maintenance path
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN.write_text(produced, encoding="utf-8", newline="\n")
            pytest.skip("golden file regenerated")

        assert GOLDEN.is_file(), (
            f"{GOLDEN} is missing. Regenerate it with UPDATE_GOLDEN=1 and commit it."
        )
        assert produced == GOLDEN.read_text(encoding="utf-8")

    def test_two_runs_of_the_same_input_are_identical(self, tmp_path: Path) -> None:
        """The property the golden file rests on. If this fails, the golden test
        is flaky rather than the report being wrong."""
        assert generate(tmp_path) == generate(tmp_path)

    def test_the_golden_file_is_valid_json_with_a_trailing_newline(self) -> None:
        text = GOLDEN.read_text(encoding="utf-8")
        assert text.endswith("\n")
        json.loads(text)

    def test_the_golden_report_loads_back_into_the_model(self) -> None:
        """A committed report has to survive a round trip, or an older report
        stops being readable by a later version."""
        from toolproof import Report

        report = Report.model_validate_json(GOLDEN.read_text(encoding="utf-8"))
        assert report.suite == "golden.suite"
        assert report.complete is True

    def test_the_golden_report_contains_every_documented_top_level_key(self) -> None:
        payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
        expected = {
            "schema_version",
            "report_id",
            "suite",
            "suite_version",
            "split",
            "created_at",
            "complete",
            "status",
            "env",
            "config_hash",
            "counts",
            "metrics",
            "outcomes",
            "confusion",
            "judge",
            "costs",
            "latency",
        }
        assert expected <= set(payload)

    def test_traces_are_not_embedded_in_the_golden_report(self) -> None:
        assert "traces" not in json.loads(GOLDEN.read_text(encoding="utf-8"))


class TestGoldenContent:
    """What the golden file asserts about the numbers, in readable form.

    These duplicate the byte comparison deliberately: when the golden diff is
    large, these say which behaviour actually changed.
    """

    def test_route_accuracy_is_the_expected_fraction(self) -> None:
        payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
        route = next(m for m in payload["metrics"] if m["name"] == "route")
        # 9 of 12 predicted routes equal their label. g-0003 is an acceptable
        # route (strict fail), g-0002 and g-0010 are genuine misroutes.
        assert route["n"] == 12
        assert route["value"] == pytest.approx(9 / 12)
        assert route["method"] == "wilson"

    def test_an_unmeasured_metric_carries_no_value(self) -> None:
        """``arguments`` was configured but no case declared tool calls, so it
        must appear with applicable=0 and no number."""
        payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
        arguments = next(m for m in payload["metrics"] if m["name"] == "arguments")
        assert arguments["applicable"] == 0
        assert arguments["value"] is None

    def test_fact_recall_uses_a_bootstrap_interval(self) -> None:
        payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
        facts = next(m for m in payload["metrics"] if m["name"] == "facts")
        assert facts["method"] == "bootstrap"
        assert facts["value"] == pytest.approx(0.5)

    def test_the_confusion_matrix_has_the_expected_misroutes(self) -> None:
        payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
        confusion = payload["confusion"]
        assert confusion["job_search"]["job_search"] == 2
        assert confusion["job_search"]["clarify"] == 1
        assert confusion["job_search"]["general"] == 1

    def test_the_cost_is_unknown_rather_than_zero(self) -> None:
        """The default pricing table is empty, so every trace has an unknown
        cost. A total of 0.0 here would be the library's own headline bug."""
        payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
        assert payload["costs"]["total_usd"] is None
        assert payload["costs"]["unknown_count"] == 12

    def test_the_run_is_complete_with_nothing_unscored(self) -> None:
        payload = json.loads(GOLDEN.read_text(encoding="utf-8"))
        assert payload["complete"] is True
        assert payload["status"] == "ok"
        assert payload["counts"]["unscored"] == 0


class TestGoldenRendering:
    def test_the_rendered_markdown_matches_its_golden_file(self) -> None:
        """The renderer has its own golden, because a README table is generated
        from it and a silent formatting change would rewrite published text."""
        from toolproof import Report
        from toolproof.report.render import render_markdown

        golden_md = GOLDEN.parent / "report.golden.md"
        report = Report.model_validate_json(GOLDEN.read_text(encoding="utf-8"))
        produced = render_markdown(report)

        if os.environ.get("UPDATE_GOLDEN") == "1":  # pragma: no cover
            golden_md.write_text(produced, encoding="utf-8", newline="\n")
            pytest.skip("golden markdown regenerated")

        assert golden_md.is_file()
        assert produced == golden_md.read_text(encoding="utf-8")
