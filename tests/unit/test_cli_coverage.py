"""``toolproof coverage``: the labelling backlog, and a non-zero exit.

An unlabelled suite must fail CI rather than running and publishing a rate over
an empty denominator. That is this library's own headline rule applied to its own
dataset, so it is enforced by an exit code and not by a warning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolproof.cli import main

CONFIG = """
[project]
name = "fixture"

[target]
entrypoint = "toolproof.evals.nextrole:adapter"

[[suite]]
name = "nextrole.routing"
path = "{path}"
split = "test"
suite_version = 1
scorers = ["route"]

[run]
seed = 1
"""


def write_cases(path: Path, counts: dict[str, int]) -> None:
    lines = []
    for branch, count in counts.items():
        for index in range(count):
            lines.append(
                json.dumps(
                    {
                        "schema_version": 1,
                        "id": f"nr-{branch}-{index:04d}",
                        "suite": "nextrole.routing",
                        "split": "test",
                        "input": {"messages": [{"role": "user", "content": "q"}]},
                        "expect": {"route": {"label": branch}},
                        "provenance": {"method": "human", "labeller": "priyank"},
                    }
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def config(tmp_path: Path) -> Path:
    cases = tmp_path / "routing.jsonl"
    write_cases(cases, {})
    config = tmp_path / "toolproof.toml"
    config.write_text(CONFIG.format(path=cases.as_posix()), encoding="utf-8")
    return config


class TestExitCodes:
    def test_an_empty_suite_exits_non_zero(self, config: Path) -> None:
        assert main(["coverage", str(config)]) != 0

    def test_allow_incomplete_reports_without_failing(self, config: Path) -> None:
        assert main(["coverage", str(config), "--allow-incomplete"]) == 0

    def test_a_fully_labelled_suite_exits_zero(self, tmp_path: Path) -> None:
        from toolproof.evals.nextrole import BRANCHES
        from toolproof.evals.suites import MIN_PER_BRANCH

        cases = tmp_path / "routing.jsonl"
        write_cases(cases, dict.fromkeys(BRANCHES, MIN_PER_BRANCH))
        config = tmp_path / "toolproof.toml"
        config.write_text(CONFIG.format(path=cases.as_posix()), encoding="utf-8")
        assert main(["coverage", str(config)]) == 0

    def test_a_missing_config_exits_non_zero(self, tmp_path: Path) -> None:
        assert main(["coverage", str(tmp_path / "absent.toml")]) != 0


class TestOutput:
    def test_names_every_branch_including_the_empty_ones(
        self, config: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The labelling backlog is the useful output, so a branch with no cases
        appears as a zero rather than being omitted."""
        main(["coverage", str(config), "--allow-incomplete"])
        out = capsys.readouterr().out
        for branch in ("job_search", "career_planning", "skill_gap"):
            assert branch in out

    def test_says_how_many_cases_are_still_needed(
        self, config: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["coverage", str(config), "--allow-incomplete"])
        out = capsys.readouterr().out
        assert "330" in out
        assert "30" in out

    def test_a_partially_labelled_suite_shows_the_shortfall(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cases = tmp_path / "routing.jsonl"
        write_cases(cases, {"job_search": 30, "general": 4})
        config = tmp_path / "toolproof.toml"
        config.write_text(CONFIG.format(path=cases.as_posix()), encoding="utf-8")
        main(["coverage", str(config), "--allow-incomplete"])
        out = capsys.readouterr().out
        assert "| job_search | 30 |" in out
        assert "| general | 4 |" in out

    def test_markdown_format_is_a_table(
        self, config: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["coverage", str(config), "--format", "markdown", "--allow-incomplete"])
        out = capsys.readouterr().out
        assert "| branch | cases | short by |" in out
