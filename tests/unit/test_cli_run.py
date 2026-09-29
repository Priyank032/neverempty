"""``toolproof run``: execute the suites a config declares.

The primary entry point in the doc's CLI table, and what `evals.yml` invokes. It
is deliberately thin — the Runner already takes every field the config carries —
so these tests are about the wiring: does the config reach the runner unchanged,
are the named scorers the ones constructed, and does a preflight refusal surface
as a non-zero exit rather than a traceback.
"""

from __future__ import annotations

import json
from pathlib import Path

from toolproof.cli import main

CONFIG = """
[project]
name = "fixture"

[target]
entrypoint = "{entrypoint}"
git_sha_from = "none"

[[suite]]
name = "demo.routing"
path = "{path}"
split = "dev"
suite_version = 1
scorers = {scorers}

[run]
repeats = {repeats}
concurrency = 2
seed = 20260929
mode = "live"
"""


def write_cases(path: Path, count: int = 3) -> None:
    rows = [
        {
            "schema_version": 1,
            "id": f"demo-{i:04d}",
            "suite": "demo.routing",
            "split": "dev",
            "input": {"messages": [{"role": "user", "content": "jobs in Pune"}]},
            "expect": {"route": {"label": "job_search"}},
        }
        for i in range(count)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def write_config(
    tmp_path: Path,
    *,
    entrypoint: str = "tests.unit.test_cli_run:good_target",
    scorers: str = '["route"]',
    repeats: int = 1,
    cases: int = 3,
) -> Path:
    cases_path = tmp_path / "demo.jsonl"
    write_cases(cases_path, cases)
    config = tmp_path / "toolproof.toml"
    config.write_text(
        CONFIG.format(
            entrypoint=entrypoint,
            path=cases_path.as_posix(),
            scorers=scorers,
            repeats=repeats,
        ),
        encoding="utf-8",
    )
    return config


# ---- targets the tests point the config at ---------------------------------


async def good_target(case: object, tracer: object) -> None:
    tracer.current_run.set_output(answer="found jobs", route="job_search")  # type: ignore[attr-defined]


async def wrong_route_target(case: object, tracer: object) -> None:
    tracer.current_run.set_output(answer="drafted", route="email_draft")  # type: ignore[attr-defined]


async def exploding_target(case: object, tracer: object) -> None:
    raise RuntimeError("the agent blew up")


NOT_CALLABLE = "not a function"


class TestItRuns:
    def test_a_suite_runs_and_exits_zero(self, tmp_path: Path) -> None:
        config = write_config(tmp_path)
        assert main(["run", str(config), "--out", str(tmp_path / "r.json")]) == 0

    def test_the_report_is_written(self, tmp_path: Path) -> None:
        config = write_config(tmp_path)
        out = tmp_path / "r.json"
        main(["run", str(config), "--out", str(out)])
        assert out.exists()
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["suite"] == "demo.routing"
        assert report["complete"] is True

    def test_the_configured_repeats_reach_the_runner(self, tmp_path: Path) -> None:
        config = write_config(tmp_path, repeats=3, cases=2)
        out = tmp_path / "r.json"
        main(["run", str(config), "--out", str(out)])
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["counts"]["cases"] == 2
        assert len(report["outcomes"]) == 6

    def test_the_named_scorer_is_the_one_that_runs(self, tmp_path: Path) -> None:
        config = write_config(tmp_path, scorers='["route"]')
        out = tmp_path / "r.json"
        main(["run", str(config), "--out", str(out)])
        report = json.loads(out.read_text(encoding="utf-8"))
        assert [m["name"] for m in report["metrics"]] == ["route"]

    def test_the_config_hash_travels_into_the_report(self, tmp_path: Path) -> None:
        """So a replay and a live run of the same config compare cleanly."""
        config = write_config(tmp_path)
        out = tmp_path / "r.json"
        main(["run", str(config), "--out", str(out)])
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["config_hash"]

    def test_a_failing_agent_still_produces_a_report(self, tmp_path: Path) -> None:
        """A broken target is a measurement, not a crash."""
        config = write_config(tmp_path, entrypoint="tests.unit.test_cli_run:exploding_target")
        out = tmp_path / "r.json"
        main(["run", str(config), "--out", str(out)])
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["counts"]["unscored"] == 3
        assert report["complete"] is False


class TestExitCodes:
    def test_a_missing_config_exits_non_zero(self, tmp_path: Path) -> None:
        assert main(["run", str(tmp_path / "absent.toml")]) != 0

    def test_an_unresolvable_entrypoint_exits_non_zero(self, tmp_path: Path) -> None:
        config = write_config(tmp_path, entrypoint="tests.unit.test_cli_run:absent")
        assert main(["run", str(config)]) != 0

    def test_a_non_callable_entrypoint_exits_non_zero(self, tmp_path: Path) -> None:
        config = write_config(tmp_path, entrypoint="tests.unit.test_cli_run:NOT_CALLABLE")
        assert main(["run", str(config)]) != 0

    def test_an_entrypoint_without_a_colon_exits_non_zero(self, tmp_path: Path) -> None:
        config = write_config(tmp_path, entrypoint="tests.unit.test_cli_run")
        assert main(["run", str(config)]) != 0

    def test_a_bad_dataset_exits_non_zero(self, tmp_path: Path) -> None:
        config = write_config(tmp_path)
        (tmp_path / "demo.jsonl").write_text("not json\n", encoding="utf-8")
        assert main(["run", str(config)]) != 0

    def test_an_incomplete_run_exits_non_zero(self, tmp_path: Path) -> None:
        """An incomplete run is a failure of the run. The CLI says so with its
        exit code rather than writing a report and returning success."""
        config = write_config(tmp_path, entrypoint="tests.unit.test_cli_run:exploding_target")
        assert main(["run", str(config), "--out", str(tmp_path / "r.json")]) != 0


class TestOverrides:
    def test_mode_can_be_overridden_from_the_command_line(self, tmp_path: Path) -> None:
        """So one config serves a live run and a replay in CI."""
        config = write_config(tmp_path)
        out = tmp_path / "r.json"
        assert main(["run", str(config), "--mode", "live", "--out", str(out)]) == 0

    def test_a_single_suite_can_be_selected(self, tmp_path: Path) -> None:
        config = write_config(tmp_path)
        out = tmp_path / "r.json"
        assert main(["run", str(config), "--suite", "demo.routing", "--out", str(out)]) == 0

    def test_selecting_an_unknown_suite_exits_non_zero(self, tmp_path: Path) -> None:
        config = write_config(tmp_path)
        assert main(["run", str(config), "--suite", "nope"]) != 0


class TestOutputPaths:
    def test_without_out_the_report_goes_beside_the_config(self, tmp_path: Path) -> None:
        config = write_config(tmp_path)
        main(["run", str(config)])
        written = list((tmp_path / "reports").glob("*.json"))
        assert len(written) == 1

    def test_multiple_suites_do_not_overwrite_each_other(self, tmp_path: Path) -> None:
        cases_a = tmp_path / "a.jsonl"
        write_cases(cases_a)
        cases_b = tmp_path / "b.jsonl"
        rows = [
            {
                "schema_version": 1,
                "id": "demo-b-0001",
                "suite": "demo.failure",
                "split": "dev",
                "input": {"messages": [{"role": "user", "content": "q"}]},
                "expect": {"route": {"label": "job_search"}},
            }
        ]
        cases_b.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

        config = tmp_path / "toolproof.toml"
        config.write_text(
            f"""
[project]
name = "fixture"

[target]
entrypoint = "tests.unit.test_cli_run:good_target"
git_sha_from = "none"

[[suite]]
name = "demo.routing"
path = "{cases_a.as_posix()}"
split = "dev"
suite_version = 1
scorers = ["route"]

[[suite]]
name = "demo.failure"
path = "{cases_b.as_posix()}"
split = "dev"
suite_version = 1
scorers = ["route"]

[run]
seed = 1
""",
            encoding="utf-8",
        )
        assert main(["run", str(config)]) == 0
        written = sorted(p.name for p in (tmp_path / "reports").glob("*.json"))
        assert len(written) == 2, written
