"""The ``compare``, ``gate`` and ``baseline`` subcommands.

The gate's exit code is the whole contract with CI: a build passes or fails on
this number, so each code has to be reachable from the command line and not only
from the library.

``gate`` deliberately does not use the CLI's own ``EXIT_INVALID``. Its codes come
from the doc's table and mean something specific, so they are passed through
unchanged rather than remapped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neverempty import CaseOutcome, Report, Score
from neverempty.cli import main
from neverempty.core.trace import Env


def write_report(
    path: Path,
    *,
    passes: int = 0,
    fails: int = 0,
    must_pass_fails: int = 0,
    complete: bool = True,
    status: str = "ok",
    suite_version: int = 1,
    flip_first: int = 0,
) -> Path:
    outcomes: list[CaseOutcome] = []
    index = 0

    def add(passed: bool, *, must_pass: bool = False) -> None:
        nonlocal index
        outcomes.append(
            CaseOutcome(
                case_id=f"c-{index:04d}",
                repeat=0,
                scored=True,
                must_pass=must_pass,
                scores={
                    "route": Score(
                        passed=passed,
                        value=1.0 if passed else 0.0,
                        detail={"expected": "job_search", "predicted": "job_search"},
                    )
                },
            )
        )
        index += 1

    for position in range(passes):
        add(position >= flip_first)
    for _ in range(fails):
        add(False)
    for _ in range(must_pass_fails):
        add(False, must_pass=True)

    report = Report.model_validate(
        {
            "suite": "nextrole.routing",
            "suite_version": suite_version,
            "split": "test",
            "created_at": "2026-09-23T10:00:00.000Z",
            "complete": complete,
            "status": status,
            "env": Env(
                neverempty_version="0.0.1",
                pricing_version="v1",
                python_version="3.12.10",
                resolved_models=["gpt-4o-2024-08-06"],
            ),
            "outcomes": outcomes,
            "counts": {"cases": index, "repeats": 1, "scored": index},
        }
    )
    return report.save(path)


class TestGateExitCodes:
    def test_an_identical_pair_exits_zero(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=30)
        assert main(["gate", str(base), str(candidate)]) == 0

    def test_a_significant_regression_exits_one(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=30, flip_first=10)
        assert main(["gate", str(base), str(candidate)]) == 1

    def test_a_must_pass_failure_exits_two(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=29, must_pass_fails=1)
        assert main(["gate", str(base), str(candidate)]) == 2

    def test_an_incomplete_candidate_exits_four(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(
            tmp_path / "cand.json", passes=30, complete=False, status="incomplete"
        )
        assert main(["gate", str(base), str(candidate)]) == 4

    def test_a_missing_baseline_exits_four(self, tmp_path: Path) -> None:
        """The doc lists a missing baseline as invalid input, not as a pass."""
        candidate = write_report(tmp_path / "cand.json", passes=30)
        assert main(["gate", str(tmp_path / "absent.json"), str(candidate)]) == 4

    def test_a_suite_version_mismatch_exits_four(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=30, suite_version=1)
        candidate = write_report(tmp_path / "cand.json", passes=30, suite_version=2)
        assert main(["gate", str(base), str(candidate)]) == 4

    def test_an_unreadable_report_exits_four(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        assert main(["gate", str(base), str(broken)]) == 4


class TestGateOutput:
    def test_the_verdict_is_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=30, flip_first=10)
        main(["gate", str(base), str(candidate)])
        assert "regression" in capsys.readouterr().out.lower()

    def test_json_output_carries_the_exit_code(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=30, flip_first=10)
        main(["gate", str(base), str(candidate), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["exit_code"] == 1
        assert payload["verdict"] == "regression"

    def test_the_summary_can_be_written_to_a_file(self, tmp_path: Path) -> None:
        """CI writes this into the job summary, so it has to land in a file
        rather than only on stdout."""
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=30)
        summary = tmp_path / "summary.md"
        main(["gate", str(base), str(candidate), "--summary", str(summary)])
        assert "Gate passed" in summary.read_text(encoding="utf-8")

    def test_a_floor_can_be_set_from_the_command_line(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=10, fails=20)
        candidate = write_report(tmp_path / "cand.json", passes=10, fails=20)
        assert main(["gate", str(base), str(candidate), "--floor", "route=0.75"]) == 1

    def test_a_malformed_floor_argument_is_a_usage_error(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=30)
        assert main(["gate", str(base), str(candidate), "--floor", "route"]) == 2


class TestCompare:
    def test_compare_prints_a_markdown_diff(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=30, flip_first=5)
        assert main(["compare", str(base), str(candidate)]) == 0
        out = capsys.readouterr().out
        assert "route" in out
        assert "5" in out

    def test_compare_refuses_a_model_change_by_default(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = Report.load(write_report(tmp_path / "cand.json", passes=30))
        drifted = candidate.model_copy(
            update={
                "env": candidate.env.model_copy(update={"resolved_models": ["gpt-4o-2024-11-20"]})
            }
        )
        path = drifted.save(tmp_path / "drift.json")
        assert main(["compare", str(base), str(path)]) == 1

    def test_the_model_change_refusal_can_be_overridden(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = Report.load(write_report(tmp_path / "cand.json", passes=30))
        drifted = candidate.model_copy(
            update={
                "env": candidate.env.model_copy(update={"resolved_models": ["gpt-4o-2024-11-20"]})
            }
        )
        path = drifted.save(tmp_path / "drift.json")
        assert main(["compare", str(base), str(path), "--allow-model-change"]) == 0

    def test_compare_emits_json_on_request(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=30, flip_first=5)
        main(["compare", str(base), str(candidate), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["regressed"] == [f"c-{i:04d}" for i in range(5)]

    def test_the_full_flip_list_is_in_the_json_even_when_printing_is_capped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The doc caps the printed list at 20 each way, full list in JSON."""
        base = write_report(tmp_path / "base.json", passes=40)
        candidate = write_report(tmp_path / "cand.json", passes=40, flip_first=25)
        main(["compare", str(base), str(candidate), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["regressed"]) == 25


class TestRender:
    def test_render_prints_markdown_for_a_report(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_report(tmp_path / "r.json", passes=30)
        assert main(["render", str(path)]) == 0
        # This fixture carries outcomes but no precomputed metrics list, so the
        # metrics table is correctly absent; the reproducibility block is not.
        assert "## Reproducibility" in capsys.readouterr().out

    def test_render_can_write_to_a_file(self, tmp_path: Path) -> None:
        path = write_report(tmp_path / "r.json", passes=30)
        out = tmp_path / "report.md"
        assert main(["render", str(path), "--out", str(out)]) == 0
        assert "## Reproducibility" in out.read_text(encoding="utf-8")

    def test_rendering_a_missing_report_fails(self, tmp_path: Path) -> None:
        assert main(["render", str(tmp_path / "absent.json")]) == 1


class TestBaselinePromote:
    def test_promoting_copies_the_candidate_to_the_baseline_path(self, tmp_path: Path) -> None:
        candidate = write_report(tmp_path / "cand.json", passes=30)
        baseline = tmp_path / "baselines" / "nextrole.json"
        assert main(["baseline", "promote", str(candidate), "--to", str(baseline)]) == 0
        assert Report.load(baseline).suite == "nextrole.routing"

    def test_promoting_an_incomplete_report_is_refused(self, tmp_path: Path) -> None:
        """A baseline is what every future run is judged against. Promoting an
        incomplete one would bake a partial measurement into the gate."""
        candidate = write_report(
            tmp_path / "cand.json", passes=30, complete=False, status="incomplete"
        )
        baseline = tmp_path / "baseline.json"
        assert main(["baseline", "promote", str(candidate), "--to", str(baseline)]) == 1
        assert not baseline.exists()

    def test_promoting_over_an_existing_baseline_needs_force(self, tmp_path: Path) -> None:
        """Explicit and reviewed, per the doc. An accidental overwrite would
        silently move the bar every future run is measured against."""
        candidate = write_report(tmp_path / "cand.json", passes=30)
        baseline = write_report(tmp_path / "baseline.json", passes=25, fails=5)
        assert main(["baseline", "promote", str(candidate), "--to", str(baseline)]) == 1
        assert main(["baseline", "promote", str(candidate), "--to", str(baseline), "--force"]) == 0


class TestHelp:
    def test_every_subcommand_is_listed_in_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        main([])
        out = capsys.readouterr().out
        for command in ("validate", "compare", "gate", "render", "baseline"):
            assert command in out

    def test_the_gate_help_documents_the_exit_codes(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Someone reading CI output needs to know what exit 3 meant."""
        with pytest.raises(SystemExit):
            main(["gate", "--help"])
        out = capsys.readouterr().out
        assert "inconclusive" in out.lower()


class TestGateFromConfig:
    """``--config`` reads ``[gate]``, so CI and a local run use one definition."""

    def write_config(self, tmp_path: Path, gate_block: str) -> Path:
        path = tmp_path / "neverempty.toml"
        path.write_text(
            '[project]\nname = "x"\n[target]\nentrypoint = "m:f"\n'
            '[[suite]]\nname = "a.b"\npath = "x.jsonl"\nsplit = "test"\n' + gate_block,
            encoding="utf-8",
        )
        return path

    def test_a_floor_from_the_config_file_fails_the_build(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=10, fails=20)
        candidate = write_report(tmp_path / "cand.json", passes=10, fails=20)
        config = self.write_config(tmp_path, "[gate]\nfloors = { route = 0.75 }\n")
        assert main(["gate", str(base), str(candidate), "--config", str(config)]) == 1

    def test_a_command_line_floor_overrides_the_config(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=30)
        config = self.write_config(tmp_path, "[gate]\nfloors = { route = 0.99 }\n")
        assert (
            main(
                [
                    "gate",
                    str(base),
                    str(candidate),
                    "--config",
                    str(config),
                    "--floor",
                    "route=0.5",
                ]
            )
            == 0
        )

    def test_an_invalid_config_is_a_usage_error_not_a_verdict(self, tmp_path: Path) -> None:
        """A broken config means the gate could not run. Reporting it as a
        quality verdict would be the wrong kind of failure entirely."""
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=30)
        config = tmp_path / "bad.toml"
        config.write_text('[project]\nname = "x"\n', encoding="utf-8")
        assert main(["gate", str(base), str(candidate), "--config", str(config)]) == 2

    def test_alpha_can_be_tightened_from_the_command_line(self, tmp_path: Path) -> None:
        """At alpha=0.01, five regressions (p=0.031) is no longer significant."""
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=30, flip_first=5)
        assert main(["gate", str(base), str(candidate)]) == 1
        assert main(["gate", str(base), str(candidate), "--paired-alpha", "0.01"]) == 0

    def test_a_non_numeric_floor_value_is_a_usage_error(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=30)
        assert main(["gate", str(base), str(candidate), "--floor", "route=high"]) == 2

    def test_the_primary_metric_can_be_named(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=30, flip_first=10)
        assert main(["gate", str(base), str(candidate), "--primary", "route"]) == 1

    def test_the_max_unstable_rate_can_be_set(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(tmp_path / "cand.json", passes=30)
        assert main(["gate", str(base), str(candidate), "--max-unstable-rate", "0.5"]) == 0


class TestBaselineUsage:
    def test_baseline_without_an_action_prints_usage(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["baseline"]) == 2
        assert "promote" in capsys.readouterr().out

    def test_promoting_a_missing_candidate_fails(self, tmp_path: Path) -> None:
        assert (
            main(
                [
                    "baseline",
                    "promote",
                    str(tmp_path / "absent.json"),
                    "--to",
                    str(tmp_path / "b.json"),
                ]
            )
            == 1
        )


class TestCompareMissingReport:
    def test_comparing_a_missing_report_fails(self, tmp_path: Path) -> None:
        base = write_report(tmp_path / "base.json", passes=30)
        assert main(["compare", str(base), str(tmp_path / "absent.json")]) == 1

    def test_a_refused_comparison_exits_nonzero_in_json_mode_too(self, tmp_path: Path) -> None:
        """The refusal has to be visible to a script, not only to a reader."""
        base = write_report(tmp_path / "base.json", passes=30)
        candidate = write_report(
            tmp_path / "cand.json", passes=30, complete=False, status="incomplete"
        )
        assert main(["compare", str(base), str(candidate), "--json"]) == 1
