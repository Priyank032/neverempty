"""``neverempty init``: a working setup in one command.

Adoption, not correctness. A stranger who installs this gets a framework and
an empty page: the first dataset they write fails on format rules, the first
config they write has no example, and the README's quickstart has to be
retyped by hand. That is the gap between "the library works" and "someone can
use it".

``init`` writes a runnable skeleton into their project -- a config, a dataset
with real example cases, and a target stub -- so the first ``neverempty run``
succeeds before they have written anything of their own. What it does *not*
write is labels: those are the user's ground truth about their own agent, and
a generated one would make every number a measure of a model agreeing with
itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neverempty.cli import main


class TestItCreatesAWorkingSetup:
    def test_it_exits_zero(self, tmp_path: Path) -> None:
        assert main(["init", "--dir", str(tmp_path)]) == 0

    def test_it_writes_a_config(self, tmp_path: Path) -> None:
        main(["init", "--dir", str(tmp_path)])
        assert (tmp_path / "evals" / "neverempty.toml").is_file()

    def test_it_writes_a_dataset(self, tmp_path: Path) -> None:
        main(["init", "--dir", str(tmp_path)])
        assert (tmp_path / "evals" / "datasets" / "routing.jsonl").is_file()

    def test_it_writes_a_target_stub(self, tmp_path: Path) -> None:
        main(["init", "--dir", str(tmp_path)])
        assert (tmp_path / "evals" / "target.py").is_file()

    def test_it_says_what_to_do_next(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["init", "--dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert "neverempty validate" in out
        assert "neverempty run" in out


class TestTheGeneratedDatasetIsValid:
    """The whole point: it has to load, or ``init`` has shipped the same
    first-dataset failure it exists to prevent."""

    def test_it_passes_validate(self, tmp_path: Path) -> None:
        main(["init", "--dir", str(tmp_path)])
        assert main(["validate", str(tmp_path / "evals" / "datasets" / "routing.jsonl")]) == 0

    def test_it_loads_through_the_api(self, tmp_path: Path) -> None:
        from neverempty import Dataset

        main(["init", "--dir", str(tmp_path)])
        dataset = Dataset.load(tmp_path / "evals" / "datasets" / "routing.jsonl", split="dev")
        assert len(dataset) >= 3

    def test_every_case_declares_an_expectation(self, tmp_path: Path) -> None:
        main(["init", "--dir", str(tmp_path)])
        lines = (
            (tmp_path / "evals" / "datasets" / "routing.jsonl")
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        )
        for line in lines:
            assert json.loads(line)["expect"]

    def test_it_includes_a_fault_suite(self, tmp_path: Path) -> None:
        """The headline metric needs fault opportunities, and a user who has
        never seen one declared will not write one. It is a separate file
        because one report describes one suite."""
        main(["init", "--dir", str(tmp_path)])
        text = (tmp_path / "evals" / "datasets" / "failure.jsonl").read_text(encoding="utf-8")
        assert '"faults"' in text

    def test_the_failure_dataset_also_validates(self, tmp_path: Path) -> None:
        main(["init", "--dir", str(tmp_path)])
        assert main(["validate", str(tmp_path / "evals" / "datasets" / "failure.jsonl")]) == 0


class TestItRefusesToOverwrite:
    def test_an_existing_file_is_not_clobbered(self, tmp_path: Path) -> None:
        config = tmp_path / "evals" / "neverempty.toml"
        config.parent.mkdir(parents=True)
        config.write_text("# mine\n", encoding="utf-8")

        assert main(["init", "--dir", str(tmp_path)]) != 0
        assert config.read_text(encoding="utf-8") == "# mine\n"

    def test_force_overwrites(self, tmp_path: Path) -> None:
        config = tmp_path / "evals" / "neverempty.toml"
        config.parent.mkdir(parents=True)
        config.write_text("# mine\n", encoding="utf-8")

        assert main(["init", "--dir", str(tmp_path), "--force"]) == 0
        assert config.read_text(encoding="utf-8") != "# mine\n"

    def test_the_refusal_names_the_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = tmp_path / "evals" / "neverempty.toml"
        config.parent.mkdir(parents=True)
        config.write_text("# mine\n", encoding="utf-8")

        main(["init", "--dir", str(tmp_path)])
        assert "neverempty.toml" in capsys.readouterr().err


class TestItWritesNoLabels:
    """Ground truth is the user's. A generated label would make every number a
    measure of one model agreeing with another."""

    def test_the_example_cases_are_marked_as_examples(self, tmp_path: Path) -> None:
        main(["init", "--dir", str(tmp_path)])
        text = (tmp_path / "evals" / "datasets" / "routing.jsonl").read_text(encoding="utf-8")
        for line in text.strip().splitlines():
            assert json.loads(line)["id"].startswith("example-")

    def test_the_config_says_the_suite_is_not_ready(self, tmp_path: Path) -> None:
        main(["init", "--dir", str(tmp_path)])
        config = (tmp_path / "evals" / "neverempty.toml").read_text(encoding="utf-8")
        assert "floors" in config


class TestTheScaffoldActuallyRuns:
    """``init`` printing "try it now" and then failing is worse than no
    scaffold: it fails in the one place a newcomer cannot debug."""

    def test_run_succeeds_on_the_generated_setup(self, tmp_path: Path) -> None:
        main(["init", "--dir", str(tmp_path)])
        assert main(["run", str(tmp_path / "evals" / "neverempty.toml")]) == 0

    def test_it_produces_a_complete_report(self, tmp_path: Path) -> None:
        from neverempty import Report

        main(["init", "--dir", str(tmp_path)])
        main(["run", str(tmp_path / "evals" / "neverempty.toml")])
        reports = sorted((tmp_path / "evals" / "reports").glob("*.json"))
        assert len(reports) == 2, "one report per suite"
        for path in reports:
            report = Report.load(path)
            assert report.complete is True
            assert report.counts.scored == report.counts.cases

    def test_the_fault_case_produces_the_headline_metric(self, tmp_path: Path) -> None:
        """The scaffold ships a fault case so misreport_as_empty has an
        opportunity. A suite with none publishes 0% from zero events."""
        from neverempty import Report

        main(["init", "--dir", str(tmp_path)])
        main(["run", str(tmp_path / "evals" / "neverempty.toml")])
        reports = sorted((tmp_path / "evals" / "reports").glob("*failure*.json"))
        assert reports, "the failure suite wrote no report"
        names = {m.name for m in Report.load(reports[0]).metrics}
        assert "misreport_as_empty" in names


class TestPathsResolveAgainstTheConfigNotTheShell:
    """``Config.resolve`` exists and says "running the CLI from a different
    directory must not change which dataset a suite names". ``run`` and
    ``coverage`` were passing the raw string instead, so a config was only
    usable from the directory above it -- which is the directory a scaffolded
    user happens to be in, and so the bug stayed invisible."""

    def test_run_works_from_an_unrelated_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "project"
        project.mkdir()
        main(["init", "--dir", str(project)])

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        assert main(["run", str(project / "evals" / "neverempty.toml")]) == 0

    def test_coverage_works_from_an_unrelated_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "project"
        project.mkdir()
        main(["init", "--dir", str(project)])

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        # Non-zero is fine -- the example suite is far short of a real one --
        # but it must be a coverage verdict, not a missing file.
        assert main(["coverage", str(project / "evals" / "neverempty.toml")]) != 2


class TestTomlWorksOnEverySupportedPython:
    """A fresh 3.10 install could run `init` and `validate` but not `run`:
    reading any TOML config needed `tomli`, which was not a dependency.

    The brief says core depends on pydantic only, and that stays true -- the
    marker means 3.11+ installs nothing extra. On 3.10 the stdlib has no
    `tomllib`, so a CLI that advertises `run` and cannot read a config there is
    advertising something it does not have.
    """

    def test_tomli_is_required_on_310_only(self) -> None:
        from neverempty.config import _toml_module

        data = _toml_module().loads(
            (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
        )
        deps = data["project"]["dependencies"]
        tomli = [d for d in deps if d.startswith("tomli")]
        assert tomli, "tomli must be declared for the 3.10 leg"
        assert "python_version < " in tomli[0], "it must be conditional, not unconditional"

    def test_core_is_still_pydantic_plus_the_marker(self) -> None:
        """The brief's rule: core depends on pydantic v2 only."""
        from neverempty.config import _toml_module

        data = _toml_module().loads(
            (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
        )
        unconditional = [d for d in data["project"]["dependencies"] if ";" not in d]
        assert unconditional == ["pydantic>=2.7,<3"]

    def test_a_config_loads_on_this_interpreter(self, tmp_path: Path) -> None:
        from neverempty.config import load_config

        main(["init", "--dir", str(tmp_path)])
        config = load_config(str(tmp_path / "evals" / "neverempty.toml"))
        assert len(config.suites) == 2
