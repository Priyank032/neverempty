"""D27: a missing dataset file crashes ``run`` with a traceback.

``validate`` catches ``(DatasetError, FileNotFoundError)``. ``run`` catches only
``DatasetError``, so a path that does not exist -- the most ordinary mistake
there is, a typo or a file not yet written -- escapes as an uncaught
``FileNotFoundError`` with a Python stack trace.

The message underneath is good ("dataset file not found: <path>"). The
traceback around it is what makes a one-character typo look like a broken
install, and a new user has no way to tell those apart.

Found by running the tool in three real repositories rather than by reading the
code: the scaffold always writes the file, so nothing in the test suite ever
exercised the path where it is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neverempty.cli import main


def _project(tmp_path: Path, *, keep_dataset: bool) -> Path:
    main(["init", "--dir", str(tmp_path)])
    if not keep_dataset:
        (tmp_path / "evals" / "datasets" / "routing.jsonl").unlink()
    return tmp_path / "evals" / "neverempty.toml"


class TestAMissingDatasetIsAnError:
    def test_it_does_not_raise(self, tmp_path: Path) -> None:
        """The defect: an uncaught FileNotFoundError and a stack trace."""
        config = _project(tmp_path, keep_dataset=False)
        assert main(["run", str(config), "--suite", "myagent.routing"]) != 0

    def test_it_exits_invalid(self, tmp_path: Path) -> None:
        """``EXIT_INVALID`` is 1 for the CLI. The gate's 4 is a different
        scale: these are a command's exit codes, not a gate verdict."""
        from neverempty.cli import EXIT_INVALID

        config = _project(tmp_path, keep_dataset=False)
        assert main(["run", str(config), "--suite", "myagent.routing"]) == EXIT_INVALID

    def test_the_message_names_the_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = _project(tmp_path, keep_dataset=False)
        main(["run", str(config), "--suite", "myagent.routing"])
        assert "routing.jsonl" in capsys.readouterr().err

    def test_the_message_names_the_suite(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With several suites configured, which one is missing matters."""
        config = _project(tmp_path, keep_dataset=False)
        main(["run", str(config), "--suite", "myagent.routing"])
        assert "myagent.routing" in capsys.readouterr().err

    def test_no_traceback_reaches_the_user(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config = _project(tmp_path, keep_dataset=False)
        main(["run", str(config), "--suite", "myagent.routing"])
        assert "Traceback" not in capsys.readouterr().err


class TestAPresentDatasetStillRuns:
    def test_the_scaffold_runs_unchanged(self, tmp_path: Path) -> None:
        config = _project(tmp_path, keep_dataset=True)
        assert main(["run", str(config), "--suite", "myagent.routing"]) == 0


class TestValidateAlreadyHandledThis:
    """Pinned so the two commands cannot drift apart again."""

    def test_validate_reports_a_missing_file_cleanly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _project(tmp_path, keep_dataset=False)
        code = main(["validate", str(tmp_path / "evals" / "datasets" / "routing.jsonl")])
        assert code != 0
        assert "Traceback" not in capsys.readouterr().err
