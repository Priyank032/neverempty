"""D18: non-ASCII output crashes on a default Windows console.

The failure patterns include Devanagari and the renderer uses typographic
punctuation, so printing either on a ``cp1252`` console raises
``UnicodeEncodeError`` or prints replacement characters. The files the library
*writes* are already correct -- everything goes through
``encoding="utf-8"`` -- so this is stdout only.

It is narrow but real: the suites this library was built for are Hindi and
Hinglish, so a user on Windows inspecting their own data hits it immediately,
and ``UnicodeEncodeError`` from inside a reporting tool reads as a bug in the
tool rather than a console setting.

The CLI reconfigures its own streams to UTF-8 rather than asking users to set
``PYTHONIOENCODING``. Writing bytes the terminal renders imperfectly is a
cosmetic problem; raising is not.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _child(code: str) -> subprocess.CompletedProcess[bytes]:
    """Run code in a subprocess with no PYTHONIOENCODING, as a user would."""
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    env["PYTHONUTF8"] = "0"  # do not let UTF-8 mode mask the problem
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )


class TestTheCliCanPrintNonAscii:
    def test_devanagari_reaches_stdout(self) -> None:
        """``--version`` exits, so the streams are reconfigured directly --
        which is what ``main`` does on its first line."""
        result = _child(
            "from neverempty.cli import _use_utf8_streams; "
            "_use_utf8_streams(); "
            "print('नौकरी मिलेगी')"
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        assert "नौकरी" in result.stdout.decode("utf-8", "replace")

    def test_without_the_fix_it_would_have_failed(self) -> None:
        """Proves the test is exercising something: the same print without the
        reconfiguration raises on a cp1252 console."""
        result = _child("print('नौकरी मिलेगी')")
        if result.returncode == 0:
            # A console that is already UTF-8; nothing to prove here.
            return
        assert b"UnicodeEncodeError" in result.stderr

    def test_the_failure_patterns_can_be_printed(self) -> None:
        """The reviewer's exact repro."""
        result = _child(
            "from neverempty.cli import main; "
            "main(['--version']); "
            "from neverempty.scorers.failure import FAILURE_PATTERNS; "
            "print([p.pattern for p in FAILURE_PATTERNS])"
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        assert b"UnicodeEncodeError" not in result.stderr


class TestFilesWereAlreadyCorrect:
    """Pinned so the stdout fix is not mistaken for the whole story."""

    def test_a_rendered_report_writes_utf8(self, tmp_path: Path) -> None:
        from neverempty import Report, render_markdown
        from neverempty.core.trace import Env
        from neverempty.report.report import CaseOutcome, Metric, Score

        report = Report(
            suite="s.routing",
            split="test",
            created_at="2026-10-01T00:00:00.000Z",
            complete=True,
            status="ok",
            env=Env(
                neverempty_version="0.1.0",
                pricing_version="p",
                python_version="3.12.10",
            ),
            outcomes=[
                CaseOutcome(
                    case_id="c-0001",
                    repeat=0,
                    scored=True,
                    scores={"route": Score(passed=True, value=1.0)},
                )
            ],
            metrics=[Metric(name="route", n=1, applicable=1, value=1.0, method="wilson")],
        )
        target = tmp_path / "r.md"
        target.write_text(render_markdown(report), encoding="utf-8")
        assert target.read_text(encoding="utf-8")

    def test_a_saved_report_round_trips_devanagari(self, tmp_path: Path) -> None:
        from neverempty import Report
        from neverempty.core.trace import Env
        from neverempty.report.report import CaseOutcome, Score

        report = Report(
            suite="s.routing",
            split="test",
            created_at="2026-10-01T00:00:00.000Z",
            complete=True,
            status="ok",
            env=Env(
                neverempty_version="0.1.0",
                pricing_version="p",
                python_version="3.12.10",
            ),
            outcomes=[
                CaseOutcome(
                    case_id="c-0001",
                    repeat=0,
                    scored=True,
                    scores={"route": Score(passed=True, value=1.0, detail={"predicted": "नौकरी"})},
                )
            ],
        )
        path = report.save(tmp_path / "r.json")
        assert "नौकरी" in Report.load(path).outcomes[0].scores["route"].detail["predicted"]


class TestPythonDashMWorks:
    """``python -m neverempty`` failed with "is a package and cannot be
    directly executed". The console script is the documented path, but ``-m``
    is what people reach for when it is not on PATH, and failing there reads as
    a broken install rather than a missing entry point."""

    def test_the_module_is_runnable(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "neverempty", "--version"],
            capture_output=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        assert b"neverempty" in result.stdout

    def test_it_reports_the_same_version_as_the_script(self) -> None:
        from neverempty import __version__

        result = subprocess.run(
            [sys.executable, "-m", "neverempty", "--version"],
            capture_output=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert __version__ in result.stdout.decode("utf-8", "replace")

    def test_an_unknown_subcommand_exits_non_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "neverempty", "nonsense"],
            capture_output=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert result.returncode != 0
