"""``neverempty readme``: publish from reports, or refuse."""

from __future__ import annotations

from pathlib import Path

import pytest

from neverempty.cli import BEGIN_MARKER, END_MARKER, main
from neverempty.report.report import Metric, Report


def build(path: Path, *, kappa: float | None = None, complete: bool = True) -> Path:
    payload: dict[str, object] = {
        "suite": "nextrole.routing",
        "split": "test",
        "created_at": "2026-09-25T10:00:00.000Z",
        "complete": complete,
        "status": "ok" if complete else "incomplete",
        "env": {
            "neverempty_version": "0.1.0",
            "pricing_version": "openai-2026-09-01",
            "python_version": "3.12.10",
            "target_git_sha": "c" * 40,
            "resolved_models": ["gpt-4o-mini-2024-07-18"],
        },
        "counts": {"cases": 330, "scored": 330, "unscored": 0},
        "metrics": [
            Metric(
                name="route",
                n=330,
                value=0.863,
                ci_low=0.822,
                ci_high=0.896,
                method="wilson",
                applicable=330,
            ).model_dump(),
            Metric(
                name="facts",
                n=330,
                value=0.91,
                ci_low=0.88,
                ci_high=0.94,
                method="wilson",
                applicable=330,
            ).model_dump(),
        ],
    }
    if kappa is not None:
        payload["judge"] = {"model": "anthropic.claude-test", "kappa": kappa}
    Report.model_validate(payload).save(path)
    return path


class TestRendering:
    def test_no_reports_prints_the_refusal(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["readme"]) == 0
        assert "No measured numbers yet" in capsys.readouterr().out

    def test_a_report_publishes_a_linked_row(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report = build(tmp_path / "r.json", kappa=0.78)
        main(["readme", str(report)])
        out = capsys.readouterr().out
        assert "86.3%" in out
        assert report.as_posix() in out

    def test_a_low_kappa_cuts_the_judge_metric(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["readme", str(build(tmp_path / "r.json", kappa=0.41))])
        out = capsys.readouterr().out
        assert "`route`" in out
        assert "`facts`" not in out

    def test_a_missing_report_exits_non_zero(self, tmp_path: Path) -> None:
        assert main(["readme", str(tmp_path / "absent.json")]) != 0


class TestCheck:
    def _write(self, readme: Path, rendered: str, *, prose: str = "") -> None:
        readme.write_text(
            BEGIN_MARKER + "\n" + rendered + "\n" + END_MARKER + "\n\n" + prose,
            encoding="utf-8",
        )

    def _rendered(self, report: Path) -> str:
        from neverempty.report.readme import load_reports, render_readme_numbers

        return render_readme_numbers(load_reports([report]))

    def test_a_matching_section_passes(self, tmp_path: Path) -> None:
        report = build(tmp_path / "r.json", kappa=0.78)
        readme = tmp_path / "README.md"
        self._write(readme, self._rendered(report))
        assert main(["readme", str(report), "--check", str(readme)]) == 0

    def test_a_stale_section_fails(self, tmp_path: Path) -> None:
        """A published number that no longer matches its report is exactly what
        this check exists to catch."""
        report = build(tmp_path / "r.json", kappa=0.78)
        readme = tmp_path / "README.md"
        self._write(readme, "## Numbers\n\nRoute accuracy is 99.9%.")
        assert main(["readme", str(report), "--check", str(readme)]) != 0

    def test_hand_written_prose_beside_the_block_is_not_drift(self, tmp_path: Path) -> None:
        """Only what is between the markers is generated, so commentary next to
        it must not make the check fail."""
        report = build(tmp_path / "r.json", kappa=0.78)
        readme = tmp_path / "README.md"
        self._write(
            readme,
            self._rendered(report),
            prose="A long hand-written note that changes often.\n",
        )
        assert main(["readme", str(report), "--check", str(readme)]) == 0

    def test_a_readme_without_the_markers_fails(self, tmp_path: Path) -> None:
        report = build(tmp_path / "r.json", kappa=0.78)
        readme = tmp_path / "README.md"
        readme.write_text("# x\n\nNothing here.\n", encoding="utf-8")
        assert main(["readme", str(report), "--check", str(readme)]) != 0

    def test_a_missing_readme_fails(self, tmp_path: Path) -> None:
        report = build(tmp_path / "r.json", kappa=0.78)
        assert main(["readme", str(report), "--check", str(tmp_path / "absent.md")]) != 0

    def test_the_repos_own_readme_is_current(self) -> None:
        """The published Numbers section must match the committed reports.

        It used to assert the opposite -- that with no reports the section says
        so -- which was the honest check while there were none. There is one
        now, so the check is that every published number still matches the
        report it was rendered from. A number in a README that has drifted from
        its report is exactly the failure this command exists to catch.
        """
        root = Path(__file__).resolve().parents[2]
        readme = root / "README.md"
        if not readme.exists():  # pragma: no cover - defensive
            pytest.skip("README.md not found")
        reports = sorted((root / "evals" / "reports").glob("*.json"))
        argv = ["readme", *[str(path) for path in reports], "--check", str(readme)]
        assert main(argv) == 0


class TestReadmeExpandsGlobsLikeValidateDoes:
    """``readme`` took its paths raw while ``validate`` expanded them.

    ``expand_paths`` exists precisely because "the shell has usually expanded
    these already, but not on Windows and not when a pattern is quoted", and
    the README's own documented command is a glob. On PowerShell and cmd --
    the platform this was developed on -- it reached ``open()`` verbatim and
    failed with a raw ``[Errno 22] Invalid argument``, so the documented
    workflow could not be run at all.
    """

    def test_a_glob_matching_several_reports_is_expanded(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reports = tmp_path / "reports"
        reports.mkdir()
        build(reports / "a.json")
        build(reports / "b.json")

        code = main(["readme", str(reports / "*.json")])
        assert code == 0
        assert "route" in capsys.readouterr().out

    def test_a_glob_matching_nothing_is_refused_not_treated_as_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An empty match must never render an empty Numbers section: that
        would publish "no numbers" as though it were a measured result."""
        code = main(["readme", str(tmp_path / "nothing" / "*.json")])
        assert code != 0
        assert "no files matched" in capsys.readouterr().err

    def test_a_literal_path_still_works(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        build(tmp_path / "one.json")
        assert main(["readme", str(tmp_path / "one.json")]) == 0
        assert "route" in capsys.readouterr().out

    def test_a_missing_literal_path_names_the_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["readme", str(tmp_path / "absent.json")])
        assert code != 0
        assert "absent.json" in capsys.readouterr().err

    def test_check_mode_also_expands(self, tmp_path: Path) -> None:
        reports = tmp_path / "reports"
        reports.mkdir()
        build(reports / "a.json")
        target = tmp_path / "README.md"
        target.write_text(f"# T\n\n{BEGIN_MARKER}\nstale\n{END_MARKER}\n", encoding="utf-8")
        # Reaches the drift comparison rather than dying on the glob.
        assert main(["readme", str(reports / "*.json"), "--check", str(target)]) != 0
