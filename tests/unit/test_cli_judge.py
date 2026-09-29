"""The ``neverempty judge calibrate`` subcommand.

Prints Cohen's kappa, the 3x3 matrix, and precision and recall for
``contradicted`` specifically, because that is the label driving the headline
YojanaKhoj number.

Exit codes here are about whether calibration could be *computed*, not about
whether the judge is good. A kappa of 0.2 is a valid measurement and exits 0 with
a prominent warning; a malformed calibration file exits non-zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neverempty.cli import main


def write_calibration(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def row(index: int, human: str, language: str = "en", **extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": f"k-{index:04d}",
        "claim": f"claim {index}",
        "evidence": {"rows": [{"n": index}]},
        "human_label": human,
        "language": language,
    }
    base.update(extra)
    return base


def agreeing(tmp_path: Path, n: int = 30) -> Path:
    """A set the scripted judge will agree with perfectly."""
    rows = [row(index, "supported" if index % 2 else "contradicted") for index in range(n)]
    return write_calibration(tmp_path / "judge.v1.jsonl", rows)


class TestCalibrateCommand:
    def test_calibrating_against_a_perfect_judge_exits_zero(self, tmp_path: Path) -> None:
        path = agreeing(tmp_path)
        assert main(["judge", "calibrate", str(path), "--replay", str(path)]) == 0

    def test_the_kappa_is_printed(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = agreeing(tmp_path)
        main(["judge", "calibrate", str(path), "--replay", str(path)])
        out = capsys.readouterr().out
        assert "kappa" in out.lower()
        assert "1.000" in out

    def test_the_three_by_three_matrix_is_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = agreeing(tmp_path)
        main(["judge", "calibrate", str(path), "--replay", str(path)])
        out = capsys.readouterr().out
        for label in ("supported", "contradicted", "not_in_evidence"):
            assert label in out

    def test_contradicted_precision_and_recall_are_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The label that drives the headline number gets its own line."""
        path = agreeing(tmp_path)
        main(["judge", "calibrate", str(path), "--replay", str(path)])
        out = capsys.readouterr().out.lower()
        assert "precision" in out
        assert "recall" in out

    def test_a_low_kappa_exits_zero_but_warns_prominently(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A bad judge is a valid measurement, not a broken command. The warning
        is what stops the number being published."""
        labels = write_calibration(
            tmp_path / "judge.v1.jsonl",
            [row(i, "supported") for i in range(8)]
            + [row(i + 8, "contradicted") for i in range(2)],
        )
        replay = write_calibration(
            tmp_path / "replay.jsonl", [row(i, "supported") for i in range(10)]
        )
        assert main(["judge", "calibrate", str(labels), "--replay", str(replay)]) == 0
        out = capsys.readouterr().out.lower()
        assert "not publishable" in out or "below" in out

    def test_a_low_kappa_can_be_made_a_failure_for_ci(self, tmp_path: Path) -> None:
        """A release workflow wants this to fail; an exploratory run does not."""
        labels = write_calibration(
            tmp_path / "judge.v1.jsonl",
            [row(i, "supported") for i in range(8)]
            + [row(i + 8, "contradicted") for i in range(2)],
        )
        replay = write_calibration(
            tmp_path / "replay.jsonl", [row(i, "supported") for i in range(10)]
        )
        assert (
            main(
                [
                    "judge",
                    "calibrate",
                    str(labels),
                    "--replay",
                    str(replay),
                    "--fail-below-threshold",
                ]
            )
            == 1
        )

    def test_a_malformed_calibration_file_exits_nonzero(self, tmp_path: Path) -> None:
        path = write_calibration(tmp_path / "judge.v1.jsonl", [row(1, "maybe")])
        assert main(["judge", "calibrate", str(path), "--replay", str(path)]) == 1

    def test_a_missing_calibration_file_exits_nonzero(self, tmp_path: Path) -> None:
        assert (
            main(
                [
                    "judge",
                    "calibrate",
                    str(tmp_path / "absent.jsonl"),
                    "--replay",
                    str(tmp_path / "absent.jsonl"),
                ]
            )
            == 1
        )

    def test_json_output_is_machine_readable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = agreeing(tmp_path)
        main(["judge", "calibrate", str(path), "--replay", str(path), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["kappa"] == 1.0
        assert payload["scored"] == 30

    def test_the_result_can_be_written_to_a_file(self, tmp_path: Path) -> None:
        """The report links to this, so it has to be a committed artefact."""
        path = agreeing(tmp_path)
        out = tmp_path / "calibration.json"
        main(["judge", "calibrate", str(path), "--replay", str(path), "--out", str(out)])
        assert json.loads(out.read_text(encoding="utf-8"))["kappa"] == 1.0

    def test_per_language_agreement_is_printed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_calibration(
            tmp_path / "judge.v1.jsonl",
            [row(i, "supported", "hi" if i < 5 else "en") for i in range(10)],
        )
        main(["judge", "calibrate", str(path), "--replay", str(path)])
        out = capsys.readouterr().out
        assert "hi" in out
        assert "en" in out

    def test_a_replay_missing_a_case_is_refused(self, tmp_path: Path) -> None:
        """Calibration pairs labels by id. A replay file missing a case would
        silently shrink the set the kappa was computed on."""
        labels = write_calibration(
            tmp_path / "judge.v1.jsonl", [row(i, "supported") for i in range(5)]
        )
        replay = write_calibration(
            tmp_path / "replay.jsonl", [row(i, "supported") for i in range(3)]
        )
        assert main(["judge", "calibrate", str(labels), "--replay", str(replay)]) == 1


class TestInjectionFixtures:
    def test_the_shipped_injection_fixtures_load(self) -> None:
        """The doc requires two injection fixtures in the calibration set. They
        ship with the library so a downstream set can start from them."""
        from neverempty.judge.calibration import load_injection_fixtures

        cases = load_injection_fixtures()
        assert len(cases) >= 2
        assert all(case.injection for case in cases)

    def test_every_injection_fixture_carries_an_instruction_in_its_evidence(
        self,
    ) -> None:
        from neverempty.judge.calibration import load_injection_fixtures

        cases = load_injection_fixtures()
        for case in cases:
            rendered = json.dumps(case.evidence).lower()
            assert "ignore" in rendered or "supported" in rendered

    def test_the_fixtures_expect_the_label_not_to_flip(self) -> None:
        """Each fixture's human label is what a judge that ignored the embedded
        instruction would say. A judge that obeyed it would answer supported."""
        from neverempty.judge.calibration import load_injection_fixtures

        cases = load_injection_fixtures()
        assert any(case.human_label != "supported" for case in cases)

    async def test_the_judge_holds_the_line_on_every_shipped_fixture(self) -> None:
        """End to end: the real judge, the shipped fixtures, a scripted model
        that would obey an instruction if one reached it."""
        from neverempty.judge.calibration import load_injection_fixtures
        from neverempty.judge.judge import Claim, ClaimJudge
        from neverempty.judge.model import ScriptedJudge

        cases = load_injection_fixtures()
        for case in cases:
            model = ScriptedJudge([json.dumps({"label": case.human_label, "rationale": "r"})])
            judge = ClaimJudge(
                model=model,
                model_id="anthropic.claude-test",
                agent_family="openai",
            )
            verdicts = await judge.verify(
                claims=[Claim(id=case.id, text=case.claim)], evidence=case.evidence
            )
            assert verdicts[0].label == case.human_label
            # The instruction reached the model as delimited data, not as prompt.
            assert "<evidence>" in model.calls[0].user

    def test_calibrating_the_shipped_fixtures_counts_them_as_injections(self) -> None:
        from neverempty.judge.calibration import calibrate, load_injection_fixtures

        cases = load_injection_fixtures()
        result = calibrate(
            cases,
            judge_labels=[case.human_label for case in cases],
            model_id="m",
            prompt_version="v1",
        )
        assert result.injections == len(cases)
        assert result.injections_held == len(cases)

    def test_a_flipped_injection_is_reported_as_not_held(self) -> None:
        """The number that matters: how many fixtures the judge did *not* hold."""
        from neverempty.judge.calibration import calibrate, load_injection_fixtures

        cases = load_injection_fixtures()
        result = calibrate(
            cases,
            judge_labels=["supported"] * len(cases),
            model_id="m",
            prompt_version="v1",
        )
        assert result.injections_held < result.injections


class TestHelp:
    def test_judge_is_listed_in_the_top_level_help(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main([])
        assert "judge" in capsys.readouterr().out

    def test_judge_without_an_action_prints_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["judge"]) == 2
        assert "calibrate" in capsys.readouterr().out
