"""``neverempty validate evals/**/*.jsonl``.

    schema + id uniqueness + split hash check

This runs in CI in a target repo, so its exit codes and its output are the
contract: a green run must mean the dataset is safe to publish numbers from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from neverempty.cli import main

from .test_dataset import case_dict, write_jsonl


def run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    code = main(["validate", *args])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


class TestExitCodes:
    def test_a_valid_dataset_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001")])
        code, output = run(capsys, str(path))
        assert code == 0
        assert "1 case" in output

    def test_an_invalid_dataset_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001", split="nope")])
        code, _ = run(capsys, str(path))
        assert code != 0

    def test_a_missing_file_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, output = run(capsys, str(tmp_path / "absent.jsonl"))
        assert code != 0
        assert "absent.jsonl" in output

    def test_no_matching_files_is_an_error_not_a_silent_pass(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A glob that matches nothing must never read as 'all datasets valid'."""
        code, output = run(capsys, str(tmp_path / "*.jsonl"))
        assert code != 0
        assert "no files" in output.lower()


class TestReporting:
    def test_it_names_every_bad_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_jsonl(
            tmp_path / "d.jsonl",
            [
                case_dict("c-0001"),
                case_dict("c-0002", split="nope"),
                case_dict("c-0003", suite="BAD"),
            ],
        )
        _, output = run(capsys, str(path))
        assert "line 2" in output
        assert "line 3" in output

    def test_it_reports_counts_per_split(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_jsonl(
            tmp_path / "d.jsonl",
            [case_dict("c-0001", "dev"), case_dict("c-0002", "test"), case_dict("c-0003", "test")],
        )
        _, output = run(capsys, str(path))
        assert "dev" in output
        assert "test" in output

    def test_it_prints_the_test_split_hash(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """So it can be pasted into config as the recorded hash."""
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001", "test")])
        _, output = run(capsys, str(path))
        from neverempty import Dataset

        assert Dataset.load(path).split_hash() in output  # type: ignore[operator]

    def test_a_dev_only_dataset_says_no_test_split_rather_than_printing_a_hash(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001", "dev")])
        _, output = run(capsys, str(path))
        assert "no test split" in output.lower()

    def test_several_files_are_each_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        a = write_jsonl(tmp_path / "routing.jsonl", [case_dict("c-0001")])
        b = write_jsonl(tmp_path / "failure.jsonl", [case_dict("c-0002")])
        code, output = run(capsys, str(a), str(b))
        assert code == 0
        assert "routing.jsonl" in output
        assert "failure.jsonl" in output

    def test_one_bad_file_among_several_fails_the_whole_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        good = write_jsonl(tmp_path / "good.jsonl", [case_dict("c-0001")])
        bad = write_jsonl(tmp_path / "bad.jsonl", [case_dict("c-0002", split="nope")])
        code, output = run(capsys, str(good), str(bad))
        assert code != 0
        assert "bad.jsonl" in output


class TestGlobs:
    def test_a_glob_expands_to_every_match(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_jsonl(tmp_path / "a.jsonl", [case_dict("c-0001")])
        write_jsonl(tmp_path / "b.jsonl", [case_dict("c-0002")])
        code, output = run(capsys, str(tmp_path / "*.jsonl"))
        assert code == 0
        assert "a.jsonl" in output
        assert "b.jsonl" in output

    def test_a_recursive_glob_descends(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        nested = tmp_path / "nextrole"
        nested.mkdir()
        write_jsonl(nested / "routing.jsonl", [case_dict("c-0001")])
        code, output = run(capsys, str(tmp_path / "**" / "*.jsonl"))
        assert code == 0
        assert "routing.jsonl" in output


class TestHashVerification:
    def test_a_matching_expected_hash_passes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from neverempty import Dataset

        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001", "test")])
        digest = Dataset.load(path).split_hash()
        assert digest is not None
        code, _ = run(capsys, str(path), "--expect-split-hash", digest)
        assert code == 0

    def test_a_mismatched_expected_hash_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The leakage guard: an edited test split must not pass CI silently."""
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001", "test")])
        code, output = run(capsys, str(path), "--expect-split-hash", "f" * 64)
        assert code != 0
        assert "suite_version" in output


class TestStrictProvenance:
    def test_a_test_case_without_provenance_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        case = case_dict("c-0001", "test")
        del case["provenance"]
        path = write_jsonl(tmp_path / "d.jsonl", [case])
        code, output = run(capsys, str(path))
        assert code != 0
        assert "provenance" in output


class TestJsonOutput:
    def test_json_mode_emits_a_machine_readable_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_jsonl(
            tmp_path / "d.jsonl", [case_dict("c-0001", "dev"), case_dict("c-0002", "test")]
        )
        code, output = run(capsys, str(path), "--json")
        assert code == 0
        payload: dict[str, Any] = json.loads(output)
        assert payload["ok"] is True
        assert payload["files"][0]["counts"]["test"] == 1

    def test_json_mode_reports_failures_too(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001", split="nope")])
        code, output = run(capsys, str(path), "--json")
        assert code != 0
        payload = json.loads(output)
        assert payload["ok"] is False
        assert payload["files"][0]["errors"]


class TestNoSubcommand:
    def test_bare_invocation_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main([]) == 0
        assert "validate" in capsys.readouterr().out

    def test_no_files_in_json_mode_is_machine_readable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["validate", str(tmp_path / "*.jsonl"), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert code != 0
        assert payload["ok"] is False
        assert "no files" in payload["error"].lower()

    def test_split_filtering_is_available_from_the_cli(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_jsonl(
            tmp_path / "d.jsonl", [case_dict("c-0001", "dev"), case_dict("c-0002", "test")]
        )
        code, output = run(capsys, str(path), "--split", "test")
        assert code == 0
        assert "1 case" in output
