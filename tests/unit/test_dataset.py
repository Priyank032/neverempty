"""Acceptance row for ``dataset``.

    unknown key rejected; duplicate id rejected; split hash mismatch rejected;
    a real 280-line file validates in under a second

M4's build row adds: bad lines rejected with line numbers. A validation error
that does not name the line is useless on a 280-line file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from neverempty import Dataset
from neverempty.dataset.loader import DatasetError, split_hash


def case_dict(case_id: str, split: str = "dev", **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": 1,
        "id": case_id,
        "suite": "nextrole.routing",
        "split": split,
        "input": {"messages": [{"role": "user", "content": f"query {case_id}"}]},
        "expect": {"route": {"label": "job_search"}},
    }
    if split == "test":
        base["provenance"] = {
            "labeller": "priyank",
            "method": "human",
            "labelled_at": "2026-09-23",
        }
    base.update(overrides)
    return base


def write_jsonl(path: Path, cases: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    return path


class TestLoading:
    def test_a_valid_file_loads_every_case(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict(f"c-{i:04d}") for i in range(5)])
        dataset = Dataset.load(path)
        assert len(dataset) == 5
        assert dataset.cases[0].id == "c-0000"

    def test_it_is_iterable_and_indexable(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict(f"c-{i:04d}") for i in range(3)])
        dataset = Dataset.load(path)
        assert [case.id for case in dataset] == ["c-0000", "c-0001", "c-0002"]
        assert dataset[1].id == "c-0001"

    def test_a_case_is_retrievable_by_id(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001"), case_dict("c-0002")])
        assert Dataset.load(path).by_id("c-0002").id == "c-0002"

    def test_an_unknown_id_raises_a_keyerror(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001")])
        with pytest.raises(KeyError, match="c-9999"):
            Dataset.load(path).by_id("c-9999")

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text(
            json.dumps(case_dict("c-0001")) + "\n\n  \n" + json.dumps(case_dict("c-0002")) + "\n",
            encoding="utf-8",
        )
        assert len(Dataset.load(path)) == 2

    def test_a_utf8_bom_is_tolerated(self, tmp_path: Path) -> None:
        """Windows editors write one; it is not a data error."""
        path = tmp_path / "d.jsonl"
        path.write_bytes(b"\xef\xbb\xbf" + json.dumps(case_dict("c-0001")).encode("utf-8"))
        assert len(Dataset.load(path)) == 1

    def test_devanagari_content_survives_loading(self, tmp_path: Path) -> None:
        case = case_dict("c-0001")
        case["input"]["messages"][0]["content"] = "मुझे पुणे में नौकरी चाहिए"
        path = write_jsonl(tmp_path / "d.jsonl", [case])
        loaded = Dataset.load(path)
        assert loaded[0].input.messages is not None
        assert "पुणे" in loaded[0].input.messages[0].content

    def test_a_missing_file_fails_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Dataset.load(tmp_path / "absent.jsonl")

    def test_an_empty_file_is_an_error(self, tmp_path: Path) -> None:
        """An empty dataset would silently produce a report over zero cases."""
        path = tmp_path / "d.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(DatasetError, match="no cases"):
            Dataset.load(path)


class TestErrorsNameTheLine:
    def test_malformed_json_names_its_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text(
            json.dumps(case_dict("c-0001")) + "\n{not json\n" + json.dumps(case_dict("c-0003")),
            encoding="utf-8",
        )
        with pytest.raises(DatasetError) as exc:
            Dataset.load(path)
        assert "line 2" in str(exc.value)

    def test_a_schema_violation_names_its_line_and_field(self, tmp_path: Path) -> None:
        path = write_jsonl(
            tmp_path / "d.jsonl",
            [case_dict("c-0001"), case_dict("c-0002", suite="Not A Suite")],
        )
        with pytest.raises(DatasetError) as exc:
            Dataset.load(path)
        message = str(exc.value)
        assert "line 2" in message
        assert "suite" in message

    def test_an_unknown_key_names_its_line(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001", expects={})])
        with pytest.raises(DatasetError) as exc:
            Dataset.load(path)
        assert "line 1" in str(exc.value)
        assert "expects" in str(exc.value)

    def test_the_file_path_is_named_too(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "routing.jsonl", [case_dict("c-0001", split="nope")])
        with pytest.raises(DatasetError, match=r"routing\.jsonl"):
            Dataset.load(path)

    def test_every_bad_line_is_reported_not_only_the_first(self, tmp_path: Path) -> None:
        """Fixing 280 lines one error per run is not a workflow."""
        path = write_jsonl(
            tmp_path / "d.jsonl",
            [
                case_dict("c-0001"),
                case_dict("c-0002", split="nope"),
                case_dict("c-0003"),
                case_dict("c-0004", suite="BAD"),
            ],
        )
        with pytest.raises(DatasetError) as exc:
            Dataset.load(path)
        message = str(exc.value)
        assert "line 2" in message
        assert "line 4" in message

    def test_a_json_array_is_rejected_with_a_hint(self, tmp_path: Path) -> None:
        """JSON vs JSONL is the most common authoring mistake."""
        path = tmp_path / "d.json"
        path.write_text(json.dumps([case_dict("c-0001")]), encoding="utf-8")
        with pytest.raises(DatasetError, match=r"one JSON object per line"):
            Dataset.load(path)


class TestDuplicateIds:
    def test_a_duplicate_id_is_rejected(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001"), case_dict("c-0001")])
        with pytest.raises(DatasetError, match="duplicate"):
            Dataset.load(path)

    def test_the_duplicate_error_names_both_lines(self, tmp_path: Path) -> None:
        path = write_jsonl(
            tmp_path / "d.jsonl",
            [case_dict("c-0001"), case_dict("c-0002"), case_dict("c-0001")],
        )
        with pytest.raises(DatasetError) as exc:
            Dataset.load(path)
        message = str(exc.value)
        assert "c-0001" in message
        assert "line 3" in message

    def test_the_same_id_in_different_suites_is_allowed(self, tmp_path: Path) -> None:
        """Ids are unique within a suite, not globally."""
        path = write_jsonl(
            tmp_path / "d.jsonl",
            [case_dict("c-0001"), case_dict("c-0001", suite="nextrole.failure")],
        )
        assert len(Dataset.load(path)) == 2


class TestSplitFiltering:
    def test_loading_can_be_restricted_to_one_split(self, tmp_path: Path) -> None:
        path = write_jsonl(
            tmp_path / "d.jsonl",
            [case_dict("c-0001", "dev"), case_dict("c-0002", "test")],
        )
        assert [c.id for c in Dataset.load(path, split="test")] == ["c-0002"]

    def test_filtering_to_an_absent_split_is_an_error(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001", "dev")])
        with pytest.raises(DatasetError, match="no cases"):
            Dataset.load(path, split="test")

    def test_the_split_property_reports_what_is_present(self, tmp_path: Path) -> None:
        path = write_jsonl(
            tmp_path / "d.jsonl",
            [case_dict("c-0001", "dev"), case_dict("c-0002", "test")],
        )
        assert Dataset.load(path).splits == {"dev", "test"}


class TestSplitHash:
    def make(self, tmp_path: Path, cases: list[dict[str, Any]]) -> Dataset:
        return Dataset.load(write_jsonl(tmp_path / "d.jsonl", cases))

    def test_the_hash_covers_only_the_test_split(self, tmp_path: Path) -> None:
        """Tuning on dev is explicitly allowed, so dev must not invalidate a
        baseline."""
        with_dev = self.make(tmp_path, [case_dict("c-0001", "test"), case_dict("c-0002", "dev")])
        without_dev = self.make(tmp_path, [case_dict("c-0001", "test")])
        assert with_dev.split_hash() == without_dev.split_hash()

    def test_editing_a_dev_case_does_not_change_the_hash(self, tmp_path: Path) -> None:
        before = self.make(
            tmp_path, [case_dict("c-0001", "test"), case_dict("c-0002", "dev")]
        ).split_hash()
        after = self.make(
            tmp_path,
            [
                case_dict("c-0001", "test"),
                case_dict("c-0002", "dev", tags=["reworded"]),
            ],
        ).split_hash()
        assert before == after

    def test_reordering_test_cases_does_not_change_the_hash(self, tmp_path: Path) -> None:
        a = self.make(
            tmp_path, [case_dict("c-0001", "test"), case_dict("c-0002", "test")]
        ).split_hash()
        b = self.make(
            tmp_path, [case_dict("c-0002", "test"), case_dict("c-0001", "test")]
        ).split_hash()
        assert a == b

    def test_reformatting_a_line_does_not_change_the_hash(self, tmp_path: Path) -> None:
        """A prettier pass over the dataset is not test-set tampering."""
        canonical = self.make(tmp_path, [case_dict("c-0001", "test")]).split_hash()
        path = tmp_path / "reformatted.jsonl"
        loose = json.dumps(case_dict("c-0001", "test"), separators=(", ", ": "))
        path.write_text(loose, encoding="utf-8")
        assert Dataset.load(path).split_hash() == canonical

    def test_editing_a_test_label_changes_the_hash(self, tmp_path: Path) -> None:
        before = self.make(tmp_path, [case_dict("c-0001", "test")]).split_hash()
        after = self.make(
            tmp_path,
            [case_dict("c-0001", "test", expect={"route": {"label": "clarify"}})],
        ).split_hash()
        assert before != after

    def test_adding_a_test_case_changes_the_hash(self, tmp_path: Path) -> None:
        before = self.make(tmp_path, [case_dict("c-0001", "test")]).split_hash()
        after = self.make(
            tmp_path, [case_dict("c-0001", "test"), case_dict("c-0002", "test")]
        ).split_hash()
        assert before != after

    def test_deleting_a_test_case_changes_the_hash(self, tmp_path: Path) -> None:
        before = self.make(
            tmp_path, [case_dict("c-0001", "test"), case_dict("c-0002", "test")]
        ).split_hash()
        after = self.make(tmp_path, [case_dict("c-0001", "test")]).split_hash()
        assert before != after

    def test_the_hash_is_a_sha256_hex_digest(self, tmp_path: Path) -> None:
        digest = self.make(tmp_path, [case_dict("c-0001", "test")]).split_hash()
        assert digest is not None
        assert len(digest) == 64
        assert all(ch in "0123456789abcdef" for ch in digest)

    def test_a_dataset_with_no_test_cases_has_no_hash(self, tmp_path: Path) -> None:
        """Absent, not a hash of nothing: an empty digest would compare equal
        between two datasets that share no cases at all."""
        assert self.make(tmp_path, [case_dict("c-0001", "dev")]).split_hash() is None

    def test_the_helper_matches_the_dataset_method(self, tmp_path: Path) -> None:
        dataset = self.make(tmp_path, [case_dict("c-0001", "test")])
        assert split_hash(dataset.cases) == dataset.split_hash()


class TestVerifyAgainstAKnownHash:
    def test_a_matching_hash_passes(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001", "test")])
        dataset = Dataset.load(path)
        dataset.verify_split_hash(dataset.split_hash())

    def test_a_mismatched_hash_is_rejected_with_both_values(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001", "test")])
        dataset = Dataset.load(path)
        with pytest.raises(DatasetError) as exc:
            dataset.verify_split_hash("f" * 64)
        message = str(exc.value)
        assert "f" * 64 in message
        assert "suite_version" in message

    def test_verifying_none_is_a_no_op(self, tmp_path: Path) -> None:
        """No recorded hash yet is not a mismatch."""
        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001", "test")])
        Dataset.load(path).verify_split_hash(None)


class TestScale:
    def test_a_280_line_file_validates_in_under_a_second(self, tmp_path: Path) -> None:
        """The doc's target dataset size, with the doc's stated budget."""
        cases = [case_dict(f"nr-route-{i:04d}", "test") for i in range(280)]
        path = write_jsonl(tmp_path / "routing.jsonl", cases)

        started = time.perf_counter()
        dataset = Dataset.load(path)
        elapsed = time.perf_counter() - started

        assert len(dataset) == 280
        assert elapsed < 1.0, f"took {elapsed:.3f}s"

    def test_hashing_280_cases_is_fast(self, tmp_path: Path) -> None:
        cases = [case_dict(f"nr-route-{i:04d}", "test") for i in range(280)]
        dataset = Dataset.load(write_jsonl(tmp_path / "d.jsonl", cases))
        started = time.perf_counter()
        dataset.split_hash()
        assert time.perf_counter() - started < 1.0


class TestMultipleFiles:
    def test_several_files_load_as_one_dataset(self, tmp_path: Path) -> None:
        a = write_jsonl(tmp_path / "routing.jsonl", [case_dict("c-0001")])
        b = write_jsonl(tmp_path / "failure.jsonl", [case_dict("c-0002")])
        assert len(Dataset.load([a, b])) == 2

    def test_a_duplicate_across_files_in_one_suite_is_rejected(self, tmp_path: Path) -> None:
        a = write_jsonl(tmp_path / "a.jsonl", [case_dict("c-0001")])
        b = write_jsonl(tmp_path / "b.jsonl", [case_dict("c-0001")])
        with pytest.raises(DatasetError, match="duplicate"):
            Dataset.load([a, b])


class TestLookupEdgeCases:
    def test_by_id_with_an_explicit_suite(self, tmp_path: Path) -> None:
        path = write_jsonl(
            tmp_path / "d.jsonl",
            [case_dict("c-0001"), case_dict("c-0001", suite="nextrole.failure")],
        )
        dataset = Dataset.load(path)
        assert dataset.by_id("c-0001", suite="nextrole.failure").suite == "nextrole.failure"

    def test_an_ambiguous_id_without_a_suite_is_an_error(self, tmp_path: Path) -> None:
        """Silently returning the first match would score the wrong case."""
        path = write_jsonl(
            tmp_path / "d.jsonl",
            [case_dict("c-0001"), case_dict("c-0001", suite="nextrole.failure")],
        )
        with pytest.raises(KeyError, match="several suites"):
            Dataset.load(path).by_id("c-0001")

    def test_filter_returns_only_that_split(self, tmp_path: Path) -> None:
        path = write_jsonl(
            tmp_path / "d.jsonl",
            [case_dict("c-0001", "dev"), case_dict("c-0002", "test")],
        )
        assert [c.id for c in Dataset.load(path).filter("test")] == ["c-0002"]

    def test_suites_reports_what_is_present(self, tmp_path: Path) -> None:
        path = write_jsonl(
            tmp_path / "d.jsonl",
            [case_dict("c-0001"), case_dict("c-0002", suite="nextrole.failure")],
        )
        assert Dataset.load(path).suites == {"nextrole.routing", "nextrole.failure"}

    def test_a_non_object_line_is_rejected_with_its_type(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text(json.dumps(case_dict("c-0001")) + "\n[1, 2, 3]\n", encoding="utf-8")
        with pytest.raises(DatasetError) as exc:
            Dataset.load(path)
        assert "line 2" in str(exc.value)
        assert "list" in str(exc.value)

    def test_load_cases_returns_the_case_list(self, tmp_path: Path) -> None:
        from neverempty.dataset.loader import load_cases

        path = write_jsonl(tmp_path / "d.jsonl", [case_dict("c-0001")])
        assert [case.id for case in load_cases(path)] == ["c-0001"]

    def test_the_case_schema_is_generatable(self) -> None:
        from neverempty.dataset.loader import schema_dict

        assert schema_dict()["title"]


class TestProvenanceValidation:
    def test_a_malformed_source_commit_is_rejected(self, tmp_path: Path) -> None:
        path = write_jsonl(
            tmp_path / "d.jsonl",
            [
                case_dict(
                    "c-0001",
                    "test",
                    provenance={"method": "generated_from_rules", "source_commit": "not-a-sha"},
                )
            ],
        )
        with pytest.raises(DatasetError, match="source_commit"):
            Dataset.load(path)

    def test_a_malformed_scheme_hash_is_rejected(self, tmp_path: Path) -> None:
        path = write_jsonl(
            tmp_path / "d.jsonl",
            [
                case_dict(
                    "c-0001",
                    "test",
                    provenance={
                        "method": "generated_from_rules",
                        "source_commit": "a" * 40,
                        "scheme_file_sha256": "tooshort",
                    },
                )
            ],
        )
        with pytest.raises(DatasetError, match="scheme_file_sha256"):
            Dataset.load(path)
