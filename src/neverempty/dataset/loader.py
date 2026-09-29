"""Loading and validating JSONL datasets.

Two properties matter more than anything else here:

1. **Every bad line is reported, with its line number.** Fixing a 280-line
   dataset one error per run is not a workflow, and a validation message that
   does not name the line is useless at that size.
2. **The test split is hashed by content.** Editing it invalidates the
   baseline, which is what stops a quietly reworded test set from turning into
   an unexplained accuracy gain.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from neverempty.dataset.case import Case, Split


class DatasetError(ValueError):
    """A dataset failed to load or validate.

    Carries every problem found, each naming its file and line, so one run
    surfaces the whole list.
    """

    def __init__(self, message: str, *, problems: list[str] | None = None) -> None:
        super().__init__(message)
        self.problems = problems or []


def split_hash(cases: Sequence[Case], split: Split = "test") -> str | None:
    """SHA-256 over the canonical content of one split, order-independent.

    Returns ``None`` when the split holds no cases. That is deliberately not a
    hash of nothing: an empty digest would compare equal between two datasets
    that share no cases at all.

    Reordering lines or reformatting the file leaves this unchanged; editing
    any field, adding a case or deleting one changes it. That is the smallest
    definition that still catches the leakage the rule exists to prevent.
    """
    selected = sorted((case for case in cases if case.split == split), key=lambda c: c.id)
    if not selected:
        return None
    digest = hashlib.sha256()
    for case in selected:
        digest.update(case.canonical_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class Dataset:
    """A validated set of cases, loaded from one or more JSONL files."""

    def __init__(self, cases: list[Case], *, sources: list[Path] | None = None) -> None:
        self.cases = cases
        self.sources = sources or []
        self._by_id = {(case.suite, case.id): case for case in cases}

    @classmethod
    def load(
        cls,
        path: str | Path | Sequence[str | Path],
        *,
        split: Split | None = None,
    ) -> Dataset:
        """Load and validate. Fails on any bad line, naming every one.

        Args:
            path: One path, or several. Each holds one JSON object per line.
            split: Keep only this split. An absent split is an error, not an
                empty dataset: a report over zero cases is worse than a failure.
        """
        paths = [Path(p) for p in ([path] if isinstance(path, (str, Path)) else list(path))]
        cases: list[Case] = []
        problems: list[str] = []
        seen: dict[tuple[str, str], tuple[Path, int]] = {}

        for file_path in paths:
            for case, line_number, problem in _read_file(file_path):
                if problem is not None:
                    problems.append(problem)
                    continue
                assert case is not None
                key = (case.suite, case.id)
                if key in seen:
                    first_path, first_line = seen[key]
                    problems.append(
                        f"{file_path.name} line {line_number}: duplicate case id "
                        f"{case.id!r} in suite {case.suite!r}, first seen at "
                        f"{first_path.name} line {first_line}. Ids are stable "
                        f"forever and never reused."
                    )
                    continue
                seen[key] = (file_path, line_number)
                cases.append(case)

        if problems:
            raise DatasetError(
                f"{len(problems)} problem(s) in {len(paths)} file(s):\n  " + "\n  ".join(problems),
                problems=problems,
            )

        if split is not None:
            cases = [case for case in cases if case.split == split]

        if not cases:
            where = f" for split {split!r}" if split else ""
            raise DatasetError(
                f"no cases loaded{where} from {', '.join(p.name for p in paths)}: "
                f"an empty dataset would produce a report over zero cases"
            )

        return cls(cases, sources=paths)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[Case]:
        return iter(self.cases)

    def __getitem__(self, index: int) -> Case:
        return self.cases[index]

    def by_id(self, case_id: str, suite: str | None = None) -> Case:
        """One case by id. ``suite`` disambiguates when ids repeat across suites."""
        if suite is not None:
            return self._by_id[(suite, case_id)]
        matches = [case for case in self.cases if case.id == case_id]
        if not matches:
            raise KeyError(f"no case with id {case_id!r}")
        if len(matches) > 1:
            raise KeyError(
                f"case id {case_id!r} appears in several suites "
                f"({sorted({c.suite for c in matches})}); pass suite="
            )
        return matches[0]

    @property
    def splits(self) -> set[str]:
        return {case.split for case in self.cases}

    @property
    def suites(self) -> set[str]:
        return {case.suite for case in self.cases}

    def filter(self, split: Split) -> list[Case]:
        return [case for case in self.cases if case.split == split]

    def counts(self) -> dict[str, int]:
        """Cases per split, for the validate summary."""
        counts: dict[str, int] = {}
        for case in self.cases:
            counts[case.split] = counts.get(case.split, 0) + 1
        return counts

    def split_hash(self, split: Split = "test") -> str | None:
        """Content hash of one split. See :func:`split_hash`."""
        return split_hash(self.cases, split)

    def verify_split_hash(self, expected: str | None, split: Split = "test") -> None:
        """Raise when the split has changed since ``expected`` was recorded.

        Passing ``None`` is a no-op: no hash recorded yet is not a mismatch.
        """
        if expected is None:
            return
        actual = self.split_hash(split)
        if actual != expected:
            raise DatasetError(
                f"{split} split hash mismatch.\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}\n"
                f"The {split} split has changed. Editing it invalidates the "
                f"committed baseline, so bump suite_version and re-promote a "
                f"baseline rather than comparing against the old one."
            )


def _read_file(path: Path) -> Iterator[tuple[Case | None, int, str | None]]:
    """Yield ``(case, line_number, problem)`` for every non-blank line."""
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")

    # utf-8-sig tolerates the BOM Windows editors write; it is not a data error.
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise DatasetError(f"{path.name} holds no cases: the file is empty")

    if text.lstrip().startswith("["):
        raise DatasetError(
            f"{path.name} looks like a JSON array. Datasets are JSONL: one JSON "
            f"object per line, no enclosing brackets and no commas between lines."
        )

    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            yield None, line_number, f"{path.name} line {line_number}: invalid JSON ({exc.msg})"
            continue

        if not isinstance(payload, dict):
            yield (
                None,
                line_number,
                f"{path.name} line {line_number}: expected a JSON object, got "
                f"{type(payload).__name__}",
            )
            continue

        try:
            yield Case.model_validate(payload), line_number, None
        except ValidationError as exc:
            yield None, line_number, _format_validation_error(path, line_number, exc)


def _format_validation_error(path: Path, line_number: int, exc: ValidationError) -> str:
    details = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        details.append(f"{location}: {error['msg']}")
    return f"{path.name} line {line_number}: " + "; ".join(details)


def load_cases(
    path: str | Path | Sequence[str | Path], *, split: Split | None = None
) -> list[Case]:
    """Convenience wrapper returning the case list."""
    return Dataset.load(path, split=split).cases


def expand_paths(patterns: Sequence[str]) -> list[Path]:
    """Expand shell-style globs, keeping literal paths that exist.

    The shell has usually expanded these already, but not on Windows and not
    when a pattern is quoted, so the CLI cannot rely on it.
    """
    found: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_file():
            found.append(candidate)
            continue
        # Anchor the glob at the deepest non-magic parent so absolute patterns work.
        anchor = candidate.anchor or "."
        relative = str(candidate)[len(candidate.anchor) :] if candidate.anchor else pattern
        found.extend(sorted(Path(anchor).glob(relative)))
    unique: list[Path] = []
    for item in found:
        if item not in unique and item.is_file():
            unique.append(item)
    return unique


def schema_dict() -> dict[str, Any]:
    """Case v1 as a JSON Schema, for the committed contract file."""
    schema: dict[str, Any] = Case.model_json_schema(mode="validation")
    return schema


__all__ = [
    "Dataset",
    "DatasetError",
    "expand_paths",
    "load_cases",
    "schema_dict",
    "split_hash",
]
