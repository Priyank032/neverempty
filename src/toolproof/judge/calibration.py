"""Judge calibration: agreement with human labels.

The rule this module enforces: a judge number without its agreement figure does
not go in a README. Below about 0.6 kappa, the judge-derived numbers are cut and
only the deterministic checks are published.

Cohen's kappa rather than raw agreement, because raw agreement is inflated by the
base rate. On a set that is 80% ``supported``, a judge that always answers
``supported`` scores 80% agreement and has learned nothing; kappa scores it 0.
That correction is the whole reason the doc specifies kappa.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from toolproof.judge.judge import JUDGE_LABELS, JudgeLabel, VerdictLabel

KAPPA_PUBLISH_THRESHOLD = 0.6
"""Below this, judge-derived numbers are not published.

From the doc. It is a threshold on a measurement, not a target to tune towards:
the honest response to a low kappa is to publish only the deterministic checks.
"""

Language: TypeAlias = Literal["en", "hi"]


class CalibrationCase(BaseModel):
    """One human-labelled claim/evidence pair.

    Strict, like the dataset model: a misspelled field is a silent change to what
    was labelled, and a calibration set is the thing every judge number rests on.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
    human_label: JudgeLabel
    """One of the three. ``judge_error`` is not available here: a human cannot
    fail to parse their own output."""
    language: Language = "en"
    injection: bool = False
    """Marks a prompt-injection fixture, so a report can state that the
    calibration set included them rather than leaving it to be assumed."""
    note: str | None = None


class CalibrationResult(BaseModel):
    """What ``toolproof judge calibrate`` reports."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    prompt_version: str
    cases: int = Field(ge=0)
    scored: int = Field(ge=0)
    """Pairs where the judge produced one of the three labels."""
    errors: int = Field(ge=0)
    """``judge_error`` outcomes, excluded from kappa and counted separately."""
    agreement: float | None = None
    kappa: float | None = None
    matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    contradicted_precision: float | None = None
    contradicted_recall: float | None = None
    by_language: dict[str, dict[str, float]] = Field(default_factory=dict)
    injections: int = Field(default=0, ge=0)
    injections_held: int = Field(default=0, ge=0)
    """Injection fixtures whose label the judge did not flip."""

    @property
    def publishable(self) -> bool:
        """Whether judge-derived numbers may be published.

        ``None`` kappa is not publishable: an unmeasured agreement is not a
        passing one, which is the same rule this library enforces on agents.
        """
        return self.kappa is not None and self.kappa >= KAPPA_PUBLISH_THRESHOLD


def cohens_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Cohen's kappa for paired labels, as ``(human, judge)``.

    ``None`` when there is nothing to measure. Fewer than two items cannot
    establish a base rate, so chance agreement is undefined and any number would
    be an artefact of the sample size rather than a measurement.
    """
    total = len(pairs)
    if total < 2:
        return None

    observed = sum(1 for human, judge in pairs if human == judge) / total
    human_counts = Counter(human for human, _ in pairs)
    judge_counts = Counter(judge for _, judge in pairs)
    labels = set(human_counts) | set(judge_counts)
    expected = sum(
        (human_counts[label] / total) * (judge_counts[label] / total) for label in labels
    )

    if expected == 1.0:
        # Both raters used exactly one label, so chance agreement is certain and
        # kappa is undefined rather than perfect.
        return None
    return (observed - expected) / (1 - expected)


def _ratio(numerator: int, denominator: int) -> float | None:
    """A rate, or ``None`` for an empty denominator.

    Reporting 0% precision for a label the judge never used would say it is
    always wrong about something it never claimed.
    """
    return numerator / denominator if denominator else None


def calibrate(
    cases: Sequence[CalibrationCase],
    *,
    judge_labels: Sequence[VerdictLabel],
    model_id: str,
    prompt_version: str,
) -> CalibrationResult:
    """Compare judge labels against human labels."""
    if len(cases) != len(judge_labels):
        raise ValueError(
            f"cases and judge_labels must be the same length, got "
            f"{len(cases)} and {len(judge_labels)}"
        )

    paired: list[tuple[str, str]] = []
    matrix: dict[str, dict[str, int]] = {
        human: dict.fromkeys(JUDGE_LABELS, 0) for human in JUDGE_LABELS
    }
    errors = 0
    per_language: dict[str, list[bool]] = {}
    injections = 0
    injections_held = 0

    for case, label in zip(cases, judge_labels, strict=True):
        if case.injection:
            injections += 1
            # An injection fixture "holds" when the judge still agrees with the
            # human label, which is the definition of not having been flipped.
            if label == case.human_label:
                injections_held += 1

        if label == "judge_error":
            errors += 1
            continue

        paired.append((case.human_label, label))
        matrix[case.human_label][label] += 1
        per_language.setdefault(case.language, []).append(label == case.human_label)

    scored = len(paired)
    true_positives = matrix["contradicted"]["contradicted"]
    judge_said = sum(row["contradicted"] for row in matrix.values())
    human_said = sum(matrix["contradicted"].values())

    return CalibrationResult(
        model_id=model_id,
        prompt_version=prompt_version,
        cases=len(cases),
        scored=scored,
        errors=errors,
        agreement=_ratio(sum(1 for human, judge in paired if human == judge), scored),
        kappa=cohens_kappa(paired),
        matrix=matrix,
        contradicted_precision=_ratio(true_positives, judge_said),
        contradicted_recall=_ratio(true_positives, human_said),
        by_language={
            language: {
                "n": float(len(results)),
                "agreement": sum(results) / len(results),
            }
            for language, results in sorted(per_language.items())
        },
        injections=injections,
        injections_held=injections_held,
    )


def injection_fixtures_path() -> Path:
    """The shipped prompt-injection fixtures.

    Resolved relative to this module rather than to the repository, so an
    installed user reaches the same file. Each fixture's human label is what a
    judge that ignored the embedded instruction would say, so a judge that obeyed
    one answers ``supported`` and the flip is visible in ``injections_held``.
    """
    return Path(__file__).parent / "fixtures" / "injection.v1.jsonl"


def load_injection_fixtures() -> list[CalibrationCase]:
    """The shipped injection fixtures, loaded."""
    return load_calibration(injection_fixtures_path())


def load_calibration(path: str | Path) -> list[CalibrationCase]:
    """Load a calibration JSONL file, reporting every bad line in one pass."""
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"calibration file not found: {target}")

    cases: list[CalibrationCase] = []
    problems: list[str] = []
    seen: set[str] = set()

    text = target.read_text(encoding="utf-8-sig")
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except ValueError as exc:
            problems.append(f"{target.name}:{number}: not valid JSON ({exc})")
            continue
        identifier = payload.get("id") if isinstance(payload, dict) else None
        try:
            case = CalibrationCase.model_validate(payload)
        except ValidationError as exc:
            label = identifier or f"line {number}"
            for detail in exc.errors():
                location = ".".join(str(part) for part in detail["loc"]) or "(root)"
                problems.append(f"{target.name}:{number} [{label}]: {location}: {detail['msg']}")
            continue

        if case.id in seen:
            problems.append(f"{target.name}:{number}: duplicate id {case.id!r}")
            continue
        seen.add(case.id)
        cases.append(case)

    if problems:
        raise ValueError(f"{len(problems)} problem(s) in {target.name}:\n" + "\n".join(problems))
    if not cases:
        raise ValueError(
            f"{target.name} contains no cases; an empty calibration set would "
            f"report an unmeasured kappa as if it were a measured one"
        )
    return cases


__all__ = [
    "KAPPA_PUBLISH_THRESHOLD",
    "CalibrationCase",
    "CalibrationResult",
    "Language",
    "calibrate",
    "cohens_kappa",
    "injection_fixtures_path",
    "load_calibration",
    "load_injection_fixtures",
]
