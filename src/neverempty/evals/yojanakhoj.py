"""YojanaKhoj consistency: does the explanation agree with the rule engine?

The ground truth here is generated, not hand-labelled: a rule engine decides
eligibility deterministically, and the question is whether the LLM's yes/no and
its prose agree with that decision. Rules are ground truth for *consistency*, not
for real-world eligibility, and the README has to say so -- a scheme's actual
rules may be wrong, and this measures agreement, not correctness.

``evaluateRule`` in that repo returns ``null`` when a field is missing, which is a
genuine three-way value. That is the interesting case: the model is being asked
about something the rules could not decide, and stating a definite yes or no
there is an overclaim regardless of which way it leans.

Two things this module refuses to score, both for the same reason:

- An item whose LLM call failed. Those paths return a canned bilingual
  explanation and a ``llm_confidence`` that is either a rescaled ranking score or
  a hardcoded constant. Scoring that text as a model answer would measure the
  fallback string, and scoring that number as a confidence would measure the
  score function.
- An item the rules could not evaluate, when asking whether the verdict is
  *correct*. It is still scored for overclaiming, which is the separate and more
  interesting question.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from neverempty.dataset.case import Case
from neverempty.report.report import Score
from neverempty.runner.runner import ScorerError
from neverempty.scorers.reading import structured_of

Verdict: TypeAlias = Literal["yes", "no", "check"]

VERDICTS: tuple[str, ...] = ("yes", "no", "check")
"""The LLM's three-way eligibility field.

Worth stating explicitly because the design doc assumed this was a boolean and
predicted that the schema could not express "cannot evaluate". It can: the
matcher declares ``enum: ['yes', 'no', 'check']`` and its prompt says to use
``check`` when unsure. So the open question is not whether the model *can* say
"cannot evaluate" but whether it *does*, which is what CATEGORY_OVERCLAIM
measures.
"""

CATEGORY_VERDICT = "verdict_contradicts_rule"
CATEGORY_OVERCLAIM = "overclaim_on_null"
CATEGORY_REASON = "reason_contradicts_trace"
CATEGORY_LANGUAGE = "explanations_disagree"
CATEGORY_CONFIDENCE = "high_confidence_on_null"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_VERDICT,
    CATEGORY_OVERCLAIM,
    CATEGORY_REASON,
    CATEGORY_LANGUAGE,
    CATEGORY_CONFIDENCE,
)
"""The doc's five consistency categories, each with its own denominator.

Reported separately rather than pooled, because they are structurally different.
The hard rule filter removes rule-false schemes before the LLM sees them, so
verdict contradictions on ineligible schemes are near zero by construction; the
null cases and the zero-match soft path are where the real risk lives. Pooling
them would let the protected categories hide the exposed ones.
"""

HIGH_CONFIDENCE = 80
"""``llm_confidence`` at or above this is a definite claim, on a 0-100 scale.

The matcher declares ``integer, minimum 0, maximum 100``, so this is on that
scale and not a probability. Converting it to one here would invent a precision
the field does not have.
"""

DEGRADED_MARKERS: tuple[str, ...] = (
    "verify at nearest jan seva kendra",
    "jan seva kendra",
)
"""Text that identifies the LLM-degraded fallback.

The fallback paths emit a fixed bilingual string rather than a model answer. An
item carrying it is excluded from scoring entirely: measuring a canned string
would be measuring the outage, and the doc's export contract requires the same
exclusion on the Node side. Matched case-insensitively against the reason text.
"""


@dataclass(frozen=True)
class ItemVerdict:
    """One scheme's LLM output, alongside what the rules decided."""

    item_id: str
    rule_result: bool | None
    verdict: str | None
    reason: str | None
    explanation_en: str | None
    explanation_hi: str | None
    confidence: int | None
    llm_failed: bool = False

    @property
    def scorable(self) -> bool:
        """False when the LLM did not actually answer.

        A failed call is an outage, not a wrong answer, and folding it in either
        direction would misreport the model.
        """
        return not self.llm_failed and self.verdict is not None


def _is_degraded(item: Mapping[str, Any]) -> bool:
    if item.get("llm_failed") is True or item.get("llm_degraded") is True:
        return True
    reason = item.get("eligibility_reason") or item.get("reason") or ""
    lowered = str(reason).casefold()
    return any(marker in lowered for marker in DEGRADED_MARKERS)


def read_items(structured: Mapping[str, Any] | None) -> list[ItemVerdict]:
    """The per-scheme LLM output from a trace's structured block.

    Defensive by design: this data crosses a language boundary, and a missing
    field has to read as "not stated" rather than raising halfway through a run.
    """
    if not structured:
        return []
    raw = structured.get("items") or structured.get("ranked") or []
    if not isinstance(raw, Sequence):
        return []

    items: list[ItemVerdict] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        item_id = entry.get("item_id") or entry.get("schemeId") or entry.get("scheme_id")
        if not isinstance(item_id, str) or not item_id:
            continue
        verdict = entry.get("eligibility_yes_no") or entry.get("verdict")
        confidence = entry.get("llm_confidence")
        items.append(
            ItemVerdict(
                item_id=item_id,
                rule_result=_tri(entry.get("rule_result")),
                verdict=str(verdict) if isinstance(verdict, str) else None,
                reason=_text(entry.get("eligibility_reason") or entry.get("reason")),
                explanation_en=_text(entry.get("explanation_en")),
                explanation_hi=_text(entry.get("explanation_hi")),
                confidence=_int(confidence),
                llm_failed=_is_degraded(entry),
            )
        )
    return items


def _tri(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _int(value: Any) -> int | None:
    """An integer confidence, or None.

    A boolean is rejected even though Python makes ``True == 1``: an item
    carrying ``llm_confidence: true`` has a bug, and reading it as 1 on a 0-100
    scale would bury it as merely very low confidence.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


class ConsistencyScorer:
    """Does the LLM's verdict agree with what the rules decided?

    One score per case, whose value is the fraction of scorable items that were
    consistent, plus a per-category breakdown in the detail. ``passed`` is the
    all-items-consistent verdict, so a case with any contradiction fails.
    """

    name = "consistency"
    requires = frozenset({"expect.items"})

    def __init__(self, *, high_confidence: int = HIGH_CONFIDENCE) -> None:
        self.high_confidence = high_confidence

    def score(self, case: Case, trace: object) -> Score | None:
        expected = case.expect.items
        if not expected:
            return None

        observed = {item.item_id: item for item in read_items(structured_of(trace))}  # type: ignore[arg-type]
        if not observed:
            raise ScorerError(
                f"case {case.id!r} expects {len(expected)} item(s) but the trace "
                f"carried no structured items. The case is unscored, which the "
                f"gate treats as a failure of the run rather than of the agent."
            )

        findings: dict[str, list[str]] = {category: [] for category in CATEGORIES}
        denominators: dict[str, int] = dict.fromkeys(CATEGORIES, 0)
        scored = 0
        consistent = 0
        skipped: list[str] = []

        for expectation in expected:
            item = observed.get(expectation.item_id)
            if item is None or not item.scorable:
                # A missing or failed item is unmeasured, never counted wrong.
                skipped.append(expectation.item_id)
                continue

            scored += 1
            problems = self._check(expectation.rule_result, item, findings, denominators)
            if not problems:
                consistent += 1

        if scored == 0:
            # Every item failed or was absent: nothing was measured, so this is
            # not-applicable rather than a zero consistency rate.
            return None

        rates = {
            category: (len(findings[category]) / denominators[category])
            for category in CATEGORIES
            if denominators[category] > 0
        }

        return Score(
            passed=consistent == scored,
            value=consistent / scored,
            detail={
                "items_expected": len(expected),
                "items_scored": scored,
                "items_skipped": skipped,
                "consistent": consistent,
                "categories": {c: findings[c] for c in CATEGORIES if findings[c]},
                "denominators": {c: n for c, n in denominators.items() if n},
                "rates": rates,
            },
        )

    def _check(
        self,
        rule_result: bool | None,
        item: ItemVerdict,
        findings: dict[str, list[str]],
        denominators: dict[str, int],
    ) -> list[str]:
        problems: list[str] = []

        if rule_result is None:
            # The rules could not decide. A definite yes or no is an overclaim;
            # 'check' is the correct answer.
            denominators[CATEGORY_OVERCLAIM] += 1
            if item.verdict in {"yes", "no"}:
                findings[CATEGORY_OVERCLAIM].append(item.item_id)
                problems.append(CATEGORY_OVERCLAIM)

            if item.confidence is not None:
                denominators[CATEGORY_CONFIDENCE] += 1
                if item.confidence >= self.high_confidence:
                    findings[CATEGORY_CONFIDENCE].append(item.item_id)
                    problems.append(CATEGORY_CONFIDENCE)
        else:
            # The rules decided, so 'check' is not wrong -- it is a refusal to
            # commit, which is not a contradiction. Only the opposite verdict is.
            denominators[CATEGORY_VERDICT] += 1
            contradicts = (rule_result and item.verdict == "no") or (
                not rule_result and item.verdict == "yes"
            )
            if contradicts:
                findings[CATEGORY_VERDICT].append(item.item_id)
                problems.append(CATEGORY_VERDICT)

        if item.explanation_en is not None and item.explanation_hi is not None:
            denominators[CATEGORY_LANGUAGE] += 1
            if _polarity(item.explanation_en) != _polarity(item.explanation_hi):
                findings[CATEGORY_LANGUAGE].append(item.item_id)
                problems.append(CATEGORY_LANGUAGE)

        return problems


_NEGATIVE_EN = ("not eligible", "do not qualify", "does not qualify", "cannot", "ineligible")
_POSITIVE_EN = ("are eligible", "you qualify", "is eligible", "eligible for")
_NEGATIVE_HI = ("पात्र नहीं", "योग्य नहीं", "नहीं है")
_POSITIVE_HI = ("पात्र हैं", "पात्र है", "योग्य हैं")


def _polarity(text: str) -> str:
    """Whether an explanation reads as eligible, ineligible, or neither.

    Deliberately crude and deliberately three-valued. This is not a semantic
    judgement: it detects the case where one language says eligible and the
    other says not, which is the doc's cross-language consistency category. An
    explanation matching neither list returns ``unknown``, and two ``unknown``
    explanations agree -- the check must not invent a disagreement out of prose
    it cannot read.
    """
    lowered = text.casefold()
    negative = any(marker in lowered for marker in _NEGATIVE_EN) or any(
        marker in text for marker in _NEGATIVE_HI
    )
    positive = any(marker in lowered for marker in _POSITIVE_EN) or any(
        marker in text for marker in _POSITIVE_HI
    )
    if negative and not positive:
        return "negative"
    if positive and not negative:
        return "positive"
    return "unknown"


def consistency(*, high_confidence: int = HIGH_CONFIDENCE) -> ConsistencyScorer:
    """The YojanaKhoj consistency scorer."""
    return ConsistencyScorer(high_confidence=high_confidence)


__all__ = [
    "CATEGORIES",
    "CATEGORY_CONFIDENCE",
    "CATEGORY_LANGUAGE",
    "CATEGORY_OVERCLAIM",
    "CATEGORY_REASON",
    "CATEGORY_VERDICT",
    "DEGRADED_MARKERS",
    "HIGH_CONFIDENCE",
    "VERDICTS",
    "ConsistencyScorer",
    "ItemVerdict",
    "consistency",
    "read_items",
]
