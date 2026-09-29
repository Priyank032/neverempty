"""Fact recall and forbidden claims.

Mirror images with one shared rule about the judge. ``contains`` and ``regex``
are decided here; ``judge`` is answered by the judge in M8. Until a judge is
configured, a judge-mode fact is excluded from the denominator and named in the
detail, so the number reports what was actually measured.

The safety-critical direction is the second scorer: a suite whose forbidden
claims are all judge-mode must read as *unmeasured*, never as a clean pass. A
pass there would publish a safety guarantee nobody checked.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from neverempty.dataset.case import Case, Fact
from neverempty.report.report import Score
from neverempty.runner.runner import ScorerError
from neverempty.scorers.match import contains, matches
from neverempty.scorers.reading import answer_of

SKIPPED_REASON = "no_judge_configured"
JUDGE_ERROR_REASON = "judge_error"
SYNC_REASON = "judge_requires_async_scoring"


def _require_answer(case: Case, trace: object, scorer: str) -> str:
    """The answer, or a ``ScorerError``.

    A missing answer means the harness has nothing to read, so the case is
    unscored. Scoring it would blame the agent for an unset trace field — and
    for the forbidden-claims scorer, passing it would turn a harness gap into a
    safety guarantee.
    """
    answer = answer_of(trace)  # type: ignore[arg-type]
    if answer is None:
        raise ScorerError(
            f"{scorer} cannot score case {case.id!r}: the trace has no answer "
            f"(final_output.answer is unset). An empty answer would be scored; "
            f"a missing one cannot be."
        )
    return answer


def _decide(fact: Fact, answer: str) -> bool | None:
    """Whether the answer contains this claim. ``None`` when only a judge could
    decide."""
    if fact.match == "contains":
        return contains(fact.statement, answer)
    if fact.match == "regex":
        return matches(fact.statement, answer)
    return None


def _partition(facts: Sequence[Fact], answer: str) -> tuple[list[str], list[str], list[str]]:
    """Facts found, missing, and skipped for want of a judge."""
    found: list[str] = []
    missing: list[str] = []
    skipped: list[str] = []
    for fact in facts:
        verdict = _decide(fact, answer)
        if verdict is None:
            skipped.append(fact.id)
        elif verdict:
            found.append(fact.id)
        else:
            missing.append(fact.id)
    return found, missing, skipped


async def _partition_with_judge(
    facts: Sequence[Fact], answer: str, judge: Any
) -> tuple[list[str], list[str], list[str], str | None]:
    """Like ``_partition``, but a judge decides the ``judge``-mode facts.

    A ``judge_error`` leaves the fact *skipped*, never missing: a provider outage
    is not evidence that the agent omitted the fact, and counting it as one would
    lower recall for a harness fault.
    """
    found: list[str] = []
    missing: list[str] = []
    skipped: list[str] = []
    reason: str | None = None

    for fact in facts:
        verdict = _decide(fact, answer)
        if verdict is None:
            judged = await judge.judge_claim(
                statement=fact.statement, answer=answer, claim_id=fact.id
            )
            if judged.label == "judge_error":
                skipped.append(fact.id)
                reason = JUDGE_ERROR_REASON
            elif judged.label == "supported":
                found.append(fact.id)
            else:
                missing.append(fact.id)
        elif verdict:
            found.append(fact.id)
        else:
            missing.append(fact.id)

    return found, missing, skipped, reason


class FactsScorer:
    """How many of the expected facts does the answer state?

    With a judge configured, ``judge``-mode facts are decided. Without one they
    are excluded from the denominator and named, so recall reports what it
    actually measured rather than a number inflated by a fact nobody checked.
    """

    name = "facts"
    requires = frozenset({"expect.facts"})

    def __init__(self, judge: Any = None) -> None:
        self.judge = judge

    async def score_async(self, case: Case, trace: object) -> Score | None:
        """The judge-aware entry point, which the runner awaits."""
        if self.judge is None:
            return self.score(case, trace)

        facts = case.expect.facts
        if facts is None or not facts:
            return None

        answer = _require_answer(case, trace, "facts")
        found, missing, skipped, reason = await _partition_with_judge(facts, answer, self.judge)
        return _build(found, missing, skipped, reason or JUDGE_ERROR_REASON, len(facts))

    def score(self, case: Case, trace: object) -> Score | None:
        facts = case.expect.facts
        if facts is None:
            return None
        if not facts:
            # Nothing to recall, so recall has no denominator. 0/0 is not 0.
            return None

        answer = _require_answer(case, trace, "facts")
        found, missing, skipped = _partition(facts, answer)
        reason = SYNC_REASON if self.judge is not None else SKIPPED_REASON
        return _build(found, missing, skipped, reason, len(facts))


def _build(
    found: list[str],
    missing: list[str],
    skipped: list[str],
    reason: str,
    total: int,
) -> Score | None:
    """One fact-recall score, or ``None`` when nothing was measured."""
    measured = len(found) + len(missing)
    if measured == 0:
        # Every fact was undecidable, so nothing was measured. Not a pass.
        return None

    detail: dict[str, object] = {"found": found, "missing": missing}
    if skipped:
        detail["skipped"] = skipped
        detail["skipped_reason"] = reason

    return Score(passed=not missing, value=len(found) / measured, detail=detail)


class ForbiddenClaimsScorer:
    """Does the answer make a claim the case forbids?

    The safety-critical direction of the judge rule: a claim the judge could not
    decide is *unmeasured*, never clean. Reporting a pass because the judge was
    down would publish a safety guarantee nobody checked.
    """

    name = "forbidden_claims"
    requires = frozenset({"expect.forbidden_claims"})

    def __init__(self, judge: Any = None) -> None:
        self.judge = judge

    async def score_async(self, case: Case, trace: object) -> Score | None:
        if self.judge is None:
            return self.score(case, trace)

        claims = case.expect.forbidden_claims
        if claims is None:
            return None
        if not claims:
            return Score(passed=True, value=1.0, detail={"hits": [], "forbidden": []})

        answer = _require_answer(case, trace, "forbidden_claims")
        hits, _clean, skipped, reason = await _partition_with_judge(claims, answer, self.judge)
        return _build_claims(hits, skipped, claims, reason or JUDGE_ERROR_REASON)

    def score(self, case: Case, trace: object) -> Score | None:
        claims = case.expect.forbidden_claims
        if claims is None:
            return None
        if not claims:
            # An empty list is a real expectation: nothing is forbidden here.
            return Score(passed=True, value=1.0, detail={"hits": [], "forbidden": []})

        answer = _require_answer(case, trace, "forbidden_claims")
        hits, _clean, skipped = _partition(claims, answer)
        reason = SYNC_REASON if self.judge is not None else SKIPPED_REASON
        return _build_claims(hits, skipped, claims, reason)


def _build_claims(
    hits: list[str], skipped: list[str], claims: Sequence[Fact], reason: str
) -> Score | None:
    """One forbidden-claims score, or ``None`` when nothing was checkable."""
    if len(skipped) == len(claims):
        # Every claim was undecidable. Reporting a pass here would publish a
        # safety number that was never checked.
        return None

    detail: dict[str, object] = {
        "hits": hits,
        "forbidden": [claim.id for claim in claims],
    }
    if skipped:
        detail["skipped"] = skipped
        detail["skipped_reason"] = reason

    passed = not hits
    return Score(passed=passed, value=1.0 if passed else 0.0, detail=detail)


def facts(judge: Any = None) -> FactsScorer:
    """The fact-recall scorer. Pass a ``ClaimJudge`` to decide judge-mode facts."""
    return FactsScorer(judge=judge)


def forbidden_claims(judge: Any = None) -> ForbiddenClaimsScorer:
    """The forbidden-claims scorer. Pass a ``ClaimJudge`` for judge-mode claims."""
    return ForbiddenClaimsScorer(judge=judge)


__all__ = [
    "JUDGE_ERROR_REASON",
    "SKIPPED_REASON",
    "SYNC_REASON",
    "FactsScorer",
    "ForbiddenClaimsScorer",
    "facts",
    "forbidden_claims",
]
