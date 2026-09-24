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

from toolproof.dataset.case import Case, Fact
from toolproof.report.report import Score
from toolproof.runner.runner import ScorerError
from toolproof.scorers.match import contains, matches
from toolproof.scorers.reading import answer_of

SKIPPED_REASON = "no_judge_configured"


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


class FactsScorer:
    """How many of the expected facts does the answer state?"""

    name = "facts"
    requires = frozenset({"expect.facts"})

    def score(self, case: Case, trace: object) -> Score | None:
        facts = case.expect.facts
        if facts is None:
            return None
        if not facts:
            # Nothing to recall, so recall has no denominator. 0/0 is not 0.
            return None

        answer = _require_answer(case, trace, "facts")
        found, missing, skipped = _partition(facts, answer)

        measured = len(found) + len(missing)
        if measured == 0:
            # Every fact needs a judge, so nothing was measured. Not a pass.
            return None

        detail: dict[str, object] = {"found": found, "missing": missing}
        if skipped:
            detail["skipped"] = skipped
            detail["skipped_reason"] = SKIPPED_REASON

        return Score(passed=not missing, value=len(found) / measured, detail=detail)


class ForbiddenClaimsScorer:
    """Does the answer make a claim the case forbids?"""

    name = "forbidden_claims"
    requires = frozenset({"expect.forbidden_claims"})

    def score(self, case: Case, trace: object) -> Score | None:
        claims = case.expect.forbidden_claims
        if claims is None:
            return None
        if not claims:
            # An empty list is a real expectation: nothing is forbidden here.
            return Score(passed=True, value=1.0, detail={"hits": [], "forbidden": []})

        answer = _require_answer(case, trace, "forbidden_claims")
        hits, _clean, skipped = _partition(claims, answer)

        if len(skipped) == len(claims):
            # Every forbidden claim needs a judge. Reporting a pass here would
            # publish a safety number that was never checked.
            return None

        detail: dict[str, object] = {
            "hits": hits,
            "forbidden": [claim.id for claim in claims],
        }
        if skipped:
            detail["skipped"] = skipped
            detail["skipped_reason"] = SKIPPED_REASON

        passed = not hits
        return Score(passed=passed, value=1.0 if passed else 0.0, detail=detail)


def facts() -> FactsScorer:
    """The fact-recall scorer."""
    return FactsScorer()


def forbidden_claims() -> ForbiddenClaimsScorer:
    """The forbidden-claims scorer."""
    return ForbiddenClaimsScorer()


__all__ = ["FactsScorer", "ForbiddenClaimsScorer", "facts", "forbidden_claims"]
