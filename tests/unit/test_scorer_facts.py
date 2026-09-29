"""The facts and forbidden-claims scorers.

Acceptance row: "not-applicable never scores as fail".

Fact recall is expected facts found over expected facts, matched by ``contains``
or ``regex``. ``judge`` is a third mode the judge answers in M8; until then a
judge fact is excluded from the denominator and named in the detail, so recall
reports what it actually measured rather than a number inflated or deflated by
a fact nobody checked.

Forbidden claims are the mirror image and collapse by any-hit: one absence claim
in three repeats is a finding, not noise.
"""

from __future__ import annotations

from typing import Any

import pytest

from neverempty import ScorerError, scorers
from neverempty.dataset.case import Case

from ._scoring import trace


def facts_case(*facts: dict[str, Any]) -> Case:
    return Case.model_validate(
        {
            "id": "nr-fact-0001",
            "suite": "nextrole.routing",
            "split": "dev",
            "input": {"messages": [{"role": "user", "content": "find me a job"}]},
            "expect": {"facts": list(facts)},
        }
    )


def claims_case(*claims: dict[str, Any]) -> Case:
    return Case.model_validate(
        {
            "id": "nr-claim-0001",
            "suite": "nextrole.failure",
            "split": "dev",
            "input": {"messages": [{"role": "user", "content": "find me a job"}]},
            "expect": {"forbidden_claims": list(claims)},
        }
    )


def score_one(subject_case: Case, subject_trace: Any, scorer: Any) -> Any:
    verdict = scorer.score(subject_case, subject_trace)
    assert verdict is not None
    return verdict


class TestFactRecall:
    def test_a_contained_statement_is_found(self) -> None:
        verdict = score_one(
            facts_case({"id": "f1", "statement": "Pune", "match": "contains"}),
            trace(answer="I found 4 roles in Pune."),
            scorers.facts(),
        )
        assert verdict.passed is True
        assert verdict.value == 1.0

    def test_contains_is_case_insensitive(self) -> None:
        """A fact is about content, not about how the model capitalised it."""
        verdict = score_one(
            facts_case({"id": "f1", "statement": "pune", "match": "contains"}),
            trace(answer="I found 4 roles in PUNE."),
            scorers.facts(),
        )
        assert verdict.passed is True

    def test_a_missing_statement_is_not_found(self) -> None:
        verdict = score_one(
            facts_case({"id": "f1", "statement": "Pune", "match": "contains"}),
            trace(answer="I found 4 roles in Mumbai."),
            scorers.facts(),
        )
        assert verdict.passed is False
        assert verdict.value == 0.0

    def test_recall_is_the_fraction_found(self) -> None:
        verdict = score_one(
            facts_case(
                {"id": "f1", "statement": "Pune", "match": "contains"},
                {"id": "f2", "statement": "remote", "match": "contains"},
                {"id": "f3", "statement": "Razorpay", "match": "contains"},
            ),
            trace(answer="Two remote roles in Pune."),
            scorers.facts(),
        )
        assert verdict.value == pytest.approx(2 / 3)
        assert verdict.passed is False

    def test_all_facts_found_passes(self) -> None:
        verdict = score_one(
            facts_case(
                {"id": "f1", "statement": "Pune", "match": "contains"},
                {"id": "f2", "statement": "remote", "match": "contains"},
            ),
            trace(answer="Two remote roles in Pune."),
            scorers.facts(),
        )
        assert verdict.passed is True
        assert verdict.value == 1.0

    def test_a_regex_fact_is_matched_as_a_pattern(self) -> None:
        verdict = score_one(
            facts_case({"id": "f1", "statement": r"\b\d+ roles?\b", "match": "regex"}),
            trace(answer="I found 4 roles."),
            scorers.facts(),
        )
        assert verdict.passed is True

    def test_a_non_matching_regex_fact_is_not_found(self) -> None:
        verdict = score_one(
            facts_case({"id": "f1", "statement": r"^\d+$", "match": "regex"}),
            trace(answer="I found four roles."),
            scorers.facts(),
        )
        assert verdict.passed is False

    def test_the_per_fact_breakdown_is_reported(self) -> None:
        verdict = score_one(
            facts_case(
                {"id": "f1", "statement": "Pune", "match": "contains"},
                {"id": "f2", "statement": "Razorpay", "match": "contains"},
            ),
            trace(answer="Roles in Pune."),
            scorers.facts(),
        )
        assert verdict.detail["found"] == ["f1"]
        assert verdict.detail["missing"] == ["f2"]

    def test_a_hindi_fact_is_found_in_a_hindi_answer(self) -> None:
        verdict = score_one(
            facts_case({"id": "f1", "statement": "नौकरी", "match": "contains"}),
            trace(answer="मुझे तीन नौकरी मिलीं।"),
            scorers.facts(),
        )
        assert verdict.passed is True

    def test_a_hindi_fact_matches_across_nfkc_equivalent_forms(self) -> None:
        """Same rule as the arguments scorer: an encoding difference between a
        human label and a model answer is not a recall failure."""
        verdict = score_one(
            facts_case({"id": "f1", "statement": "ड़", "match": "contains"}),
            trace(answer="ड़ है"),
            scorers.facts(),
        )
        assert verdict.passed is True


class TestNoAnswer:
    def test_an_absent_answer_raises_rather_than_scoring_zero(self) -> None:
        """No answer means the harness has nothing to read. Scoring that as 0%
        recall would blame the agent for a missing trace field."""
        with pytest.raises(ScorerError, match="no answer"):
            scorers.facts().score(
                facts_case({"id": "f1", "statement": "Pune", "match": "contains"}), trace()
            )

    def test_an_empty_answer_is_a_real_answer_and_scores_zero(self) -> None:
        """An agent that replied with nothing did reply. That is a finding, and
        it is distinguishable from a missing field only because one is ``None``
        and the other is ``""``."""
        verdict = score_one(
            facts_case({"id": "f1", "statement": "Pune", "match": "contains"}),
            trace(answer=""),
            scorers.facts(),
        )
        assert verdict.passed is False
        assert verdict.value == 0.0


class TestJudgeFacts:
    def test_a_judge_fact_is_excluded_from_recall_and_named(self) -> None:
        """M8 supplies the judge. Until then a judge fact is unmeasured, so it
        leaves the denominator rather than counting as found or missing."""
        verdict = score_one(
            facts_case(
                {"id": "f1", "statement": "Pune", "match": "contains"},
                {"id": "f2", "statement": "The tone is professional", "match": "judge"},
            ),
            trace(answer="Roles in Pune."),
            scorers.facts(),
        )
        assert verdict.value == 1.0
        assert verdict.detail["skipped"] == ["f2"]
        assert verdict.detail["skipped_reason"] == "no_judge_configured"

    def test_a_case_of_only_judge_facts_is_not_applicable(self) -> None:
        """Nothing was measured, so there is no number to report. Not a pass."""
        verdict = scorers.facts().score(
            facts_case({"id": "f1", "statement": "The tone is professional", "match": "judge"}),
            trace(answer="Roles in Pune."),
        )
        assert verdict is None

    def test_deterministic_facts_still_fail_alongside_a_skipped_judge_fact(self) -> None:
        verdict = score_one(
            facts_case(
                {"id": "f1", "statement": "Razorpay", "match": "contains"},
                {"id": "f2", "statement": "The tone is professional", "match": "judge"},
            ),
            trace(answer="Roles in Pune."),
            scorers.facts(),
        )
        assert verdict.passed is False
        assert verdict.detail["skipped"] == ["f2"]


class TestForbiddenClaims:
    def test_an_answer_making_no_forbidden_claim_passes(self) -> None:
        verdict = score_one(
            claims_case({"id": "absence", "statement": "no matching jobs", "match": "contains"}),
            trace(answer="The job search tool failed, so I could not check."),
            scorers.forbidden_claims(),
        )
        assert verdict.passed is True
        assert verdict.detail["hits"] == []

    def test_an_answer_making_a_forbidden_claim_fails(self) -> None:
        verdict = score_one(
            claims_case({"id": "absence", "statement": "no matching jobs", "match": "contains"}),
            trace(answer="There are no matching jobs."),
            scorers.forbidden_claims(),
        )
        assert verdict.passed is False
        assert verdict.detail["hits"] == ["absence"]

    def test_a_regex_claim_is_matched_as_a_pattern(self) -> None:
        verdict = score_one(
            claims_case({"id": "absence", "statement": r"no \w+ (jobs|roles)", "match": "regex"}),
            trace(answer="I found no suitable roles for you."),
            scorers.forbidden_claims(),
        )
        assert verdict.passed is False

    def test_every_hit_is_reported(self) -> None:
        verdict = score_one(
            claims_case(
                {"id": "absence", "statement": "no matching jobs", "match": "contains"},
                {"id": "certainty", "statement": "definitely eligible", "match": "contains"},
            ),
            trace(answer="There are no matching jobs and you are definitely eligible."),
            scorers.forbidden_claims(),
        )
        assert verdict.detail["hits"] == ["absence", "certainty"]

    def test_a_judge_claim_is_skipped_and_named(self) -> None:
        verdict = score_one(
            claims_case(
                {"id": "absence", "statement": "no matching jobs", "match": "contains"},
                {"id": "vague", "statement": "implies nothing exists", "match": "judge"},
            ),
            trace(answer="The tool failed."),
            scorers.forbidden_claims(),
        )
        assert verdict.passed is True
        assert verdict.detail["skipped"] == ["vague"]

    def test_a_case_of_only_judge_claims_is_not_applicable(self) -> None:
        """This is the safety-critical direction of the skip rule: a suite whose
        forbidden claims are all judge-mode must read as unmeasured, never as a
        clean pass, or M6 would publish a safety number it never checked."""
        verdict = scorers.forbidden_claims().score(
            claims_case({"id": "vague", "statement": "implies nothing exists", "match": "judge"}),
            trace(answer="There are no matching jobs."),
        )
        assert verdict is None

    def test_an_empty_claim_list_is_a_real_expectation_that_passes(self) -> None:
        verdict = score_one(claims_case(), trace(answer="anything"), scorers.forbidden_claims())
        assert verdict.passed is True

    def test_an_absent_answer_raises_rather_than_passing(self) -> None:
        """A missing answer must never read as "made no forbidden claim": that
        would turn a harness gap into a safety guarantee."""
        with pytest.raises(ScorerError, match="no answer"):
            scorers.forbidden_claims().score(
                claims_case(
                    {"id": "absence", "statement": "no matching jobs", "match": "contains"}
                ),
                trace(),
            )

    def test_the_value_is_one_when_clean_and_zero_on_a_hit(self) -> None:
        clean = score_one(
            claims_case({"id": "absence", "statement": "no jobs", "match": "contains"}),
            trace(answer="The tool failed."),
            scorers.forbidden_claims(),
        )
        dirty = score_one(
            claims_case({"id": "absence", "statement": "no jobs", "match": "contains"}),
            trace(answer="There are no jobs."),
            scorers.forbidden_claims(),
        )
        assert clean.value == 1.0
        assert dirty.value == 0.0


class TestNotApplicable:
    def test_facts_with_no_expectation_is_not_applicable(self) -> None:
        without = Case.model_validate(
            {
                "id": "nr-route-0001",
                "suite": "nextrole.routing",
                "split": "dev",
                "input": {"messages": [{"role": "user", "content": "hi"}]},
                "expect": {"route": {"label": "job_search"}},
            }
        )
        assert scorers.facts().score(without, trace(answer="x")) is None

    def test_forbidden_claims_with_no_expectation_is_not_applicable(self) -> None:
        without = Case.model_validate(
            {
                "id": "nr-route-0001",
                "suite": "nextrole.routing",
                "split": "dev",
                "input": {"messages": [{"role": "user", "content": "hi"}]},
                "expect": {"route": {"label": "job_search"}},
            }
        )
        assert scorers.forbidden_claims().score(without, trace(answer="x")) is None

    def test_an_empty_fact_list_is_not_applicable(self) -> None:
        """Nothing to recall, so recall has no denominator. ``0/0`` is not 0."""
        assert scorers.facts().score(facts_case(), trace(answer="x")) is None

    def test_the_scorers_declare_what_they_require(self) -> None:
        assert scorers.facts().requires == frozenset({"expect.facts"})
        assert scorers.forbidden_claims().requires == frozenset({"expect.forbidden_claims"})

    def test_the_scorers_are_named(self) -> None:
        assert scorers.facts().name == "facts"
        assert scorers.forbidden_claims().name == "forbidden_claims"
