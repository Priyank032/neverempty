"""YojanaKhoj consistency scoring.

M10's acceptance row: "Node traces validate, cache bypass asserted, per-category
rates produced". This covers the third, and the per-category denominators are the
point: the doc's trap #4 is reading a low pooled contradiction rate as a win when
the hard rule filter has already removed the schemes that could contradict.
"""

from __future__ import annotations

import pytest

from neverempty.dataset.case import Case
from neverempty.evals.yojanakhoj import (
    CATEGORY_CONFIDENCE,
    CATEGORY_LANGUAGE,
    CATEGORY_OVERCLAIM,
    CATEGORY_VERDICT,
    VERDICTS,
    consistency,
    read_items,
)
from neverempty.runner.runner import ScorerError

from ._scoring import trace


def case(*items: tuple[str, bool | None]) -> Case:
    return Case.model_validate(
        {
            "id": "yk-p07-pm-kisan",
            "suite": "yojanakhoj.consistency",
            "split": "test",
            "input": {"payload": {"lang": "en"}},
            "expect": {
                "items": [{"item_id": item_id, "rule_result": rule} for item_id, rule in items]
            },
            "provenance": {
                "method": "generated_from_rules",
                "source_commit": "a" * 40,
            },
        }
    )


def item(
    item_id: str,
    verdict: str,
    *,
    confidence: int | None = None,
    reason: str | None = None,
    en: str | None = None,
    hi: str | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {"schemeId": item_id, "eligibility_yes_no": verdict}
    if confidence is not None:
        entry["llm_confidence"] = confidence
    if reason is not None:
        entry["eligibility_reason"] = reason
    if en is not None:
        entry["explanation_en"] = en
    if hi is not None:
        entry["explanation_hi"] = hi
    return entry


def structured(*items: dict[str, object]) -> dict[str, object]:
    return {"items": list(items)}


class TestTheSchemaIsThreeWay:
    def test_check_is_a_real_verdict(self) -> None:
        """The design doc predicted a boolean schema that could not express
        'cannot evaluate'. The matcher declares enum yes/no/check, so it can."""
        assert VERDICTS == ("yes", "no", "check")


class TestReadItems:
    def test_reads_the_matchers_field_names(self) -> None:
        items = read_items(structured(item("pm-kisan", "yes", confidence=90)))
        assert items[0].item_id == "pm-kisan"
        assert items[0].verdict == "yes"
        assert items[0].confidence == 90

    def test_no_structured_output_is_empty_not_an_error(self) -> None:
        assert read_items(None) == []

    def test_an_item_without_an_id_is_skipped(self) -> None:
        assert read_items({"items": [{"eligibility_yes_no": "yes"}]}) == []

    def test_a_boolean_confidence_is_not_a_number(self) -> None:
        items = read_items({"items": [{"schemeId": "x", "llm_confidence": True}]})
        assert items[0].confidence is None

    def test_the_degraded_fallback_is_detected_by_its_canned_text(self) -> None:
        """The LLM-failure path emits a fixed bilingual string. Scoring it as a
        model answer would measure the outage, not the model."""
        items = read_items(
            structured(item("x", "check", reason="Verify at nearest Jan Seva Kendra."))
        )
        assert items[0].llm_failed
        assert not items[0].scorable

    def test_an_explicit_failure_flag_is_honoured(self) -> None:
        items = read_items({"items": [{"schemeId": "x", "llm_failed": True}]})
        assert not items[0].scorable


class TestOverclaimOnNull:
    """The doc's most interesting case: the rules could not decide."""

    def test_a_definite_yes_on_a_null_rule_is_an_overclaim(self) -> None:
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", None)), trace(structured=structured(item("pm-kisan", "yes")))
        )
        assert result is not None
        assert result.passed is False
        assert CATEGORY_OVERCLAIM in result.detail["categories"]

    def test_a_definite_no_on_a_null_rule_is_also_an_overclaim(self) -> None:
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", None)), trace(structured=structured(item("pm-kisan", "no")))
        )
        assert result is not None
        assert result.passed is False

    def test_check_on_a_null_rule_is_correct(self) -> None:
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", None)),
            trace(structured=structured(item("pm-kisan", "check"))),
        )
        assert result is not None
        assert result.passed is True

    def test_high_confidence_on_a_null_rule_is_its_own_finding(self) -> None:
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", None)),
            trace(structured=structured(item("pm-kisan", "check", confidence=95))),
        )
        assert result is not None
        assert CATEGORY_CONFIDENCE in result.detail["categories"]

    def test_low_confidence_on_a_null_rule_is_fine(self) -> None:
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", None)),
            trace(structured=structured(item("pm-kisan", "check", confidence=30))),
        )
        assert result is not None
        assert result.passed is True

    def test_confidence_is_on_the_zero_to_hundred_scale(self) -> None:
        """The matcher declares integer 0-100. Treating it as a probability
        would make 95 read as certainty beyond the scale."""
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", None)),
            trace(structured=structured(item("pm-kisan", "check", confidence=1))),
        )
        assert result is not None
        assert result.passed is True


class TestVerdictContradiction:
    def test_no_against_an_eligible_rule_contradicts(self) -> None:
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", True)), trace(structured=structured(item("pm-kisan", "no")))
        )
        assert result is not None
        assert CATEGORY_VERDICT in result.detail["categories"]

    def test_yes_against_an_ineligible_rule_contradicts(self) -> None:
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", False)),
            trace(structured=structured(item("pm-kisan", "yes"))),
        )
        assert result is not None
        assert CATEGORY_VERDICT in result.detail["categories"]

    def test_check_against_a_decided_rule_is_not_a_contradiction(self) -> None:
        """Declining to commit is not the same as saying the opposite."""
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", True)),
            trace(structured=structured(item("pm-kisan", "check"))),
        )
        assert result is not None
        assert result.passed is True

    def test_agreement_passes(self) -> None:
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", True)),
            trace(structured=structured(item("pm-kisan", "yes"))),
        )
        assert result is not None
        assert result.passed is True
        assert result.value == 1.0


class TestCrossLanguage:
    def test_english_eligible_and_hindi_ineligible_disagree(self) -> None:
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", True)),
            trace(
                structured=structured(
                    item(
                        "pm-kisan",
                        "yes",
                        en="You are eligible for this scheme.",
                        hi="आप इस योजना के लिए पात्र नहीं हैं।",
                    )
                )
            ),
        )
        assert result is not None
        assert CATEGORY_LANGUAGE in result.detail["categories"]

    def test_both_languages_agreeing_passes(self) -> None:
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", True)),
            trace(
                structured=structured(
                    item(
                        "pm-kisan",
                        "yes",
                        en="You are eligible for this scheme.",
                        hi="आप इस योजना के लिए पात्र हैं।",
                    )
                )
            ),
        )
        assert result is not None
        assert result.passed is True

    def test_unreadable_prose_does_not_invent_a_disagreement(self) -> None:
        """Two explanations the crude polarity check cannot read agree, because
        a check that cannot decide must not report a finding."""
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", True)),
            trace(
                structured=structured(
                    item("pm-kisan", "yes", en="Details follow.", hi="विवरण नीचे है।")
                )
            ),
        )
        assert result is not None
        assert result.passed is True

    def test_one_language_missing_is_not_checked(self) -> None:
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", True)),
            trace(structured=structured(item("pm-kisan", "yes", en="You are eligible."))),
        )
        assert result is not None
        assert CATEGORY_LANGUAGE not in result.detail.get("categories", {})


class TestExclusions:
    def test_a_failed_llm_item_is_skipped_not_counted_wrong(self) -> None:
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", True), ("nsap", True)),
            trace(
                structured=structured(
                    item("pm-kisan", "yes"),
                    item("nsap", "check", reason="Verify at nearest Jan Seva Kendra."),
                )
            ),
        )
        assert result is not None
        assert result.detail["items_scored"] == 1
        assert "nsap" in result.detail["items_skipped"]
        assert result.passed is True

    def test_every_item_failing_is_not_applicable_not_zero(self) -> None:
        """A case where the LLM never answered measures nothing. Returning 0%
        would publish an outage as a consistency failure."""
        scorer = consistency()
        result = scorer.score(
            case(("pm-kisan", True)),
            trace(
                structured=structured(
                    item("pm-kisan", "yes", reason="Verify at nearest Jan Seva Kendra.")
                )
            ),
        )
        assert result is None

    def test_a_case_without_item_expectations_is_not_applicable(self) -> None:
        scorer = consistency()
        bare = Case.model_validate(
            {
                "id": "yk-x",
                "suite": "yojanakhoj.consistency",
                "split": "dev",
                "input": {"payload": {}},
                "expect": {"forbidden_tools": []},
            }
        )
        assert scorer.score(bare, trace(structured=structured())) is None

    def test_no_structured_items_at_all_raises(self) -> None:
        """Expecting items and receiving none is a harness gap, so the case is
        unscored rather than scored zero."""
        scorer = consistency()
        with pytest.raises(ScorerError, match="no structured items"):
            scorer.score(case(("pm-kisan", True)), trace(structured=None))


class TestPerCategoryDenominators:
    def test_each_category_carries_its_own_denominator(self) -> None:
        """The doc's trap #4: pooling lets a structurally protected category
        hide an exposed one."""
        scorer = consistency()
        result = scorer.score(
            case(("a", True), ("b", None), ("c", None)),
            trace(
                structured=structured(
                    item("a", "yes"),
                    item("b", "yes"),
                    item("c", "check"),
                )
            ),
        )
        assert result is not None
        denominators = result.detail["denominators"]
        assert denominators[CATEGORY_VERDICT] == 1
        assert denominators[CATEGORY_OVERCLAIM] == 2

    def test_the_overclaim_rate_is_over_null_cases_only(self) -> None:
        scorer = consistency()
        result = scorer.score(
            case(("a", True), ("b", None), ("c", None)),
            trace(structured=structured(item("a", "yes"), item("b", "yes"), item("c", "check"))),
        )
        assert result is not None
        assert result.detail["rates"][CATEGORY_OVERCLAIM] == pytest.approx(0.5)

    def test_the_value_is_the_fraction_of_consistent_items(self) -> None:
        scorer = consistency()
        result = scorer.score(
            case(("a", True), ("b", True), ("c", True), ("d", True)),
            trace(
                structured=structured(
                    item("a", "yes"),
                    item("b", "yes"),
                    item("c", "yes"),
                    item("d", "no"),
                )
            ),
        )
        assert result is not None
        assert result.value == pytest.approx(0.75)
        assert result.passed is False
