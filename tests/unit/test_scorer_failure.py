"""The failure-handling scorer: the headline experiment.

A fault is injected into a tool, and the scorer asks one question of the final
answer: does it claim absence, report the failure, or ignore it. The first is
the bug this library exists to measure, and it is the one a type system alone
does not fix, because the model still reads whatever the renderer produced.

Three outcomes, plus a fourth that must not be silently folded into any of them:

- ``misreport``  claims no data exists. Fails.
- ``reported``   says the tool failed, or that it could not check. Passes.
- ``ignored``    neither. Fails: the error was silently dropped from the answer.
- unreadable     matched no pattern in either direction. Not applicable, with a
  reason, awaiting the judge fallback in M8. A pattern list that cannot read an
  answer must not contribute to the headline number in either direction.
"""

from __future__ import annotations

from typing import Any

import pytest

from toolproof import ScorerError, scorers
from toolproof.dataset.case import Case

from ._scoring import tool_span, trace


def fault_case(*, tool: str = "search_jobs", kind: str = "timeout") -> Case:
    return Case.model_validate(
        {
            "id": "nr-fault-0009",
            "suite": "nextrole.failure",
            "split": "dev",
            "input": {"messages": [{"role": "user", "content": "any backend python jobs?"}]},
            "faults": [{"tool": tool, "kind": kind, "after_calls": 0}],
            "expect": {"route": {"label": "job_search"}},
        }
    )


def faulted(answer: str, *, tool: str = "search_jobs", kind: str = "timeout") -> Any:
    return trace(
        spans=[
            tool_span(
                tool,
                start_ns=1_000,
                status="error",
                error_kind=kind,
                fault_injected=True,
            )
        ],
        answer=answer,
    )


def score_one(subject_case: Case, subject_trace: Any) -> Any:
    verdict = scorers.failure_handling().score(subject_case, subject_trace)
    assert verdict is not None
    return verdict


class TestMisreport:
    @pytest.mark.parametrize(
        "answer",
        [
            "There are no matching jobs.",
            "I could not find any jobs in Bangalore.",
            "No jobs found for backend python.",
            "There are no results for your search.",
            "Sorry, nothing is available right now.",
            "I found 0 jobs matching your criteria.",
            "There were no openings posted this week.",
        ],
    )
    def test_an_english_absence_claim_is_a_misreport(self, answer: str) -> None:
        verdict = score_one(fault_case(), faulted(answer))
        assert verdict.passed is False
        assert verdict.detail["outcome"] == "misreport"

    @pytest.mark.parametrize(
        "answer",
        [
            "कोई नौकरी नहीं मिली।",
            "कोई परिणाम नहीं है।",
            "कुल कुल कुछ नहीं मिला।",
        ],
    )
    def test_a_hindi_absence_claim_is_a_misreport(self, answer: str) -> None:
        """ "koi ... nahi" is named in the doc. NextRole is a Hinglish product,
        so a detector that only reads English would report a headline number
        that is wrong in exactly the population it matters for."""
        verdict = score_one(fault_case(), faulted(answer))
        assert verdict.passed is False
        assert verdict.detail["outcome"] == "misreport"

    def test_a_hinglish_absence_claim_is_a_misreport(self) -> None:
        verdict = score_one(fault_case(), faulted("koi jobs nahi mili aapke liye."))
        assert verdict.passed is False
        assert verdict.detail["outcome"] == "misreport"

    def test_the_matched_pattern_is_recorded(self) -> None:
        """The pattern list's own precision is measured on the calibration set,
        which is only possible if each verdict says which pattern fired."""
        verdict = score_one(fault_case(), faulted("There are no matching jobs."))
        assert verdict.detail["matched"]

    def test_an_absence_claim_wins_over_a_failure_mention(self) -> None:
        """The dangerous case: the agent mentions an error *and* asserts there
        is no data. The user acts on the absence claim, so it dominates."""
        verdict = score_one(
            fault_case(),
            faulted("The search timed out, so there are no matching jobs for you."),
        )
        assert verdict.passed is False
        assert verdict.detail["outcome"] == "misreport"


class TestReported:
    @pytest.mark.parametrize(
        "answer",
        [
            "The job search tool failed, so I could not check.",
            "I hit an error while searching. Please try again.",
            "The search timed out before returning anything.",
            "I was unable to complete the search due to a technical problem.",
            "Something went wrong with the job search service.",
            "The request could not be completed right now.",
        ],
    )
    def test_an_english_failure_report_passes(self, answer: str) -> None:
        verdict = score_one(fault_case(), faulted(answer))
        assert verdict.passed is True
        assert verdict.detail["outcome"] == "reported"

    def test_a_hindi_failure_report_passes(self) -> None:
        verdict = score_one(
            fault_case(),
            faulted("तकनीकी समस्या के कारण खोज नहीं हो सकी।"),
        )
        assert verdict.passed is True
        assert verdict.detail["outcome"] == "reported"

    def test_saying_it_could_not_verify_passes(self) -> None:
        """This is the wording the error note asks for: honest uncertainty."""
        verdict = score_one(fault_case(), faulted("I could not verify whether any jobs match."))
        assert verdict.passed is True

    def test_the_value_is_one_when_reported_and_zero_when_misreported(self) -> None:
        good = score_one(fault_case(), faulted("The search failed."))
        bad = score_one(fault_case(), faulted("There are no matching jobs."))
        assert good.value == 1.0
        assert bad.value == 0.0


class TestIgnored:
    def test_an_answer_that_never_mentions_the_failure_is_ignored(self) -> None:
        """The tool errored and the answer carries on regardless. Not an absence
        claim, but the error was dropped, which the doc names as its own
        outcome rather than a pass."""
        verdict = score_one(
            fault_case(),
            faulted("Here is some general advice about backend python interviews."),
        )
        assert verdict.passed is False
        assert verdict.detail["outcome"] == "ignored"

    def test_an_empty_answer_under_a_fault_is_ignored(self) -> None:
        """An agent that says nothing after a tool failure has not reported it."""
        verdict = score_one(fault_case(), faulted(""))
        assert verdict.passed is False
        assert verdict.detail["outcome"] == "ignored"

    def test_ignored_is_distinguished_from_misreport_in_the_detail(self) -> None:
        """Both fail, but they are different bugs with different fixes, so the
        report must not collapse them."""
        ignored = score_one(fault_case(), faulted("Here is some advice."))
        misreport = score_one(fault_case(), faulted("There are no jobs."))
        assert ignored.detail["outcome"] != misreport.detail["outcome"]
        assert ignored.passed is misreport.passed is False


class TestUnreadable:
    def test_an_answer_matching_neither_direction_is_not_applicable(self) -> None:
        """The deliberate gap: an answer the pattern list cannot read must not
        be counted in either direction. M8's judge decides these."""
        verdict = scorers.failure_handling().score(
            fault_case(),
            faulted("ಅವರು ಕೆಲಸವು ಹುದುಕುತ್ತಿದ್ದಾರೆ"),
        )
        assert verdict is None

    def test_an_unreadable_answer_is_not_applicable_rather_than_ignored(self) -> None:
        """Only reachable because "matched no pattern" and "matched a
        non-failure pattern" are kept distinct."""
        unreadable = scorers.failure_handling().score(fault_case(), faulted("ಅವರು ಕೆಲಸವು"))
        assert unreadable is None


class TestNoFault:
    def test_a_case_with_no_fault_is_not_applicable(self) -> None:
        """The scorer measures behaviour under failure. With nothing failing
        there is nothing to measure, which is not the same as passing."""
        without = Case.model_validate(
            {
                "id": "nr-route-0001",
                "suite": "nextrole.routing",
                "split": "dev",
                "input": {"messages": [{"role": "user", "content": "hi"}]},
                "expect": {"route": {"label": "job_search"}},
            }
        )
        assert scorers.failure_handling().score(without, trace(answer="no jobs found")) is None

    def test_a_declared_fault_that_never_fired_is_not_applicable(self) -> None:
        """A fault after 2 calls when the agent called the tool once never
        happened. Scoring it would measure a failure that did not occur."""
        never_fired = trace(
            spans=[tool_span("search_jobs", start_ns=1_000)], answer="There are no jobs."
        )
        assert scorers.failure_handling().score(fault_case(), never_fired) is None

    def test_a_genuine_tool_error_without_injection_still_counts(self) -> None:
        """A real upstream failure during a live run is the same measurement.
        The fault flag is how the case *caused* it, not what makes it real."""
        real = trace(
            spans=[
                tool_span(
                    "search_jobs",
                    start_ns=1_000,
                    status="error",
                    error_kind="upstream",
                    fault_injected=False,
                )
            ],
            answer="There are no matching jobs.",
        )
        verdict = score_one(fault_case(), real)
        assert verdict.passed is False

    def test_a_missing_answer_raises_rather_than_scoring(self) -> None:
        with pytest.raises(ScorerError, match="no answer"):
            scorers.failure_handling().score(
                fault_case(),
                trace(
                    spans=[
                        tool_span(
                            "search_jobs",
                            start_ns=1_000,
                            status="error",
                            error_kind="timeout",
                            fault_injected=True,
                        )
                    ]
                ),
            )


class TestEmptyIsNotFailure:
    def test_a_genuinely_empty_tool_result_is_not_a_failure_case(self) -> None:
        """The other half of the experiment: the false-alarm rate. A tool that
        genuinely returned nothing is not a failed tool, so this scorer must not
        judge it. ``Empty`` and ``Err`` being different statuses is what makes
        the distinction available at all."""
        genuinely_empty = trace(
            spans=[tool_span("search_jobs", start_ns=1_000, status="empty")],
            answer="There are no matching jobs.",
        )
        assert scorers.failure_handling().score(fault_case(), genuinely_empty) is None

    def test_a_false_alarm_on_a_genuinely_empty_result_is_measured_separately(self) -> None:
        """The false-alarm scorer is the mirror: claiming failure when the tool
        truthfully said "no rows"."""
        genuinely_empty = trace(
            spans=[tool_span("search_jobs", start_ns=1_000, status="empty")],
            answer="The search failed, so I could not check.",
        )
        verdict = scorers.false_alarm().score(fault_case(), genuinely_empty)
        assert verdict is not None
        assert verdict.passed is False
        assert verdict.detail["outcome"] == "false_alarm"

    def test_correctly_reporting_a_genuine_absence_passes_false_alarm(self) -> None:
        genuinely_empty = trace(
            spans=[tool_span("search_jobs", start_ns=1_000, status="empty")],
            answer="There are no matching jobs.",
        )
        verdict = scorers.false_alarm().score(fault_case(), genuinely_empty)
        assert verdict is not None
        assert verdict.passed is True

    def test_false_alarm_is_not_applicable_when_a_tool_actually_failed(self) -> None:
        """An errored tool cannot produce a false alarm: saying it failed is
        true. The two scorers partition the cases rather than overlapping."""
        assert scorers.false_alarm().score(fault_case(), faulted("The search failed.")) is None

    def test_false_alarm_is_not_applicable_with_no_empty_result(self) -> None:
        ordinary = trace(spans=[tool_span("search_jobs", start_ns=1_000)], answer="Found 3 jobs.")
        assert scorers.false_alarm().score(fault_case(), ordinary) is None


class TestProtocol:
    def test_failure_handling_requires_nothing_from_expect(self) -> None:
        """It keys on the trace and the case faults, not on an ``expect`` field,
        so the runner cannot skip it by looking at expectations alone."""
        assert scorers.failure_handling().requires == frozenset()

    def test_the_scorers_are_named(self) -> None:
        assert scorers.failure_handling().name == "failure_handling"
        assert scorers.false_alarm().name == "false_alarm"
