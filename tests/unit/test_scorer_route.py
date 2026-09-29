"""The route scorer.

Acceptance row: "route prediction source fallback order ... not-applicable
never scores as fail".

The fallback order is fixed by the doc: ``graph.next`` from the probe span,
then the last node span before ``finalize``, then ``final_output.route``. If
none exists, ``ScorerError``, which makes the case unscored. The order matters
because the three sources can disagree, and a scorer that silently preferred a
different one would make two runs of the same suite mean different things.
"""

from __future__ import annotations

import pytest

from neverempty import ScorerError, scorers
from neverempty.dataset.case import Case

from ._scoring import node_span, probe_span, trace


def case(**route: object) -> Case:
    return Case.model_validate(
        {
            "id": "nr-route-0142",
            "suite": "nextrole.routing",
            "split": "dev",
            "input": {"messages": [{"role": "user", "content": "find me a job"}]},
            "expect": {"route": route},
        }
    )


class TestPredictionSourceOrder:
    def test_the_probe_span_wins_over_a_node_span(self) -> None:
        """``graph.next`` from the probe is the decision the agent made; a node
        span is what happened to execute, which differs under a fallback edge."""
        subject = trace(
            spans=[
                probe_span(start_ns=1_000, route="job_search"),
                node_span("email_draft", start_ns=2_000),
                node_span("finalize", start_ns=3_000),
            ]
        )
        verdict = scorers.route().score(case(label="job_search"), subject)
        assert verdict is not None
        assert verdict.passed is True
        assert verdict.detail["source"] == "graph.next"
        assert verdict.detail["predicted"] == "job_search"

    def test_the_probe_span_wins_over_final_output(self) -> None:
        subject = trace(spans=[probe_span(start_ns=1_000, route="job_search")], route="email_draft")
        verdict = scorers.route().score(case(label="job_search"), subject)
        assert verdict is not None
        assert verdict.passed is True
        assert verdict.detail["source"] == "graph.next"

    def test_the_last_node_before_finalize_is_used_when_there_is_no_probe(self) -> None:
        subject = trace(
            spans=[
                node_span("safety_check", start_ns=1_000),
                node_span("classify_intent", start_ns=2_000),
                node_span("job_search", start_ns=3_000),
                node_span("finalize", start_ns=4_000),
            ]
        )
        verdict = scorers.route().score(case(label="job_search"), subject)
        assert verdict is not None
        assert verdict.passed is True
        assert verdict.detail["source"] == "node_span"

    def test_nodes_after_finalize_are_ignored(self) -> None:
        """A cleanup node running after ``finalize`` is not the route."""
        subject = trace(
            spans=[
                node_span("job_search", start_ns=1_000),
                node_span("finalize", start_ns=2_000),
                node_span("telemetry_flush", start_ns=3_000),
            ]
        )
        verdict = scorers.route().score(case(label="job_search"), subject)
        assert verdict is not None
        assert verdict.detail["predicted"] == "job_search"

    def test_the_last_node_is_used_when_finalize_never_ran(self) -> None:
        subject = trace(
            spans=[
                node_span("classify_intent", start_ns=1_000),
                node_span("job_search", start_ns=2_000),
            ]
        )
        verdict = scorers.route().score(case(label="job_search"), subject)
        assert verdict is not None
        assert verdict.detail["predicted"] == "job_search"
        assert verdict.detail["source"] == "node_span"

    def test_plumbing_nodes_are_not_routes(self) -> None:
        """``safety_check`` and ``classify_intent`` run on every path, so
        neither can be the branch the agent chose."""
        subject = trace(
            spans=[node_span("safety_check", start_ns=1_000)],
            route="job_search",
        )
        verdict = scorers.route().score(case(label="job_search"), subject)
        assert verdict is not None
        assert verdict.detail["source"] == "final_output.route"

    def test_final_output_is_the_last_resort(self) -> None:
        subject = trace(route="job_search")
        verdict = scorers.route().score(case(label="job_search"), subject)
        assert verdict is not None
        assert verdict.passed is True
        assert verdict.detail["source"] == "final_output.route"

    def test_no_source_at_all_raises_scorer_error(self) -> None:
        """Marks the case unscored. ``gate`` treats that as a failure of the
        run, not of the agent: the harness could not see what happened."""
        with pytest.raises(ScorerError, match="no route prediction"):
            scorers.route().score(case(label="job_search"), trace())

    def test_the_error_names_the_sources_it_looked_for(self) -> None:
        with pytest.raises(ScorerError) as exc:
            scorers.route().score(case(label="job_search"), trace())
        message = str(exc.value)
        assert "graph.next" in message
        assert "final_output.route" in message

    def test_an_empty_string_route_is_not_a_prediction(self) -> None:
        """Failure must never be representable as empty: an empty route is a
        missing route, so the scorer falls through rather than scoring a miss."""
        subject = trace(spans=[probe_span(start_ns=1_000, route="")], route="job_search")
        verdict = scorers.route().score(case(label="job_search"), subject)
        assert verdict is not None
        assert verdict.detail["source"] == "final_output.route"


class TestStrictAndLenient:
    def test_a_wrong_route_fails_strictly(self) -> None:
        subject = trace(route="email_draft")
        verdict = scorers.route().score(case(label="job_search"), subject)
        assert verdict is not None
        assert verdict.passed is False
        assert verdict.value == 0.0

    def test_an_acceptable_route_fails_strict_but_passes_lenient(self) -> None:
        subject = trace(route="email_draft")
        verdict = scorers.route().score(case(label="followup", acceptable=["email_draft"]), subject)
        assert verdict is not None
        assert verdict.passed is False
        assert verdict.detail["lenient"] is True

    def test_the_label_itself_passes_lenient_too(self) -> None:
        subject = trace(route="followup")
        verdict = scorers.route().score(case(label="followup", acceptable=["email_draft"]), subject)
        assert verdict is not None
        assert verdict.passed is True
        assert verdict.detail["lenient"] is True

    def test_a_route_outside_label_and_acceptable_fails_both(self) -> None:
        subject = trace(route="salary_negotiation")
        verdict = scorers.route().score(case(label="followup", acceptable=["email_draft"]), subject)
        assert verdict is not None
        assert verdict.passed is False
        assert verdict.detail["lenient"] is False

    def test_with_no_acceptable_set_lenient_equals_strict(self) -> None:
        subject = trace(route="job_search")
        verdict = scorers.route().score(case(label="job_search"), subject)
        assert verdict is not None
        assert verdict.detail["lenient"] == verdict.passed

    def test_the_expected_label_is_reported(self) -> None:
        verdict = scorers.route().score(case(label="job_search"), trace(route="email_draft"))
        assert verdict is not None
        assert verdict.detail["expected"] == "job_search"


class TestNotApplicable:
    def test_a_case_with_no_route_expectation_is_not_applicable(self) -> None:
        """Never a fail: the case did not ask."""
        without = Case.model_validate(
            {
                "id": "nr-tool-0001",
                "suite": "nextrole.routing",
                "split": "dev",
                "input": {"messages": [{"role": "user", "content": "hi"}]},
                "expect": {"forbidden_tools": []},
            }
        )
        assert scorers.route().score(without, trace(route="job_search")) is None

    def test_the_scorer_declares_what_it_requires(self) -> None:
        """So the runner can skip it before calling it at all."""
        assert scorers.route().requires == frozenset({"expect.route"})

    def test_the_scorer_is_named(self) -> None:
        assert scorers.route().name == "route"


class TestDisagreement:
    def test_a_probe_disagreement_is_surfaced_not_resolved(self) -> None:
        """``route_probe`` records both the graph edge and the state field. A
        disagreement is a finding about the agent, so it travels in the detail
        rather than changing which source wins."""
        subject = trace(
            spans=[
                probe_span(
                    start_ns=1_000,
                    route="job_search",
                    **{"graph.state_route": "email_draft", "graph.route_disagreement": True},
                )
            ]
        )
        verdict = scorers.route().score(case(label="job_search"), subject)
        assert verdict is not None
        assert verdict.passed is True
        assert verdict.detail["state_route"] == "email_draft"
        assert verdict.detail["route_disagreement"] is True
