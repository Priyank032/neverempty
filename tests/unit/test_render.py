"""The Markdown renderer.

Acceptance row: "renderer refuses a metric with ``applicable=0``; percentages
suppressed below n=10".

The renderer reads only the report structure, never the outcomes or traces. That
is the constraint that makes the README trustworthy: a table in a README can
only contain numbers that are in a committed report, because the renderer has no
other source for them.

"Refuses" means it prints "not measured" rather than a number. It does not raise:
a report legitimately contains unmeasured metrics, and a renderer that crashed
on one would make partial reports unreadable.
"""

from __future__ import annotations

from neverempty import Metric, Report
from neverempty.core.trace import Env
from neverempty.report.gate import GateConfig, gate
from neverempty.report.render import (
    INDICATIVE_MIN_N,
    SUPPRESS_BELOW_N,
    render_gate,
    render_markdown,
)


def env() -> Env:
    return Env(
        neverempty_version="0.0.1",
        pricing_version="empty-2026-09-23",
        python_version="3.12.10",
        target_git_sha="abc1234",
        resolved_models=["gpt-4o-2024-08-06"],
        concurrency=8,
        seed=20260921,
    )


def report(**overrides: object) -> Report:
    fields: dict[str, object] = {
        "suite": "nextrole.routing",
        "split": "test",
        "created_at": "2026-09-23T10:00:00.000Z",
        "complete": True,
        "status": "ok",
        "env": env(),
    }
    fields.update(overrides)
    return Report.model_validate(fields)


class TestNotMeasured:
    def test_a_metric_with_applicable_zero_prints_not_measured(self) -> None:
        """Never 0%, and never a blank cell that a reader fills in optimistically."""
        subject = report(
            metrics=[Metric(name="facts", n=0, applicable=0, note="no fact expectations")]
        )
        rendered = render_markdown(subject)
        assert "not measured" in rendered
        assert "0.0%" not in rendered
        assert "0%" not in rendered

    def test_the_reason_it_was_not_measured_is_printed(self) -> None:
        subject = report(
            metrics=[Metric(name="facts", n=0, applicable=0, note="no fact expectations")]
        )
        assert "no fact expectations" in render_markdown(subject)

    def test_a_measured_metric_prints_its_percentage(self) -> None:
        subject = report(
            metrics=[
                Metric(
                    name="route",
                    n=210,
                    value=0.852,
                    ci_low=0.798,
                    ci_high=0.894,
                    method="wilson",
                    applicable=210,
                )
            ]
        )
        rendered = render_markdown(subject)
        assert "85.2%" in rendered
        assert "79.8%" in rendered
        assert "89.4%" in rendered

    def test_a_genuinely_zero_metric_prints_zero_not_not_measured(self) -> None:
        """0% misreport is a result, and the best possible one. It must be
        distinguishable from "we did not check"."""
        subject = report(
            metrics=[
                Metric(
                    name="failure_handling",
                    n=30,
                    value=0.0,
                    ci_low=0.0,
                    ci_high=0.11,
                    method="wilson",
                    applicable=30,
                )
            ]
        )
        rendered = render_markdown(subject)
        assert "0.0%" in rendered
        # Scoped to the metrics table: an absent latency sample legitimately
        # renders as "not measured" elsewhere in the same document.
        metrics_row = next(
            line for line in rendered.splitlines() if line.startswith("| failure_handling")
        )
        assert "0.0%" in metrics_row
        assert "not measured" not in metrics_row

    def test_the_renderer_does_not_raise_on_an_unmeasured_metric(self) -> None:
        """A partial report has to stay readable; that is when you most need it."""
        subject = report(
            status="incomplete",
            complete=False,
            metrics=[Metric(name="route", n=0, applicable=0)],
        )
        assert render_markdown(subject)


class TestSuppressionBelowTen:
    def test_a_percentage_is_suppressed_below_n_of_ten(self) -> None:
        """At n=9 a percentage is a number with two significant figures and
        almost no information. Printing it invites a claim nobody can defend."""
        subject = report(
            metrics=[
                Metric(
                    name="route",
                    n=9,
                    value=0.888,
                    ci_low=0.55,
                    ci_high=0.98,
                    method="wilson",
                    applicable=9,
                )
            ]
        )
        rendered = render_markdown(subject)
        assert "88.8%" not in rendered
        assert "8/9" in rendered or "n=9" in rendered

    def test_the_raw_counts_are_still_shown_when_suppressed(self) -> None:
        """Suppressing the percentage must not suppress the evidence."""
        subject = report(
            metrics=[Metric(name="route", n=9, value=0.888, method="wilson", applicable=9)]
        )
        assert "9" in render_markdown(subject)

    def test_the_threshold_is_ten(self) -> None:
        assert SUPPRESS_BELOW_N == 10

    def test_at_n_of_ten_the_percentage_is_printed(self) -> None:
        subject = report(
            metrics=[
                Metric(
                    name="route",
                    n=10,
                    value=0.9,
                    ci_low=0.60,
                    ci_high=0.98,
                    method="wilson",
                    applicable=10,
                )
            ]
        )
        assert "90.0%" in render_markdown(subject)

    def test_a_small_n_is_labelled_rather_than_silently_shown(self) -> None:
        subject = report(
            metrics=[Metric(name="route", n=9, value=0.888, method="wilson", applicable=9)]
        )
        assert "too few" in render_markdown(subject).lower()


class TestIndicativeLabelling:
    def test_a_metric_under_fifty_is_labelled_indicative(self) -> None:
        """The doc labels per-branch figures indicative under n=50, because even
        at n=30 a real 5-point drop and noise look identical."""
        subject = report(
            metrics=[
                Metric(
                    name="route",
                    n=30,
                    value=0.867,
                    ci_low=0.703,
                    ci_high=0.947,
                    method="wilson",
                    applicable=30,
                )
            ]
        )
        assert "indicative" in render_markdown(subject).lower()

    def test_a_metric_at_or_above_fifty_is_not_labelled_indicative(self) -> None:
        subject = report(
            metrics=[
                Metric(
                    name="route",
                    n=210,
                    value=0.852,
                    ci_low=0.798,
                    ci_high=0.894,
                    method="wilson",
                    applicable=210,
                )
            ]
        )
        assert "indicative" not in render_markdown(subject).lower()

    def test_the_indicative_threshold_is_fifty(self) -> None:
        assert INDICATIVE_MIN_N == 50


class TestReproducibilityBlock:
    def test_the_git_sha_and_model_are_printed(self) -> None:
        """A number without its provenance cannot go in a README. The doc makes
        this block a hard requirement for the renderer."""
        rendered = render_markdown(report())
        assert "abc1234" in rendered
        assert "gpt-4o-2024-08-06" in rendered

    def test_the_pricing_and_neverempty_versions_are_printed(self) -> None:
        rendered = render_markdown(report())
        assert "empty-2026-09-23" in rendered
        assert "0.0.1" in rendered

    def test_the_seed_and_concurrency_are_printed(self) -> None:
        """p95 under 8-way concurrency is not production p95, so the number is
        never shown without the concurrency beside it."""
        rendered = render_markdown(report())
        assert "20260921" in rendered
        assert "8" in rendered

    def test_a_report_without_a_git_sha_says_so(self) -> None:
        """Silence would read as "clean checkout" rather than "not recorded"."""
        bare = report(
            env=Env(
                neverempty_version="0.0.1",
                pricing_version="empty-2026-09-23",
                python_version="3.12.10",
            )
        )
        rendered = render_markdown(bare)
        assert "unknown" in rendered.lower() or "not recorded" in rendered.lower()

    def test_a_dirty_working_tree_is_flagged(self) -> None:
        """A number produced from uncommitted code is not reproducible, and the
        report must say so rather than look clean."""
        dirty = report(
            env=Env(
                neverempty_version="0.0.1",
                pricing_version="empty-2026-09-23",
                python_version="3.12.10",
                target_git_sha="abc1234",
                target_dirty=True,
            )
        )
        assert "dirty" in render_markdown(dirty).lower()


class TestStatusAndCompleteness:
    def test_an_incomplete_report_is_flagged_prominently(self) -> None:
        subject = report(status="incomplete", complete=False)
        rendered = render_markdown(subject)
        assert "incomplete" in rendered.lower()

    def test_a_budget_aborted_report_is_flagged(self) -> None:
        subject = report(status="aborted_budget", complete=False)
        assert "aborted" in render_markdown(subject).lower()

    def test_an_unknown_cost_count_is_printed(self) -> None:
        """ "cost unknown for 12/210 traces" rather than a confident total."""
        subject = report(
            counts={"cases": 210, "repeats": 1, "scored": 210},
            costs={"total_usd": None, "unknown_count": 12, "pricing_version": "v1"},
        )
        rendered = render_markdown(subject)
        assert "12" in rendered
        assert "unknown" in rendered.lower()

    def test_a_known_total_cost_is_printed(self) -> None:
        subject = report(costs={"total_usd": 1.2345, "mean_usd": 0.0059, "pricing_version": "v1"})
        assert "1.23" in render_markdown(subject)

    def test_unscored_case_ids_are_listed(self) -> None:
        """The doc requires them listed by id, so a reader can go look."""
        subject = report(
            complete=False,
            status="incomplete",
            outcomes=[
                {"case_id": "nr-route-0007", "repeat": 0, "scored": False},
                {"case_id": "nr-route-0008", "repeat": 0, "scored": True},
            ],
        )
        rendered = render_markdown(subject)
        assert "nr-route-0007" in rendered
        assert "nr-route-0008" not in rendered


class TestConfusionMatrix:
    def test_the_matrix_is_rendered_with_counts(self) -> None:
        subject = report(
            confusion={
                "job_search": {"job_search": 28, "clarify": 2},
                "followup": {"followup": 25, "email_draft": 5},
            }
        )
        rendered = render_markdown(subject)
        assert "job_search" in rendered
        assert "28" in rendered

    def test_percentages_are_not_printed_for_a_row_below_ten(self) -> None:
        """The doc is explicit: the renderer refuses percentages for rows with
        n below 10. A 2-of-3 row printed as 67% is a misleading number."""
        subject = report(confusion={"clarify": {"clarify": 2, "general": 1}})
        rendered = render_markdown(subject)
        assert "66.7%" not in rendered
        assert "67%" not in rendered

    def test_percentages_are_printed_for_a_row_at_ten_or_above(self) -> None:
        subject = report(confusion={"job_search": {"job_search": 9, "clarify": 1}})
        assert "90.0%" in render_markdown(subject)

    def test_an_empty_matrix_is_omitted_rather_than_printed_blank(self) -> None:
        rendered = render_markdown(report(confusion={}))
        assert "Confusion" not in rendered

    def test_the_unscored_column_is_rendered(self) -> None:
        subject = report(confusion={"job_search": {"job_search": 20, "unscored": 3}})
        rendered = render_markdown(subject)
        assert "unscored" in rendered
        assert "3" in rendered


class TestDeterminism:
    def test_the_same_report_renders_identically(self) -> None:
        subject = report(
            metrics=[
                Metric(name="route", n=210, value=0.85, method="wilson", applicable=210),
                Metric(name="facts", n=100, value=0.7, method="bootstrap", applicable=100),
            ],
            confusion={"job_search": {"job_search": 28}},
        )
        assert render_markdown(subject) == render_markdown(subject)

    def test_metric_order_does_not_depend_on_input_order(self) -> None:
        """Two reports with the same metrics in a different order render the
        same, so a README diff shows content changes rather than reordering."""
        first = report(
            metrics=[
                Metric(name="route", n=100, value=0.9, method="wilson", applicable=100),
                Metric(name="facts", n=100, value=0.7, method="bootstrap", applicable=100),
            ]
        )
        second = report(
            metrics=[
                Metric(name="facts", n=100, value=0.7, method="bootstrap", applicable=100),
                Metric(name="route", n=100, value=0.9, method="wilson", applicable=100),
            ]
        )
        assert render_markdown(first) == render_markdown(second)

    def test_the_output_ends_with_a_single_newline(self) -> None:
        rendered = render_markdown(report())
        assert rendered.endswith("\n")
        assert not rendered.endswith("\n\n")


class TestRenderGate:
    def test_a_passing_gate_renders_its_verdict(self) -> None:
        subject = report(
            outcomes=[
                {
                    "case_id": f"c-{i:04d}",
                    "repeat": 0,
                    "scored": True,
                    "scores": {"route": {"passed": True, "value": 1.0}},
                }
                for i in range(30)
            ]
        )
        rendered = render_gate(gate(subject, subject, GateConfig()))
        assert "pass" in rendered.lower()
        assert "exit 0" in rendered.lower()

    def test_an_inconclusive_gate_says_rerun_rather_than_regression(self) -> None:
        """The doc is explicit that code 3 must not be misread as a quality
        regression, so the rendered text has to distinguish them."""
        from neverempty.report.gate import GateResult

        result = GateResult(
            verdict="inconclusive",
            exit_code=3,
            reason="inconclusive, rerun or reduce noise: 30.0% of cases are unstable",
            unstable_rate=0.30,
        )
        rendered = render_gate(result).lower()
        assert "inconclusive" in rendered
        assert "rerun" in rendered

    def test_an_invalid_gate_is_labelled_infrastructure(self) -> None:
        """Code 4 is an infrastructure failure and the doc requires it labelled
        as such, so nobody reads it as the agent getting worse."""
        from neverempty.report.gate import GateResult

        result = GateResult(
            verdict="invalid",
            exit_code=4,
            reason="candidate report is not complete",
        )
        rendered = render_gate(result).lower()
        assert "infrastructure" in rendered or "invalid input" in rendered

    def test_the_mcnemar_counts_are_rendered(self) -> None:
        from neverempty.report.gate import GateResult, McNemarDTO

        result = GateResult(
            verdict="regression",
            exit_code=1,
            reason="significant regression",
            mcnemar=McNemarDTO(b=10, c=1, p_value=0.0059, significant=True, alpha=0.05),
        )
        rendered = render_gate(result)
        assert "10" in rendered
        assert "0.0059" in rendered or "0.006" in rendered

    def test_warnings_are_rendered_separately_from_the_verdict(self) -> None:
        from neverempty.report.gate import GateResult

        result = GateResult(
            verdict="pass",
            exit_code=0,
            reason="no significant regression",
            warnings=["p95 latency rose from 100ms to 200ms"],
        )
        rendered = render_gate(result)
        assert "latency" in rendered
        assert "pass" in rendered.lower()
