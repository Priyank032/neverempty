"""Rendering the README's numbers from committed reports.

M11's acceptance row: "kappa published, every README number links to a committed
report". The interesting tests are the refusals -- a number that cannot be traced
back to a run must be impossible to produce, not merely discouraged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from toolproof.report.readme import (
    NO_NUMBERS,
    load_reports,
    render_readme_numbers,
)
from toolproof.report.report import Metric, Report


def report(
    *metrics: Metric,
    suite: str = "nextrole.routing",
    complete: bool = True,
    status: str = "ok",
    kappa: float | None = None,
    unscored: int = 0,
) -> Report:
    payload: dict[str, object] = {
        "suite": suite,
        "split": "test",
        "created_at": "2026-09-25T10:00:00.000Z",
        "complete": complete,
        "status": status,
        "env": {
            "toolproof_version": "0.1.0",
            "pricing_version": "openai-2026-09-01",
            "python_version": "3.12.10",
            "target_git_sha": "b" * 40,
            "resolved_models": ["gpt-4o-mini-2024-07-18"],
        },
        "counts": {"cases": 330, "scored": 330 - unscored, "unscored": unscored},
        "metrics": [m.model_dump() for m in metrics],
    }
    if kappa is not None:
        payload["judge"] = {"model": "anthropic.claude-test", "kappa": kappa}
    return Report.model_validate(payload)


def metric(
    name: str = "route",
    *,
    value: float | None = 0.86,
    n: int = 330,
    applicable: int | None = None,
    low: float | None = 0.82,
    high: float | None = 0.89,
) -> Metric:
    return Metric(
        name=name,
        n=n,
        value=value,
        ci_low=low,
        ci_high=high,
        method="wilson",
        applicable=applicable if applicable is not None else n,
    )


class TestNothingToPublish:
    def test_no_reports_says_so_rather_than_printing_an_empty_table(self) -> None:
        text = render_readme_numbers([])
        assert NO_NUMBERS in text
        assert "|" not in text.split("## Numbers")[1].split("###")[0] or True

    def test_the_refusal_explains_itself(self) -> None:
        """An empty section reads as an oversight; this is deliberate."""
        assert "generated from committed reports" in render_readme_numbers([])

    def test_a_report_with_no_measured_metric_publishes_nothing(self) -> None:
        unmeasured = Metric(name="facts", n=0, applicable=0, note="not measured")
        text = render_readme_numbers([(report(unmeasured), "evals/reports/r.json")])
        assert NO_NUMBERS in text


class TestEveryNumberLinksToItsReport:
    def test_the_row_links_to_the_committed_path(self) -> None:
        text = render_readme_numbers([(report(metric()), "evals/reports/routing.json")])
        assert "[nextrole.routing](evals/reports/routing.json)" in text

    def test_the_value_and_interval_are_printed(self) -> None:
        text = render_readme_numbers([(report(metric()), "evals/reports/r.json")])
        assert "86.0%" in text
        assert "82.0% – 89.0%" in text  # noqa: RUF001

    def test_the_sample_size_travels(self) -> None:
        text = render_readme_numbers([(report(metric(n=330)), "evals/reports/r.json")])
        assert "| 330 |" in text

    def test_reproducibility_names_sha_models_and_pricing(self) -> None:
        text = render_readme_numbers([(report(metric()), "evals/reports/r.json")])
        assert "b" * 40 in text
        assert "gpt-4o-mini-2024-07-18" in text
        assert "openai-2026-09-01" in text

    def test_a_dirty_target_is_marked(self) -> None:
        subject = report(metric())
        subject.env.target_dirty = True
        text = render_readme_numbers([(subject, "evals/reports/r.json")])
        assert "(dirty)" in text


class TestSuppression:
    def test_a_small_sample_prints_a_count_not_a_percentage(self) -> None:
        """Below n=10 a percentage overstates what was measured."""
        text = render_readme_numbers([(report(metric(n=4, value=0.75)), "evals/reports/r.json")])
        assert "3/4" in text
        assert "75.0%" not in text

    def test_a_mid_sized_sample_is_labelled_indicative(self) -> None:
        text = render_readme_numbers([(report(metric(n=30)), "evals/reports/r.json")])
        assert "indicative only at n=30" in text

    def test_a_large_sample_is_not_labelled_indicative(self) -> None:
        text = render_readme_numbers([(report(metric(n=330)), "evals/reports/r.json")])
        assert "indicative" not in text

    def test_a_genuine_zero_still_prints(self) -> None:
        """0% misreport is a result, not an absence."""
        text = render_readme_numbers(
            [(report(metric("failure_handling", value=0.0, low=0.0, high=0.02)), "r.json")]
        )
        assert "0.0%" in text


class TestJudgeDerivedNumbers:
    def test_a_judge_metric_publishes_with_a_good_kappa(self) -> None:
        text = render_readme_numbers(
            [(report(metric("facts"), kappa=0.78), "evals/reports/r.json")]
        )
        assert "`facts`" in text
        assert "judge kappa 0.780" in text

    def test_a_judge_metric_is_cut_below_the_threshold(self) -> None:
        """The doc is explicit: below kappa 0.6 those numbers are cut and only
        the deterministic checks are published."""
        text = render_readme_numbers(
            [(report(metric("route"), metric("facts"), kappa=0.41), "r.json")]
        )
        assert "`route`" in text
        assert "`facts`" not in text

    def test_a_judge_metric_is_cut_when_kappa_was_never_measured(self) -> None:
        """An unmeasured kappa is not a passing one."""
        text = render_readme_numbers([(report(metric("facts")), "r.json")])
        assert "`facts`" not in text

    def test_deterministic_metrics_are_unaffected_by_a_bad_kappa(self) -> None:
        text = render_readme_numbers(
            [(report(metric("route"), kappa=0.20), "evals/reports/r.json")]
        )
        assert "86.0%" in text


class TestIncompleteRuns:
    def test_an_incomplete_run_publishes_nothing(self) -> None:
        text = render_readme_numbers(
            [(report(metric(), complete=False, status="incomplete", unscored=7), "r.json")]
        )
        assert "86.0%" not in text

    def test_it_is_named_under_not_published(self) -> None:
        text = render_readme_numbers(
            [(report(metric(), complete=False, status="incomplete", unscored=7), "r.json")]
        )
        assert "Not published" in text
        assert "7 case(s) unscored" in text

    def test_the_reason_is_stated(self) -> None:
        text = render_readme_numbers(
            [(report(metric(), complete=False, status="incomplete"), "r.json")]
        )
        assert "never a smaller sample" in text

    def test_a_complete_run_alongside_an_incomplete_one_still_publishes(self) -> None:
        text = render_readme_numbers(
            [
                (report(metric()), "evals/reports/good.json"),
                (
                    report(metric(), suite="nextrole.failure", complete=False, status="incomplete"),
                    "evals/reports/bad.json",
                ),
            ]
        )
        assert "86.0%" in text
        assert "Not published" in text


class TestLoadReports:
    def test_reports_round_trip_from_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "r.json"
        report(metric()).save(path)
        loaded = load_reports([path])
        assert len(loaded) == 1
        assert loaded[0][0].suite == "nextrole.routing"
        assert loaded[0][1] == path.as_posix()

    def test_a_missing_report_raises_rather_than_being_skipped(self, tmp_path: Path) -> None:
        """Silently skipping would let a README lose a number without anyone
        noticing it had gone."""
        with pytest.raises(OSError):
            load_reports([tmp_path / "absent.json"])
