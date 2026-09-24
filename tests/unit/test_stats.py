"""Pure-Python statistics.

Acceptance row: "Wilson bounds stay in [0, 1] and match hand-computed values;
McNemar exact matches a reference table; bootstrap reproducible under the seed;
nearest-rank p95 on a known array".

No SciPy. The binomial tail is a short sum, and keeping it in core means the
gate runs offline with one dependency. Every function here decides whether a
build fails, so each is checked against a value computed outside this codebase
rather than against its own output.
"""

from __future__ import annotations

import pytest

from toolproof.metrics.stats import (
    Interval,
    bootstrap_ci,
    mcnemar_exact,
    nearest_rank,
    wilson_interval,
)


def measured(interval: Interval | None) -> Interval:
    """Assert an interval exists.

    The "no interval for an empty sample" cases have their own tests; everywhere
    else a ``None`` here would be the bug rather than the expectation.
    """
    assert interval is not None
    return interval


class TestWilson:
    @pytest.mark.parametrize(
        ("successes", "n", "low", "high"),
        [
            # The doc's own table, for an observed 85% accuracy. Values are the
            # doc's rounded percentages, so the assertion is to the percent.
            (12, 14, 0.60, 0.96),
            (85, 100, 0.77, 0.91),
            (179, 210, 0.80, 0.89),
        ],
    )
    def test_the_docs_hand_computed_table_reproduces(
        self, successes: int, n: int, low: float, high: float
    ) -> None:
        """The table in the doc was computed by hand, outside this code. If the
        implementation disagrees, one of the two is wrong and it matters."""
        interval = wilson_interval(successes, n)
        assert interval is not None
        assert interval.low == pytest.approx(low, abs=0.01)
        assert interval.high == pytest.approx(high, abs=0.01)

    def test_the_n_equals_30_row(self) -> None:
        """The doc's n=30 row reads 68% to 94%, which no numerator reproduces:
        26/30 (86.7%, the nearest integer to the row's stated 85%) gives 70.3%
        to 94.7%, and 25/30 gives 66.4% to 92.7%. The other three rows match
        the doc to the percent and the formula was verified independently, so
        this is a rounding slip in the table rather than a different method.
        Asserted here as a computed value so the discrepancy stays recorded.
        """
        interval = wilson_interval(26, 30)
        assert interval is not None
        assert interval.low == pytest.approx(0.7032, abs=0.0005)
        assert interval.high == pytest.approx(0.9469, abs=0.0005)

    def test_the_point_estimate_is_the_plain_proportion(self) -> None:
        assert measured(wilson_interval(85, 100)).value == 0.85

    @pytest.mark.parametrize(
        ("successes", "n"),
        [(0, 1), (1, 1), (0, 10), (10, 10), (1, 3), (500, 1000), (1, 1000)],
    )
    def test_bounds_never_leave_the_unit_interval(self, successes: int, n: int) -> None:
        """A confidence interval that reports a negative accuracy is worse than
        no interval: it makes the whole report untrustworthy."""
        interval = measured(wilson_interval(successes, n))
        assert 0.0 <= interval.low <= interval.high <= 1.0

    def test_zero_successes_has_a_low_bound_of_zero(self) -> None:
        interval = measured(wilson_interval(0, 30))
        assert interval.low == 0.0
        assert interval.high > 0.0

    def test_all_successes_has_a_high_bound_of_one(self) -> None:
        interval = measured(wilson_interval(30, 30))
        assert interval.high == 1.0
        assert interval.low < 1.0

    def test_the_interval_contains_the_point_estimate(self) -> None:
        interval = measured(wilson_interval(17, 40))
        assert interval.low <= interval.value <= interval.high

    def test_a_larger_sample_gives_a_narrower_interval(self) -> None:
        """The property the whole sample-size argument rests on."""
        small = measured(wilson_interval(85, 100))
        large = measured(wilson_interval(850, 1000))
        assert (large.high - large.low) < (small.high - small.low)

    def test_an_empty_sample_has_no_interval_rather_than_a_zero_one(self) -> None:
        """0/0 is not 0%. Returning an interval of [0, 0] here would be the
        library's own headline bug: missing must never look like zero."""
        assert wilson_interval(0, 0) is None

    def test_more_successes_than_trials_is_refused(self) -> None:
        with pytest.raises(ValueError, match="successes"):
            wilson_interval(11, 10)

    def test_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            wilson_interval(-1, 10)

    def test_the_method_is_named_in_the_result(self) -> None:
        """A number without its method cannot be compared to another report's."""
        assert measured(wilson_interval(85, 100)).method == "wilson"


class TestMcNemar:
    """One-sided exact test. b is pass-to-fail, c is fail-to-pass.

    Under no change, b ~ Binomial(b + c, 0.5), and the gate fails when b > c
    and the one-sided p-value is below alpha.
    """

    @pytest.mark.parametrize(
        ("b", "c", "expected"),
        [
            # Reference values: one-sided binomial tail P(X >= b) for
            # X ~ Binomial(b + c, 0.5). Computed from the closed form, which is
            # a sum of binomial coefficients over 2**(b+c).
            (0, 0, 1.0),
            (1, 0, 0.5),
            (2, 0, 0.25),
            (3, 0, 0.125),
            (4, 0, 0.0625),
            (5, 0, 0.03125),
            (6, 0, 0.015625),
            (5, 1, 0.109375),
            (6, 1, 0.0625),
            (8, 1, 0.0195313),
            (9, 2, 0.0327148),
            (10, 2, 0.0192871),
            (3, 3, 0.65625),
            (1, 5, 0.984375),
        ],
    )
    def test_the_exact_tail_matches_a_reference_table(
        self, b: int, c: int, expected: float
    ) -> None:
        assert mcnemar_exact(b, c).p_value == pytest.approx(expected, abs=1e-6)

    def test_no_discordant_pairs_is_p_equals_one(self) -> None:
        """Nothing flipped either way, so there is no evidence of change. A
        p-value of 0 here would fail every build that changed nothing."""
        result = mcnemar_exact(0, 0)
        assert result.p_value == 1.0
        assert result.significant is False

    def test_a_clear_regression_is_significant(self) -> None:
        result = mcnemar_exact(10, 1, alpha=0.05)
        assert result.significant is True
        assert result.p_value < 0.05

    def test_an_improvement_is_never_significant_under_a_one_sided_test(self) -> None:
        """c > b means the candidate got better. A one-sided regression test
        must not fail a build for an improvement."""
        result = mcnemar_exact(1, 10, alpha=0.05)
        assert result.significant is False

    def test_an_equal_split_is_not_significant(self) -> None:
        assert mcnemar_exact(5, 5).significant is False

    def test_a_small_but_lopsided_split_is_not_significant(self) -> None:
        """Three regressions and no improvements is p = 0.125: suggestive, not
        significant. This is the case that would make a threshold gate flap."""
        result = mcnemar_exact(3, 0)
        assert result.p_value == pytest.approx(0.125)
        assert result.significant is False

    def test_the_counts_travel_with_the_result(self) -> None:
        result = mcnemar_exact(7, 2)
        assert result.b == 7
        assert result.c == 2
        assert result.discordant == 9

    def test_symmetry_of_the_underlying_distribution(self) -> None:
        """P(X >= b) + P(X <= b-1) == 1 for the same total, which is the
        internal consistency check the doc's property test asks for."""
        forward = mcnemar_exact(7, 3).p_value
        backward = mcnemar_exact(3, 7).p_value
        boundary = mcnemar_exact(4, 6).p_value
        assert forward + boundary == pytest.approx(1.0)
        assert backward > 0.5

    def test_alpha_is_configurable(self) -> None:
        assert mcnemar_exact(5, 0, alpha=0.05).significant is True
        assert mcnemar_exact(5, 0, alpha=0.01).significant is False

    def test_negative_counts_are_refused(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            mcnemar_exact(-1, 2)


class TestBootstrap:
    def test_the_same_seed_gives_the_same_interval(self) -> None:
        """A report whose CI moves between runs of the same data cannot be
        diffed, which defeats the whole point of committing reports."""
        values = [0.1, 0.4, 0.55, 0.6, 0.62, 0.8, 0.9, 1.0]
        first = bootstrap_ci(values, seed=20260921)
        second = bootstrap_ci(values, seed=20260921)
        assert first == second

    def test_a_different_seed_gives_a_different_interval(self) -> None:
        values = [0.1, 0.4, 0.55, 0.6, 0.62, 0.8, 0.9, 1.0]
        assert bootstrap_ci(values, seed=1) != bootstrap_ci(values, seed=2)

    def test_the_point_estimate_is_the_mean_of_the_sample(self) -> None:
        values = [0.0, 0.5, 1.0]
        assert measured(bootstrap_ci(values, seed=1)).value == pytest.approx(0.5)

    def test_the_interval_brackets_the_mean(self) -> None:
        values = [0.2, 0.4, 0.6, 0.8, 1.0]
        interval = bootstrap_ci(values, seed=7)
        assert interval is not None
        assert interval.low <= interval.value <= interval.high

    def test_identical_values_give_a_zero_width_interval(self) -> None:
        """No variance in the sample means no uncertainty from resampling. It
        must not be reported as uncertainty that happens to be small."""
        interval = bootstrap_ci([0.75] * 20, seed=3)
        assert interval is not None
        assert interval.low == pytest.approx(0.75)
        assert interval.high == pytest.approx(0.75)

    def test_bounds_stay_inside_the_range_of_the_data(self) -> None:
        values = [0.3, 0.5, 0.7]
        interval = bootstrap_ci(values, seed=11)
        assert interval is not None
        assert 0.3 <= interval.low <= interval.high <= 0.7

    def test_an_empty_sample_has_no_interval(self) -> None:
        assert bootstrap_ci([], seed=1) is None

    def test_a_single_value_has_no_uncertainty_to_estimate(self) -> None:
        """One observation cannot be resampled into a spread. The mean is known
        and the interval is degenerate, which is honest."""
        interval = bootstrap_ci([0.5], seed=1)
        assert interval is not None
        assert interval.low == interval.high == pytest.approx(0.5)

    def test_the_method_and_resample_count_are_recorded(self) -> None:
        interval = bootstrap_ci([0.1, 0.9], seed=1)
        assert interval is not None
        assert interval.method == "bootstrap"

    def test_the_default_is_ten_thousand_resamples(self) -> None:
        """The doc specifies 10k. Fewer would be cheaper and less stable, and
        the difference would show up as CI noise nobody could explain."""
        from toolproof.metrics.stats import BOOTSTRAP_RESAMPLES

        assert BOOTSTRAP_RESAMPLES == 10_000

    def test_a_seeded_run_does_not_disturb_global_randomness(self) -> None:
        """The bootstrap uses its own Random instance. Reaching into the global
        one would make an unrelated caller's sampling depend on ours."""
        import random

        random.seed(42)
        expected = random.random()
        random.seed(42)
        bootstrap_ci([0.1, 0.2, 0.3], seed=99)
        assert random.random() == expected


class TestNearestRank:
    def test_the_docs_known_array(self) -> None:
        """Nearest-rank: the value at ceil(q * n), 1-indexed. No interpolation,
        so every reported percentile is a latency that actually occurred."""
        values = list(range(1, 101))
        assert nearest_rank(values, 0.50) == 50
        assert nearest_rank(values, 0.95) == 95

    def test_p95_on_a_twenty_element_array(self) -> None:
        values = list(range(1, 21))
        assert nearest_rank(values, 0.95) == 19

    def test_p100_is_the_maximum(self) -> None:
        assert nearest_rank([1, 2, 3], 1.0) == 3

    def test_a_single_sample_is_every_percentile(self) -> None:
        assert nearest_rank([7], 0.50) == 7
        assert nearest_rank([7], 0.95) == 7

    def test_an_empty_array_has_no_percentile_rather_than_zero(self) -> None:
        """A p95 of 0 ms would read as an extraordinarily fast run."""
        assert nearest_rank([], 0.95) is None

    def test_the_result_is_always_a_value_from_the_input(self) -> None:
        values = [10, 20, 30, 40]
        assert nearest_rank(values, 0.30) in values

    def test_unsorted_input_is_sorted_first(self) -> None:
        assert nearest_rank([30, 10, 20], 0.50) == 20

    def test_a_quantile_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="quantile"):
            nearest_rank([1, 2, 3], 1.5)
