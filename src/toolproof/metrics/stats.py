"""Pure-Python statistics for the report and the gate.

No SciPy, and no NumPy. The binomial tail is a short sum and Wilson is a closed
form, so the gate runs offline with pydantic as the only dependency. That is not
minimalism for its own sake: a CI gate that needs a scientific stack is a gate
someone eventually disables.

Every function here can fail a build, so each returns ``None`` rather than a
number when there is nothing to measure. An empty sample has no percentile and
no interval; reporting ``0`` for either would read as a real, excellent result.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, comb, sqrt

Z_95 = 1.959963984540054
"""Two-sided 95% normal quantile.

Spelled out rather than computed, because it is part of the published contract:
a report says ``method="wilson"`` and a reader has to be able to reproduce the
bounds exactly.
"""

BOOTSTRAP_RESAMPLES = 10_000
"""The doc's resample count. Fewer would be cheaper and visibly less stable."""


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval and the method that made it.

    The method name travels with the number because two reports can only be
    compared when they were computed the same way.
    """

    value: float
    low: float
    high: float
    method: str
    n: int


@dataclass(frozen=True)
class McNemarResult:
    """The paired regression test's verdict.

    ``b`` is pass-to-fail and ``c`` is fail-to-pass, both counted on the same
    frozen cases. Comparing outcomes per case rather than two proportions is
    what makes the test sensitive enough to be useful at n=210.
    """

    b: int
    c: int
    p_value: float
    significant: bool
    alpha: float

    @property
    def discordant(self) -> int:
        return self.b + self.c


def wilson_interval(successes: int, n: int, *, z: float = Z_95) -> Interval | None:
    """Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation because the normal one produces
    bounds outside [0, 1] at the extremes, and per-branch accuracy at n=14 is
    exactly that regime.

    Returns ``None`` for an empty sample: 0/0 is "not measured", not 0%.
    """
    if successes < 0 or n < 0:
        raise ValueError(f"counts must not be negative, got successes={successes}, n={n}")
    if successes > n:
        raise ValueError(f"successes ({successes}) cannot exceed n ({n})")
    if n == 0:
        return None

    proportion = successes / n
    denominator = 1 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    margin = z * sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator

    low = max(0.0, centre - margin)
    high = min(1.0, centre + margin)
    # At p=0 and p=1 the closed form lands within one ULP of the exact bound
    # (0.9999999999999999 for 30/30), which a reader reproducing the number by
    # hand would report as 1. The algebra is exact at the extremes, so the
    # bound is pinned rather than left as float noise.
    if successes == 0:
        low = 0.0
    if successes == n:
        high = 1.0

    return Interval(value=proportion, low=low, high=high, method="wilson", n=n)


def binomial_tail(successes: int, trials: int) -> float:
    """``P(X >= successes)`` for ``X ~ Binomial(trials, 0.5)``.

    Exact, via integer binomial coefficients, so there is no floating-point
    accumulation to argue about when a p-value lands near alpha.
    """
    if trials == 0:
        return 1.0
    total: int = sum(comb(trials, k) for k in range(successes, trials + 1))
    outcomes: int = 2**trials
    return total / outcomes


def mcnemar_exact(b: int, c: int, *, alpha: float = 0.05) -> McNemarResult:
    """One-sided exact McNemar test for a regression.

    One-sided deliberately: the gate exists to catch the candidate getting
    worse. An improvement (``c > b``) must never fail a build, which a
    two-sided test would allow.

    With no discordant pairs the p-value is 1: nothing flipped, so there is no
    evidence of change.
    """
    if b < 0 or c < 0:
        raise ValueError(f"counts must not be negative, got b={b}, c={c}")

    p_value = binomial_tail(b, b + c)
    return McNemarResult(
        b=b,
        c=c,
        p_value=p_value,
        significant=b > c and p_value < alpha,
        alpha=alpha,
    )


def bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
) -> Interval | None:
    """Percentile bootstrap interval for the mean of a continuous metric.

    Used for fact recall, which is a mean of per-case fractions rather than a
    count of successes, so Wilson does not apply.

    The RNG is a local ``random.Random``, never the module-level one: seeding
    the global RNG would make an unrelated caller's sampling depend on ours.
    """
    if not values:
        return None

    sample = list(values)
    point = statistics.fmean(sample)
    size = len(sample)

    if size == 1:
        # One observation cannot be resampled into a spread. The mean is known
        # exactly and the interval is degenerate, which is the honest report.
        return Interval(value=point, low=point, high=point, method="bootstrap", n=1)

    rng = random.Random(seed)  # noqa: S311 - seeded resampling, not crypto
    means = sorted(statistics.fmean(rng.choices(sample, k=size)) for _ in range(resamples))
    tail = (1.0 - confidence) / 2.0
    low = means[_percentile_index(len(means), tail)]
    high = means[_percentile_index(len(means), 1.0 - tail)]
    return Interval(value=point, low=low, high=high, method="bootstrap", n=size)


def _percentile_index(count: int, quantile: float) -> int:
    """Index into a sorted list for a percentile, clamped to the list."""
    position = round(quantile * (count - 1))
    return min(max(position, 0), count - 1)


def nearest_rank(values: Sequence[float], quantile: float) -> float | None:
    """Nearest-rank percentile: the value at ``ceil(q * n)``, 1-indexed.

    No interpolation, so every reported percentile is a latency that actually
    occurred. An interpolated p95 is a number no request ever took.

    Returns ``None`` for an empty sample, never 0: a p95 of 0 ms would read as
    an extraordinarily fast run.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be within [0, 1], got {quantile}")
    if not values:
        return None

    ordered = sorted(values)
    if quantile == 0.0:
        return ordered[0]
    rank = ceil(len(ordered) * quantile)
    return ordered[min(rank, len(ordered)) - 1]


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "Z_95",
    "Interval",
    "McNemarResult",
    "binomial_tail",
    "bootstrap_ci",
    "mcnemar_exact",
    "nearest_rank",
    "wilson_interval",
]
