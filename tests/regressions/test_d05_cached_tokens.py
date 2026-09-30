"""D5/D6: the cost path can be silently wrong in both directions.

D5 -- OpenAI's ``prompt_tokens`` already *includes* ``prompt_tokens_details.
cached_tokens``; the cached count is a subset, not an addition. ``cost_for``
charged the full input count at the full rate and then added the cached count
again at the cached rate, overstating a heavily cached call by about 3.9x.

The two providers disagree, which is why a single global rule is wrong:

- OpenAI  ``prompt_tokens``  includes  ``cached_tokens``   (subset)
- Bedrock ``inputTokens``    excludes  ``cacheReadInputTokens`` (separate)

So the ambiguity is normalised at the adapter boundary, where the provider is
known, and ``Usage.cached_input_tokens`` is documented as a subset of
``input_tokens``. That is the only reading under which the field name is true,
and it is checked rather than assumed.

D6 -- a trace with two LLM calls where only one reported tokens summed the one
it had and reported a confident total, so a partially observed run was priced
as a cheap one. Missing must never look like a number.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neverempty.core.trace import Trace, Usage
from neverempty.tracer.pricing import ModelPrice, Pricing


def _pricing() -> Pricing:
    return Pricing(
        version="test-2026-09-01",
        models={
            "m": ModelPrice(
                input_usd_per_mtok=3.0,
                output_usd_per_mtok=10.0,
                cached_input_usd_per_mtok=0.30,
                as_of="2026-09-01",
                source_url="https://example.invalid/pricing",
            )
        },
    )


class TestCachedTokensAreASubsetNotAnAddition:
    def test_the_reviewers_case_is_priced_correctly(self) -> None:
        """1000 prompt tokens of which 800 were cached, at $3 / $0.30."""
        usage = Usage(input_tokens=1000, output_tokens=0, cached_input_tokens=800)
        cost = _pricing().cost_for(usage, ["m"])
        # 200 fresh at $3/Mtok + 800 cached at $0.30/Mtok
        assert cost.usd == pytest.approx((200 * 3.0 + 800 * 0.30) / 1_000_000)

    def test_it_is_not_the_old_overcharge(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=0, cached_input_tokens=800)
        cost = _pricing().cost_for(usage, ["m"])
        assert cost.usd is not None
        assert cost.usd == pytest.approx(0.00084)
        assert cost.usd != pytest.approx(0.00324)

    def test_no_cached_tokens_prices_the_whole_input_fresh(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=0)
        cost = _pricing().cost_for(usage, ["m"])
        assert cost.usd == pytest.approx(1000 * 3.0 / 1_000_000)

    def test_a_fully_cached_call_charges_only_the_cached_rate(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=0, cached_input_tokens=1000)
        cost = _pricing().cost_for(usage, ["m"])
        assert cost.usd == pytest.approx(1000 * 0.30 / 1_000_000)

    def test_output_tokens_are_unaffected(self) -> None:
        usage = Usage(input_tokens=0, output_tokens=500)
        cost = _pricing().cost_for(usage, ["m"])
        assert cost.usd == pytest.approx(500 * 10.0 / 1_000_000)

    def test_a_table_without_a_cached_rate_charges_the_input_rate(self) -> None:
        """No cached price means no cached discount, not a free ride."""
        pricing = Pricing(
            version="t",
            models={
                "m": ModelPrice(
                    input_usd_per_mtok=3.0,
                    output_usd_per_mtok=10.0,
                    as_of="2026-09-01",
                    source_url="https://example.invalid/pricing",
                )
            },
        )
        usage = Usage(input_tokens=1000, output_tokens=0, cached_input_tokens=800)
        assert pricing.cost_for(usage, ["m"]).usd == pytest.approx(1000 * 3.0 / 1_000_000)


class TestMoreCachedThanInputIsRefused:
    """A subset cannot exceed its superset. Accepting it silently produced a
    negative fresh-token count, and a negative cost is not a cheap run."""

    def test_cached_above_input_is_a_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="cached_input_tokens"):
            Usage(input_tokens=100, output_tokens=0, cached_input_tokens=101)

    def test_cached_equal_to_input_is_fine(self) -> None:
        """A fully cached call is ordinary, not a contradiction."""
        usage = Usage(input_tokens=100, output_tokens=0, cached_input_tokens=100)
        assert usage.cached_input_tokens == 100

    def test_cached_without_input_is_still_refused(self) -> None:
        with pytest.raises(ValidationError, match="cached_input_tokens"):
            Usage(output_tokens=5, cached_input_tokens=10)

    def test_the_cost_is_never_negative(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=0, cached_input_tokens=1000)
        cost = _pricing().cost_for(usage, ["m"])
        assert cost.usd is not None
        assert cost.usd >= 0


class TestPartialUsageIsNotACheapRun:
    """D6: a trace whose LLM calls did not all report tokens.

    Two calls, one reporting 500/100 and one reporting nothing, produced
    ``usage_missing: False`` and a confident total -- a partially observed run
    priced as a cheap one. The cost is not merely imprecise, it is a different
    number from the truth with nothing marking it, which is this library's own
    rule ("missing never looks like zero") broken in its own cost path.
    """

    async def test_an_unreported_call_makes_the_cost_unknown(self) -> None:
        trace = await _two_calls(second_reports=False)
        assert trace.cost.usd is None
        assert trace.cost.unknown_reason == "usage_missing"

    async def test_the_usage_is_flagged_as_partial(self) -> None:
        """Distinct from ``usage_missing``: part of the run *was* observed, so
        the totals are real but describe less work than happened."""
        trace = await _two_calls(second_reports=False)
        assert trace.usage.usage_partial is True
        assert trace.usage.usage_missing is False

    async def test_the_observed_counts_are_still_recorded(self) -> None:
        """The partial measurement is kept; only the claim about totals goes."""
        trace = await _two_calls(second_reports=False)
        assert trace.usage.input_tokens == 500

    async def test_two_reporting_calls_still_price_normally(self) -> None:
        trace = await _two_calls(second_reports=True)
        assert trace.usage.usage_missing is False
        assert trace.cost.usd == pytest.approx((1000 * 3.0 + 200 * 10.0) / 1_000_000)

    async def test_a_run_with_no_llm_calls_is_unaffected(self) -> None:
        """No call means nothing was missed, which is not the same as a gap."""
        from neverempty import Tracer
        from neverempty.tracer.sinks import MemorySink

        sink = MemorySink()
        tracer = Tracer(sink=sink, pricing=_pricing())
        async with tracer.run():
            pass
        assert sink.traces[0].usage.usage_missing is True
        assert sink.traces[0].cost.unknown_reason == "no_model_recorded"


async def _two_calls(*, second_reports: bool) -> Trace:
    from neverempty import Tracer
    from neverempty.tracer.sinks import MemorySink

    sink = MemorySink()
    tracer = Tracer(sink=sink, pricing=_pricing())
    async with tracer.run():
        with tracer.span("llm", name="call1") as span:
            span.record_usage(model="m", input_tokens=500, output_tokens=100)
        with tracer.span("llm", name="call2") as span:
            if second_reports:
                span.record_usage(model="m", input_tokens=500, output_tokens=100)
            else:
                span.record_usage(model="m")
    return sink.traces[0]
