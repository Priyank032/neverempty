"""Acceptance row for ``pricing``.

    unknown model yields ``usd=null`` plus ``unknown_reason``, never 0; pricing
    version recorded

This is the library's own principle turned on itself: missing must never look
like zero. A cost of 0.0 is a claim that the call was free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neverempty import Pricing
from neverempty.core.trace import Usage
from neverempty.tracer.pricing import ModelPrice


def table() -> Pricing:
    return Pricing(
        version="test-2026-09-23",
        models={
            "gpt-4o-2024-08-06": ModelPrice(
                input_usd_per_mtok=2.50,
                output_usd_per_mtok=10.00,
                cached_input_usd_per_mtok=1.25,
                as_of="2026-09-23",
                source_url="https://example.invalid/pricing",
            )
        },
    )


class TestDefaultTable:
    def test_the_default_table_ships_empty(self) -> None:
        """No price ships that cannot be cited. An unverifiable number behind
        every cost figure is worse than an explicit null."""
        assert Pricing.default().models == {}

    def test_the_default_version_names_itself_as_empty(self) -> None:
        assert "empty" in Pricing.default().version

    def test_the_default_version_carries_a_date(self) -> None:
        assert any(ch.isdigit() for ch in Pricing.default().version)

    def test_every_cost_is_unknown_under_the_default_table(self) -> None:
        cost = Pricing.default().cost_for(Usage(input_tokens=1000, output_tokens=500), ["gpt-4o"])
        assert cost.usd is None
        assert cost.unknown_reason == "model_not_in_pricing_table"


class TestUnknownModel:
    def test_an_unknown_model_yields_null_and_a_reason(self) -> None:
        cost = table().cost_for(Usage(input_tokens=100, output_tokens=50), ["mystery-model"])
        assert cost.usd is None
        assert cost.unknown_reason == "model_not_in_pricing_table"

    def test_an_unknown_model_is_never_priced_at_zero(self) -> None:
        assert table().cost_for(Usage(input_tokens=1, output_tokens=1), ["nope"]).usd != 0.0

    def test_the_unknown_model_name_is_named_in_the_reason_detail(self) -> None:
        cost = table().cost_for(Usage(input_tokens=1, output_tokens=1), ["mystery-model"])
        assert cost.unknown_models == ["mystery-model"]

    def test_one_unknown_model_among_several_makes_the_whole_cost_unknown(self) -> None:
        """A partial total is a wrong total, and it would be indistinguishable
        from a cheap run."""
        cost = table().cost_for(
            Usage(input_tokens=100, output_tokens=50),
            ["gpt-4o-2024-08-06", "mystery-model"],
        )
        assert cost.usd is None
        assert cost.unknown_models == ["mystery-model"]


class TestMissingUsage:
    def test_missing_usage_yields_null_with_its_own_reason(self) -> None:
        cost = table().cost_for(Usage(), ["gpt-4o-2024-08-06"])
        assert cost.usd is None
        assert cost.unknown_reason == "usage_missing"

    def test_a_half_reported_usage_is_still_unknown(self) -> None:
        cost = table().cost_for(Usage(input_tokens=100, output_tokens=None), ["gpt-4o-2024-08-06"])
        assert cost.usd is None
        assert cost.unknown_reason == "usage_missing"

    def test_no_models_at_all_yields_null(self) -> None:
        cost = table().cost_for(Usage(input_tokens=1, output_tokens=1), [])
        assert cost.usd is None
        assert cost.unknown_reason == "no_model_recorded"


class TestKnownCost:
    def test_a_known_model_is_priced_from_its_per_million_rates(self) -> None:
        cost = table().cost_for(
            Usage(input_tokens=1_000_000, output_tokens=1_000_000),
            ["gpt-4o-2024-08-06"],
        )
        assert cost.usd == pytest.approx(12.50)
        assert cost.unknown_reason is None

    def test_a_partial_million_scales_linearly(self) -> None:
        cost = table().cost_for(Usage(input_tokens=1_000, output_tokens=500), ["gpt-4o-2024-08-06"])
        assert cost.usd == pytest.approx(2.50 / 1000 + 10.00 / 2000)

    def test_cached_input_tokens_are_priced_at_the_cached_rate(self) -> None:
        cost = table().cost_for(
            Usage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=1_000_000),
            ["gpt-4o-2024-08-06"],
        )
        assert cost.usd == pytest.approx(2.50 + 1.25)

    def test_a_genuinely_free_call_costs_zero_with_no_reason(self) -> None:
        """Zero is legitimate when it was measured, not when it was missing."""
        cost = table().cost_for(Usage(input_tokens=0, output_tokens=0), ["gpt-4o-2024-08-06"])
        assert cost.usd == 0.0
        assert cost.unknown_reason is None

    def test_the_pricing_version_is_recorded_on_every_cost(self) -> None:
        for usage in (Usage(), Usage(input_tokens=1, output_tokens=1)):
            cost = table().cost_for(usage, ["gpt-4o-2024-08-06"])
            assert cost.pricing_version == "test-2026-09-23"


class TestProvenance:
    def test_each_price_carries_an_as_of_date_and_a_source_url(self) -> None:
        price = table().models["gpt-4o-2024-08-06"]
        assert price.as_of == "2026-09-23"
        assert price.source_url.startswith("http")

    def test_a_price_without_a_source_url_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="source_url"):
            ModelPrice(
                input_usd_per_mtok=1.0,
                output_usd_per_mtok=1.0,
                as_of="2026-09-23",
                source_url="",
            )

    def test_a_price_without_an_as_of_date_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="as_of"):
            ModelPrice(
                input_usd_per_mtok=1.0,
                output_usd_per_mtok=1.0,
                as_of="",
                source_url="https://example.invalid",
            )

    def test_a_negative_rate_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="input_usd_per_mtok"):
            ModelPrice(
                input_usd_per_mtok=-1.0,
                output_usd_per_mtok=1.0,
                as_of="2026-09-23",
                source_url="https://example.invalid",
            )


class TestLoadingFromFile:
    def test_a_table_loads_from_json(self, tmp_path: Path) -> None:
        path = tmp_path / "pricing.json"
        path.write_text(
            json.dumps(
                {
                    "version": "mine-2026-09-23",
                    "models": {
                        "my-model": {
                            "input_usd_per_mtok": 1.0,
                            "output_usd_per_mtok": 2.0,
                            "as_of": "2026-09-23",
                            "source_url": "https://example.invalid/prices",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        pricing = Pricing.from_file(path)
        assert pricing.version == "mine-2026-09-23"
        cost = pricing.cost_for(Usage(input_tokens=1_000_000, output_tokens=0), ["my-model"])
        assert cost.usd == pytest.approx(1.0)

    def test_a_malformed_file_fails_loudly(self, tmp_path: Path) -> None:
        path = tmp_path / "pricing.json"
        path.write_text('{"version": "x"}', encoding="utf-8")
        with pytest.raises(ValueError, match="models"):
            Pricing.from_file(path)

    def test_a_missing_file_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Pricing.from_file(tmp_path / "absent.json")

    def test_a_round_trip_through_json_preserves_the_table(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        path.write_text(table().model_dump_json(), encoding="utf-8")
        assert Pricing.from_file(path).version == table().version
