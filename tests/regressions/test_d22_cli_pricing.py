"""D22: a CLI run can never price anything, so ``max_cost_usd`` never works.

The config has ``max_cost_usd`` but no way to declare prices. The runner builds
a Tracer with the empty pricing table, every cost comes back null, and the D4
fix then correctly refuses to run -- a budget that cannot be measured against
is not a budget. The result is that ``max_cost_usd`` in a TOML config aborts
every run on its second batch, whatever the number.

Found running the NextRole routing suite against the live router: it stopped
after 3 of 330 cases with ``status=aborted_budget`` and
``pricing_version='empty-2026-09-23'``.

D4 is behaving correctly. The gap is that there was no way to give it prices.
``[pricing]`` closes it, and the fields ``ModelPrice`` already requires --
``as_of`` and ``source_url`` -- carry over, because an undated price cannot be
audited and an uncited one cannot be checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neverempty.config import ConfigError, load_config

BASE = """\
[project]
name = "demo"

[target]
entrypoint = "m:f"

[[suite]]
name = "demo.routing"
path = "datasets/r.jsonl"
split = "dev"
scorers = ["route"]
"""

PRICING = """\

[pricing]
version = "openai-2026-09-01"

[pricing.models."gpt-4o"]
input_usd_per_mtok = 2.50
output_usd_per_mtok = 10.00
cached_input_usd_per_mtok = 1.25
as_of = "2026-09-01"
source_url = "https://openai.com/api/pricing/"
"""


def _config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "neverempty.toml"
    path.write_text(body, encoding="utf-8")
    return path


class TestPricesCanBeDeclared:
    def test_a_config_with_prices_loads(self, tmp_path: Path) -> None:
        config = load_config(str(_config(tmp_path, BASE + PRICING)))
        assert config.pricing is not None

    def test_the_version_travels(self, tmp_path: Path) -> None:
        config = load_config(str(_config(tmp_path, BASE + PRICING)))
        assert config.pricing is not None
        assert config.pricing.version == "openai-2026-09-01"

    def test_a_model_is_priced(self, tmp_path: Path) -> None:
        from neverempty.core.trace import Usage

        config = load_config(str(_config(tmp_path, BASE + PRICING)))
        assert config.pricing is not None
        cost = config.pricing.cost_for(Usage(input_tokens=1_000_000, output_tokens=0), ["gpt-4o"])
        assert cost.usd == pytest.approx(2.50)

    def test_cached_tokens_are_a_subset(self, tmp_path: Path) -> None:
        """The convention settled in D5 has to hold through the config too."""
        from neverempty.core.trace import Usage

        config = load_config(str(_config(tmp_path, BASE + PRICING)))
        assert config.pricing is not None
        cost = config.pricing.cost_for(
            Usage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=1_000_000),
            ["gpt-4o"],
        )
        assert cost.usd == pytest.approx(1.25)


class TestAnUnauditablePriceIsRefused:
    """``ModelPrice`` already requires these; the config must not bypass them."""

    def test_a_price_without_as_of_is_refused(self, tmp_path: Path) -> None:
        body = BASE + PRICING.replace('as_of = "2026-09-01"\n', "")
        with pytest.raises(ConfigError, match="as_of"):
            load_config(str(_config(tmp_path, body)))

    def test_a_price_without_a_source_is_refused(self, tmp_path: Path) -> None:
        body = BASE + PRICING.replace('source_url = "https://openai.com/api/pricing/"\n', "")
        with pytest.raises(ConfigError, match="source_url"):
            load_config(str(_config(tmp_path, body)))

    def test_a_negative_price_is_refused(self, tmp_path: Path) -> None:
        body = BASE + PRICING.replace("input_usd_per_mtok = 2.50", "input_usd_per_mtok = -1")
        with pytest.raises(ConfigError):
            load_config(str(_config(tmp_path, body)))


class TestNoPricingSectionIsStillValid:
    """Pricing is optional: a run without a budget does not need it."""

    def test_a_config_without_prices_loads(self, tmp_path: Path) -> None:
        config = load_config(str(_config(tmp_path, BASE)))
        assert config.pricing is None
