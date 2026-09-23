"""Cost, or an explicit null saying why it is unknown.

The default table is **empty**, and that is the design, not an omission. A
price this library cannot cite would put an unverifiable number behind every
cost figure in every report, which is precisely the failure the library exists
to prevent, turned on itself. So until a table is supplied, every cost is null
with ``model_not_in_pricing_table``, and a report says "cost unknown for 12/210
traces" rather than printing a total that is quietly wrong.

Supply your own, with a source and a date per model::

    Pricing.from_file("evals/pricing.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from toolproof.core.trace import Cost, Usage

DEFAULT_VERSION = "empty-2026-09-23"
"""Version string of the empty default table. Recorded in every report, so a
reader can tell that no prices were configured."""

TOKENS_PER_MTOK = 1_000_000


class ModelPrice(BaseModel):
    """Per-model rates, with the provenance that makes them checkable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_usd_per_mtok: float = Field(ge=0)
    output_usd_per_mtok: float = Field(ge=0)
    cached_input_usd_per_mtok: float | None = Field(default=None, ge=0)
    as_of: str
    source_url: str

    @field_validator("as_of")
    @classmethod
    def _as_of_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("as_of is required: an undated price cannot be audited")
        return value

    @field_validator("source_url")
    @classmethod
    def _source_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_url is required: an uncited price cannot be checked")
        return value


class Pricing(BaseModel):
    """A versioned pricing table. The version travels with every report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    models: dict[str, ModelPrice] = Field(default_factory=dict)

    @classmethod
    def default(cls) -> Pricing:
        """The empty table. Every cost comes back null with a reason."""
        return cls(version=DEFAULT_VERSION, models={})

    @classmethod
    def from_file(cls, path: str | Path) -> Pricing:
        """Load a table from JSON. Fails loudly on anything malformed."""
        target = Path(path)
        raw = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "models" not in raw:
            raise ValueError(
                f"{target} is not a pricing table: expected an object with 'version' "
                f"and 'models' keys"
            )
        return cls.model_validate(raw)

    def cost_for(self, usage: Usage, models: list[str]) -> Cost:
        """Price ``usage`` against every model that contributed to it.

        Returns a null cost with a reason whenever the answer would otherwise be
        a guess: no model recorded, usage not reported by the provider, or any
        model absent from the table. One unknown model makes the whole total
        unknown, because a partial total is indistinguishable from a cheap run.
        """
        if not models:
            return Cost(
                usd=None,
                pricing_version=self.version,
                unknown_reason="no_model_recorded",
            )

        unknown = [model for model in models if model not in self.models]
        if unknown:
            return Cost(
                usd=None,
                pricing_version=self.version,
                unknown_reason="model_not_in_pricing_table",
                unknown_models=sorted(set(unknown)),
            )

        if not usage.complete:
            return Cost(
                usd=None,
                pricing_version=self.version,
                unknown_reason="usage_missing",
            )

        # Several models in one trace cannot be attributed per call from the
        # aggregate, so the first is used and the rest are recorded. A trace
        # that mixes models should be priced per span; that lands with the
        # report in M7.
        price = self.models[models[0]]
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        cached = usage.cached_input_tokens or 0

        total = (
            input_tokens * price.input_usd_per_mtok
            + output_tokens * price.output_usd_per_mtok
            + cached * (price.cached_input_usd_per_mtok or price.input_usd_per_mtok)
        ) / TOKENS_PER_MTOK

        return Cost(usd=total, pricing_version=self.version, unknown_reason=None)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()


__all__ = ["DEFAULT_VERSION", "ModelPrice", "Pricing"]
