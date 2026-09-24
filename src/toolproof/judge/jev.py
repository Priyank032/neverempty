"""Jev as a judge backend: typed decisions with a stated confidence.

Jev (TypeSafe AI) returns a choice from a fixed option set plus a confidence,
rather than free text. Claim verification is exactly that shape -- three labels
and a confidence -- so it fits behind ``JudgeModel`` with no change to the judge.

**It is not run or endorsed here.** Access is waitlisted, the architecture is
undisclosed, critics call it a repackaged zero-shot classifier, and the vendor's
calibration claim is a claim. This module exists so that the comparison the doc
wants can be run the moment access exists, and so that the result decides whether
the backend earns its place. Nothing in core imports it, and no number produced
by it may be published without a kappa beside it, on the same 60 human labels
every other judge is held to.

No SDK is imported. The caller passes a client object and this adapts it, which
is the same duck-typing choice the provider instrumentation makes: core stays at
one dependency, and the same code path is exercised by a fake with the real
shape.

The confidence is the point. A cheap judge is only worth escalating from if its
confidence means something, and that is measured by the reliability curve, not
taken from the vendor. ``JevJudge`` records every confidence it saw so the curve
can be built from a real run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from toolproof.judge.model import JudgeError

JEV_FAMILY = "jev"
"""Its own model family.

Which satisfies the different-family rule against an OpenAI agent, the same way
Bedrock does. Recorded explicitly rather than inferred from a model id, because
``infer_family`` knows nothing about this vendor.
"""

DEFAULT_LABELS: tuple[str, ...] = ("supported", "contradicted", "not_in_evidence")
"""The judge's three labels, as options for a typed decision.

Deliberately the same three the text judge uses. A backend that offered a
different label set would not be comparable against it, which is the entire
purpose of running this.
"""

MAX_OPTIONS = 255
"""Jev's documented ceiling on a choice decision.

Checked rather than trusted: passing more options than the vendor accepts would
fail at the call, and failing here names the reason.
"""


class JevClient(Protocol):
    """The shape this backend needs from a Jev client.

    A protocol rather than an import: the SDK is not a dependency, and a fake
    with this shape exercises the same code path as the real thing.
    """

    async def decide(
        self,
        *,
        prompt: str,
        options: Sequence[str],
        model: str,
    ) -> Any: ...


@dataclass(frozen=True)
class JevDecision:
    """One typed decision, as this backend reads it."""

    label: str
    confidence: float


@dataclass
class JevJudge:
    """A ``JudgeModel`` backed by a typed-decision API.

    Implements ``complete`` by making a choice decision over the label set and
    returning the JSON object the judge's parser already expects. That is what
    lets the same prompts, the same retries and the same cache serve both
    backends, so a comparison between them is a comparison of the models rather
    than of two different harnesses.
    """

    client: JevClient
    model: str
    labels: tuple[str, ...] = DEFAULT_LABELS
    confidences: list[float] = field(default_factory=list)
    """Every confidence seen, in call order.

    Kept so a reliability curve can be built from a real run: the vendor's
    calibration claim is not evidence, and this is the data that would settle it.
    """

    def __post_init__(self) -> None:
        if not self.labels:
            raise JudgeError("JevJudge needs at least one label option")
        if len(self.labels) > MAX_OPTIONS:
            raise JudgeError(
                f"JevJudge was given {len(self.labels)} options, above the "
                f"documented ceiling of {MAX_OPTIONS}"
            )
        if len(set(self.labels)) != len(self.labels):
            raise JudgeError(f"duplicate label options: {self.labels}")

    @property
    def family(self) -> str:
        return JEV_FAMILY

    async def complete(self, *, system: str, user: str, temperature: float) -> str:
        """Make a typed decision and render it as the judge's JSON shape.

        ``temperature`` is accepted and ignored: a typed-decision API has no
        sampling temperature to set. Ignoring it silently would be wrong for a
        text model, but here there is nothing to set, and raising would make the
        backend unusable behind a protocol whose whole point is substitutability.
        The judge calls at temperature 0 regardless.
        """
        decision = await self._decide(f"{system}\n\n{user}")
        self.confidences.append(decision.confidence)
        # Rendered as the same JSON the text judge's parser reads, so the parser,
        # the retry policy and the cache are shared rather than duplicated.
        return json.dumps(
            {
                "label": decision.label,
                "rationale": f"typed decision, confidence {decision.confidence:.3f}",
                "confidence": decision.confidence,
            }
        )

    async def _decide(self, prompt: str) -> JevDecision:
        try:
            raw = await self.client.decide(
                prompt=prompt, options=list(self.labels), model=self.model
            )
        except (TimeoutError, OSError) as exc:
            # Narrow on purpose: CancelledError, KeyboardInterrupt and
            # SystemExit are not caught here and propagate untouched.
            raise JudgeError(f"Jev call failed: {exc}") from exc

        label = _read(raw, ("label", "choice", "decision", "value"))
        confidence = _read(raw, ("confidence", "score", "probability"))

        if not isinstance(label, str) or label not in self.labels:
            raise JudgeError(
                f"Jev returned {label!r}, which is not one of {list(self.labels)}. "
                f"A typed decision outside its own option set is a backend fault, "
                f"not a fourth opinion."
            )
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise JudgeError(
                f"Jev returned no usable confidence for {label!r} (got "
                f"{confidence!r}). The confidence is the reason to use this "
                f"backend, so a decision without one is not accepted."
            )
        value = float(confidence)
        if not 0.0 <= value <= 1.0:
            raise JudgeError(
                f"Jev returned a confidence of {value!r}, outside [0, 1]. It is not "
                f"rescaled: a reliability curve computed from an undeclared scale "
                f"would not be reproducible."
            )
        return JevDecision(label=label, confidence=value)


def _read(payload: Any, names: Sequence[str]) -> Any:
    """Read the first present field, from an object or a mapping.

    The SDK's exact response shape is not pinned here, so both attribute and key
    access are tried. That is looser than this codebase usually allows, and the
    reason is stated rather than hidden: the vendor is waitlisted, the shape is
    unverified, and guessing one accessor would fail at the first real call.
    """
    for name in names:
        if isinstance(payload, dict):
            if name in payload:
                return payload[name]
        elif hasattr(payload, name):
            return getattr(payload, name)
    return None


__all__ = [
    "DEFAULT_LABELS",
    "JEV_FAMILY",
    "MAX_OPTIONS",
    "JevClient",
    "JevDecision",
    "JevJudge",
]
