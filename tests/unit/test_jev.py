"""The Jev judge backend.

M13's acceptance row: "both reliability curves published with n and cost; core
untouched, backend is an extra". The curves need vendor access, which does not
exist yet. What is testable now is the half that must hold regardless: the
backend sits behind JudgeModel, core does not import it, and it refuses a
decision it cannot trust rather than producing a number.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest

from toolproof.judge.jev import (
    DEFAULT_LABELS,
    JEV_FAMILY,
    MAX_OPTIONS,
    JevJudge,
)
from toolproof.judge.model import JudgeError


class FakeJev:
    """A client with the shape the protocol declares."""

    def __init__(self, *responses: Any, raises: BaseException | None = None) -> None:
        self.responses = list(responses)
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    async def decide(self, *, prompt: str, options: Sequence[str], model: str) -> Any:
        self.calls.append({"prompt": prompt, "options": options, "model": model})
        if self.raises is not None:
            raise self.raises
        if not self.responses:
            raise AssertionError("FakeJev ran out of queued responses")
        return self.responses.pop(0)


def judge(*responses: Any, **kwargs: Any) -> JevJudge:
    kwargs.setdefault("model", "jev-1")
    return JevJudge(client=FakeJev(*responses), **kwargs)


class TestItIsAJudgeModel:
    async def test_complete_returns_the_judges_json_shape(self) -> None:
        """Rendering the same JSON the text judge's parser reads is what lets the
        parser, the retries and the cache be shared between backends."""
        subject = judge({"label": "supported", "confidence": 0.91})
        raw = await subject.complete(system="sys", user="usr", temperature=0.0)
        payload = json.loads(raw)
        assert payload["label"] == "supported"
        assert payload["confidence"] == pytest.approx(0.91)

    async def test_it_satisfies_the_protocol_structurally(self) -> None:
        from toolproof.judge.model import JudgeModel

        subject = judge({"label": "supported", "confidence": 0.5})
        assert isinstance(subject, JudgeModel)

    async def test_the_prompt_carries_both_system_and_user(self) -> None:
        client = FakeJev({"label": "supported", "confidence": 0.5})
        subject = JevJudge(client=client, model="jev-1")
        await subject.complete(system="SYSTEM TEXT", user="USER TEXT", temperature=0.0)
        assert "SYSTEM TEXT" in client.calls[0]["prompt"]
        assert "USER TEXT" in client.calls[0]["prompt"]

    async def test_the_option_set_is_the_judges_three_labels(self) -> None:
        """A backend offering a different label set would not be comparable
        against the text judge, which is the whole point of running it."""
        client = FakeJev({"label": "contradicted", "confidence": 0.7})
        subject = JevJudge(client=client, model="jev-1")
        await subject.complete(system="s", user="u", temperature=0.0)
        assert client.calls[0]["options"] == list(DEFAULT_LABELS)

    async def test_the_pinned_model_id_travels(self) -> None:
        client = FakeJev({"label": "supported", "confidence": 0.5})
        subject = JevJudge(client=client, model="jev-1-2026-09-15")
        await subject.complete(system="s", user="u", temperature=0.0)
        assert client.calls[0]["model"] == "jev-1-2026-09-15"


class TestFamily:
    def test_it_declares_its_own_family(self) -> None:
        """Which satisfies the different-family rule against an OpenAI agent."""
        assert judge().family == JEV_FAMILY
        assert JEV_FAMILY not in {"openai", "anthropic"}


class TestConfidenceIsRecorded:
    async def test_every_confidence_is_kept_in_call_order(self) -> None:
        """The vendor's calibration claim is a claim; this is the data that
        would settle it."""
        subject = judge(
            {"label": "supported", "confidence": 0.9},
            {"label": "contradicted", "confidence": 0.4},
        )
        await subject.complete(system="s", user="u", temperature=0.0)
        await subject.complete(system="s", user="u", temperature=0.0)
        assert subject.confidences == [pytest.approx(0.9), pytest.approx(0.4)]

    async def test_the_recorded_confidences_feed_a_reliability_curve(self) -> None:
        from toolproof.metrics.reliability import reliability_curve

        subject = judge(*({"label": "supported", "confidence": 0.95},) * 4)
        for _ in range(4):
            await subject.complete(system="s", user="u", temperature=0.0)
        curve = reliability_curve([(c, True) for c in subject.confidences])
        assert curve.n == 4
        assert curve.buckets[0].label == "0.9-1.0"


class TestItRefusesWhatItCannotTrust:
    async def test_a_label_outside_the_option_set_raises(self) -> None:
        """A typed decision outside its own option set is a backend fault, not a
        fourth opinion."""
        subject = judge({"label": "probably_fine", "confidence": 0.9})
        with pytest.raises(JudgeError, match="not one of"):
            await subject.complete(system="s", user="u", temperature=0.0)

    async def test_a_missing_confidence_raises(self) -> None:
        """The confidence is the reason to use this backend at all."""
        subject = judge({"label": "supported"})
        with pytest.raises(JudgeError, match="confidence"):
            await subject.complete(system="s", user="u", temperature=0.0)

    async def test_a_boolean_confidence_is_not_a_number(self) -> None:
        subject = judge({"label": "supported", "confidence": True})
        with pytest.raises(JudgeError, match="confidence"):
            await subject.complete(system="s", user="u", temperature=0.0)

    async def test_a_confidence_outside_zero_to_one_is_not_rescaled(self) -> None:
        """A curve computed from an undeclared scale is not reproducible."""
        subject = judge({"label": "supported", "confidence": 95})
        with pytest.raises(JudgeError, match="outside"):
            await subject.complete(system="s", user="u", temperature=0.0)

    async def test_a_transport_failure_becomes_a_judge_error(self) -> None:
        subject = JevJudge(client=FakeJev(raises=TimeoutError("timed out")), model="jev-1")
        with pytest.raises(JudgeError, match="Jev call failed"):
            await subject.complete(system="s", user="u", temperature=0.0)

    async def test_cancellation_propagates_untouched(self) -> None:
        """One of the library's non-negotiables, enforced in every backend."""
        subject = JevJudge(client=FakeJev(raises=asyncio.CancelledError()), model="jev-1")
        with pytest.raises(asyncio.CancelledError):
            await subject.complete(system="s", user="u", temperature=0.0)

    async def test_keyboard_interrupt_propagates_untouched(self) -> None:
        subject = JevJudge(client=FakeJev(raises=KeyboardInterrupt()), model="jev-1")
        with pytest.raises(KeyboardInterrupt):
            await subject.complete(system="s", user="u", temperature=0.0)


class TestResponseShapes:
    """The SDK's shape is unverified, so both accessors are accepted."""

    async def test_a_mapping_response_is_read(self) -> None:
        subject = judge({"label": "supported", "confidence": 0.8})
        assert (
            json.loads(await subject.complete(system="s", user="u", temperature=0.0))["label"]
            == "supported"
        )

    async def test_an_object_response_is_read(self) -> None:
        class Decision:
            label = "contradicted"
            confidence = 0.6

        subject = judge(Decision())
        payload = json.loads(await subject.complete(system="s", user="u", temperature=0.0))
        assert payload["label"] == "contradicted"

    async def test_alternate_field_names_are_read(self) -> None:
        subject = judge({"choice": "not_in_evidence", "score": 0.55})
        payload = json.loads(await subject.complete(system="s", user="u", temperature=0.0))
        assert payload["label"] == "not_in_evidence"


class TestConstruction:
    def test_an_empty_option_set_is_refused(self) -> None:
        with pytest.raises(JudgeError, match="at least one label"):
            JevJudge(client=FakeJev(), model="jev-1", labels=())

    def test_duplicate_options_are_refused(self) -> None:
        with pytest.raises(JudgeError, match="duplicate"):
            JevJudge(client=FakeJev(), model="jev-1", labels=("a", "b", "a"))

    def test_more_than_the_documented_ceiling_is_refused(self) -> None:
        labels = tuple(f"o{i}" for i in range(MAX_OPTIONS + 1))
        with pytest.raises(JudgeError, match="ceiling"):
            JevJudge(client=FakeJev(), model="jev-1", labels=labels)

    def test_the_ceiling_is_the_documented_255(self) -> None:
        assert MAX_OPTIONS == 255


class TestCoreIsUntouched:
    def test_no_sdk_is_imported(self) -> None:
        """The vendor is waitlisted; importing its SDK would make an unavailable
        dependency mandatory for anyone reading this module."""
        import sys

        import toolproof.judge.jev  # noqa: F401

        loaded = [
            name for name in sys.modules if name.startswith("jev") and name != "toolproof.judge.jev"
        ]
        assert loaded == []

    def test_the_public_surface_does_not_export_it(self) -> None:
        """An unrun, unvalidated backend is not part of the promised API."""
        import toolproof

        assert "JevJudge" not in toolproof.__all__
