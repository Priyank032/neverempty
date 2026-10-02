"""D24: provider-prefixed model ids infer no family, so a gateway is unusable.

``infer_family`` matches on prefix, which fits a bare id like ``gpt-4o``. Every
aggregator namespaces instead -- OpenRouter, Together, LiteLLM and Bedrock all
write ``<provider>/<model>`` or ``<provider>.<model>`` -- so::

    anthropic/claude-sonnet-4   -> None
    openai/gpt-4o               -> None
    meta-llama/llama-3.3-70b    -> None

and the judge then refuses to start, because a judge whose family is unknown
cannot be checked against the agent's. That refusal is right and the inference
was wrong: the family is right there in the string.

Getting this correct matters more than convenience. The whole point of the
family check is that a model must not grade its own output, and a gateway makes
it *easy* to accidentally run both sides on the same underlying model while the
ids look different.
"""

from __future__ import annotations

import pytest

from neverempty.judge.judge import infer_family


class TestAGatewayPrefixIsUnderstood:
    @pytest.mark.parametrize(
        ("model_id", "family"),
        [
            ("anthropic/claude-sonnet-4", "anthropic"),
            ("openai/gpt-4o", "openai"),
            ("openai/o3-mini", "openai"),
            ("google/gemini-2.5-pro", "google"),
            ("mistralai/mistral-large", "mistral"),
            ("cohere/command-r-plus", "cohere"),
        ],
    )
    def test_the_provider_segment_names_the_family(self, model_id: str, family: str) -> None:
        assert infer_family(model_id) == family

    @pytest.mark.parametrize(
        "model_id",
        ["meta-llama/llama-3.3-70b-instruct", "meta/llama-3.1-8b"],
    )
    def test_a_hyphenated_vendor_segment_is_read(self, model_id: str) -> None:
        assert infer_family(model_id) == "meta"

    def test_a_bedrock_style_dotted_id_still_works(self) -> None:
        assert infer_family("anthropic.claude-3-5-sonnet-20241022-v2:0") == "anthropic"


class TestTheModelSegmentIsTheFallback:
    """When the vendor segment is unknown, the model name may still say."""

    def test_an_unknown_vendor_with_a_known_model(self) -> None:
        assert infer_family("someproxy/claude-sonnet-4") == "anthropic"

    def test_an_unknown_vendor_and_model_is_still_unknown(self) -> None:
        """Refusing beats guessing: an unknown family cannot be checked."""
        assert infer_family("acme/internal-model-v2") is None


class TestBareIdsAreUnchanged:
    @pytest.mark.parametrize(
        ("model_id", "family"),
        [
            ("gpt-4o", "openai"),
            ("claude-sonnet-5", "anthropic"),
            ("gemini-2.5-pro", "google"),
            ("mistral-large-latest", "mistral"),
        ],
    )
    def test_they_resolve_as_before(self, model_id: str, family: str) -> None:
        assert infer_family(model_id) == family


class TestTheFamilyCheckStillBites:
    """A gateway makes it easy to run both sides on one model while the ids
    look different. The check has to survive that."""

    def test_a_judge_matching_the_agents_family_is_refused(self) -> None:
        from neverempty.judge.judge import ClaimJudge

        class Scripted:
            async def complete(self, *, system: str, user: str, temperature: float) -> str:
                return "{}"

        with pytest.raises(Exception, match="family"):
            ClaimJudge(
                model=Scripted(),
                model_id="openai/gpt-4o",
                agent_family="openai",
            )

    def test_a_different_family_through_a_gateway_is_accepted(self) -> None:
        from neverempty.judge.judge import ClaimJudge

        class Scripted:
            async def complete(self, *, system: str, user: str, temperature: float) -> str:
                return "{}"

        judge = ClaimJudge(
            model=Scripted(),
            model_id="anthropic/claude-sonnet-4",
            agent_family="openai",
        )
        assert judge is not None


class TestTheFamiliesAGatewayServes:
    """A gateway routes to far more vendors than a direct SDK does, and each
    one a judge might run on has to be nameable -- otherwise the family check
    cannot tell whether the judge shares the agent's model."""

    @pytest.mark.parametrize(
        ("model_id", "family"),
        [
            ("deepseek/deepseek-chat", "deepseek"),
            ("qwen/qwen-2.5-72b-instruct", "qwen"),
            ("x-ai/grok-2", "xai"),
            ("amazon/nova-pro-v1", "amazon"),
        ],
    )
    def test_each_is_recognised(self, model_id: str, family: str) -> None:
        assert infer_family(model_id) == family

    def test_an_unlisted_vendor_still_refuses(self) -> None:
        """The list grows by evidence, not by guessing."""
        assert infer_family("acme/internal-model-v2") is None


class TestTheDocumentedGatewayBindingWorks:
    """``docs/writing-labels.md`` tells a reader to pass the gateway's own id
    verbatim. If that construction fails, the advice is worse than none."""

    class _Gateway:
        """The documented shape, with the network replaced."""

        def __init__(self, model: str) -> None:
            self._model = model

        async def complete(self, *, system: str, user: str, temperature: float) -> str:
            return '{"label": "supported", "rationale": "ok"}'

    def test_an_openrouter_id_constructs(self) -> None:
        from neverempty.judge.judge import ClaimJudge

        judge = ClaimJudge(
            model=self._Gateway("anthropic/claude-sonnet-4"),
            model_id="anthropic/claude-sonnet-4",
            agent_family="openai",
        )
        assert judge is not None

    async def test_it_judges(self) -> None:
        from neverempty.judge.judge import Claim, ClaimJudge

        judge = ClaimJudge(
            model=self._Gateway("anthropic/claude-sonnet-4"),
            model_id="anthropic/claude-sonnet-4",
            agent_family="openai",
        )
        verdicts = await judge.verify(claims=[Claim(id="c1", text="x")], evidence={"e": "y"})
        assert verdicts[0].label == "supported"

    def test_the_same_family_through_a_gateway_is_still_refused(self) -> None:
        """The failure a gateway makes easy: both sides on one model."""
        from neverempty.judge.judge import ClaimJudge

        with pytest.raises(Exception, match="family"):
            ClaimJudge(
                model=self._Gateway("openai/gpt-4o"),
                model_id="openai/gpt-4o",
                agent_family="openai",
            )

    def test_the_docs_warn_about_silent_routing(self) -> None:
        from pathlib import Path

        text = (Path(__file__).resolve().parents[2] / "docs" / "writing-labels.md").read_text(
            encoding="utf-8"
        )
        assert "Silent routing" in text
        assert "base_url" in text

    def test_the_docs_say_the_binding_is_unverified(self) -> None:
        """No request has been sent to a gateway from this repository. The
        snippet proves construction and the family check, not the network."""
        from pathlib import Path

        text = (Path(__file__).resolve().parents[2] / "docs" / "writing-labels.md").read_text(
            encoding="utf-8"
        )
        assert "Unverified against a live gateway" in text
