"""The config entrypoint and the recording stubs."""

from __future__ import annotations

import pytest

from neverempty.core.results import Ok
from neverempty.evals.nextrole import SIDE_EFFECT_TOOLS
from neverempty.evals.targets import STUBS, RecordingStub, run_nextrole


class TestRecordingStub:
    def test_records_its_calls(self) -> None:
        stub = RecordingStub("send_gmail")
        stub(to="a@example.com", subject="hi")
        stub(to="b@example.com", subject="there")
        assert len(stub.calls) == 2
        assert stub.calls[0]["to"] == "a@example.com"

    def test_returns_an_explicit_ok_not_none(self) -> None:
        """A stub returning None would raise under strict mode, and a falsy
        return would be exactly the ambiguity this library removes."""
        result = RecordingStub("send_gmail")(to="a@example.com")
        assert isinstance(result, Ok)
        assert result.value["stubbed"] == "send_gmail"

    def test_the_call_index_travels_so_after_calls_faults_are_checkable(self) -> None:
        stub = RecordingStub("save_application")
        first = stub(job="x")
        second = stub(job="y")
        assert isinstance(first, Ok)
        assert isinstance(second, Ok)
        assert first.value["call_index"] == 0
        assert second.value["call_index"] == 1

    async def test_async_call_records_the_same_way(self) -> None:
        stub = RecordingStub("send_gmail")
        result = await stub.acall(to="a@example.com")
        assert isinstance(result, Ok)
        assert len(stub.calls) == 1

    def test_reset_clears_the_record(self) -> None:
        stub = RecordingStub("send_gmail")
        stub(to="a@example.com")
        stub.reset()
        assert stub.calls == []

    def test_no_network_is_reachable_from_a_stub(self) -> None:
        """The stub takes arbitrary kwargs and returns without doing anything.

        Asserted by construction rather than by mocking a socket: there is no
        branch in ``__call__`` that could reach out.
        """
        stub = RecordingStub("send_gmail")
        result = stub(to="real.recruiter@company.com", body="please hire me")
        assert isinstance(result, Ok)
        assert result.value == {"stubbed": "send_gmail", "call_index": 0}


class TestStubs:
    def test_every_side_effect_tool_has_a_stub(self) -> None:
        """The runner refuses to start otherwise, which is the check that stops a
        fault run from emailing a real recruiter."""
        for name in SIDE_EFFECT_TOOLS:
            assert name in STUBS

    def test_stubs_are_populated_at_import(self) -> None:
        """Preflight reads this mapping before any case runs, so it cannot be
        lazily empty at that moment."""
        assert len(STUBS) == len(SIDE_EFFECT_TOOLS)


class TestEntrypoint:
    async def test_without_the_env_var_it_says_what_to_set(self) -> None:
        """A missing agent repo is a setup problem with an obvious fix, and must
        not read as a neverempty bug."""
        from neverempty.evals import targets

        targets._adapter = None
        with pytest.raises(RuntimeError, match="NEXTROLE_GRAPH_FACTORY"):
            await run_nextrole(object(), object())  # type: ignore[arg-type]

    async def test_a_malformed_spec_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from neverempty.evals import targets

        targets._adapter = None
        monkeypatch.setenv("NEXTROLE_GRAPH_FACTORY", "no_colon_here")
        with pytest.raises(RuntimeError, match="module:name"):
            await run_nextrole(object(), object())  # type: ignore[arg-type]

    async def test_a_missing_attribute_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from neverempty.evals import targets

        targets._adapter = None
        monkeypatch.setenv("NEXTROLE_GRAPH_FACTORY", "json:not_a_thing")
        with pytest.raises(RuntimeError, match="not_a_thing"):
            await run_nextrole(object(), object())  # type: ignore[arg-type]

    async def test_a_non_callable_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from neverempty.evals import targets

        targets._adapter = None
        monkeypatch.setenv("NEXTROLE_GRAPH_FACTORY", "json:__doc__")
        with pytest.raises(RuntimeError, match="not callable"):
            await run_nextrole(object(), object())  # type: ignore[arg-type]

    def test_importing_targets_needs_no_agent_repo(self) -> None:
        """neverempty's own tests, wheel build and ``validate`` must all work
        without ai-career-copilot on the path."""
        import importlib

        module = importlib.import_module("neverempty.evals.targets")
        assert module.run_nextrole is not None


class _FakeRouter:
    """Records the call and answers like ``IntentRouterAgent.classify_intent``."""

    def __init__(
        self, intent: object = "job_search", confidence: float = 0.9, clarify: str | None = None
    ) -> None:
        self.intent = intent
        self.confidence = confidence
        self.clarify = clarify
        self.calls: list[dict[str, object]] = []

    async def classify_intent(self, **kwargs: object) -> object:
        from types import SimpleNamespace

        self.calls.append(kwargs)
        return SimpleNamespace(
            intent=self.intent,
            confidence=self.confidence,
            requires_clarification=self.clarify is not None,
            clarification_question=self.clarify,
        )


def _case(messages: list[dict[str, str]], suite: str = "nextrole.routing") -> object:
    from neverempty.dataset.case import Case

    return Case.model_validate(
        {
            "id": "nr-route-9999",
            "suite": suite,
            "split": "dev",
            "input": {"messages": messages},
            "expect": {"route": {"label": "job_search"}},
        }
    )


class _FakeRun:
    def __init__(self) -> None:
        self.output: dict[str, object] = {}

    def set_output(self, **kwargs: object) -> None:
        self.output.update(kwargs)


class TestLiveRouter:
    async def test_routes_the_final_user_turn_with_history_including_it(self) -> None:
        """chat.py saves the user message before loading history, so the live
        router sees the current turn in both places; the target must match."""
        from neverempty.evals.nextrole_live import route_live

        router = _FakeRouter()
        case = _case(
            [
                {"role": "user", "content": "show me python jobs in pune"},
                {"role": "assistant", "content": "Here are 8 roles"},
                {"role": "user", "content": "only remote ones"},
            ]
        )
        intent, confidence = await route_live(case, router)  # type: ignore[arg-type]
        assert (intent, confidence) == ("job_search", 0.9)
        call = router.calls[0]
        assert call["message"] == "only remote ones"
        history = call["conversation_history"]
        assert [m.content for m in history] == [  # type: ignore[attr-defined]
            "show me python jobs in pune",
            "Here are 8 roles",
            "only remote ones",
        ]
        assert call["user_profile"] is None
        assert call["context"] is None

    async def test_an_enum_intent_is_reported_by_value(self) -> None:
        import enum

        from neverempty.evals.nextrole_live import route_live

        class Intent(str, enum.Enum):
            FOLLOWUP = "followup"

        intent, _ = await route_live(
            _case([{"role": "user", "content": "x"}]),  # type: ignore[arg-type]
            _FakeRouter(intent=Intent.FOLLOWUP),
        )
        assert intent == "followup"

    @pytest.mark.parametrize(
        "fallback",
        [
            "I encountered an issue understanding your request. Could you try again?",
            "I'm not sure I understood that. Could you please rephrase your request?",
        ],
    )
    async def test_the_router_error_fallback_raises_instead_of_scoring_general(
        self, fallback: str
    ) -> None:
        """Scored as a route, an outage would count as correct on every general
        case: the misreport this library exists to catch, inside the eval."""
        from neverempty.evals.nextrole_live import RouterFailedError, route_live

        router = _FakeRouter(intent="general", confidence=0.3, clarify=fallback)
        with pytest.raises(RouterFailedError):
            await route_live(_case([{"role": "user", "content": "x"}]), router)  # type: ignore[arg-type]

    async def test_a_genuine_clarification_is_still_a_route(self) -> None:
        from neverempty.evals.nextrole_live import route_live

        router = _FakeRouter(
            intent="job_search", confidence=0.5, clarify="Which city are you looking in?"
        )
        intent, _ = await route_live(_case([{"role": "user", "content": "jobs pls"}]), router)  # type: ignore[arg-type]
        assert intent == "job_search"

    async def test_the_entrypoint_records_route_and_confidence(self) -> None:
        from types import SimpleNamespace

        from neverempty.evals import targets

        targets._router = _FakeRouter(intent="salary_estimate", confidence=0.8)
        run = _FakeRun()
        try:
            await targets.run_nextrole_router(
                _case([{"role": "user", "content": "sde2 pay in blr"}]),  # type: ignore[arg-type]
                SimpleNamespace(current_run=run),
            )
        finally:
            targets._router = None
        assert run.output == {"route": "salary_estimate", "structured": {"confidence": 0.8}}

    async def test_the_entrypoint_refuses_a_failure_suite(self) -> None:
        """A failure case must execute a branch; this target never does."""
        from neverempty.evals import targets

        with pytest.raises(RuntimeError, match="routing suites"):
            await targets.run_nextrole_router(
                _case([{"role": "user", "content": "x"}], suite="nextrole.failure"),  # type: ignore[arg-type]
                object(),
            )
