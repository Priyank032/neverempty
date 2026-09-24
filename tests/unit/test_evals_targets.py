"""The config entrypoint and the recording stubs."""

from __future__ import annotations

import pytest

from toolproof.core.results import Ok
from toolproof.evals.nextrole import SIDE_EFFECT_TOOLS
from toolproof.evals.targets import STUBS, RecordingStub, run_nextrole


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
        not read as a toolproof bug."""
        from toolproof.evals import targets

        targets._adapter = None
        with pytest.raises(RuntimeError, match="NEXTROLE_GRAPH_FACTORY"):
            await run_nextrole(object(), object())  # type: ignore[arg-type]

    async def test_a_malformed_spec_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from toolproof.evals import targets

        targets._adapter = None
        monkeypatch.setenv("NEXTROLE_GRAPH_FACTORY", "no_colon_here")
        with pytest.raises(RuntimeError, match="module:name"):
            await run_nextrole(object(), object())  # type: ignore[arg-type]

    async def test_a_missing_attribute_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from toolproof.evals import targets

        targets._adapter = None
        monkeypatch.setenv("NEXTROLE_GRAPH_FACTORY", "json:not_a_thing")
        with pytest.raises(RuntimeError, match="not_a_thing"):
            await run_nextrole(object(), object())  # type: ignore[arg-type]

    async def test_a_non_callable_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from toolproof.evals import targets

        targets._adapter = None
        monkeypatch.setenv("NEXTROLE_GRAPH_FACTORY", "json:__doc__")
        with pytest.raises(RuntimeError, match="not callable"):
            await run_nextrole(object(), object())  # type: ignore[arg-type]

    def test_importing_targets_needs_no_agent_repo(self) -> None:
        """toolproof's own tests, wheel build and ``validate`` must all work
        without ai-career-copilot on the path."""
        import importlib

        module = importlib.import_module("toolproof.evals.targets")
        assert module.run_nextrole is not None
