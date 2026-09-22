"""Acceptance row for ``core/results``.

every variant round-trips through JSON; ``truncated`` survives; pattern
matching covers all three; ``status`` is the only discriminator
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from toolproof import Empty, Err, Ok, ToolResult


def _roundtrip(result: ToolResult[Any]) -> dict[str, Any]:
    """Serialize and parse back, the way a sink or a trace reader would."""
    payload: dict[str, Any] = json.loads(result.to_json())
    return payload


class TestRoundTrip:
    def test_ok_round_trips(self) -> None:
        result = Ok(value=[{"id": 1}])
        assert _roundtrip(result) == {
            "status": "ok",
            "value": [{"id": 1}],
            "truncated": False,
        }

    def test_truncated_survives_the_round_trip(self) -> None:
        """The sibling bug: 100 rows of a larger set reported as 'there are 100'."""
        result = Ok(value=list(range(100)), truncated=True)
        assert _roundtrip(result)["truncated"] is True
        assert Ok[Any].model_validate(_roundtrip(result)).truncated is True

    def test_empty_round_trips_with_its_reason(self) -> None:
        result = Empty(reason="no rows matched the filter")
        assert _roundtrip(result) == {
            "status": "empty",
            "reason": "no rows matched the filter",
        }

    def test_empty_reason_defaults_to_null_not_empty_string(self) -> None:
        """Missing must not look like a value; null is the absence marker."""
        assert _roundtrip(Empty())["reason"] is None

    def test_err_round_trips(self) -> None:
        result = Err(
            kind="timeout",
            message="upstream took longer than 8.0s",
            retryable=True,
            cause="TimeoutError",
        )
        assert _roundtrip(result) == {
            "status": "error",
            "kind": "timeout",
            "message": "upstream took longer than 8.0s",
            "retryable": True,
            "cause": "TimeoutError",
        }

    @pytest.mark.parametrize(
        "value",
        [0, False, "", [], {}, None, 0.0],
        ids=["zero", "false", "empty-str", "empty-list", "empty-dict", "none", "zero-float"],
    )
    def test_falsy_values_survive_as_ok(self, value: object) -> None:
        """No falsiness inference anywhere: these are legitimate Ok values."""
        result: Ok[object] = Ok(value=value)
        assert result.status == "ok"
        assert _roundtrip(result)["value"] == value
        assert Ok[Any].model_validate(_roundtrip(result)).value == value


class TestDiscriminator:
    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (Ok(value=1), "ok"),
            (Empty(), "empty"),
            (Err(kind="timeout", message="m", retryable=True), "error"),
        ],
    )
    def test_status_literal_is_fixed_per_variant(
        self, result: ToolResult[Any], expected: str
    ) -> None:
        assert result.status == expected

    def test_status_cannot_be_reassigned(self) -> None:
        """Results are frozen: an Err can never be mutated into an Ok."""
        result = Err(kind="timeout", message="m", retryable=True)
        mutable: Any = result
        with pytest.raises((ValueError, TypeError)):
            mutable.status = "ok"

    def test_status_is_the_only_discriminator(self) -> None:
        """A parser keying on `status` alone can reconstruct any variant."""
        results: list[ToolResult[Any]] = [
            Ok(value=[1], truncated=True),
            Empty(reason="none matched"),
            Err(kind="upstream", message="503", retryable=True, cause="HTTPError"),
        ]
        by_status: dict[str, type[BaseModel]] = {
            "ok": Ok[Any],
            "empty": Empty,
            "error": Err,
        }
        for original in results:
            payload = _roundtrip(original)
            rebuilt = by_status[payload["status"]].model_validate(payload)
            assert rebuilt == original

    def test_wrong_status_for_a_variant_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="status"):
            Empty.model_validate({"status": "ok"})


class TestPatternMatching:
    @staticmethod
    def classify(result: ToolResult[Any]) -> str:
        match result:
            case Ok(value=v, truncated=True):
                return f"partial:{len(v)}"
            case Ok(value=v):
                return f"ok:{v}"
            case Empty(reason=r):
                return f"empty:{r}"
            case Err(kind="timeout"):
                return "timeout"
            case Err(kind=k):
                return f"error:{k}"

    def test_match_covers_every_variant(self) -> None:
        assert self.classify(Ok(value=7)) == "ok:7"
        assert self.classify(Ok(value=[1, 2], truncated=True)) == "partial:2"
        assert self.classify(Empty(reason="none")) == "empty:none"
        assert self.classify(Err(kind="timeout", message="m", retryable=True)) == "timeout"
        assert (
            self.classify(Err(kind="permission", message="m", retryable=False))
            == "error:permission"
        )

    def test_matching_a_falsy_ok_still_enters_the_ok_branch(self) -> None:
        """The bug in one line: `if result:` would skip this."""
        assert self.classify(Ok(value=0)) == "ok:0"


class TestModelFacingRenderer:
    def test_ok_renders_data_and_truncation_flag(self) -> None:
        rendered = json.loads(Ok(value=[{"id": 1}]).to_model())
        assert rendered == {"status": "ok", "data": [{"id": 1}], "truncated": False}

    def test_truncated_ok_tells_the_model_the_set_is_partial(self) -> None:
        rendered = json.loads(Ok(value=list(range(100)), truncated=True).to_model())
        assert rendered["truncated"] is True
        assert "note" in rendered
        assert "more" in rendered["note"].lower()

    def test_empty_states_success_and_absence_in_words(self) -> None:
        rendered = json.loads(Empty().to_model())
        assert rendered["status"] == "empty"
        note = rendered["note"].lower()
        assert "succeeded" in note
        assert "no matching records" in note

    def test_error_renders_exactly_status_kind_and_note(self) -> None:
        """Message and cause stay in the trace; they never reach the model."""
        rendered = json.loads(
            Err(
                kind="timeout",
                message="connect to db-prod-7.internal timed out",
                retryable=True,
                cause="TimeoutError",
            ).to_model()
        )
        assert set(rendered) == {"status", "kind", "note"}
        assert rendered["kind"] == "timeout"
        assert "db-prod-7.internal" not in json.dumps(rendered)
        assert "TimeoutError" not in json.dumps(rendered)

    def test_error_note_forbids_claiming_absence(self) -> None:
        """This string is the fix for the bug the library is named after."""
        note = json.loads(Err(kind="timeout", message="m", retryable=True).to_model())["note"]
        lowered = note.lower()
        assert "the tool failed" in lowered
        assert "do not say that no data exists" in lowered
        assert "you do not know whether" in lowered

    @pytest.mark.parametrize(
        "kind",
        ["timeout", "upstream", "validation", "permission", "rate_limit", "exception"],
    )
    def test_every_error_kind_renders_a_refusal_to_claim_absence(self, kind: str) -> None:
        rendered = json.loads(Err(kind=kind, message="m", retryable=False).to_model())  # type: ignore[arg-type]
        assert "no data exists" in rendered["note"].lower()

    def test_rendered_error_is_never_empty_looking(self) -> None:
        """An Err must not serialise to something a model reads as 'no results'."""
        rendered = Err(kind="upstream", message="m", retryable=True).to_model()
        assert rendered.strip() not in {"[]", "{}", "", "null", "0"}
        assert "error" in rendered

    def test_renderer_output_is_valid_json_for_every_variant(self) -> None:
        for result in (
            Ok(value={"k": "v"}),
            Empty(reason="none"),
            Err(kind="rate_limit", message="429", retryable=True),
        ):
            assert isinstance(json.loads(result.to_model()), dict)


class TestTypeAliasUsability:
    """``ToolResult`` must be usable in the two places the doc shows it."""

    def test_tool_result_is_subscriptable_as_a_return_annotation(self) -> None:
        assert ToolResult[list[dict[str, int]]] is not None

    def test_subscripting_survives_get_type_hints(self) -> None:
        from typing import get_type_hints

        def f() -> ToolResult[list[int]]:  # pragma: no cover - annotation only
            raise NotImplementedError

        assert get_type_hints(f)["return"] is not None

    def test_any_tool_result_builds_a_pydantic_tagged_union(self) -> None:
        """A field typed this way dispatches on ``status``, not by trial parsing."""
        from pydantic import BaseModel

        from toolproof import AnyToolResult

        class Holder(BaseModel):
            result: AnyToolResult

        cases: list[tuple[dict[str, Any], type]] = [
            ({"status": "ok", "value": [1], "truncated": True}, Ok),
            ({"status": "empty", "reason": "none"}, Empty),
            (
                {"status": "error", "kind": "timeout", "message": "m", "retryable": True},
                Err,
            ),
        ]
        for payload, expected in cases:
            assert isinstance(Holder(result=payload).result, expected)  # type: ignore[arg-type]

    def test_an_unknown_status_is_rejected_by_the_union(self) -> None:
        from pydantic import BaseModel, ValidationError

        from toolproof import AnyToolResult

        class Holder(BaseModel):
            result: AnyToolResult

        with pytest.raises(ValidationError):
            Holder(result={"status": "partial", "value": 1})  # type: ignore[arg-type]

    def test_fault_injected_never_reaches_serialized_output(self) -> None:
        """It is span metadata, not part of the result contract."""
        result = Err(kind="timeout", message="m", retryable=True, fault_injected=True)
        assert result.fault_injected is True
        assert "fault_injected" not in result.to_json()
        assert "fault_injected" not in result.to_model()
