"""Acceptance row for ``core/tool``.

    each row of the classification table; predicate raising becomes
    ``validation``; ``CancelledError`` propagates; strict mode raises on
    undeclared empty; sync and async parity; signature preserved; under 1 ms
    overhead

The overhead benchmark lives in ``tests/unit/test_tool_overhead.py``.
"""

from __future__ import annotations

import asyncio
import inspect
import socket
import time
from typing import Any

import pytest

from toolproof import (
    AmbiguousEmptyError,
    Empty,
    Err,
    Ok,
    ToolResult,
    tool,
)


class HTTPStatusError(Exception):
    """Stands in for httpx.HTTPStatusError, which core must not depend on."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class TestClassificationTable:
    """One test per row of the default error classification table."""

    async def _raise(self, exc: BaseException) -> Err:
        @tool(never_empty=True)
        async def failing() -> str:
            raise exc

        result = await failing()
        assert isinstance(result, Err), f"expected Err, got {result!r}"
        return result

    @pytest.mark.parametrize(
        ("exc", "kind", "retryable"),
        [
            (asyncio.TimeoutError(), "timeout", True),
            (TimeoutError(), "timeout", True),
            (HTTPStatusError(429), "rate_limit", True),
            (HTTPStatusError(503), "upstream", True),
            (HTTPStatusError(500), "upstream", True),
            (ConnectionError("refused"), "upstream", True),
            (HTTPStatusError(401), "permission", False),
            (HTTPStatusError(403), "permission", False),
            (HTTPStatusError(404), "validation", False),
            (HTTPStatusError(422), "validation", False),
            (ValueError("bad"), "validation", False),
            (KeyError("missing"), "validation", False),
            (TypeError("wrong"), "validation", False),
            (RuntimeError("something else"), "exception", False),
        ],
        ids=[
            "asyncio-timeout",
            "builtin-timeout",
            "http-429",
            "http-503",
            "http-500",
            "connection-error",
            "http-401",
            "http-403",
            "http-404",
            "http-422",
            "value-error",
            "key-error",
            "type-error",
            "unknown",
        ],
    )
    async def test_table_row(self, exc: Exception, kind: str, retryable: bool) -> None:
        result = await self._raise(exc)
        assert result.kind == kind
        assert result.retryable is retryable

    async def test_socket_oserror_is_upstream(self) -> None:
        result = await self._raise(socket.gaierror("name resolution failed"))
        assert result.kind == "upstream"
        assert result.retryable is True

    async def test_cause_records_the_class_name_never_the_traceback(self) -> None:
        result = await self._raise(ValueError("secret path /home/me/keys.txt"))
        assert result.cause == "ValueError"
        assert "Traceback" not in (result.message or "")
        assert "/home/me/keys.txt" not in result.cause

    async def test_pydantic_validation_error_is_validation(self) -> None:
        from pydantic import BaseModel, ValidationError

        class M(BaseModel):
            x: int

        with pytest.raises(ValidationError) as caught:
            M.model_validate({"x": "not-an-int"})
        result = await self._raise(caught.value)
        assert result.kind == "validation"
        assert result.retryable is False


class TestClassifyErrorOverride:
    async def test_per_tool_hook_wins_over_the_table(self) -> None:
        def classify(exc: BaseException) -> Err | None:
            if isinstance(exc, ValueError):
                return Err(kind="rate_limit", message="mapped", retryable=True)
            return None

        @tool(never_empty=True, classify_error=classify)
        async def failing() -> str:
            raise ValueError("x")

        result = await failing()
        assert isinstance(result, Err)
        assert result.kind == "rate_limit"

    async def test_returning_none_falls_through_to_the_table(self) -> None:
        def classify(exc: BaseException) -> Err | None:
            return None

        @tool(never_empty=True, classify_error=classify)
        async def failing() -> str:
            raise ValueError("x")

        result = await failing()
        assert isinstance(result, Err)
        assert result.kind == "validation"


class TestNeverRaises:
    async def test_a_tool_never_raises_an_ordinary_exception(self) -> None:
        @tool(never_empty=True)
        async def boom() -> str:
            raise RuntimeError("kaboom")

        assert isinstance(await boom(), Err)


class TestCancellationPropagates:
    """Converting a cancellation into Err would make a budget abort look like
    an agent failure, which corrupts every number downstream."""

    @pytest.mark.parametrize("exc_type", [asyncio.CancelledError, KeyboardInterrupt, SystemExit])
    async def test_async_propagates_untouched(self, exc_type: type[BaseException]) -> None:
        @tool(never_empty=True)
        async def failing() -> str:
            raise exc_type()

        with pytest.raises(exc_type):
            await failing()

    @pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
    def test_sync_propagates_untouched(self, exc_type: type[BaseException]) -> None:
        @tool(never_empty=True)
        def failing() -> str:
            raise exc_type()

        with pytest.raises(exc_type):
            failing()

    async def test_real_task_cancellation_is_not_swallowed(self) -> None:
        started = asyncio.Event()

        @tool(never_empty=True)
        async def slow() -> str:
            started.set()
            await asyncio.sleep(3600)
            return "never"

        task = asyncio.create_task(slow())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


class TestPassThrough:
    """Rule 2: an explicit Ok, Empty or Err is returned untouched."""

    async def test_explicit_ok_passes_through(self) -> None:
        marker = Ok(value=[1, 2], truncated=True)

        @tool()
        async def t() -> ToolResult[list[int]]:
            return marker

        assert await t() is marker

    async def test_explicit_empty_passes_through_without_predicates(self) -> None:
        @tool()
        async def t() -> ToolResult[list[int]]:
            return Empty(reason="author decided")

        result = await t()
        assert isinstance(result, Empty)
        assert result.reason == "author decided"

    async def test_explicit_err_passes_through_unclassified(self) -> None:
        @tool()
        async def t() -> ToolResult[str]:
            return Err(kind="permission", message="denied", retryable=False)

        result = await t()
        assert isinstance(result, Err)
        assert result.kind == "permission"


class TestNoFalsinessInference:
    """Rule 3: a plain return becomes Empty only when empty_when says so."""

    @pytest.mark.parametrize(
        "value",
        [0, False, "", 0.0],
        ids=["zero", "false", "empty-string", "zero-float"],
    )
    async def test_falsy_scalars_are_ok_not_empty(self, value: object) -> None:
        @tool(never_empty=True)
        async def t() -> object:
            return value

        result = await t()
        assert isinstance(result, Ok)
        assert result.value == value

    async def test_empty_when_true_produces_empty(self) -> None:
        @tool(empty_when=lambda rows: len(rows) == 0)
        async def t() -> list[int]:
            return []

        assert isinstance(await t(), Empty)

    async def test_empty_when_false_keeps_a_falsy_value_as_ok(self) -> None:
        """empty_when is the only authority, even when the value is falsy."""

        @tool(empty_when=lambda n: n == -1)
        async def t() -> int:
            return 0

        result = await t()
        assert isinstance(result, Ok)
        assert result.value == 0

    async def test_empty_when_can_mark_a_truthy_value_empty(self) -> None:
        @tool(empty_when=lambda payload: payload["count"] == 0)
        async def t() -> dict[str, int]:
            return {"count": 0}

        assert isinstance(await t(), Empty)


class TestStrictMode:
    """Rule 4: an undeclared empty return is an authoring bug, raised loudly."""

    @pytest.mark.parametrize(
        "value",
        [None, [], {}, ()],
        ids=["none", "empty-list", "empty-dict", "empty-tuple"],
    )
    async def test_undeclared_empty_raises(self, value: object) -> None:
        @tool()
        async def t() -> object:
            return value

        with pytest.raises(AmbiguousEmptyError) as exc:
            await t()
        assert "t" in str(exc.value)
        assert "empty_when" in str(exc.value)

    def test_sync_undeclared_empty_raises(self) -> None:
        @tool()
        def t() -> list[int]:
            return []

        with pytest.raises(AmbiguousEmptyError):
            t()

    async def test_empty_when_satisfies_strict_mode(self) -> None:
        @tool(empty_when=lambda rows: len(rows) == 0)
        async def t() -> list[int]:
            return []

        assert isinstance(await t(), Empty)

    async def test_never_empty_satisfies_strict_mode(self) -> None:
        @tool(never_empty=True)
        async def t() -> list[int]:
            return []

        result = await t()
        assert isinstance(result, Ok)
        assert result.value == []

    async def test_strict_false_downgrades_to_ok_without_raising(self) -> None:
        """Production path: record the ambiguity, keep serving."""

        @tool(strict=False)
        async def t() -> list[int]:
            return []

        result = await t()
        assert isinstance(result, Ok)
        assert result.value == []

    async def test_falsy_scalars_do_not_trip_strict_mode(self) -> None:
        """0 and False are unambiguous values, not candidates for empty."""

        @tool()
        async def t() -> int:
            return 0

        assert isinstance(await t(), Ok)

    def test_declaring_both_empty_when_and_never_empty_is_an_authoring_error(self) -> None:
        with pytest.raises(ValueError, match="never_empty"):

            @tool(empty_when=lambda v: True, never_empty=True)
            async def t() -> list[int]:
                return []


class TestPredicateFailures:
    """Rule 5: a raising predicate is a validation error, never a silent Ok."""

    async def test_raising_empty_when_becomes_validation_error(self) -> None:
        @tool(empty_when=lambda rows: rows["missing"] == 0)
        async def t() -> list[int]:
            return [1]

        result = await t()
        assert isinstance(result, Err)
        assert result.kind == "validation"
        assert "empty_when" in result.message

    async def test_raising_truncated_when_becomes_validation_error(self) -> None:
        @tool(never_empty=True, truncated_when=lambda rows: 1 / 0 > 0)
        async def t() -> list[int]:
            return [1]

        result = await t()
        assert isinstance(result, Err)
        assert result.kind == "validation"
        assert "truncated_when" in result.message

    async def test_a_raising_predicate_never_yields_ok(self) -> None:
        @tool(empty_when=lambda v: 1 / 0 == 0)
        async def t() -> list[int]:
            return []

        assert not isinstance(await t(), Ok)


class TestTruncation:
    async def test_truncated_when_sets_the_flag(self) -> None:
        @tool(never_empty=True, truncated_when=lambda rows: len(rows) >= 100)
        async def t() -> list[int]:
            return list(range(100))

        result = await t()
        assert isinstance(result, Ok)
        assert result.truncated is True

    async def test_not_truncated_leaves_the_flag_false(self) -> None:
        @tool(never_empty=True, truncated_when=lambda rows: len(rows) >= 100)
        async def t() -> list[int]:
            return [1]

        result = await t()
        assert isinstance(result, Ok)
        assert result.truncated is False

    async def test_empty_wins_over_truncated(self) -> None:
        @tool(
            empty_when=lambda rows: len(rows) == 0,
            truncated_when=lambda rows: True,
        )
        async def t() -> list[int]:
            return []

        assert isinstance(await t(), Empty)


class TestTimeout:
    async def test_async_timeout_returns_err_timeout(self) -> None:
        @tool(never_empty=True, timeout_s=0.02)
        async def slow() -> str:
            await asyncio.sleep(5)
            return "never"

        result = await slow()
        assert isinstance(result, Err)
        assert result.kind == "timeout"
        assert result.retryable is True

    async def test_a_fast_tool_under_its_timeout_succeeds(self) -> None:
        @tool(never_empty=True, timeout_s=5.0)
        async def fast() -> str:
            return "done"

        result = await fast()
        assert isinstance(result, Ok)
        assert result.value == "done"

    def test_sync_timeout_is_advisory_and_documented_as_such(self) -> None:
        assert "advisory" in (tool.__doc__ or "").lower()


class TestSyncAsyncParity:
    @pytest.mark.parametrize(
        ("empty_when", "returned", "expected"),
        [
            (None, "value", Ok),
            (lambda v: len(v) == 0, [], Empty),
            (lambda v: len(v) == 0, [1], Ok),
        ],
        ids=["ok", "empty", "non-empty"],
    )
    async def test_same_inputs_give_the_same_variant(
        self,
        empty_when: Any,
        returned: Any,
        expected: type,
    ) -> None:
        kwargs: dict[str, Any] = {"empty_when": empty_when} if empty_when else {"never_empty": True}

        @tool(**kwargs)
        def sync_tool() -> Any:
            return returned

        @tool(**kwargs)
        async def async_tool() -> Any:
            return returned

        sync_result = sync_tool()
        async_result = await async_tool()
        assert isinstance(sync_result, expected)
        assert isinstance(async_result, expected)
        assert sync_result == async_result

    def test_sync_tool_returns_directly_not_a_coroutine(self) -> None:
        @tool(never_empty=True)
        def t() -> int:
            return 1

        assert not inspect.isawaitable(t())

    async def test_async_tool_stays_awaitable(self) -> None:
        @tool(never_empty=True)
        async def t() -> int:
            return 1

        coro = t()
        assert inspect.isawaitable(coro)
        assert isinstance(await coro, Ok)

    def test_errors_classify_identically_across_sync_and_async(self) -> None:
        @tool(never_empty=True)
        def sync_tool() -> str:
            raise ValueError("x")

        @tool(never_empty=True)
        async def async_tool() -> str:
            raise ValueError("x")

        sync_result = sync_tool()
        async_result = asyncio.run(async_tool())
        assert isinstance(sync_result, Err)
        assert isinstance(async_result, Err)
        assert sync_result.kind == async_result.kind == "validation"


class TestSignaturePreservation:
    """Rule 6: LangChain and OpenAI schema generation must see the original."""

    def test_name_doc_and_wrapped_are_preserved(self) -> None:
        @tool(never_empty=True)
        async def query_emissions(org_id: str, quarter: str = "Q1") -> list[dict[str, int]]:
            """Look up emissions for an org."""
            return [{"t": 1}]

        assert query_emissions.__name__ == "query_emissions"
        assert query_emissions.__doc__ == "Look up emissions for an org."
        assert hasattr(query_emissions, "__wrapped__")

    def test_signature_is_unchanged(self) -> None:
        @tool(never_empty=True)
        async def f(org_id: str, quarter: str = "Q1", *, limit: int = 10) -> list[int]:
            return [1]

        sig = inspect.signature(f)
        assert list(sig.parameters) == ["org_id", "quarter", "limit"]
        assert sig.parameters["quarter"].default == "Q1"
        assert sig.parameters["limit"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_type_hints_resolve_through_the_wrapper(self) -> None:
        @tool(never_empty=True)
        async def f(org_id: str, quarter: int) -> list[int]:
            return [1]

        hints = __import__("typing").get_type_hints(f)
        assert hints["org_id"] is str
        assert hints["quarter"] is int

    async def test_arguments_pass_through_positionally_and_by_keyword(self) -> None:
        @tool(never_empty=True)
        async def f(a: int, b: int = 2, *args: int, **kwargs: int) -> dict[str, Any]:
            return {"a": a, "b": b, "args": args, "kwargs": kwargs}

        result = await f(1, 5, 9, c=3)
        assert isinstance(result, Ok)
        assert result.value == {"a": 1, "b": 5, "args": (9,), "kwargs": {"c": 3}}

    def test_tool_name_defaults_to_the_function_name(self) -> None:
        @tool(never_empty=True)
        async def search_jobs() -> list[int]:
            return [1]

        assert search_jobs.toolproof_name == "search_jobs"

    def test_tool_name_is_overridable(self) -> None:
        @tool(name="search", never_empty=True)
        async def search_jobs() -> list[int]:
            return [1]

        assert search_jobs.toolproof_name == "search"


class TestNoTracerRequired:
    """Rule 7: outside a tracer context the wrapper works and records nothing."""

    async def test_works_with_no_tracer_active(self) -> None:
        @tool(never_empty=True)
        async def t() -> int:
            return 1

        assert isinstance(await t(), Ok)

    def test_decorating_has_no_import_time_side_effects(self) -> None:
        calls: list[str] = []

        @tool(never_empty=True)
        async def t() -> int:
            calls.append("ran")
            return 1

        assert calls == []


class TestSideEffectTag:
    def test_side_effect_defaults_to_false(self) -> None:
        @tool(never_empty=True)
        async def t() -> int:
            return 1

        assert t.toolproof_side_effect is False

    def test_side_effect_tag_is_recorded_for_runner_preflight(self) -> None:
        @tool(never_empty=True, side_effect=True)
        async def send_gmail() -> int:
            return 1

        assert send_gmail.toolproof_side_effect is True


class TestRetries:
    async def test_retries_are_off_by_default(self) -> None:
        attempts = 0

        @tool(never_empty=True)
        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            raise ConnectionError("down")

        assert isinstance(await flaky(), Err)
        assert attempts == 1

    async def test_only_retryable_errors_are_retried(self) -> None:
        attempts = 0

        @tool(never_empty=True, retries=2, retry_backoff_s=0.0)
        async def not_retryable() -> str:
            nonlocal attempts
            attempts += 1
            raise ValueError("bad input")

        result = await not_retryable()
        assert isinstance(result, Err)
        assert result.retryable is False
        assert attempts == 1

    async def test_retryable_errors_are_retried_up_to_the_limit(self) -> None:
        attempts = 0

        @tool(never_empty=True, retries=2, retry_backoff_s=0.0)
        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            raise ConnectionError("down")

        assert isinstance(await flaky(), Err)
        assert attempts == 3  # one attempt plus two retries

    async def test_a_retry_that_succeeds_returns_ok(self) -> None:
        attempts = 0

        @tool(never_empty=True, retries=3, retry_backoff_s=0.0)
        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("down")
            return "recovered"

        result = await flaky()
        assert isinstance(result, Ok)
        assert result.value == "recovered"
        assert attempts == 3


class TestSyncEdgeCases:
    """Sync paths that differ from the async ones."""

    def test_sync_retries_retryable_errors(self) -> None:
        attempts = 0

        @tool(never_empty=True, retries=2, retry_backoff_s=0.0)
        def flaky() -> str:
            nonlocal attempts
            attempts += 1
            raise ConnectionError("down")

        assert isinstance(flaky(), Err)
        assert attempts == 3

    def test_sync_retry_can_recover(self) -> None:
        attempts = 0

        @tool(never_empty=True, retries=2, retry_backoff_s=0.0)
        def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ConnectionError("down")
            return "recovered"

        result = flaky()
        assert isinstance(result, Ok)
        assert result.value == "recovered"

    def test_sync_overrunning_its_advisory_timeout_reports_timeout(self) -> None:
        """The call completes, but the caller is told the deadline passed."""

        @tool(never_empty=True, timeout_s=0.001)
        def slow() -> str:
            time.sleep(0.05)
            return "late"

        result = slow()
        assert isinstance(result, Err)
        assert result.kind == "timeout"
        assert "advisory" in result.message
        assert "ran to completion" in result.message

    def test_sync_within_its_timeout_returns_ok(self) -> None:
        @tool(never_empty=True, timeout_s=5.0)
        def fast() -> str:
            return "quick"

        assert isinstance(fast(), Ok)

    def test_sync_predicate_failure_is_a_validation_error(self) -> None:
        @tool(empty_when=lambda rows: rows["nope"] == 0)
        def t() -> list[int]:
            return [1]

        result = t()
        assert isinstance(result, Err)
        assert result.kind == "validation"


class TestBackoff:
    def test_zero_backoff_returns_zero(self) -> None:
        @tool(never_empty=True, retries=1, retry_backoff_s=0.0)
        def t() -> int:
            return 1

        assert t.__wrapped__ is not None

    def test_backoff_grows_and_stays_within_its_jitter_band(self) -> None:
        from toolproof.core.tool import _ToolSpec

        spec = _ToolSpec(
            name="t",
            empty_when=None,
            truncated_when=None,
            never_empty=True,
            strict=True,
            timeout_s=None,
            retries=3,
            retry_backoff_s=0.1,
            side_effect=False,
            classify_error=None,
        )
        for attempt in range(4):
            base = 0.1 * (2**attempt)
            delay = spec.backoff_for(attempt)
            assert base * 0.5 <= delay <= base, f"attempt {attempt}: {delay}"

    def test_non_positive_backoff_never_sleeps(self) -> None:
        from toolproof.core.tool import _ToolSpec

        spec = _ToolSpec(
            name="t",
            empty_when=None,
            truncated_when=None,
            never_empty=True,
            strict=True,
            timeout_s=None,
            retries=1,
            retry_backoff_s=0.0,
            side_effect=False,
            classify_error=None,
        )
        assert spec.backoff_for(5) == 0.0


class TestDecoratorArgumentValidation:
    def test_negative_retries_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="retries"):
            tool(retries=-1)

    def test_negative_after_calls_is_rejected(self) -> None:
        from toolproof import FaultSpec

        with pytest.raises(ValueError, match="after_calls"):
            FaultSpec(tool="t", kind="timeout", after_calls=-1)


class TestPredicateCancellation:
    """A predicate that is cancelled must not become a validation error."""

    async def test_cancellation_inside_empty_when_propagates(self) -> None:
        def cancel(_: object) -> bool:
            raise asyncio.CancelledError

        @tool(empty_when=cancel)
        async def t() -> list[int]:
            return [1]

        with pytest.raises(asyncio.CancelledError):
            await t()

    async def test_cancellation_inside_truncated_when_propagates(self) -> None:
        def cancel(_: object) -> bool:
            raise KeyboardInterrupt

        @tool(never_empty=True, truncated_when=cancel)
        async def t() -> list[int]:
            return [1]

        with pytest.raises(KeyboardInterrupt):
            await t()


class TestTruncatedFaultOnNonOk:
    async def test_a_truncated_fault_on_an_empty_result_flags_but_does_not_truncate(
        self,
    ) -> None:
        from toolproof import FaultSpec, fault_scope

        @tool(empty_when=lambda rows: len(rows) == 0)
        async def t() -> list[int]:
            return []

        with fault_scope([FaultSpec(tool="t", kind="truncated")]):
            result = await t()

        assert isinstance(result, Empty)
        assert result.fault_injected is True
