"""D14: control-flow exceptions become results, and a KeyError is misclassified.

Two separate findings.

**GeneratorExit.** ``UNCATCHABLE`` holds the three the doc names as
non-negotiable -- ``CancelledError``, ``KeyboardInterrupt``, ``SystemExit``.
``GeneratorExit`` is not among them, and it is not an ordinary error either:
Python throws it into a generator or coroutine during teardown, and swallowing
it into an ``Err`` corrupts the cleanup it is part of. Reproduced: a tool
cancelled mid-await that raises ``GeneratorExit`` while unwinding returned
``Err(kind="exception")`` instead of propagating, so the cancellation was
lost -- which is the spirit of the non-negotiable even though the doc's list
does not name this exception.

A *custom* ``BaseException`` subclass is left alone deliberately. Someone who
subclasses ``BaseException`` for an ordinary domain error should still get a
typed failure, and there is no way to tell that intent from a control-flow
signal except by naming the ones Python itself uses.

**KeyError.** Classified ``validation``, alongside ``ValueError`` and
``TypeError``. A ``KeyError`` from a dict lookup in the tool body is a bug in
the tool, not a validation failure of the caller's arguments, and the two are
different things to a reader deciding whether to fix their input or file a bug.
The reviewer is right, but the classification table is the doc's (line 675 and
the table above it) and ``validation`` is where it puts ``KeyError``. Changing
it would be a silent change to every published error_kind number, so it is left
alone and the reasoning recorded here.
"""

from __future__ import annotations

import asyncio

import pytest

from neverempty import Err, tool
from neverempty.core.classify import classify, is_uncatchable


class TestGeneratorExitPropagates:
    def test_it_is_uncatchable(self) -> None:
        assert is_uncatchable(GeneratorExit())

    async def test_a_tool_raising_it_does_not_return_a_result(self) -> None:
        @tool(never_empty=True)
        async def t() -> int:
            raise GeneratorExit

        with pytest.raises(GeneratorExit):
            await t()

    async def test_a_cancelled_tool_that_raises_it_still_cancels(self) -> None:
        """The reproduction: cleanup turned a cancellation into an Err."""

        @tool(never_empty=True)
        async def t() -> int:
            try:
                await asyncio.sleep(10)
            except BaseException:
                raise GeneratorExit from None
            return 1

        task = asyncio.create_task(t())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises((asyncio.CancelledError, GeneratorExit)):
            await task


class TestTheThreeNonNegotiablesAreUnchanged:
    async def test_cancelled_error_propagates(self) -> None:
        @tool(never_empty=True)
        async def t() -> int:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await t()

    async def test_keyboard_interrupt_propagates(self) -> None:
        @tool(never_empty=True)
        async def t() -> int:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            await t()

    async def test_system_exit_propagates(self) -> None:
        @tool(never_empty=True)
        async def t() -> int:
            raise SystemExit

        with pytest.raises(SystemExit):
            await t()


class TestACustomBaseExceptionStillBecomesAResult:
    """Deliberate: a domain error that happens to subclass BaseException is
    not a control-flow signal, and the wrapper cannot read intent."""

    async def test_it_is_a_typed_failure(self) -> None:
        class DomainError(BaseException):
            pass

        @tool(never_empty=True)
        async def t() -> int:
            raise DomainError("upstream said no")

        result = await t()
        assert isinstance(result, Err)
        assert result.cause == "DomainError"


class TestTheClassificationTableIsUnchanged:
    """Pinned so a future change to these is a deliberate one. Changing a kind
    silently rewrites every published error_kind number."""

    @pytest.mark.parametrize(
        ("exc", "kind"),
        [
            (KeyError("k"), "validation"),
            (ValueError("v"), "validation"),
            (TypeError("t"), "validation"),
            (TimeoutError(), "timeout"),
            (ConnectionError(), "upstream"),
            (PermissionError(), "permission"),
            (RuntimeError("r"), "exception"),
        ],
    )
    def test_the_kind_is_stable(self, exc: BaseException, kind: str) -> None:
        assert classify(exc).kind == kind
