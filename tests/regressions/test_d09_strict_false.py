"""D9: ``strict=False`` records nothing, contradicting its own docstring.

The ``@tool`` docstring says, of ``strict``:

    Set false in production to keep serving; the ambiguity is then recorded
    on the span instead.

Nothing was recorded. A tool returning ``[]`` without declaring ``empty_when``
produced an ordinary ``Ok``, the model read ``{"status": "ok", "data": []}``,
and the span carried no trace of the ambiguity at all -- so the one mode the
docstring recommends for production reproduced the exact bug the library
exists to prevent, invisibly.

The non-strict path's existing comment is right that the value must not be
promoted to ``Empty``: claiming absence the author never declared is the bug.
What was missing is the other half of the promise -- saying so on the span,
where an eval can find it.
"""

from __future__ import annotations

import json
from typing import Any

from neverempty import Ok, Tracer, tool
from neverempty.tracer.sinks import MemorySink


async def _span_of(value: Any, **kwargs: Any) -> Any:
    sink = MemorySink()
    tracer = Tracer(sink=sink)

    @tool(strict=False, **kwargs)
    async def t() -> Any:
        return value

    async with tracer.run():
        await t()
    return sink.traces[0].spans[0]


class TestTheAmbiguityReachesTheSpan:
    async def test_an_empty_list_is_flagged(self) -> None:
        span = await _span_of([])
        assert span.attributes["tool.ambiguous_empty"] is True

    async def test_none_is_flagged(self) -> None:
        span = await _span_of(None)
        assert span.attributes["tool.ambiguous_empty"] is True

    async def test_an_empty_dict_is_flagged(self) -> None:
        span = await _span_of({})
        assert span.attributes["tool.ambiguous_empty"] is True

    async def test_the_span_still_reports_ok(self) -> None:
        """The result is ``Ok``, so the span must not claim an error. The flag
        is a separate fact: the value was ambiguous, not that it failed."""
        span = await _span_of([])
        assert span.status == "ok"
        assert span.attributes["tool.status"] == "ok"


class TestUnambiguousValuesCarryNoFlag:
    async def test_a_non_empty_result_is_not_flagged(self) -> None:
        span = await _span_of([1, 2, 3])
        assert span.attributes.get("tool.ambiguous_empty", False) is False

    async def test_a_declared_empty_is_not_ambiguous(self) -> None:
        """``empty_when`` means the author answered the question."""
        span = await _span_of([], empty_when=lambda rows: len(rows) == 0)
        assert span.attributes.get("tool.ambiguous_empty", False) is False
        assert span.status == "empty"

    async def test_never_empty_is_not_ambiguous(self) -> None:
        span = await _span_of([], never_empty=True)
        assert span.attributes.get("tool.ambiguous_empty", False) is False

    async def test_a_falsy_but_unambiguous_value_is_not_flagged(self) -> None:
        """``0`` and ``""`` are ordinary values, not empty-looking ones."""
        for value in (0, "", False):
            span = await _span_of(value)
            assert span.attributes.get("tool.ambiguous_empty", False) is False, value


class TestTheResultItselfIsUnchanged:
    """Non-strict mode keeps serving. Only the silence is fixed."""

    async def test_the_value_still_reaches_the_model(self) -> None:
        @tool(strict=False)
        async def t() -> Any:
            return []

        result = await t()
        assert isinstance(result, Ok)
        assert json.loads(result.to_model()) == {
            "status": "ok",
            "data": [],
            "truncated": False,
        }

    async def test_it_is_not_promoted_to_empty(self) -> None:
        """Claiming absence the author never declared is the original bug."""

        @tool(strict=False)
        async def t() -> Any:
            return []

        assert (await t()).status == "ok"

    async def test_strict_mode_still_raises(self) -> None:
        import pytest

        from neverempty import AmbiguousEmptyError

        @tool()
        async def t() -> Any:
            return []

        with pytest.raises(AmbiguousEmptyError):
            await t()
