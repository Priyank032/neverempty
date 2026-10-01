"""D10: the trace and the model can disagree about what happened.

A tool returning a value JSON cannot represent produced ``Ok``, a span marked
``ok``, and a ``to_model()`` payload saying ``status: error``. Any scorer
reading the trace then sees a different world from the one the model saw, and
the report would score the case as a successful tool call that the model was
told had failed.

This is a regression from the D6 fix. Refusing to hand the model a repr
labelled ``ok`` was right; doing it in ``to_model()`` was the wrong layer,
because by then the result and its span already exist and say otherwise. A
result's status is decided once, where the result is made.

``to_model()`` keeps its guard as a backstop -- a hand-built ``Ok`` never
passed through the wrapper can still reach it -- but the wrapper no longer
lets one through.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

from neverempty import Err, Ok, Tracer, tool
from neverempty.tracer.sinks import MemorySink


class TestTheWrapperDecidesSerializabilityOnce:
    async def test_an_unserializable_return_becomes_an_err(self) -> None:
        @tool(never_empty=True)
        async def t() -> Any:
            return {"when": datetime.datetime(2026, 1, 1)}

        result = await t()
        assert isinstance(result, Err)
        assert result.kind == "validation"

    async def test_the_span_agrees_with_the_result(self) -> None:
        sink = MemorySink()
        tracer = Tracer(sink=sink)

        @tool(never_empty=True)
        async def t() -> Any:
            return {"when": datetime.datetime(2026, 1, 1)}

        async with tracer.run():
            result = await t()

        span = sink.traces[0].spans[0]
        assert span.status == "error"
        assert span.attributes["tool.status"] == "error"
        assert result.status == "error"

    async def test_the_model_sees_the_same_thing(self) -> None:
        @tool(never_empty=True)
        async def t() -> Any:
            return {"when": datetime.datetime(2026, 1, 1)}

        rendered = json.loads((await t()).to_model())
        assert rendered["status"] == "error"

    async def test_the_message_names_the_tool(self) -> None:
        """A serialization failure is a bug in the tool, so it has to be findable."""

        @tool(never_empty=True)
        async def search_jobs() -> Any:
            return {"when": datetime.datetime(2026, 1, 1)}

        result = await search_jobs()
        assert isinstance(result, Err)
        assert "search_jobs" in result.message


class TestOrdinaryValuesAreUntouched:
    async def test_a_serializable_dict_is_ok(self) -> None:
        @tool(never_empty=True)
        async def t() -> Any:
            return {"a": 1, "b": [2, 3], "c": None}

        result = await t()
        assert isinstance(result, Ok)
        assert result.value == {"a": 1, "b": [2, 3], "c": None}

    async def test_falsy_values_still_pass(self) -> None:
        for value in (0, "", False, 0.0):

            @tool(never_empty=True)
            async def t(v: Any = value) -> Any:
                return v

            assert isinstance(await t(), Ok)

    async def test_an_explicit_empty_is_untouched(self) -> None:
        from neverempty import Empty

        @tool()
        async def t() -> Any:
            return Empty(reason="no rows")

        assert isinstance(await t(), Empty)

    async def test_nan_is_refused_because_it_is_not_valid_json(self) -> None:
        """A reader that rejects NaN would see a different payload than one
        that accepts it, so the result cannot claim to be ok."""

        @tool(never_empty=True)
        async def t() -> Any:
            return {"v": float("nan")}

        assert isinstance(await t(), Err)


class TestToModelKeepsItsBackstop:
    """A hand-built ``Ok`` never passed through the wrapper still must not
    render a repr labelled ok."""

    def test_a_hand_built_ok_still_renders_as_an_error(self) -> None:
        class Row:
            pass

        rendered = json.loads(Ok(value={"obj": Row()}).to_model())
        assert rendered["status"] == "error"
        assert "0x" not in json.dumps(rendered)


class TestTheCachedRenderIsIdenticalToTheUncachedOne:
    """``to_model`` splices the wrapper's serialized text instead of encoding
    the payload again. The two paths must produce the same bytes, or the model
    would read something different depending on how the result was built."""

    PAYLOADS: tuple[Any, ...] = (
        {"a": 1, "b": [1, 2], "c": None},
        [],
        {},
        0,
        "",
        False,
        [{"x": "ü"}],
        {"nested": {"deep": [1, {"k": "v"}]}},
        r'quote"and\slash',
        {"é": "ü"},
        [None, True, 1.5],
    )

    def test_every_payload_renders_identically(self) -> None:
        for payload in self.PAYLOADS:
            cached = Ok(
                value=payload,
                serialized_value=json.dumps(payload, ensure_ascii=False),
            ).to_model()
            assert cached == Ok(value=payload).to_model(), payload

    def test_the_truncated_flag_matches_on_both_paths(self) -> None:
        for truncated in (True, False):
            cached = Ok(value=[1], truncated=truncated, serialized_value="[1]").to_model()
            assert cached == Ok(value=[1], truncated=truncated).to_model()

    def test_the_cache_never_reaches_the_trace(self) -> None:
        """It is a cache of ``value``; serializing it would double the payload."""
        result = Ok(value=[1, 2], serialized_value="[1, 2]")
        assert "serialized_value" not in json.loads(result.to_json())


class TestTheOverheadStaysWithinBudget:
    """Doc line 678: under 1 ms per call. The serializability check costs one
    ``json.dumps``, so the budget is asserted on a realistic payload rather
    than the trivial int the existing benchmark uses -- that is why this
    regression was invisible to it."""

    async def test_a_hundred_row_result_stays_under_a_millisecond(self) -> None:
        import statistics
        import time

        rows = [{"i": i, "name": f"job {i}", "city": "Pune"} for i in range(100)]

        @tool(never_empty=True)
        async def decorated() -> Any:
            return rows

        async def bare() -> Any:
            return rows

        await decorated()
        wrapped: list[float] = []
        plain: list[float] = []
        for _ in range(200):
            start = time.perf_counter()
            await bare()
            plain.append(time.perf_counter() - start)
            start = time.perf_counter()
            await decorated()
            wrapped.append(time.perf_counter() - start)

        overhead = statistics.median(wrapped) - statistics.median(plain)
        assert overhead < 1e-3, f"{overhead * 1e6:.0f}us per call exceeds the 1ms budget"
