"""D21: ordinary standard-library values became tool errors.

``UUID``, ``datetime``, ``date``, ``Decimal`` and ``Enum`` are what a
database-backed tool returns. Refusing them pushed every author into
hand-serializing before returning, which is friction severe enough to push
people away from ``@tool`` altogether -- and the whole value of this library
sits behind that decorator.

Reported from the field: wrapping NextRole's tools meant excluding
``get_user_profile`` because its rows carry a ``UUID`` and a ``datetime``.

The D10 fix was right that a value JSON cannot represent must not reach a model
labelled ``ok``. It was too strict about what cannot be represented. These five
types have one obvious, lossless JSON spelling each, and ``json.dumps``
declines them only because it will not choose a convention for you. Choosing it
once, here, is better than every tool author choosing it separately.

What stays refused is anything whose JSON form would be a guess: an arbitrary
object, a set, a complex number. Those still become ``Err(validation)``, and
``NaN`` still does too.
"""

from __future__ import annotations

import datetime
import decimal
import enum
import json
import uuid
from typing import Any

from neverempty import Err, Ok, tool


class Colour(enum.Enum):
    RED = "red"


class TestStandardTypesSerializeLosslessly:
    async def test_a_uuid_becomes_its_canonical_string(self) -> None:
        value = uuid.UUID("12345678-1234-5678-1234-567812345678")

        @tool(never_empty=True)
        async def t() -> Any:
            return {"id": value}

        rendered = json.loads((await t()).to_model())
        assert rendered["status"] == "ok"
        assert rendered["data"]["id"] == "12345678-1234-5678-1234-567812345678"

    async def test_a_datetime_becomes_iso_8601(self) -> None:
        when = datetime.datetime(2026, 10, 2, 14, 30, tzinfo=datetime.timezone.utc)

        @tool(never_empty=True)
        async def t() -> Any:
            return {"when": when}

        rendered = json.loads((await t()).to_model())
        assert rendered["data"]["when"] == when.isoformat()

    async def test_a_date_becomes_iso_8601(self) -> None:
        @tool(never_empty=True)
        async def t() -> Any:
            return {"on": datetime.date(2026, 10, 2)}

        assert json.loads((await t()).to_model())["data"]["on"] == "2026-10-02"

    async def test_a_decimal_becomes_a_string_not_a_float(self) -> None:
        """A float would lose the precision Decimal exists to keep."""

        @tool(never_empty=True)
        async def t() -> Any:
            return {"salary": decimal.Decimal("1800000.50")}

        rendered = json.loads((await t()).to_model())
        assert rendered["data"]["salary"] == "1800000.50"

    async def test_an_enum_becomes_its_value(self) -> None:
        @tool(never_empty=True)
        async def t() -> Any:
            return {"colour": Colour.RED}

        assert json.loads((await t()).to_model())["data"]["colour"] == "red"

    async def test_a_realistic_orm_row_passes(self) -> None:
        """The reported case: a profile row with an id and a timestamp."""

        @tool(empty_when=lambda rows: len(rows) == 0)
        async def get_user_profile() -> Any:
            return [
                {
                    "id": uuid.uuid4(),
                    "created_at": datetime.datetime.now(datetime.timezone.utc),
                    "headline": "Backend Engineer",
                }
            ]

        result = await get_user_profile()
        assert isinstance(result, Ok)
        assert json.loads(result.to_model())["status"] == "ok"


class TestGenuinelyUnrepresentableValuesAreStillRefused:
    """The D10 guarantee: a value whose JSON form would be a guess never
    reaches a model wearing ``status: ok``."""

    async def test_an_arbitrary_object_is_still_an_error(self) -> None:
        class Row:
            pass

        @tool(never_empty=True)
        async def t() -> Any:
            return {"row": Row()}

        assert isinstance(await t(), Err)

    async def test_a_set_is_still_an_error(self) -> None:
        """List or sorted list? That is the caller's decision, not ours."""

        @tool(never_empty=True)
        async def t() -> Any:
            return {"tags": {"a", "b"}}

        assert isinstance(await t(), Err)

    async def test_nan_is_still_an_error(self) -> None:
        @tool(never_empty=True)
        async def t() -> Any:
            return {"v": float("nan")}

        assert isinstance(await t(), Err)

    async def test_the_model_never_sees_a_memory_address(self) -> None:
        class Row:
            pass

        @tool(never_empty=True)
        async def t() -> Any:
            return {"row": Row()}

        assert "0x" not in (await t()).to_model()


class TestTheValueIsUnchangedForScorers:
    """Only the model's view is serialized; scorers read the real objects."""

    async def test_the_result_keeps_the_original_types(self) -> None:
        when = datetime.datetime(2026, 10, 2, tzinfo=datetime.timezone.utc)

        @tool(never_empty=True)
        async def t() -> Any:
            return {"when": when}

        result = await t()
        assert isinstance(result, Ok)
        assert result.value["when"] is when


class TestTheTraceAgreesWithTheModel:
    """D10's invariant has to survive this change."""

    async def test_a_serializable_row_is_ok_everywhere(self) -> None:
        from neverempty import Tracer
        from neverempty.tracer.sinks import MemorySink

        sink = MemorySink()
        tracer = Tracer(sink=sink)

        @tool(never_empty=True)
        async def t() -> Any:
            return {"id": uuid.uuid4()}

        async with tracer.run():
            result = await t()

        span = sink.traces[0].spans[0]
        assert result.status == "ok"
        assert span.status == "ok"
        assert json.loads(result.to_model())["status"] == "ok"


class TestAHandBuiltOkRendersTheSameWay:
    """``to_model`` has an uncached path for an ``Ok`` that never passed
    through the wrapper. The same value must render the same way through
    either, or the output depends on how the result was constructed."""

    def test_a_uuid_renders_identically(self) -> None:
        value = {"id": uuid.UUID("12345678-1234-5678-1234-567812345678")}
        rendered = json.loads(Ok(value=value).to_model())
        assert rendered["status"] == "ok"
        assert rendered["data"]["id"] == "12345678-1234-5678-1234-567812345678"

    def test_a_datetime_renders_identically(self) -> None:
        when = datetime.datetime(2026, 10, 2, tzinfo=datetime.timezone.utc)
        rendered = json.loads(Ok(value={"when": when}).to_model())
        assert rendered["data"]["when"] == when.isoformat()

    def test_an_arbitrary_object_is_still_refused(self) -> None:
        class Row:
            pass

        assert json.loads(Ok(value={"row": Row()}).to_model())["status"] == "error"


class TestPydanticModelsSerialize:
    """A typed return is good practice, and a harness that turns it into an
    error punishes it. Found wrapping NextRole's InterviewPrepAgent, whose
    ``prepare`` returns an ``InterviewPrepResult`` -- it would have been an Err
    on every call."""

    async def test_a_model_becomes_its_json_form(self) -> None:
        from pydantic import BaseModel

        class Question(BaseModel):
            text: str

        class Result(BaseModel):
            role: str
            questions: list[Question]

        @tool(never_empty=True)
        async def prepare() -> Any:
            return Result(role="Backend", questions=[Question(text="why?")])

        rendered = json.loads((await prepare()).to_model())
        assert rendered["status"] == "ok"
        assert rendered["data"] == {
            "role": "Backend",
            "questions": [{"text": "why?"}],
        }

    async def test_nested_standard_types_inside_a_model_work(self) -> None:
        """``mode="json"`` handles the UUID and datetime inside, so the two
        rules compose rather than fighting."""
        from pydantic import BaseModel

        class Row(BaseModel):
            id: uuid.UUID
            created_at: datetime.datetime

        @tool(never_empty=True)
        async def fetch() -> Any:
            return Row(
                id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
                created_at=datetime.datetime(2026, 10, 2, tzinfo=datetime.timezone.utc),
            )

        rendered = json.loads((await fetch()).to_model())
        assert rendered["data"]["id"] == "12345678-1234-5678-1234-567812345678"
        assert rendered["data"]["created_at"].startswith("2026-10-02")

    async def test_an_arbitrary_object_is_still_refused(self) -> None:
        """The guard must not have widened into "anything with a method"."""

        class NotAModel:
            def model_dump(self) -> str:
                return "not a dict"

        @tool(never_empty=True)
        async def t() -> Any:
            return {"x": NotAModel()}

        # Checked against pydantic's BaseModel, not duck-typed on the method
        # name: a class that merely has ``model_dump`` is not a model, and
        # trusting whatever it returns would be the "guess at a convention"
        # this serializer exists to avoid.
        rendered = json.loads((await t()).to_model())
        assert rendered["status"] == "error"
