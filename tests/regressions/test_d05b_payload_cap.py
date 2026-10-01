"""D5 (second review, D16): no default cap on what a tool hands the model.

A tool returning 200k items put 1.49 MB into the tool message with
``truncated: false``. The flag existed and nothing set it without configuration
the reviewer could not find, so a runaway tool silently consumed the context
window and the model was told the result was complete.

**Why 64 KB.** The design doc defines no default (its eight wrapper rules make
``truncated`` author-declared), so the number is chosen here and the reasoning
belongs with it. Measured against real payloads:

    job listing   382 B/row    50 rows = 18 KB
    govt scheme   299 B/row    50 rows = 14 KB

and against a model's context, at roughly 4 bytes per token:

     8 KB  ->  2k tokens   1.6% of a 128k window   40 rows
    16 KB  ->  4k tokens   3.1%                    80 rows
    64 KB  -> 16k tokens  12.5%                   320 rows

8 KB -- which matches the existing ``tool.args`` span cap and was the obvious
first guess -- would truncate an ordinary 50-row job search. A cap that fires
on normal results gets turned off, and a disabled cap protects nothing.

64 KB is high enough that no reasonable single result reaches it and low enough
that one runaway tool cannot eat an eighth of the context. It is a backstop
against a tool with no limit, not a replacement for ``truncated_when``, which
remains the right way to express a deliberate cap because only the author knows
what a meaningful page of their data is.

The value is ``DEFAULT_PAYLOAD_CAP_BYTES`` and ``payload_cap_bytes=`` overrides
it per tool, including ``None`` to disable it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from neverempty import Ok, tool
from neverempty.core.results import DEFAULT_PAYLOAD_CAP_BYTES


def _rows(count: int, pad: int = 200) -> list[dict[str, Any]]:
    return [{"i": index, "pad": "x" * pad} for index in range(count)]


class TestARunawayResultIsCapped:
    async def test_a_large_result_is_marked_truncated(self) -> None:
        @tool(never_empty=True)
        async def big() -> Any:
            return _rows(5000)

        rendered = json.loads((await big()).to_model())
        assert rendered["truncated"] is True

    async def test_the_model_is_told_the_result_is_incomplete(self) -> None:
        @tool(never_empty=True)
        async def big() -> Any:
            return _rows(5000)

        rendered = json.loads((await big()).to_model())
        assert "note" in rendered
        assert "incomplete" in rendered["note"]

    async def test_what_reaches_the_model_is_bounded(self) -> None:
        @tool(never_empty=True)
        async def big() -> Any:
            return _rows(5000)

        rendered = (await big()).to_model()
        # The envelope and note add a little; the payload itself is capped.
        assert len(rendered.encode("utf-8")) < DEFAULT_PAYLOAD_CAP_BYTES + 2_000

    async def test_the_kept_rows_are_still_valid_json(self) -> None:
        """Truncating must not hand the model a broken document."""

        @tool(never_empty=True)
        async def big() -> Any:
            return _rows(5000)

        rendered = json.loads((await big()).to_model())
        assert isinstance(rendered["data"], list)
        assert rendered["data"][0]["i"] == 0

    async def test_the_full_value_is_still_on_the_result(self) -> None:
        """The cap is about what the model reads, not what the harness keeps."""

        @tool(never_empty=True)
        async def big() -> Any:
            return _rows(5000)

        result = await big()
        assert isinstance(result, Ok)
        assert len(result.value) == 5000


class TestOrdinaryResultsAreUntouched:
    @pytest.mark.parametrize("count", [1, 10, 50, 100])
    async def test_a_realistic_result_is_not_truncated(self, count: int) -> None:
        """50 job listings is 18 KB. Capping that would make the flag noise."""

        @tool(never_empty=True)
        async def search(n: int = count) -> Any:
            return [
                {
                    "id": f"j-{index}",
                    "title": "Senior Backend Engineer",
                    "company": "Acme Technologies Pvt Ltd",
                    "city": "Pune",
                    "skills": ["python", "fastapi", "postgres", "aws"],
                }
                for index in range(n)
            ]

        rendered = json.loads((await search()).to_model())
        assert rendered["truncated"] is False
        assert "note" not in rendered

    async def test_a_small_result_is_byte_identical_to_before(self) -> None:
        @tool(never_empty=True)
        async def small() -> Any:
            return {"a": 1, "b": [2, 3]}

        assert json.loads((await small()).to_model()) == {
            "status": "ok",
            "data": {"a": 1, "b": [2, 3]},
            "truncated": False,
        }


class TestTheCapIsConfigurable:
    async def test_a_lower_cap_fires_earlier(self) -> None:
        @tool(never_empty=True, payload_cap_bytes=1_000)
        async def rows() -> Any:
            return _rows(50)

        assert json.loads((await rows()).to_model())["truncated"] is True

    async def test_none_disables_it(self) -> None:
        """An author who means it can opt out."""

        @tool(never_empty=True, payload_cap_bytes=None)
        async def rows() -> Any:
            return _rows(5000)

        rendered = json.loads((await rows()).to_model())
        assert rendered["truncated"] is False
        assert len(rendered["data"]) == 5000

    async def test_truncated_when_still_wins(self) -> None:
        """An author's declared cap is the point; the default is a backstop."""

        @tool(never_empty=True, truncated_when=lambda rows: len(rows) >= 10)
        async def rows() -> Any:
            return _rows(20)

        assert json.loads((await rows()).to_model())["truncated"] is True


class TestTheNumberIsStated:
    def test_the_default_is_64_kb(self) -> None:
        assert DEFAULT_PAYLOAD_CAP_BYTES == 64 * 1024

    def test_the_docs_give_the_number_and_the_reason(self) -> None:
        from pathlib import Path

        text = (Path(__file__).resolve().parents[2] / "docs" / "getting-started.md").read_text(
            encoding="utf-8"
        )
        assert "64 KB" in text


class TestACapThatKeepsNothingIsAFailure:
    """A single item larger than the cap left ``{"status": "ok", "data": []}``
    in front of the model -- the exact shape this library exists to make
    unrepresentable, produced by the safety feature itself.

    The note said the result was incomplete, but a model reading an empty list
    under ``status: ok`` can still conclude there is nothing. Zero items kept is
    not a truncated result, it is a result that could not be delivered, so it
    becomes an ``Err`` and the renderer's refusal note applies.
    """

    async def test_one_oversized_item_is_an_error_not_an_empty_list(self) -> None:
        from neverempty import Err

        @tool(never_empty=True)
        async def t() -> Any:
            return [{"i": 0, "pad": "x" * 500_000}]

        result = await t()
        assert isinstance(result, Err)
        assert result.kind == "validation"

    async def test_the_model_never_reads_an_empty_list_from_a_cap(self) -> None:
        @tool(never_empty=True)
        async def t() -> Any:
            return [{"i": 0, "pad": "x" * 500_000}]

        rendered = json.loads((await t()).to_model())
        assert rendered["status"] == "error"
        assert rendered.get("data") != []

    async def test_the_message_says_what_to_do(self) -> None:
        from neverempty import Err

        @tool(never_empty=True)
        async def t() -> Any:
            return [{"i": 0, "pad": "x" * 500_000}]

        result = await t()
        assert isinstance(result, Err)
        assert "payload_cap_bytes" in result.message

    async def test_a_leading_huge_item_followed_by_small_ones_also_errors(
        self,
    ) -> None:
        """Keeping only the tail would silently reorder the author's result."""
        from neverempty import Err

        @tool(never_empty=True)
        async def t() -> Any:
            return [{"i": 0, "pad": "x" * 500_000}, *({"i": n} for n in range(100))]

        assert isinstance(await t(), Err)

    async def test_one_item_that_does_fit_is_kept_normally(self) -> None:
        """The boundary: a single large-but-fitting item is an ordinary result."""

        @tool(never_empty=True)
        async def t() -> Any:
            return [{"i": 0, "pad": "x" * 60_000}]

        rendered = json.loads((await t()).to_model())
        assert rendered["status"] == "ok"
        assert len(rendered["data"]) == 1

    async def test_an_oversized_non_list_is_still_flagged_not_dropped(self) -> None:
        """A dict or string cannot be trimmed honestly, so it passes whole with
        the flag set -- too long is better than silently altered."""

        @tool(never_empty=True)
        async def t() -> Any:
            return {"report": "x" * 200_000}

        rendered = json.loads((await t()).to_model())
        assert rendered["status"] == "ok"
        assert rendered["truncated"] is True
        assert len(rendered["data"]["report"]) == 200_000
