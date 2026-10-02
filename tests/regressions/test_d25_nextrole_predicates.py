"""D25: the empty_when predicates proposed for NextRole's seven tools.

Each is derived from the agent's real return statements and exercised here
against those shapes, because a wrong predicate is a silent mismeasurement --
the failure this library exists to prevent, introduced by the instrumentation.

The one that matters most is ``FollowupScheduler``: zero pending follow-ups is
a true, common answer, so it must be ``Empty`` and never ``Err``. Getting that
backwards would make the honest case look broken on every clean run.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from neverempty import Empty, Err, Ok, tool


class InterviewQuestion(BaseModel):
    text: str = "Tell me about a hard bug."


class InterviewPrepResult(BaseModel):
    """The real shape: a pydantic model, not a dict."""

    role: str = "Backend Engineer"
    company: str = "Acme"
    questions: list[InterviewQuestion] = []
    preparation_tips: list[str] = []


class TestSearchShapedTools:
    """1 and 2: a real search can find nothing, and that is a true answer."""

    async def test_job_search_with_results_is_ok(self) -> None:
        @tool(empty_when=lambda r: not r.get("jobs"), strict=False)
        async def search_jobs() -> Any:
            return {"jobs": [{"title": "Backend Engineer"}], "total_count": 1}

        assert isinstance(await search_jobs(), Ok)

    async def test_job_search_with_no_results_is_empty(self) -> None:
        """Not an error: the query ran and the world contains no matches."""

        @tool(empty_when=lambda r: not r.get("jobs"), strict=False)
        async def search_jobs() -> Any:
            return {"jobs": [], "total_count": 0}

        assert isinstance(await search_jobs(), Empty)

    async def test_blog_search_behaves_the_same(self) -> None:
        @tool(empty_when=lambda r: not r.get("articles"), strict=False)
        async def search_articles() -> Any:
            return {"articles": [], "total_count": 0}

        assert isinstance(await search_articles(), Empty)


class TestNeverEmptyTools:
    """3, 4 and 7: there is no "none exists" outcome, so failure is failure."""

    @pytest.mark.parametrize(
        ("name", "payload"),
        [
            ("generate_email", {"email": {"subject": "Hi", "body": "..."}}),
            ("analyze", {"response": "Your profile is strong in backend."}),
            ("respond", {"response": "Hello! I'm NextRole AI."}),
        ],
    )
    async def test_a_result_is_ok(self, name: str, payload: dict[str, Any]) -> None:
        @tool(never_empty=True, strict=False)
        async def produce() -> Any:
            return payload

        assert isinstance(await produce(), Ok)

    async def test_a_raise_is_an_err_not_an_empty(self) -> None:
        @tool(never_empty=True, strict=False)
        async def generate_email() -> Any:
            raise TimeoutError("upstream timed out")

        result = await generate_email()
        assert isinstance(result, Err)
        assert result.kind == "timeout"


class TestTheFollowupCountIsTheTrickyOne:
    """5: zero pending follow-ups is a true answer, not a failure.

    Getting this backwards would mark the honest, common case as an error on
    every clean run -- noise that would drown the signal the suite exists for.
    """

    async def test_zero_pending_is_empty_not_error(self) -> None:
        @tool(empty_when=lambda r: r.get("pending_followups") == 0, strict=False)
        async def handle_followup() -> Any:
            return {"response": "Nothing to follow up on.", "pending_followups": 0}

        assert isinstance(await handle_followup(), Empty)

    async def test_some_pending_is_ok(self) -> None:
        @tool(empty_when=lambda r: r.get("pending_followups") == 0, strict=False)
        async def handle_followup() -> Any:
            return {"response": "You have 3.", "pending_followups": 3}

        assert isinstance(await handle_followup(), Ok)

    async def test_the_error_path_omits_the_count_entirely(self) -> None:
        """The real agent does this, and it is correct: no count at all beats
        a count of zero, because zero is a claim and absence of the field is
        not."""

        @tool(empty_when=lambda r: r.get("pending_followups") == 0, strict=False)
        async def handle_followup() -> Any:
            return {"response": "I had trouble checking.", "error": "timeout"}

        # ``.get`` returns None, which is not ``== 0``, so this is Ok and the
        # error field carries the truth. empty_payload scores it ``reported``.
        assert isinstance(await handle_followup(), Ok)


class TestTheTypedResultNeedsAttributeAccess:
    """6: InterviewPrepResult is a pydantic model, so ``.get`` would raise."""

    async def test_questions_present_is_ok(self) -> None:
        @tool(empty_when=lambda r: not r.questions, strict=False)
        async def prepare() -> Any:
            return InterviewPrepResult(questions=[InterviewQuestion()])

        assert isinstance(await prepare(), Ok)

    async def test_no_questions_is_empty(self) -> None:
        @tool(empty_when=lambda r: not r.questions, strict=False)
        async def prepare() -> Any:
            return InterviewPrepResult(questions=[])

        assert isinstance(await prepare(), Empty)

    async def test_a_dict_predicate_would_have_failed_here(self) -> None:
        """Why the row is called out separately: ``.get`` on a model raises,
        and a raising predicate is an Err -- a silent mismeasurement if nobody
        checked."""

        @tool(empty_when=lambda r: not r.get("questions"), strict=False)
        async def prepare() -> Any:
            return InterviewPrepResult(questions=[InterviewQuestion()])

        result = await prepare()
        assert isinstance(result, Err)
        assert result.kind == "validation"


class TestTheDocumentedTableMatches:
    def test_every_method_is_listed(self) -> None:
        from pathlib import Path

        text = (
            Path(__file__).resolve().parents[2] / "docs" / "prompts" / "wrap-nextrole-tools.md"
        ).read_text(encoding="utf-8")
        for method in (
            "search_jobs",
            "search_articles",
            "generate_email",
            "analyze",
            "handle_followup",
            "prepare",
            "respond",
        ):
            assert method in text, method

    def test_it_states_the_inertness_baseline(self) -> None:
        """The first commit is only safe if the number it must reproduce is
        written down before the change."""
        from pathlib import Path

        text = (
            Path(__file__).resolve().parents[2] / "docs" / "prompts" / "wrap-nextrole-tools.md"
        ).read_text(encoding="utf-8")
        assert "91.2%" in text
