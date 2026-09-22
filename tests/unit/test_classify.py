"""Classification paths that the table tests do not reach directly."""

from __future__ import annotations

import asyncio

import pytest

from toolproof import Err
from toolproof.core.classify import classify, is_uncatchable


class ResponseLike:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class RequestsStyleError(Exception):
    """requests and several SDKs hang the status off a response object."""

    def __init__(self, status_code: int) -> None:
        super().__init__("request failed")
        self.response = ResponseLike(status_code)


class StatusAttrError(Exception):
    """aiohttp spells it ``status`` rather than ``status_code``."""

    def __init__(self, status: int) -> None:
        super().__init__("request failed")
        self.status = status


class TestStructuralHttpDetection:
    """Core cannot import httpx or a provider SDK, so status is read
    structurally. These are the shapes that appear in the wild."""

    @pytest.mark.parametrize(
        ("status", "kind"),
        [(429, "rate_limit"), (503, "upstream"), (403, "permission"), (400, "validation")],
    )
    def test_status_on_a_nested_response_object(self, status: int, kind: str) -> None:
        assert classify(RequestsStyleError(status)).kind == kind

    def test_status_attribute_spelled_status(self) -> None:
        assert classify(StatusAttrError(429)).kind == "rate_limit"

    def test_a_nonsense_status_value_is_ignored(self) -> None:
        class OutOfRangeStatusError(ValueError):
            status_code = 9999

        assert classify(OutOfRangeStatusError("bad")).kind == "validation"

    def test_a_non_integer_status_is_ignored(self) -> None:
        class StringStatusError(ValueError):
            status_code = "500"

        assert classify(StringStatusError("bad")).kind == "validation"

    def test_a_response_without_a_status_is_ignored(self) -> None:
        class StatuslessResponseError(ValueError):
            response = object()

        assert classify(StatuslessResponseError("bad")).kind == "validation"


class TestTypeBasedClassification:
    def test_permission_error_is_permission(self) -> None:
        result = classify(PermissionError("denied"))
        assert result.kind == "permission"
        assert result.retryable is False

    def test_a_plain_oserror_is_upstream(self) -> None:
        result = classify(OSError("device failure"))
        assert result.kind == "upstream"
        assert result.retryable is True

    def test_an_ssl_error_is_upstream(self) -> None:
        import ssl

        assert classify(ssl.SSLError("handshake failed")).kind == "upstream"


class TestMessages:
    def test_the_status_is_prefixed_when_absent_from_the_text(self) -> None:
        assert classify(RequestsStyleError(503)).message.startswith("HTTP 503:")

    def test_the_status_is_not_duplicated_when_already_present(self) -> None:
        class UpstreamFailedError(Exception):
            status_code = 503

            def __str__(self) -> str:
                return "HTTP 503 from upstream"

        message = classify(UpstreamFailedError()).message
        assert message.count("503") == 1

    def test_an_empty_message_falls_back_to_the_class_name(self) -> None:
        class SilentError(RuntimeError):
            def __str__(self) -> str:
                return "   "

        assert classify(SilentError()).message == "SilentError"

    def test_the_cause_is_always_the_class_name(self) -> None:
        assert classify(RequestsStyleError(500)).cause == "RequestsStyleError"


class TestUncatchable:
    @pytest.mark.parametrize("exc", [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit()])
    def test_control_flow_exceptions_are_uncatchable(self, exc: BaseException) -> None:
        assert is_uncatchable(exc) is True

    def test_ordinary_exceptions_are_catchable(self) -> None:
        assert is_uncatchable(ValueError("x")) is False

    def test_classify_refuses_to_convert_an_uncatchable(self) -> None:
        with pytest.raises(KeyboardInterrupt):
            classify(KeyboardInterrupt())


class TestReturnShape:
    def test_classify_always_returns_an_err(self) -> None:
        assert isinstance(classify(RuntimeError("x")), Err)

    def test_retryable_matches_the_kind(self) -> None:
        for exc, expected in [
            (TimeoutError(), True),
            (ConnectionError(), True),
            (RequestsStyleError(429), True),
            (ValueError(), False),
            (PermissionError(), False),
            (RuntimeError(), False),
        ]:
            assert classify(exc).retryable is expected, exc
