"""Unit tests for the KeiroLabs Web Search Provider."""

from unittest.mock import MagicMock, patch

import pytest
from hermes_keirolabs_web_search.provider import (
    KeiroLabsWebSearchProvider,
    _parse_retry_after,
    _reset_client_for_tests,
)


@pytest.fixture(autouse=True)
def reset_client():
    _reset_client_for_tests()
    yield
    _reset_client_for_tests()


def test_is_available_without_key(monkeypatch):
    monkeypatch.delenv("KEIROSLABS_API_KEY", raising=False)
    provider = KeiroLabsWebSearchProvider()
    assert not provider.is_available()


def test_is_available_with_key(monkeypatch):
    monkeypatch.setenv("KEIROSLABS_API_KEY", "test-key")
    provider = KeiroLabsWebSearchProvider()
    assert provider.is_available()


def test_supports_capabilities():
    provider = KeiroLabsWebSearchProvider()
    assert provider.supports_search()
    assert provider.supports_extract()


def test_search_unconfigured(monkeypatch):
    monkeypatch.delenv("KEIROSLABS_API_KEY", raising=False)
    provider = KeiroLabsWebSearchProvider()
    result = provider.search("test query")
    assert not result["success"]
    assert "not set" in result["error"]


def test_extract_unconfigured(monkeypatch):
    monkeypatch.delenv("KEIROSLABS_API_KEY", raising=False)
    provider = KeiroLabsWebSearchProvider()
    results = provider.extract(["https://example.com"])
    assert len(results) == 1
    assert "not set" in results[0]["error"]


def test_search_success(monkeypatch):
    monkeypatch.setenv("KEIROSLABS_API_KEY", "test-key")
    provider = KeiroLabsWebSearchProvider()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.is_success = True
    mock_response.json.return_value = {
        "answer": "This is a search answer.",
        "sources": [
            {
                "url": "https://example.com/source",
                "title": "Example Source",
                "snippet": "A relevant snippet",
            },
        ],
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch(
        "hermes_keirolabs_web_search.provider._get_client",
        return_value=mock_client,
    ):
        result = provider.search("test query")

        assert result["success"]
        assert len(result["data"]["web"]) == 2

        ans = result["data"]["web"][0]
        assert ans["title"] == "KeiroLabs Answer"
        assert ans["description"] == "This is a search answer."

        src = result["data"]["web"][1]
        assert src["url"] == "https://example.com/source"
        assert src["title"] == "Example Source"
        assert src["description"] == "A relevant snippet"


def test_extract_success(monkeypatch):
    monkeypatch.setenv("KEIROSLABS_API_KEY", "test-key")
    provider = KeiroLabsWebSearchProvider()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.is_success = True
    mock_response.json.return_value = {
        "title": "Extracted Page",
        "content": "This is extracted content.",
        "metadata": {"sourceURL": "https://example.com/test"},
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch(
        "hermes_keirolabs_web_search.provider._get_client",
        return_value=mock_client,
    ):
        results = provider.extract(["https://example.com/test"])

        assert len(results) == 1
        res = results[0]
        assert res["url"] == "https://example.com/test"
        assert res["content"] == "This is extracted content."
        assert "error" not in res


def test_setup_schema():
    provider = KeiroLabsWebSearchProvider()
    schema = provider.get_setup_schema()
    assert schema["name"] == "KeiroLabs"
    assert len(schema["env_vars"]) == 1
    assert schema["env_vars"][0]["key"] == "KEIROSLABS_API_KEY"


def test_api_error_401(monkeypatch):
    monkeypatch.setenv("KEIROSLABS_API_KEY", "test-key")
    provider = KeiroLabsWebSearchProvider()

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.is_success = False

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch(
        "hermes_keirolabs_web_search.provider._get_client",
        return_value=mock_client,
    ):
        result = provider.search("test query")

        assert not result["success"]
        assert "Invalid KEIROSLABS_API_KEY" in result["error"]


def test_api_error_429_with_retry_after_seconds(monkeypatch):
    monkeypatch.setenv("KEIROSLABS_API_KEY", "test-key")
    provider = KeiroLabsWebSearchProvider()

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {"Retry-After": "120"}
    mock_response.is_success = False

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch(
        "hermes_keirolabs_web_search.provider._get_client",
        return_value=mock_client,
    ):
        result = provider.search("test query")

        assert not result["success"]
        assert "rate limit exceeded" in result["error"]
        assert "120 seconds" in result["error"]


def test_api_error_429_with_retry_after_date(monkeypatch):
    monkeypatch.setenv("KEIROSLABS_API_KEY", "test-key")
    provider = KeiroLabsWebSearchProvider()

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}
    mock_response.is_success = False

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with patch(
        "hermes_keirolabs_web_search.provider._get_client",
        return_value=mock_client,
    ):
        result = provider.search("test query")

        assert not result["success"]
        assert "rate limit exceeded" in result["error"]
        assert "2026-10-21" in result["error"]


def test_parse_retry_after_seconds():
    assert _parse_retry_after("120") == "120 seconds"


def test_parse_retry_after_none():
    assert _parse_retry_after(None) == "unknown delay"


def test_parse_retry_after_empty():
    assert _parse_retry_after("") == "unknown delay"
