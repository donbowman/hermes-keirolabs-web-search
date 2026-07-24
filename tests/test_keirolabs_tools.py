"""Unit tests for KeiroLabs tools."""

import json
from unittest.mock import MagicMock, patch

import pytest
from hermes_keirolabs_web_search.tools import (
    _check_available,
    _handle_lite,
    _handle_fast,
    _handle_search_endpoint,
    _handle_answer,
    _handle_research,
    _handle_batch,
    _handle_extract,
)


def test_check_available_without_key(monkeypatch):
    monkeypatch.delenv("KEIROLABS_API_KEY", raising=False)
    assert not _check_available()


def test_check_available_with_key(monkeypatch):
    monkeypatch.setenv("KEIROLABS_API_KEY", "test-key")
    assert _check_available()


def test_handle_lite(monkeypatch):
    monkeypatch.setenv("KEIROLABS_API_KEY", "test-key")
    mock_data = {"answer": "Lite result", "sources": [], "creditsRemaining": 100}

    with patch(
        "hermes_keirolabs_web_search.provider.call_keirolabs_api",
        return_value=mock_data,
    ):
        result = json.loads(_handle_lite({"query": "test"}))
        assert len(result["results"]) == 1
        assert result["results"][0]["answer"] == "Lite result"
        assert result["credits_remaining"] == 100


def test_handle_fast(monkeypatch):
    monkeypatch.setenv("KEIROLABS_API_KEY", "test-key")
    mock_data = {"text": "Fast result", "sources": [], "creditsRemaining": 100}

    with patch(
        "hermes_keirolabs_web_search.provider.call_keirolabs_api",
        return_value=mock_data,
    ):
        result = json.loads(_handle_fast({"query": "test"}))
        assert len(result["results"]) == 1
        assert result["results"][0]["answer"] == "Fast result"
        assert result["credits_remaining"] == 100


def test_handle_search_endpoint(monkeypatch):
    monkeypatch.setenv("KEIROLABS_API_KEY", "test-key")
    mock_data = {
        "data": "Standard search result",
        "sources": [
            {"url": "https://example.com", "title": "Example", "snippet": "A snippet"},
        ],
        "creditsRemaining": 50,
    }

    with patch(
        "hermes_keirolabs_web_search.provider.call_keirolabs_api",
        return_value=mock_data,
    ):
        result = json.loads(_handle_search_endpoint({"query": "test"}))
        assert len(result["results"]) == 2
        assert result["results"][0]["answer"] == "Standard search result"
        assert result["results"][1]["url"] == "https://example.com"
        assert result["credits_remaining"] == 50


def test_handle_answer(monkeypatch):
    monkeypatch.setenv("KEIROLABS_API_KEY", "test-key")
    mock_data = {"answer": "Detailed answer", "sources": [], "creditsRemaining": 10}

    with patch(
        "hermes_keirolabs_web_search.provider.call_keirolabs_api",
        return_value=mock_data,
    ):
        result = json.loads(_handle_answer({"query": "test"}))
        assert len(result["results"]) == 1
        assert result["results"][0]["answer"] == "Detailed answer"
        assert result["credits_remaining"] == 10


def test_handle_research(monkeypatch):
    monkeypatch.setenv("KEIROLABS_API_KEY", "test-key")
    mock_data = {"text": "Deep research", "sources": [], "creditsRemaining": 5}

    with patch(
        "hermes_keirolabs_web_search.provider.call_keirolabs_api",
        return_value=mock_data,
    ):
        result = json.loads(_handle_research({"query": "test"}))
        assert len(result["results"]) == 1
        assert result["results"][0]["answer"] == "Deep research"
        assert result["credits_remaining"] == 5


def test_handle_batch(monkeypatch):
    monkeypatch.setenv("KEIROLABS_API_KEY", "test-key")
    mock_data = {"results": [{"query": "a", "data": "x"}, {"query": "b", "data": "y"}]}

    with patch(
        "hermes_keirolabs_web_search.provider.call_keirolabs_api",
        return_value=mock_data,
    ):
        result = json.loads(_handle_batch({"queries": ["a", "b"]}))
        assert len(result["results"]) == 2
        assert result["results"][0]["query"] == "a"


def test_handle_batch_missing_queries():
    result = json.loads(_handle_batch({}))
    assert "error" in result
    assert "required" in result["error"]


def test_handle_batch_wrong_type():
    result = json.loads(_handle_batch({"queries": "not-a-list"}))
    assert "error" in result


def test_handle_extract(monkeypatch):
    monkeypatch.setenv("KEIROLABS_API_KEY", "test-key")
    mock_data = {
        "title": "Extracted Page",
        "content": "Full page content here.",
        "metadata": {"sourceURL": "https://example.com"},
    }

    with patch(
        "hermes_keirolabs_web_search.provider.call_keirolabs_api",
        return_value=mock_data,
    ):
        result = json.loads(_handle_extract({"url": "https://example.com"}))
        assert result["url"] == "https://example.com"
        assert result["title"] == "Extracted Page"
        assert result["content"] == "Full page content here."


def test_handle_extract_missing_url():
    result = json.loads(_handle_extract({}))
    assert "error" in result
    assert "required" in result["error"]


def test_handle_lite_missing_query():
    result = json.loads(_handle_lite({}))
    assert "error" in result
    assert "required" in result["error"]


def test_handle_lite_api_error(monkeypatch):
    monkeypatch.setenv("KEIROLABS_API_KEY", "test-key")

    with patch(
        "hermes_keirolabs_web_search.provider.call_keirolabs_api",
        side_effect=ValueError("Invalid API key"),
    ):
        result = json.loads(_handle_lite({"query": "test"}))
        assert result["error"] == "Invalid API key"
