"""KeiroLabs web search + content extraction provider.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Uses ``httpx``
to call the KeiroLabs API.

Config keys this provider responds to::

    web:
      search_backend: "keirolabs"     # explicit per-capability
      extract_backend: "keirolabs"    # explicit per-capability
      backend: "keirolabs"            # shared fallback for both
      keirolabs_endpoint: "lite"      # "lite", "fast", "search", "answer"
      keirolabs_base_url: "https://kierolabs.space/api"  # optional override

Env var::

    KEIROSLABS_API_KEY=...
"""

from __future__ import annotations

import email.utils
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_KEIROLABS_CLIENT: httpx.Client | None = None

_ENDPOINT_MAP: Dict[str, str] = {
    "lite": "/v2/lite",
    "fast": "/v2/fast",
    "search": "/search",
    "answer": "/answer",
    "research": "/research",
    "extract": "/web-crawler",
    "batch": "/v2/batch",
}


# -- module-level helpers shared with tools.py ---------------------------------


def _get_client() -> httpx.Client:
    """Lazy-create and cache an httpx Client.

    Raises ``ValueError`` when ``KEIROSLABS_API_KEY`` is unset.
    """
    global _KEIROLABS_CLIENT

    if _KEIROLABS_CLIENT is not None:
        return _KEIROLABS_CLIENT

    api_key = os.getenv("KEIROSLABS_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "KEIROSLABS_API_KEY environment variable not set. "
            "Get your API key at https://platform.keirolabs.cloud"
        )

    _KEIROLABS_CLIENT = httpx.Client(
        timeout=httpx.Timeout(60.0),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "hermes-keirolabs-web-search/1.0.0",
        },
    )
    return _KEIROLABS_CLIENT


def _reset_client_for_tests() -> None:
    """Drop the cached client so tests can re-instantiate cleanly."""
    global _KEIROLABS_CLIENT
    if _KEIROLABS_CLIENT is not None:
        _KEIROLABS_CLIENT.close()
    _KEIROLABS_CLIENT = None


def _get_api_key() -> str:
    return os.getenv("KEIROSLABS_API_KEY", "").strip()


def _get_base_url() -> str:
    try:
        from hermes_cli.config import load_config
        cfg = load_config().get("web", {})
        return cfg.get("keirolabs_base_url", "https://kierolabs.space/api")
    except Exception:
        return "https://kierolabs.space/api"


def _endpoint_path(endpoint: str) -> str:
    return _ENDPOINT_MAP.get(endpoint, f"/v2/{endpoint}")


def _parse_retry_after(header: str | None) -> str:
    """Parse a Retry-After header into a human-readable delay string."""
    if not header:
        return "unknown delay"
    header = header.strip()
    try:
        seconds = int(header)
        return f"{seconds} seconds"
    except ValueError:
        pass
    try:
        retry_date = email.utils.parsedate_to_datetime(header)
        now = datetime.now(timezone.utc)
        delta = (retry_date - now).total_seconds()
        if delta > 0:
            return f"{delta:.0f} seconds (until {retry_date.isoformat()})"
        return f"{retry_date.isoformat()}"
    except (ValueError, TypeError, OverflowError):
        pass
    return header


def call_keirolabs_api(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call a KeiroLabs API endpoint. Raises ``ValueError`` on errors."""
    client = _get_client()
    base = _get_base_url()
    path = _endpoint_path(endpoint)

    url = f"{base}{path}"
    payload["apiKey"] = _get_api_key()

    try:
        resp = client.post(url, json=payload)
    except httpx.TimeoutException:
        raise ValueError("KeiroLabs API request timed out")
    except httpx.ConnectError:
        raise ValueError(f"Failed to connect to KeiroLabs API at {url}")

    if resp.status_code == 401:
        raise ValueError("Invalid KEIROSLABS_API_KEY")
    if resp.status_code == 402:
        raise ValueError("KeiroLabs API: out of credits")
    if resp.status_code == 403:
        raise ValueError(f"KeiroLabs API access denied: {resp.text}")
    if resp.status_code == 429:
        retry = _parse_retry_after(resp.headers.get("Retry-After"))
        raise ValueError(f"KeiroLabs API rate limit exceeded — retry after {retry}")
    if resp.status_code >= 500:
        raise ValueError(f"KeiroLabs API server error: {resp.status_code}")
    if not resp.is_success:
        raise ValueError(
            f"KeiroLabs API error {resp.status_code}: {resp.text}"
        )

    return resp.json()


# -- provider class ------------------------------------------------------------


class KeiroLabsWebSearchProvider(WebSearchProvider):
    """KeiroLabs search + extract provider."""

    @property
    def name(self) -> str:
        return "keirolabs"

    @property
    def display_name(self) -> str:
        return "KeiroLabs"

    def is_available(self) -> bool:
        return bool(os.getenv("KEIROSLABS_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def _get_config(self) -> Dict[str, Any]:
        try:
            from hermes_cli.config import load_config
            cfg = load_config().get("web", {})
            return {
                "endpoint": cfg.get("keirolabs_endpoint", "lite"),
                "base_url": cfg.get("keirolabs_base_url", "https://kierolabs.space/api"),
            }
        except Exception:
            return {
                "endpoint": "lite",
                "base_url": "https://kierolabs.space/api",
            }

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a web search via KeiroLabs API."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("KeiroLabs search: '%s'", query)

            endpoint = self._get_config()["endpoint"]
            data = call_keirolabs_api(endpoint, {"query": query})

            web_results: List[Dict[str, Any]] = []

            answer_text = data.get("answer", data.get("text", data.get("data", "")))
            if isinstance(answer_text, str) and answer_text.strip():
                web_results.append(
                    {
                        "url": "",
                        "title": "KeiroLabs Answer",
                        "description": answer_text,
                        "position": 1,
                    }
                )

            sources = data.get("sources") or data.get("results") or []
            if isinstance(sources, list):
                for source in sources:
                    url = (
                        source.get("url")
                        or source.get("link")
                        or source.get("href", "")
                    )
                    title = source.get("title", "")
                    snippet = source.get("snippet") or source.get("content") or source.get("description", "")

                    web_results.append(
                        {
                            "url": url,
                            "title": title,
                            "description": snippet or "Supporting source",
                            "position": len(web_results) + 1,
                        }
                    )

            if not web_results:
                web_results.append(
                    {
                        "url": "",
                        "title": "No Results",
                        "description": "KeiroLabs did not return a response.",
                        "position": 1,
                    }
                )

            web_results = web_results[:limit]
            return {"success": True, "data": {"web": web_results}}

        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.warning("KeiroLabs search error: %s", exc)
            return {"success": False, "error": f"KeiroLabs search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via KeiroLabs API."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            results: List[Dict[str, Any]] = []
            for url in urls:
                logger.info("KeiroLabs extract: %s", url)
                try:
                    data = call_keirolabs_api("extract", {"url": url})

                    content = data.get("content") or data.get("data") or data.get("text", "")
                    if isinstance(content, dict):
                        content = str(content)

                    title = data.get("title", "")
                    results.append(
                        {
                            "url": url,
                            "title": title or f"KeiroLabs Extraction: {url}",
                            "content": content,
                            "raw_content": content,
                            "metadata": data.get("metadata", {"sourceURL": url}),
                        }
                    )
                except Exception as inner_exc:
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "error": str(inner_exc),
                        }
                    )

            return results
        except ValueError as exc:
            return [
                {"url": u, "title": "", "content": "", "error": str(exc)}
                for u in urls
            ]
        except Exception as exc:
            logger.warning("KeiroLabs extract error: %s", exc)
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "error": f"KeiroLabs extract failed: {exc}",
                }
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "KeiroLabs",
            "badge": "free tier available",
            "tag": (
                "Web search and content extraction using KeiroLabs API. "
                "Cost-efficient web scraping with lite/fast/search/answer endpoints. "
                "Requires KEIROSLABS_API_KEY."
            ),
            "env_vars": [
                {
                    "key": "KEIROSLABS_API_KEY",
                    "prompt": "KeiroLabs API key",
                    "url": "https://platform.keirolabs.cloud",
                },
            ],
        }


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    provider = KeiroLabsWebSearchProvider()

    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        urls = sys.argv[2:]
        print(f"Extracting {urls}...")
        results = provider.extract(urls)
        for r in results:
            print(f"\n--- {r.get('title', r.get('url'))} ---")
            print(r.get("content", r.get("error")))
    else:
        query = (
            " ".join(sys.argv[1:])
            if len(sys.argv) > 1
            else "latest news on artificial intelligence"
        )
        print(f"Searching for: {query}")
        result = provider.search(query)
        print(json.dumps(result, indent=2))
