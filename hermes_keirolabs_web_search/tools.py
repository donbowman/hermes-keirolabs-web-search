"""KeiroLabs API tools for Hermes.

Exposes the full KeiroLabs API surface as discrete tools the agent can call
by name, independent of the web_search / web_extract provider pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

TOOLSET = "keirolabs"

_SEARCH_SCHEMA_BASE: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query string",
        },
    },
    "required": ["query"],
}

_EXTRACT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "URL to extract content from",
        },
    },
    "required": ["url"],
}

_BATCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of search queries to batch together",
        },
    },
    "required": ["queries"],
}


def _check_available() -> bool:
    return bool(os.getenv("KEIROSLABS_API_KEY", "").strip())


def _make_schema(name: str, desc: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "description": desc,
        "parameters": params,
    }


# -- schemas ------------------------------------------------------------------

LITE_SCHEMA = _make_schema(
    "web_search_keirolabs_lite",
    "Search the web using KeiroLabs lite endpoint — fast lightweight search.",
    _SEARCH_SCHEMA_BASE,
)

FAST_SCHEMA = _make_schema(
    "web_search_keirolabs_fast",
    "Search the web using KeiroLabs fast endpoint — minimal latency search.",
    _SEARCH_SCHEMA_BASE,
)

SEARCH_SCHEMA = _make_schema(
    "web_search_keirolabs_search",
    "Search the web using KeiroLabs standard search endpoint — full search results.",
    _SEARCH_SCHEMA_BASE,
)

ANSWER_SCHEMA = _make_schema(
    "web_search_keirolabs_answer",
    "Search the web using KeiroLabs answer endpoint — detailed answer generation.",
    _SEARCH_SCHEMA_BASE,
)

RESEARCH_SCHEMA = _make_schema(
    "web_search_keirolabs_research",
    "Search the web using KeiroLabs research endpoint — in-depth research.",
    _SEARCH_SCHEMA_BASE,
)

BATCH_SCHEMA = _make_schema(
    "web_search_keirolabs_batch",
    "Search the web using KeiroLabs batch endpoint — submit multiple queries at once.",
    _BATCH_SCHEMA,
)

EXTRACT_SCHEMA = _make_schema(
    "web_extract_keirolabs_extract",
    "Extract full page content from a URL using KeiroLabs.",
    _EXTRACT_SCHEMA,
)


# -- handlers -----------------------------------------------------------------


def _handle_search(endpoint: str, args: Dict[str, Any]) -> str:
    from hermes_keirolabs_web_search.provider import call_keirolabs_api

    query = args.get("query", "")
    if not query:
        return json.dumps({"error": "query is required"})

    try:
        data = call_keirolabs_api(endpoint, {"query": query})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.exception("KeiroLabs %s search failed", endpoint)
        return json.dumps({"error": str(exc)})

    results: List[Dict[str, Any]] = []

    answer_text = data.get("answer", data.get("text", data.get("data", "")))
    if isinstance(answer_text, str) and answer_text.strip():
        results.append(
            {"title": "KeiroLabs Answer", "answer": answer_text}
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
            snippet = (
                source.get("snippet")
                or source.get("content")
                or source.get("description", "")
            )
            results.append(
                {"url": url, "title": title, "snippet": snippet}
            )

    if not results:
        return json.dumps({"error": "No results returned from KeiroLabs"})

    return json.dumps({"results": results, "credits_remaining": data.get("creditsRemaining")})


def _handle_lite(args: Dict[str, Any]) -> str:
    return _handle_search("lite", args)


def _handle_fast(args: Dict[str, Any]) -> str:
    return _handle_search("fast", args)


def _handle_search_endpoint(args: Dict[str, Any]) -> str:
    return _handle_search("search", args)


def _handle_answer(args: Dict[str, Any]) -> str:
    return _handle_search("answer", args)


def _handle_research(args: Dict[str, Any]) -> str:
    return _handle_search("research", args)


def _handle_batch(args: Dict[str, Any]) -> str:
    from hermes_keirolabs_web_search.provider import call_keirolabs_api

    queries = args.get("queries", [])
    if not queries:
        return json.dumps({"error": "queries list is required"})
    if not isinstance(queries, list):
        return json.dumps({"error": "queries must be a list of strings"})

    try:
        data = call_keirolabs_api("batch", {"queries": queries})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.exception("KeiroLabs batch search failed")
        return json.dumps({"error": str(exc)})

    return json.dumps(data)


def _handle_extract(args: Dict[str, Any]) -> str:
    from hermes_keirolabs_web_search.provider import call_keirolabs_api

    url = args.get("url", "")
    if not url:
        return json.dumps({"error": "url is required"})

    try:
        data = call_keirolabs_api("extract", {"url": url})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.exception("KeiroLabs extract failed")
        return json.dumps({"error": str(exc)})

    content = data.get("content") or data.get("data") or data.get("text", "")
    if isinstance(content, dict):
        content = str(content)

    return json.dumps(
        {
            "url": url,
            "title": data.get("title", ""),
            "content": content,
            "metadata": data.get("metadata", {}),
        }
    )


# -- registration table -------------------------------------------------------

_TOOLS: List[Dict[str, Any]] = [
    {"schema": LITE_SCHEMA, "handler": _handle_lite},
    {"schema": FAST_SCHEMA, "handler": _handle_fast},
    {"schema": SEARCH_SCHEMA, "handler": _handle_search_endpoint},
    {"schema": ANSWER_SCHEMA, "handler": _handle_answer},
    {"schema": RESEARCH_SCHEMA, "handler": _handle_research},
    {"schema": BATCH_SCHEMA, "handler": _handle_batch},
    {"schema": EXTRACT_SCHEMA, "handler": _handle_extract},
]


def register_tools(ctx) -> None:
    """Register all KeiroLabs tools via the plugin context."""
    for entry in _TOOLS:
        schema = entry["schema"]
        handler = entry["handler"]
        ctx.register_tool(
            name=schema["name"],
            toolset=TOOLSET,
            schema=schema,
            handler=lambda args, h=handler, **kw: h(args),
            check_fn=_check_available,
            requires_env=["KEIROSLABS_API_KEY"],
            description=schema["description"],
        )
    logger.debug("Registered %d KeiroLabs tools", len(_TOOLS))
