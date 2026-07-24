"""KeiroLabs web search + extract plugin for Hermes.

Backed by the KeiroLabs search API using ``httpx``.

Activates when ``web.search_backend: keirolabs``, ``web.extract_backend: keirolabs``,
or ``web.backend: keirolabs`` is set in ``config.yaml`` and
``KEIROLABS_API_KEY`` is configured.
"""

from __future__ import annotations

from hermes_keirolabs_web_search.provider import KeiroLabsWebSearchProvider
from hermes_keirolabs_web_search.tools import register_tools


def register(ctx) -> None:
    """Register the KeiroLabs provider and tools with the Hermes plugin context."""
    ctx.register_web_search_provider(KeiroLabsWebSearchProvider())
    register_tools(ctx)
