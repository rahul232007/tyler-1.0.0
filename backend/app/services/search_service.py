"""
JARVIS - Web Search Service using DuckDuckGo.
Clean abstraction over duckduckgo_search with:
- Result title, URL, snippet extraction
- Timeout handling
- Graceful failure (returns empty list, never raises to caller)
- No sensitive data logged
"""

from __future__ import annotations

import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SearchResult:
    """Single web search result."""

    __slots__ = ("snippet", "title", "url")

    def __init__(self, title: str, url: str, snippet: str) -> None:
        self.title = title
        self.url = url
        self.snippet = snippet

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


async def web_search(
    query: str,
    max_results: int | None = None,
    region: str = "in-en",  # India English
) -> list[SearchResult]:
    """
    Search the web using DuckDuckGo and return structured results.
    Returns empty list on any failure — never raises to caller.

    Args:
        query: Search query string
        max_results: Maximum number of results (default from settings)
        region: DuckDuckGo region code
    """
    if not settings.web_search_enabled:
        logger.info("Web search is disabled (WEB_SEARCH_ENABLED=false).")
        return []

    if not query or not query.strip():
        return []

    limit = max_results or settings.web_search_max_results
    logger.info("Web search | query_len=%d | max_results=%d", len(query), limit)

    try:
        from duckduckgo_search import DDGS

        results: list[SearchResult] = []
        with DDGS(timeout=10) as ddgs:
            for item in ddgs.text(
                query.strip(),
                region=region,
                safesearch="moderate",
                max_results=limit,
            ):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                url = str(item.get("href", item.get("url", ""))).strip()
                snippet = str(item.get("body", item.get("snippet", ""))).strip()
                if url:
                    results.append(SearchResult(title=title, url=url, snippet=snippet))
        logger.info("Web search returned %d result(s).", len(results))
        return results

    except ImportError:
        logger.error(
            "duckduckgo_search not installed. Run: pip install duckduckgo-search"
        )
        return []
    except Exception as exc:
        logger.warning("Web search failed: %s — %s", type(exc).__name__, str(exc)[:100])
        return []


def format_search_results_for_context(results: list[SearchResult]) -> str:
    """Format search results as a readable string for LLM context injection."""
    if not results:
        return ""
    lines = ["Web search results:"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title}")
        if r.snippet:
            lines.append(f"   {r.snippet[:200]}")
        lines.append(f"   Source: {r.url}")
    return "\n".join(lines)
