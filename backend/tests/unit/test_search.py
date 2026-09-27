"""Unit tests for Search Service."""

from app.services.search_service import SearchResult, format_search_results_for_context


def test_search_result_formatting():
    results = [
        SearchResult(
            title="FastAPI Documentation",
            url="https://fastapi.tiangolo.com",
            snippet="High performance, easy to learn, fast to code, ready for production",
        ),
        SearchResult(
            title="Python Official Site",
            url="https://www.python.org",
            snippet="Python is a programming language that lets you work quickly",
        ),
    ]

    formatted = format_search_results_for_context(results)
    assert "FastAPI Documentation" in formatted
    assert "https://fastapi.tiangolo.com" in formatted
    assert "Python Official Site" in formatted


def test_search_result_empty():
    assert format_search_results_for_context([]) == ""
