from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Annotated, Optional

from langchain.tools import tool

from .event_store import EventStore, parse_date_bound, validate_date_range

if TYPE_CHECKING:
    from .engine import Engine


def _log_tool(name: str, reason: str, **params) -> None:
    print(f"\n[memorae:tool] ▶ {name}", file=sys.stderr, flush=True)
    print(f"  reason: {reason}", file=sys.stderr, flush=True)
    for key, val in params.items():
        if key == "reason" or val is None:
            continue
        print(f"  {key}: {val}", file=sys.stderr, flush=True)


def _empty_search_result(*, limit: int = 30, error: str | None = None) -> dict:
    out = {
        "events": [],
        "total_matched": 0,
        "returned": 0,
        "hidden_due_to_limit": 0,
        "limit": limit,
    }
    if error:
        out["error"] = error
    return out


def build_event_tools(engine: "Engine") -> list:
    store: EventStore = engine.store

    def _rag():
        return engine.ensure_rag()

    @tool
    def get_available_sources(
        reason: Annotated[str, "One sentence: why you are calling this tool right now"],
    ) -> str:
        """Return every event source in the memory stream (e.g. whatsapp, slack, gmail) with event counts."""
        _log_tool("get_available_sources", reason)
        sources = store.get_sources()
        counts = store.source_counts()
        result = {
            "sources": sources,
            "counts": counts,
            "total_visible_events": len(store._visible),
            "scenario_now": store.now.isoformat().replace("+00:00", "Z"),
        }
        print(f"  → {len(sources)} sources, {len(store._visible)} visible events", file=sys.stderr, flush=True)
        return json.dumps(result, indent=2)

    @tool
    def search_event_by_source(
        reason: Annotated[str, "One sentence: why you are calling this tool right now"],
        source_name: Annotated[str, "Source to filter by, e.g. whatsapp, slack, gmail"],
        start_date: Annotated[Optional[str], "Optional ISO start date/datetime (inclusive)"] = None,
        end_date: Annotated[Optional[str], "Optional ISO end date/datetime (inclusive)"] = None,
        limit: Annotated[int, "Max events to return"] = 50,
    ) -> str:
        """Get events from a specific source. Optionally filter by date range. Results sorted oldest-first."""
        _log_tool("search_event_by_source", reason, source_name=source_name,
                  start_date=start_date, end_date=end_date, limit=limit)
        result = store.search_by_source(source_name, start_date, end_date, limit)
        print(f"  → matched={result.get('total_matched')}, returned={result.get('returned')}, "
              f"hidden={result.get('hidden_due_to_limit', 0)}", file=sys.stderr, flush=True)
        return json.dumps(result, indent=2)

    @tool
    def get_event_by_keyword(
        reason: Annotated[str, "One sentence: why you are calling this tool right now"],
        keyword: Annotated[str, "Word or phrase to search for in event content (person name, topic, project)"],
        start_date: Annotated[Optional[str], "Optional ISO start date/datetime"] = None,
        end_date: Annotated[Optional[str], "Optional ISO end date/datetime"] = None,
        limit: Annotated[int, "Max events to return"] = 50,
        source_name: Annotated[Optional[str], "Optional source filter"] = None,
    ) -> str:
        """Fast keyword search over event content. Use for people, topics, project names. Optional date/source filters."""
        _log_tool("get_event_by_keyword", reason, keyword=keyword,
                  start_date=start_date, end_date=end_date, limit=limit, source_name=source_name)
        result = store.search_by_keyword(keyword, start_date, end_date, limit, source_name)
        print(f"  → matched={result.get('total_matched')}, returned={result.get('returned')}, "
              f"hidden={result.get('hidden_due_to_limit', 0)}", file=sys.stderr, flush=True)
        return json.dumps(result, indent=2)

    @tool
    def search_event_by_date(
        reason: Annotated[str, "One sentence: why you are calling this tool right now"],
        start_date: Annotated[str, "ISO start date/datetime (inclusive)"],
        end_date: Annotated[str, "ISO end date/datetime (inclusive)"],
        limit: Annotated[int, "Max events to return"] = 50,
    ) -> str:
        """Return all events within a date range, sorted chronologically. Use for 'last week', 'yesterday', etc."""
        _log_tool("search_event_by_date", reason, start_date=start_date, end_date=end_date, limit=limit)
        result = store.search_by_date(start_date, end_date, limit)
        print(f"  → matched={result.get('total_matched')}, returned={result.get('returned')}, "
              f"hidden={result.get('hidden_due_to_limit', 0)}", file=sys.stderr, flush=True)
        return json.dumps(result, indent=2)

    @tool
    def search_event_by_query(
        reason: Annotated[str, "One sentence: why you are calling this tool right now"],
        query: Annotated[str, "Natural-language semantic search query"],
        source_names: Annotated[Optional[list[str]], "Optional list of sources to filter"] = None,
        start_date: Annotated[Optional[str], "Optional ISO start date/datetime"] = None,
        end_date: Annotated[Optional[str], "Optional ISO end date/datetime"] = None,
        limit: Annotated[int, "Max events to return"] = 30,
    ) -> str:
        """Semantic (RAG) search — last resort only. Use when dates are unknown and you must find an answer, or when get_event_by_keyword returned no matches or too many noisy matches. Prefer search_event_by_date and get_event_by_keyword first."""
        _log_tool("search_event_by_query", reason, query=query,
                  source_names=source_names, start_date=start_date, end_date=end_date, limit=limit)

        query = (query or "").strip()
        if not query:
            print("  → empty query", file=sys.stderr, flush=True)
            return json.dumps(_empty_search_result(
                limit=limit,
                error="query is required — use get_event_by_keyword or search_event_by_date instead",
            ))

        rag = _rag()
        if rag is None:
            print("  → RAG unavailable", file=sys.stderr, flush=True)
            return json.dumps(_empty_search_result(
                limit=limit,
                error="RAG index unavailable — use get_event_by_keyword or search_event_by_date instead",
            ))

        top_k = max(limit * 3, int(__import__("os").environ.get("MEMORAE_RAG_TOP_K", "60")))
        try:
            hits = rag.query(query, top_k=top_k)
        except Exception as e:
            print(f"  → RAG query failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return json.dumps(_empty_search_result(
                limit=limit,
                error=(
                    f"Semantic search failed ({type(e).__name__}: {e}). "
                    "Fall back to get_event_by_keyword or search_event_by_date."
                ),
            ))

        scores = {h.idx: h.score for h in hits}
        indices = [h.idx for h in hits]

        start = parse_date_bound(start_date)
        end = parse_date_bound(end_date, end_of_day=True)
        _, _, err = validate_date_range(start, end)
        if err:
            print(f"  → error: {err}", file=sys.stderr, flush=True)
            return json.dumps(_empty_search_result(limit=limit, error=err))

        try:
            result = store.search_by_indices(
                indices,
                sources=source_names,
                start=start,
                end=end,
                limit=limit,
                scores=scores,
            )
        except Exception as e:
            print(f"  → result filter failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return json.dumps(_empty_search_result(
                limit=limit,
                error=(
                    f"Semantic search results could not be filtered ({type(e).__name__}: {e}). "
                    "Fall back to get_event_by_keyword or search_event_by_date."
                ),
            ))

        result["query"] = query
        result["rag_hits"] = len(hits)
        if not hits:
            result["notice"] = (
                "Semantic search returned no matches — try get_event_by_keyword "
                "or search_event_by_date."
            )
        print(f"  → rag_hits={len(hits)}, matched={result.get('total_matched')}, "
              f"returned={result.get('returned')}", file=sys.stderr, flush=True)
        return json.dumps(result, indent=2)

    return [
        get_available_sources,
        search_event_by_source,
        get_event_by_keyword,
        search_event_by_date,
        search_event_by_query,
    ]
