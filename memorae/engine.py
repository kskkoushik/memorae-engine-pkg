"""
engine.py
---------
Loads raw events + optional RAG index. The agent discovers context via tools —
no regex enrichment or pre-built commitment ledger on the agent path.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .event_store import EventStore, parse_ts
from .rag import EventIndex

DEFAULT_NOW = datetime(2026, 4, 13, 3, 0, 0, tzinfo=timezone.utc)


def _load_env() -> None:
    """Ensure .env is loaded from package root (works regardless of cwd)."""
    try:
        from dotenv import load_dotenv
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        load_dotenv(os.path.join(root, ".env"))
    except ImportError:
        pass


def _try_build_rag(store: EventStore) -> EventIndex | None:
    if os.environ.get("MEMORAE_RAG") != "1":
        print("[engine] RAG disabled (MEMORAE_RAG != 1)", file=sys.stderr, flush=True)
        return None
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("[engine] RAG disabled (OPENROUTER_API_KEY missing)", file=sys.stderr, flush=True)
        return None
    try:
        idx = EventIndex.build_from_store(store)
        if idx:
            stats = idx.stats()
            print(
                f"[engine] RAG ready | indexed={stats.get('indexed', 0)} | "
                f"model={stats.get('embed_model', '?')}",
                file=sys.stderr, flush=True,
            )
        return idx
    except Exception as e:
        print(
            f"[engine] RAG index build failed ({type(e).__name__}: {str(e)[:200]})",
            file=sys.stderr, flush=True,
        )
        return None


@dataclass
class Engine:
    store: EventStore
    rag_index: object | None = field(default=None, repr=False)
    now: datetime = field(default_factory=lambda: DEFAULT_NOW)

    @classmethod
    def from_events_file(cls, path: str, now: datetime = DEFAULT_NOW) -> "Engine":
        _load_env()
        store = EventStore.from_file(path, now=now)
        rag_index = _try_build_rag(store)
        return cls(store=store, rag_index=rag_index, now=now)

    def ensure_rag(self) -> EventIndex | None:
        """Load RAG index on demand if startup build was skipped or failed."""
        if self.rag_index is not None:
            return self.rag_index
        _load_env()
        self.rag_index = _try_build_rag(self.store)
        return self.rag_index

    @property
    def mem(self):
        """Shim for legacy API endpoints that expect .mem.owner / .mem.now."""
        return self.store

    def agent(self):
        from .agent import MemoryAgent
        return MemoryAgent(self)

    def chat(self, query: str, budget: int = 2500) -> dict:
        return self.agent().chat(query, budget=budget)

    def rag_stats(self) -> dict:
        if self.rag_index is None:
            return {"enabled": False, "indexed": 0}
        return self.rag_index.stats()

    def event_detail(self, idx: int) -> dict | None:
        if idx < 0 or idx >= len(self.store.events):
            return None
        e = self.store.events[idx]
        return {
            **e.to_dict(),
            "future": e.ts > self.now,
        }

    # Legacy stubs (old inspectable pipeline removed from agent path)
    def answer(self, query: str, budget_tokens: int = 1500, use_llm: bool = True) -> dict:
        return self.chat(query, budget=budget_tokens)

    def ledger(self) -> list[dict]:
        return []

    def topics(self) -> list[dict]:
        return []

    def run_suite(self, use_llm: bool = True) -> list[dict]:
        queries = [
            "What should I focus on today?",
            "What commitments am I at risk of missing?",
            "What have I been procrastinating on?",
            "Summarize everything related to the UIE proposal.",
        ]
        return [self.answer(q, use_llm=use_llm) for q in queries]
