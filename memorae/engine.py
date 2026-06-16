from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .event_store import EventStore
from .rag import EventIndex

DEFAULT_NOW = datetime(2026, 4, 13, 3, 0, 0, tzinfo=timezone.utc)


def _load_env() -> None:
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
                file=sys.stderr,
                flush=True,
            )
        return idx
    except Exception as e:
        print(
            f"[engine] RAG index build failed ({type(e).__name__}: {str(e)[:200]})",
            file=sys.stderr,
            flush=True,
        )
        return None


@dataclass
class Engine:
    store: EventStore
    rag_index: EventIndex | None = field(default=None, repr=False)
    now: datetime = field(default_factory=lambda: DEFAULT_NOW)

    @classmethod
    def from_events_file(cls, path: str, now: datetime = DEFAULT_NOW) -> "Engine":
        _load_env()
        store = EventStore.from_file(path, now=now)
        return cls(store=store, rag_index=_try_build_rag(store), now=now)

    def ensure_rag(self) -> EventIndex | None:
        if self.rag_index is not None:
            return self.rag_index
        _load_env()
        self.rag_index = _try_build_rag(self.store)
        return self.rag_index

    def agent(self):
        from .agent import MemoryAgent
        return MemoryAgent(self)

    def chat(self, query: str) -> dict:
        return self.agent().chat(query)

    def rag_stats(self) -> dict:
        if self.rag_index is None:
            return {"enabled": False, "indexed": 0}
        return self.rag_index.stats()

    def event_detail(self, idx: int) -> dict | None:
        if idx < 0 or idx >= len(self.store.events):
            return None
        e = self.store.events[idx]
        return {**e.to_dict(), "future": e.ts > self.now}
