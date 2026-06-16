#!/usr/bin/env python3
"""FastAPI server for the Memorae agent. Run: pip install -r requirements.txt && python api.py"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from memorae.engine import Engine, DEFAULT_NOW

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

_events = os.environ.get("MEMORAE_EVENTS", os.path.join(HERE, "memorae_mock_events.json"))
if not os.path.isabs(_events):
    _events = os.path.normpath(os.path.join(HERE, _events))
EVENTS = _events
INDEX = os.path.join(HERE, "web", "index.html")

app = FastAPI(title="Memorae Memory Agent", version="3.0")
_engine: Engine | None = None


def engine() -> Engine:
    global _engine
    if _engine is None:
        now = DEFAULT_NOW
        override = os.environ.get("MEMORAE_NOW")
        if override:
            s = override.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            now = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        if not os.path.exists(EVENTS):
            raise RuntimeError(f"events file not found: {EVENTS}")
        _engine = Engine.from_events_file(EVENTS, now=now)
        if _engine.rag_index is None:
            print(
                "[api] WARNING: RAG not loaded — semantic search retries on first query",
                file=sys.stderr,
                flush=True,
            )
    return _engine


class ChatIn(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    if not os.path.exists(INDEX):
        return HTMLResponse("<h1>web/index.html missing</h1>", status_code=500)
    with open(INDEX, encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health() -> dict:
    eng = engine()
    llm_on = os.environ.get("MEMORAE_LLM") == "1" and bool(os.environ.get("OPENROUTER_API_KEY"))
    rag = eng.rag_stats()
    return {
        "ok": True,
        "now": eng.now.isoformat().replace("+00:00", "Z"),
        "owner": eng.store.owner,
        "events": len(eng.store._visible),
        "sources": eng.store.get_sources(),
        "llm_enabled": llm_on,
        "llm_model": os.environ.get("MEMORAE_LLM_MODEL", "moonshotai/kimi-k2-thinking"),
        "rag_enabled": rag.get("enabled", False),
        "rag_indexed": rag.get("indexed", 0),
        "mode": "agentic_langchain",
    }


@app.post("/api/chat/stream")
async def chat_stream(body: ChatIn):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query is empty")

    agent = engine().agent()

    async def event_gen():
        async for event in agent.astream_events(body.query):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/memory")
def memory() -> dict:
    eng = engine()
    return {
        "now": eng.now.isoformat().replace("+00:00", "Z"),
        "owner": eng.store.owner,
        "stats": eng.store.stats(),
        "sources": eng.store.source_counts(),
    }


@app.get("/api/event/{idx}")
def event(idx: int) -> dict:
    eng = engine()
    detail = eng.event_detail(idx)
    if detail is None:
        raise HTTPException(status_code=404, detail="event index out of range")
    return detail


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
