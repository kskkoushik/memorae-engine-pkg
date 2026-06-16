# Memorae

Agentic personal memory over a raw message stream (WhatsApp, Slack, Gmail, calendar, notes). The LLM reads events through tools — no pre-labelled commitments or regex enrichment.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env   # set OPENROUTER_API_KEY
python api.py          # http://127.0.0.1:8000
```

## Architecture

```
memorae_mock_events.json → EventStore → optional ChromaDB RAG → LangChain agent → SSE → web UI
```

| Module | Role |
|--------|------|
| `event_store.py` | Load/filter raw events (date, source, keyword) |
| `event_tools.py` | Five LangChain tools for the agent |
| `rag.py` | ChromaDB + OpenRouter embeddings |
| `agent.py` | LangChain agent + streaming |
| `prompts.py` | System prompt |
| `engine.py` | Wires store, RAG, and agent |
| `api.py` | FastAPI + web UI |

## API

- `GET /` — web UI
- `POST /api/chat/stream` — SSE agent stream (`{"query": "..."}`)
- `GET /api/health` — status
- `GET /api/memory` — event stats
- `GET /api/event/{idx}` — single event

## Config (`.env`)

See `.env.example` for `OPENROUTER_API_KEY`, `MEMORAE_LLM`, `MEMORAE_RAG`, `MEMORAE_EVENTS`, `MEMORAE_NOW`, etc.
