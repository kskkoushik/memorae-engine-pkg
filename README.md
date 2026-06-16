# Memorae

**Live demo:** [https://kskkoushik135--memorae-web.modal.run](https://kskkoushik135--memorae-web.modal.run)

**Memorae** is an agentic personal-memory assistant. It reads a real message stream — WhatsApp, Slack, Gmail, calendar, notes, reminders — and answers questions that require **judgement over time**: what matters today, what is slipping, what changed, what is still open.

There are no pre-labelled commitments, no regex enrichment pipeline, and no hidden state the model can hallucinate from. The LLM discovers everything by **calling tools** over raw events, then synthesizes a grounded answer with an optional audit trail.

---

## Why this exists

Personal message streams are messy. Deadlines move. Amounts get superseded. Noise sits next to signal. A keyword search finds "UIE" everywhere; a vector search finds vaguely related text. Neither alone answers:

> *"What should I focus on today?"*

Memorae combines **structured retrieval** (date, source, keyword) with **optional semantic search** (ChromaDB RAG), orchestrated by a LangChain agent that follows explicit rules: date-first, keyword-second, semantic-last, evidence-only answers.

---

## Architecture at a glance

```
memorae_mock_events.json
        │
        ▼
   EventStore ───────────────► keyword index + date/source filters
        │
        ▼ (optional, MEMORAE_RAG=1)
   ChromaDB RAG ────────────► OpenRouter embeddings (precomputed)
        │
        ▼
   LangChain Agent ──────────► 5 tools · system prompt · tool loop
        │
        ▼
   FastAPI SSE stream ───────► web UI (tools · thinking · answer · explanation)
```

| Component | File | Role |
|-----------|------|------|
| Event store | `memorae/event_store.py` | Load JSON events, filter by date/source/keyword, pack results |
| RAG index | `memorae/rag.py` | ChromaDB + OpenRouter embeddings for semantic search |
| Tools | `memorae/event_tools.py` | Five LangChain tools the agent calls |
| Agent | `memorae/agent.py` | LangChain `create_agent` + streaming event mapping |
| Prompt | `memorae/prompts.py` | System prompt: rules, routing, response format |
| Engine | `memorae/engine.py` | Wires store, RAG, agent; lazy RAG retry |
| API | `api.py` | FastAPI + SSE endpoint + health |
| UI | `web/index.html` | Streaming chat with tool chips and explanation toggle |
| Deploy | `modal_app.py` | Modal ASGI deployment with persistent Chroma volume |

For the full end-to-end design — indexing, tool loop, example walkthrough, agent rules — see **[DESIGN.md](DESIGN.md)**.

For how to evaluate the system — offline, online, regression, subjective rubrics — see **[EVALUATION.md](EVALUATION.md)**.

---

## Quick start (local)

**Requirements:** Python 3.10+, OpenRouter API key

```bash
pip install -r requirements.txt
cp .env.example .env          # add OPENROUTER_API_KEY
python api.py                 # http://127.0.0.1:8000
```

Open the browser UI, ask a question, and watch tool calls stream live.

### Health check

```bash
curl http://127.0.0.1:8000/api/health
```

Expected fields: `ok`, `owner`, `events`, `rag_enabled`, `rag_indexed`, `llm_enabled`.

---

## Quick start (Modal)

```bash
pip install modal
modal token set --token-id <ID> --token-secret <SECRET>
modal secret create memorae-env --from-dotenv .env
modal deploy modal_app.py
```

The deployed URL is printed on success (e.g. `https://<workspace>--memorae-web.modal.run`). RAG index persists on a Modal Volume at `/data/chroma`.

---

## Data model

Each event is a raw message with three fields:

```json
{
  "timestamp": "2026-04-01T04:45:00Z",
  "source": "whatsapp",
  "content": "Aarav: I promised Nina the UIE proposal v3 by Friday Apr 10 15:00 IST; ..."
}
```

- **`timestamp`** — ISO-8601 UTC. Events after scenario `now` are invisible.
- **`source`** — channel name (`whatsapp`, `slack`, `gmail`, `calendar`, …).
- **`content`** — full message text. All meaning (deadlines, names, status updates) lives here.

The mock dataset (`memorae_mock_events.json`) has ~200 events; 164 are visible at the default scenario time `2026-04-13T03:00:00Z`. Owner is inferred as **Aarav** from first-person messages.

---

## Agent tools

Every tool requires a `reason` string (logged and shown in the UI).

| Tool | Purpose | When to use |
|------|---------|-------------|
| `get_available_sources` | List channels + counts | First turn, "what do you have access to?" |
| `search_event_by_date` | Events in a date range | **Default first step** for time-shaped questions |
| `search_event_by_source` | One channel, optional dates | "Slack this week", "Gmail from Nina" |
| `get_event_by_keyword` | Word/phrase search (fuzzy) | Named person, project, topic (`UIE`, `Nina`) |
| `search_event_by_query` | Semantic RAG search | **Last resort** — unknown dates or keyword failure |

All search tools return:

```json
{
  "total_matched": 42,
  "returned": 30,
  "hidden_due_to_limit": 12,
  "limit": 30,
  "events": [ { "idx": 0, "timestamp": "...", "source": "...", "content": "..." } ]
}
```

If `hidden_due_to_limit > 0`, the agent must narrow or re-query — never answer "everything about X" from a truncated slice.

---

## Agent rules (summary)

The full rules live in `memorae/prompts.py`. Core principles:

1. **Evidence only** — every claim must trace to a tool result event.
2. **Future is invisible** — never cite events after scenario `now`.
3. **Newer wins** — later messages supersede earlier ones (deadline moves, completions).
4. **Date-first retrieval** — compute windows from `now`; adaptive 3d → 7d → 14d for open questions.
5. **Semantic search is last resort** — only when dates are unknown or keyword search fails (0 hits / too many noisy hits).
6. **Every tool call needs `reason`** — one sentence explaining why.
7. **Answer + optional `<explanation>`** — human answer first; audit block with cited `idx` values when relevant.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/api/chat/stream` | SSE agent stream. Body: `{"query": "..."}` |
| `GET` | `/api/health` | Engine status |
| `GET` | `/api/memory` | Event stats + source counts |
| `GET` | `/api/event/{idx}` | Single event by index |

### SSE event types

| Type | Meaning |
|------|---------|
| `meta` | Owner, now, event count, RAG status |
| `phase` | UI phase label (thinking, tool, writing) |
| `tool_start` | Tool name, reason, input params |
| `tool_end` | Tool result summary (matched/returned/hidden) |
| `reasoning` | Model reasoning stream (thinking models) |
| `token` | Answer text delta |
| `error` | Error message |
| `done` | Stream complete |

---

## Configuration

Copy `.env.example` to `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | — | LLM + embeddings |
| `MEMORAE_LLM` | `1` | Enable agent |
| `MEMORAE_LLM_MODEL` | `moonshotai/kimi-k2-thinking` | Chat model |
| `MEMORAE_RAG` | `1` | Enable Chroma index |
| `MEMORAE_EMBED_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Chroma storage |
| `MEMORAE_RAG_TOP_K` | `60` | Max semantic hits before filtering |
| `MEMORAE_EVENTS` | `./memorae_mock_events.json` | Events file |
| `MEMORAE_NOW` | `2026-04-13T03:00:00Z` | Scenario "now" |
| `MEMORAE_RECURSION_LIMIT` | `40` | Max agent tool-loop steps |

---

## Project structure

```
memorae-engine-pkg/
├── api.py                      FastAPI server
├── modal_app.py                Modal deployment
├── memorae/
│   ├── engine.py               Store + RAG + agent wiring
│   ├── event_store.py          Raw event loading and search
│   ├── event_tools.py          LangChain tools
│   ├── rag.py                  ChromaDB + embeddings
│   ├── agent.py                LangChain agent + SSE mapping
│   └── prompts.py              System prompt
├── web/
│   └── index.html              Streaming chat UI
├── memorae_mock_events.json    Mock message stream
├── requirements.txt
├── README.md                   This file
├── DESIGN.md                   Architecture and process design
└── EVALUATION.md               Evaluation framework
```

---

## Example queries

The mock dataset is built to stress-test real memory scenarios:

| Query | What it tests |
|-------|---------------|
| *What should I focus on today?* | Priority ranking, adaptive window, deadline awareness |
| *What commitments am I at risk of missing?* | Overdue / due-soon detection from prose |
| *What have I been procrastinating on?* | Repeated nudges, stale open loops |
| *Summarize everything related to the UIE proposal.* | Topic thread, supersession ($42k → $48.5k), deadline moves |

---

## Optimization question

> **If latency must drop below 2 seconds and cost must fall 80%, what would you change?**

Today a typical query costs **2–4 LLM round-trips** (plan → tool → read → answer), plus **0–1 embedding calls** if semantic search runs, on a thinking model. Cold-start RAG build can add 30–60s once per container.

### What we'd change (ranked by impact)

| Change | Latency | Cost | Quality tradeoff |
|--------|---------|------|------------------|
| **Route to a small fast model** (e.g. Haiku / GPT-4o-mini) for tool planning; reserve thinking model only for final synthesis | −40–60% | −70% | Slightly weaker multi-step planning; mitigated by strong prompt + tool schemas |
| **Kill semantic search on the hot path** | −200–800ms | −embedding cost | Lose paraphrase recall; acceptable if keyword + date cover 95% of queries (our prompt already treats RAG as last resort) |
| **Cap tool loop at 2 rounds** (down from 40) with a hard "read then answer" policy | −30–50% | −30–50% | May miss edge cases needing a third narrow pass; add regression tests |
| **Pre-warm containers** (Modal `min_containers=1`, keep engine singleton hot) | Eliminates cold start | +idle cost | None for warm requests |
| **Cache tool results** keyed by `(query_hash, date_window, tool, params)` TTL 5–15 min | −50% on repeat/similar queries | −50% LLM input tokens on cache hit | Stale if new events arrive — use TTL + invalidate on ingest |
| **Precompute daily summaries** per source/topic at ingest time ("Apr 12 Slack: UIE deadline moved to Apr 13") | −60% tokens per query | −60% | Summaries can miss nuance; agent reads raw events only when summary is insufficient |
| **Tiered memory** — hot: last 7 days in RAM index; warm: keyword index for full history; cold: RAG only offline | Faster date scans | Cheaper storage/compute | Older open loops need explicit keyword or widened window |
| **Shrink tool payloads** — return `idx + snippet` first; full `content` only on second fetch by index | −20% tokens | −20% | Extra tool round if snippets insufficient |
| **Streaming-first UX** — show first token at 500ms even if tools still running (parallel plan + prefetch today's events) | Perceived <2s | Neutral | Complexity in partial-answer handling |

### Recommended 2s / −80% stack

1. **Haiku-class router** for tool selection (1 round, max 2 tools).
2. **No RAG at runtime** — pre-built keyword + date indexes only; rebuild RAG offline for eval, not prod hot path.
3. **Precomputed "open loops" digest** refreshed hourly (deadline, person, status, last mention idx) — agent reads digest + fetches 5–10 raw events max.
4. **Container always warm** + **Chroma on persistent volume** (already on Modal).
5. **Aggressive caching** of `(today's events, owner focus query)` — most morning "what should I focus on" queries hit cache.

### What we would not sacrifice

- **Evidence grounding** — every claim still maps to event `idx`.
- **Future-leak guard** — events after `now` never enter context.
- **Supersession** — agent still resolves "newer wins" by reading chronologically within a thread.
- **Regression suite** — see [EVALUATION.md](EVALUATION.md); faster/cheaper changes that break traps are rejected.

The fundamental tradeoff: **agentic flexibility vs. precomputed structure**. Under tight latency/cost, shift left — precompute priorities at ingest, use the LLM only to narrate a small, verified slice.

---

## License

Internal / assessment project. See repository for terms.
