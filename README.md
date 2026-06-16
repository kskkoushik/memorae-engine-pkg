# Memorae — Personal Memory Query Engine

A runnable memory engine that ingests a stream of timestamped personal events
(Slack, Gmail, WhatsApp, calendar, notes, reminders) and answers questions that
need **judgement over time**, not just keyword recall:

1. What should I focus on today?
2. What commitments am I at risk of missing?
3. What have I been procrastinating on?
4. Summarize everything related to the UIE proposal.

It is built so the hard parts of the problem — *which deadline is current*, *what
was superseded*, *what is noise*, *what is not yet knowable* — are handled by an
inspectable, deterministic core, with embeddings and an LLM as optional add-ons.

---

## Run it

No dependencies. Standard library only. Python 3.10+.

```bash
cd memorae-engine

# Answer the four assessment queries and write structured output to outputs/answers.json
python run.py

# Deterministic only (this is also the default if no API key is set)
python run.py --plain

# Ask an ad-hoc question
python run.py "what is on my plate before Thursday?"

# Run the evaluation harness (offline correctness + trap regressions + rubric)
python evaluate.py
```

The events file defaults to `/mnt/user-data/uploads/memorae_mock_events.json`.
Override with `--events path.json` or `MEMORAE_EVENTS=path.json`. The scenario
"now" defaults to `2026-04-13T03:00:00Z`; override with `--now <ISO>`.

### Web console (optional)

A FastAPI server exposes the engine and serves a single-page console. This is the
only part of the project with third-party dependencies; the engine core and CLI
stay pure stdlib.

```bash
pip install -r requirements-web.txt      # fastapi + uvicorn
python api.py                            # http://127.0.0.1:8000
```

The console has two views:

- **Ask** — the four canonical questions plus a free-text box. Each answer renders
  with a *priority rail* (commitments plotted relative to "now": overdue to the
  left, due-today/≤48h in amber, later in grey), the grounded answer, the exact
  **evidence** events used with their selection reason, the full **reasoning**
  (signals, why-selected, why-ignored, contradictions resolved, uncertainty), and
  the retrieval **pipeline** counts with resolved facts (e.g. `$48.5k` with the
  retired `$42k` struck through). An LLM-polish toggle and a context-budget slider
  are exposed.
- **Memory** — the memory layer itself: stats, the full commitment ledger on the
  priority rail, the resolved ledger table, and the anchor-token topics.

API endpoints: `POST /api/query`, `GET /api/suite`, `GET /api/memory`,
`GET /api/event/{idx}`, `GET /api/health`. LLM polish obeys the same
`MEMORAE_LLM=1` + provider-key contract; `/api/query` also accepts `use_llm` per
request.

### Optional LLM polish

The default output is fully deterministic. To have an LLM rewrite the *already
selected* context into more natural prose (it may not add facts), set
`MEMORAE_LLM=1` plus a provider key. Two providers are supported out of the box,
both called over `urllib` (no SDK install):

```bash
# Option A — OpenRouter (e.g. Kimi K2)
export MEMORAE_LLM=1
export OPENROUTER_API_KEY=sk-or-...
# model defaults to moonshotai/kimi-k2-thinking; override if you like:
# export MEMORAE_LLM_MODEL=moonshotai/kimi-k2-thinking
python run.py

# Option B — Anthropic
export MEMORAE_LLM=1
export ANTHROPIC_API_KEY=sk-ant-...
# export MEMORAE_LLM_MODEL=claude-sonnet-4-6
python run.py
```

Provider is auto-detected from whichever key is present (OpenRouter takes
priority). Only the already-selected context and resolved facts are sent; the
model is instructed not to introduce new facts and to preserve every date and
resolved value. At most one short call per query. Any error (network, auth,
timeout) prints a one-line reason to stderr and falls back to the deterministic
answer, so a query never fails because polish failed. With the LLM off there is
no network use at all.

Note: if you run inside a network-restricted sandbox, the provider host must be
on the egress allowlist or the call will 403 and fall back to deterministic.

---

## What you get back

`run.py` prints a readable digest and writes the full structured record per query
to `outputs/answers.json`, matching the suggested schema:

```json
{
  "query": "...",
  "intent": "focus_today",
  "answer": "...",
  "selected_context": [{ "idx": 0, "timestamp": "...", "source": "...",
                          "content": "...", "why": "...", "score": 0.0 }],
  "reasoning": {
    "signals_considered": ["..."],
    "why_selected": ["ev0: ...", "..."],
    "why_ignored": "future-dated events, completed items, deduplicated chatter, ...",
    "contradictions_resolved": ["deadline moved Apr 10 -> Apr 13 ...",
                                "amount: now $48.5k (was $42k) ..."],
    "uncertainty": "..."
  },
  "debug": { "now": "...", "funnel": {...}, "resolved_facts": [...], "owner_detected": "Aarav" }
}
```

---

## How it works (one screen)

```
events.json
   │
   ▼  enrich.py        per-event signal extraction (no labels written back)
   │                   actor/channel · obligation? · lifecycle status ·
   │                   deadlines (IST-aware, reschedule-aware) · importance · noise
   ▼  memory.py        the memory substrate
   │                   · dedup of repeated low-signal chatter
   │                   · soft topic clusters (anchor-token membership)
   │                   · Commitment ledger: one obligation tracked across many
   │                     mentions, with CURRENT status + due resolved by recency
   │                     and explicit update markers (this is where supersession
   │                     lives: "now due Apr 13" beats "by Apr 10")
   ▼  retrieval.py     query → intent → funnel
   │                   Stage 0 metadata prefilter (enforces ts ≤ now, drops noise)
   │                   Stage 1 query-specific structural scoring
   │                   Stage 2 optional embedding rerank (pluggable, off here)
   │                   Stage 3 budget-aware packing (token estimate = chars/4)
   │                   + fact resolver for topic queries ($42k→$48.5k, name, blocker)
   ▼  synth.py         deterministic grounded answer; optional LLM polish over the
   │                   SAME selected context (cannot introduce new events)
   ▼  engine.py        orchestrates the 4 steps into the inspectable record above
```

The owner of the stream ("Aarav") and the topic anchors ("uie", "southridge",
"rubric", ...) are **discovered from the data at runtime**, not hard-coded.

See `DESIGN.md` for the retrieval/memory architecture, contradiction and recency
handling, failure modes, the scaling path to 100k-token budgets, and the
latency/cost optimisation discussion.

---

## Layout

```
memorae-engine/
├── run.py                 CLI entry point
├── evaluate.py            offline correctness + trap regressions + rubric (no pytest)
├── api.py                 FastAPI server (optional web console)
├── web/
│   └── index.html         single-page console (priority rail + evidence + reasoning)
├── README.md
├── DESIGN.md
├── requirements.txt       core: intentionally empty (stdlib only)
├── requirements-web.txt   web console only: fastapi + uvicorn
├── memorae/
│   ├── timeutil.py        IST-aware deadline extraction + resolution
│   ├── enrich.py          per-event signal extraction
│   ├── memory.py          dedup · topics · commitment ledger (supersession)
│   ├── retrieval.py       intent · funnel · fact resolution · budget packing
│   ├── synth.py           deterministic answers + optional LLM polish
│   └── engine.py          orchestration → inspectable 4-step record
└── outputs/
    └── answers.json       written by `python run.py`
```
