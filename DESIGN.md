# Memorae — Design

This document covers the retrieval architecture, the memory architecture, the
context strategy, how contradictions and recency are handled, failure modes, and
the scaling and latency/cost optimisation discussion.

The central design bet: for this problem, **the bottleneck is reasoning over
time, not semantic recall.** A vector search finds the UIE events trivially —
they all say "UIE". What it cannot do is tell you that the deadline moved from
Apr 10 to Apr 13, that the procurement figure is now $48.5k and not $42k, that
"clause 8 approved" is progress rather than completion, or that 36 of the 200
events are in the future and must not be treated as known. So the core is a
deterministic, inspectable memory layer; embeddings and an LLM are optional
stages bolted onto a correct spine, not the spine itself.

---

## 1. Memory architecture

Raw events are immutable. Everything below is derived at runtime; nothing is
written back to the dataset, and no manual labels are added in preprocessing.

**Per-event enrichment (`enrich.py`).** Each event is annotated with signals
extracted by generic linguistic rules, not per-row labels:

- **actor / channel** — who authored it, in which channel.
- **owner detection** — the stream owner is whoever speaks in the first person
  most often. Learned from authorship ("Aarav"), never hard-coded.
- **obligation?** — is this the owner's to-do? Distinguishes self-commitments
  ("I promised", "I still need to"), imperatives directed at the owner
  ("Aarav: pay…"), and asks assigned by others ("can you…", "I need your…"),
  from things that merely look like tasks but are not (preferences, third-party
  actions).
- **lifecycle status** — `open / update / done / cancel`. Crucially, *unblock*
  language ("approved now", "no longer blocked") is routed to **update**, not
  done: a dependency clearing is progress, the action is still owed.
- **deadlines** — see §3. IST-aware and reschedule-aware.
- **importance** — stakeholder, money, legal, deadline, and consequence cues each
  add weight.
- **noise** — facilities chatter, newsletters, receipts, OTPs, digests score high.

**The memory substrate (`memory.py`)** organises enriched events into three
structures:

1. **Dedup groups.** Repeated low-signal chatter (the coffee-machine and
   air-conditioning spam) is collapsed by normalised body so it cannot flood a
   context window. 47 events collapse into 18 representatives here. Real
   obligations are never collapsed.

2. **Soft topic clusters.** Threads are built by **anchor-token membership**, not
   transitive union-find. An *anchor* is a token in a document-frequency band
   (appears in ≥2 but not too many events), with the owner's name, source names,
   channel names, email domains, person names, generic process words, and
   noise-only tokens removed. Each thread-sized anchor ("uie", "southridge",
   "rubric") seeds a topic = the events that mention it; an event may belong to
   several topics, and a small group mostly contained in a larger one is folded
   in ("sow"/"redlines" → "southridge", "appendix" → "uie"). This deliberately
   avoids the failure mode where one cross-cutting event ("Apr 12 UIE block
   removed, conflicts with apartment maintenance") chains every thread into one
   blob — that event simply lands in both the UIE and apartment topics, which is
   correct.

3. **The commitment ledger** (the heart of the system). One obligation is tracked
   across all its mentions. Obligations are merged when they share a specific
   topical noun anchor; then each commitment **resolves its own current state**:
   - **status** by recency + explicit markers, ignoring calendar reschedules and
     any marker from an off-topic event;
   - **due date** by letting updates beat originals (§3);
   - **slipping** if raised on multiple days, or flagged "still/again", or aged
     past its first deadline while still open;
   - **importance / consequence** carried up from its member events.

   This is what lets the engine say "UIE proposal, due in 6h, still open" from a
   pile of mentions that individually say Apr 10, Apr 13, "appendix missing",
   "add failure modes", and "block removed".

---

## 2. Retrieval architecture and context strategy

Retrieval (`retrieval.py`) is a **funnel**, so the identical code path is correct
at 200 events or 200k:

- **Stage 0 — intent + metadata prefilter.** The query is classified
  (focus / at-risk / procrastination / topic-summary / generic). The prefilter
  enforces `timestamp ≤ now` (future events are never knowable), and drops pure
  noise. This is the cheapest stage and removes the most material.
- **Stage 1 — query-specific structural scoring.** Each intent has its own scorer
  over the *resolved ledger*, not raw text:
  - *focus*: urgency × importance × actionability; overdue and due-today win,
    things with neither a deadline nor importance are dropped as background.
  - *at-risk*: open **and** (overdue **or** due within 48h), consequence-weighted.
  - *procrastination*: recurrence ≥ 2 days or explicit slip language, ranked by
    how long it has been dragging.
  - *topic-summary*: topic membership → chronological timeline → fact resolution.
- **Stage 2 — optional embedding rerank.** A pluggable slot. Off in this build
  because at 200 events lexical + structural scoring already separates signal
  cleanly; on at scale (see §5). It reranks Stage 1 survivors; it is never the
  first-pass recall mechanism.
- **Stage 3 — budget-aware packing.** Token cost is estimated as `chars / 4`.
  Highest-scoring items are kept greedily under the budget, then presented
  chronologically. The funnel counts (entered → matched → selected → tokens used)
  are returned for inspection.

**"Choose the right context, not the biggest."** The answer layer only ever sees
Stage 3's output. For Q1–Q3 that is typically 6–15 *lead* events (one per
commitment, the most informative recent mention), not every mention — the ledger
has already done the aggregation. For Q4 it is the thread's timeline minus noise
and future events, with the resolved facts surfaced first. Keeping generation
separate from retrieval is what guarantees groundedness: the synthesizer
physically cannot pull in an event the retriever did not select.

---

## 3. Contradiction and recency handling

This is the part embeddings cannot do, so it is explicit.

**Deadlines.** `timeutil.py` resolves explicit dates ("Mon Apr 13 15:00 IST"),
weekday names, and relative phrases ("tonight", "EOD", "before the review")
anchored to the event's **IST-local** date, because people in the stream write in
IST (UTC+5:30). Reschedule phrasing is handled directly: "moved **from** Apr 10
**to** Apr 13" resolves to the *target* (the later date), not the date being
superseded — a bug that, left unhandled, makes the UIE proposal look overdue
instead of due today.

**Supersession at the commitment level.** When a commitment has several deadline
signals, an explicit **update** ("ignore my earlier deadline; now due Apr 13")
beats an original ("by Friday Apr 10") regardless of which event is longer or
more detailed. Recency plus the update marker decide. The resolution is logged
("due updated by ev108 → Apr 13") and surfaced in `contradictions_resolved`.

**Supersession at the fact level (topic queries).** A dedicated resolver walks
the thread oldest→newest and reports the **current** value of any explicitly
corrected fact, plus what it replaced:
- amount: `$48.5k (was $42k; do not use the old value)`,
- external name: `Unified Intelligence Engine` (with UIE noted as internal),
- blocker: the dependency state change ("clause 8 approved", "waiting on
  diagrams, not procurement").

**Status contradictions.** "Approved now / no longer blocked" is treated as an
unblock (still open), not completion. Calendar cancellations are treated as
schedule changes and are never allowed to close a deliverable. A completion or
cancellation marker only counts if it comes from an event actually about that
commitment (a member, or one carrying the commitment's core anchor) — this stops
a "done" in a loosely-related thread from closing the wrong item.

---

## 4. Failure modes (known, and how they are bounded)

- **Lexical anchors miss a paraphrase.** A thread referred to with entirely
  different words than its anchor token could be under-recalled. Bounded today by
  topic folding and by the retriever's query-anchor expansion; the principled fix
  at scale is the Stage 2 embedding rerank.
- **Greedy commitment merge can mis-group.** Two obligations sharing a single
  rare but generic noun can over-merge (e.g. "maintenance" links a portal
  maintenance window to apartment maintenance), and a thread whose mentions use
  different rarest anchors can under-merge. The design biases toward
  **under-merging** because a mis-resolved status/due (over-merge) is worse for
  the user than the same item appearing twice. Both directions are covered by
  regression tests.
- **Deadline inference from vague phrasing.** Relative phrases produce
  lower-confidence deadlines; these are marked `[approx]` in the answer and
  flagged in `uncertainty`, never presented as hard dates.
- **Owner mis-detection** would mis-scope every obligation. It is derived from
  first-person authorship frequency; a stream with no clear first-person owner
  would degrade gracefully to treating imperatives generically.
- **Over-aggressive noise filtering** could drop a real item phrased casually.
  The noise score is overridden whenever an event is an obligation or carries a
  high-confidence deadline, so a casually-worded real task is retained.

Every one of these has at least one assertion in `evaluate.py`, so a regression
shows up as a failed check rather than a silently worse answer.

---

## 5. Scaling to production numbers

The brief's production scale is ~10k messages + ~1k notes + ~500 reminders with a
100k-token budget. The funnel is designed for exactly this.

- **Stage 0** stays O(n) and does the heavy lifting: a time filter plus cheap
  metadata predicates discard the large majority before any scoring. Anchors,
  document frequencies, dedup signatures, the topic index, and the commitment
  ledger are all **precomputed incrementally** as events arrive, so a query does
  not rebuild them.
- **Stage 1** scores only the prefiltered survivors against the ledger, which is
  already aggregated (commitments, not raw mentions), so the candidate set the
  scorer sees grows far more slowly than the raw event count.
- **Stage 2** turns on: embed events once at ingest, store vectors in an ANN
  index, and rerank Stage 1 survivors by query similarity. This is where
  embeddings earn their place — recall for paraphrased threads — without ever
  being the system's source of truth for state.
- **Stage 3** packs to the 100k budget with cluster-summary compression: instead
  of 40 near-identical chatter lines, the context carries one line plus a count
  ("38 facilities-chatter messages collapsed"). The ledger means a commitment
  contributes one resolved row, not its full mention history, unless the user
  drills in.

Memory tiering matches access patterns: a hot tier of open commitments and recent
events kept resolved in memory; a warm tier of the last N weeks on fast storage;
a cold tier of older events behind the ANN index, pulled only when a topic query
reaches back for them.

---

## 6. Optimisation: <2s latency and ~80% cost reduction

Targets: end-to-end latency under 2 seconds, and roughly an 80% cut in model
cost. The lever is to **spend tokens and model calls only where they change the
answer**, because the deterministic core is already correct without them.

**Retrieval and context.**
- Keep the funnel: Stage 0 + precomputed structures mean a query touches a small
  candidate set, so retrieval is well under the latency budget on its own.
- Pack tight. The budget packer and cluster-summary compression already minimise
  context size; smaller context is both faster and cheaper per call.

**Caching.**
- Cache the resolved ledger and topic index between queries (they change only
  when new events arrive), so repeated questions are near-instant.
- Cache LLM polish keyed by the *selected-context fingerprint*: identical
  selected context returns the cached prose with no model call. Focus/at-risk
  answers asked repeatedly through a day hit this cache until the underlying
  commitments change.

**Summaries / precompute.**
- Precompute per-thread rolling summaries and per-commitment one-liners at
  ingest. A topic query then assembles cached summaries plus the live fact
  resolver, rather than re-reading the whole thread through a model.

**Model routing.**
- Default to **no model**: the deterministic synthesizer answers Q1–Q3 fully and
  is the single biggest cost saving. Reserve a model call for cases that benefit
  from natural-language fusion (long topic summaries, ambiguous ad-hoc queries).
- When a model is used, route by difficulty: a small/cheap model for short
  polish, escalating to a larger model only for genuinely synthesis-heavy
  answers. Cap output tokens (already done at ~700).

**Memory tiers (cost of recall).**
- Embed and index once at ingest, not per query. Keep hot commitments resolved in
  memory so the common questions never touch the vector store at all.

**The quality trade-off, stated honestly.** Every saving above trades some
fluency or recall for speed and cost. The deterministic answers are correct and
specific but read more like a structured briefing than prose; the LLM polish buys
nicer language at the price of a call. Cluster compression and summary precompute
can hide a detail a user happens to want, which is why the system always exposes
the funnel and lets the user drill into a thread for the full timeline. The
design's stance is that for a personal assistant, being *fast, cheap, and right
about the current state* matters more than being eloquent, so the defaults
optimise for the former and treat eloquence as an opt-in.
