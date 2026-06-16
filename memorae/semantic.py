"""
semantic.py
-----------
Primary retrieval path: meaning-first via RAG, enriched with memory-layer context.
No keyword intent classification — semantic search is always step one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .retrieval import _resolve_facts, _pack, Selected
from .timeutil import humanize_delta

if TYPE_CHECKING:
    from .memory import Memory
    from .rag import EventIndex, RagHit


@dataclass
class SemanticContext:
    query: str
    hits: list["RagHit"]
    selected: list[Selected]
    facts: list[dict] = field(default_factory=list)
    commitments: list[dict] = field(default_factory=list)
    funnel: dict = field(default_factory=dict)


def retrieve_semantic(
    mem: "Memory",
    query: str,
    rag_index: "EventIndex",
    budget_tokens: int = 2500,
) -> SemanticContext:
    """RAG-primary retrieval: semantic hits → enrich → budget pack."""
    top_k = int(__import__("os").environ.get("MEMORAE_RAG_TOP_K", "20"))
    hits = rag_index.query(query, top_k=top_k)

    cands: list[Selected] = []
    seen: set[int] = set()
    for hit in hits:
        if hit.idx in seen or hit.idx < 0 or hit.idx >= len(mem.events):
            continue
        ev = mem.events[hit.idx]
        if ev.ts > mem.now:
            continue
        seen.add(hit.idx)
        cands.append(Selected(
            idx=hit.idx,
            score=hit.score,
            reason=f"semantic match ({hit.score:.2f})",
            event=ev,
        ))

    kept, used = _pack(cands, budget_tokens)
    event_idxs = {s.idx for s in kept}
    facts = _resolve_facts([s.event for s in kept])

    # Attach commitment context for selected events
    commitments = _commitments_for_events(mem, event_idxs)

    return SemanticContext(
        query=query,
        hits=hits,
        selected=kept,
        facts=facts,
        commitments=commitments,
        funnel={
            "mode": "semantic_primary",
            "rag_enabled": True,
            "rag_hits": len(hits),
            "events_selected": len(kept),
            "approx_tokens_used": used,
            "token_budget": budget_tokens,
            "facts_resolved": len(facts),
        },
    )


def _commitments_for_events(mem: "Memory", event_idxs: set[int]) -> list[dict]:
    from .retrieval import _commitment_label
    out = []
    now = mem.now
    for c in mem.commitments:
        if not (set(c.members) & event_idxs) and c.status != "open":
            continue
        if c.status != "open" and not (set(c.members) & event_idxs):
            continue
        due_str = ""
        if c.due:
            due_str = humanize_delta(c.due, now)
        out.append({
            "label": _commitment_label(c),
            "status": c.status,
            "due": c.due.isoformat().replace("+00:00", "Z") if c.due else None,
            "due_human": due_str,
            "overdue": c.overdue(now),
            "imminent": c.imminent(now, 48),
            "slipping": c.slipping,
            "importance": round(c.importance, 2),
            "members": c.members,
        })
    out.sort(key=lambda d: (not d["overdue"], not d["imminent"], d["due"] or "z"))
    return out[:12]


def format_context_for_llm(ctx: SemanticContext, mem: "Memory") -> str:
    """Serialize retrieved context for the LLM."""
    lines = [f"Query: {ctx.query}", f"Scenario now: {mem.now.isoformat()}", ""]
    if ctx.facts:
        lines.append("Resolved facts (use current values):")
        for f in ctx.facts:
            line = f"  - {f['fact']}: {f['current']}"
            if f.get("superseded"):
                line += f" (was {f['superseded']})"
            lines.append(line)
        lines.append("")
    if ctx.commitments:
        lines.append("Related open commitments:")
        for c in ctx.commitments[:8]:
            flags = []
            if c["overdue"]:
                flags.append("OVERDUE")
            elif c["imminent"]:
                flags.append("DUE_SOON")
            if c["slipping"]:
                flags.append("slipping")
            flag = f" [{', '.join(flags)}]" if flags else ""
            due = f" — {c['due_human']}" if c.get("due_human") else ""
            lines.append(f"  - {c['label']}{due}{flag}")
        lines.append("")
    lines.append("Retrieved events (semantic search):")
    for s in ctx.selected:
        e = s.event
        lines.append(
            f"  [ev{e.idx} | {e.source} | {e.ts.strftime('%Y-%m-%d %H:%M UTC')}] "
            f"{e.content[:400]}"
        )
    return "\n".join(lines)
