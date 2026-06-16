"""
retrieval.py
------------
Turns a natural-language query into a *small, right* slice of memory.

The retriever is a funnel, so the same code path works at 200 events or 200k:

  Stage 0  intent + metadata prefilter   (cheap, enforces ts <= now, drops noise)
  Stage 1  structural / lexical scoring   (query-specific signal; no model needed)
  Stage 2  optional embedding rerank      (pluggable; off by default)
  Stage 3  budget-aware packing           (keep the most useful, summarise the rest)

Only Stage 3's output is shown to the answer layer. Every stage records how many
items entered and left, so the funnel is inspectable (the "scaling story").

Nothing here calls a model. Q1/Q2/Q3 reason over the resolved Commitment ledger;
Q4 reasons over a topic's event timeline plus a fact resolver that applies the
supersessions ($42k -> $48.5k, internal name -> external name, blocker changes).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .enrich import Event
from .memory import Memory, Commitment
from .timeutil import humanize_delta

if TYPE_CHECKING:
    from .rag import EventIndex


# ---------------------------------------------------------------------------
# Query intent
# ---------------------------------------------------------------------------
FOCUS = "focus_today"
RISK = "at_risk"
PROCRASTINATION = "procrastination"
TOPIC = "topic_summary"
GENERIC = "generic"

_INTENT_CUES = [
    (FOCUS, re.compile(r"\b(focus|today|priorit|most important|work on (today|now)|"
                       r"what should i (do|work)|top of)", re.I)),
    (RISK, re.compile(r"\b(risk|at risk|miss|slip|drop the ball|behind on|"
                      r"deadline|overdue|due soon|about to)", re.I)),
    (PROCRASTINATION, re.compile(r"\b(procrastinat|putting off|avoiding|keep "
                                 r"(forgetting|pushing)|been meaning|stalling|"
                                 r"never get|dragging)", re.I)),
]


def classify(query: str) -> str:
    q = query.lower()
    # an explicit topic phrase ("about X", "related to X", "summarize X") is a topic query
    if re.search(r"\b(summari[sz]e|everything (about|related)|recap|status of|"
                 r"catch me up|tell me about|what.*happening with)\b", q):
        return TOPIC
    for intent, rx in _INTENT_CUES:
        if rx.search(q):
            return intent
    return GENERIC


# ---------------------------------------------------------------------------
# Selected-item container (what we hand to the answer layer)
# ---------------------------------------------------------------------------
@dataclass
class Selected:
    """One unit of selected context, with the reasoning trail attached."""
    idx: int
    score: float
    reason: str
    event: Event

    def to_dict(self) -> dict:
        d = self.event.to_context_dict()
        d["why"] = self.reason
        d["score"] = round(self.score, 3)
        return d


@dataclass
class Retrieved:
    intent: str
    query: str
    selected: list[Selected]
    funnel: dict
    commitments: list[Commitment] = field(default_factory=list)   # for Q1/Q2/Q3
    facts: list[dict] = field(default_factory=list)               # for Q4
    topic_label: str | None = None
    dropped_examples: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Budget packing  (token estimate = chars / 4)
# ---------------------------------------------------------------------------
def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _pack(cands: list[Selected], budget_tokens: int) -> tuple[list[Selected], int]:
    """Greedily keep highest-scoring items under the token budget."""
    kept, used = [], 0
    for s in sorted(cands, key=lambda x: -x.score):
        cost = _est_tokens(s.event.content)
        if used + cost > budget_tokens:
            continue
        kept.append(s)
        used += cost
    kept.sort(key=lambda s: s.event.ts)     # present chronologically
    return kept, used


# ---------------------------------------------------------------------------
# Commitment-centric queries  (Q1 focus, Q2 risk, Q3 procrastination)
# ---------------------------------------------------------------------------
def _commitment_events(mem: Memory, com: Commitment) -> list[int]:
    return sorted(set(com.members))


def _score_focus(com: Commitment, mem: Memory) -> tuple[float, str] | None:
    """Q1: what deserves attention *today*. Reward urgency x importance x being
    actionable now. Exclude done/cancelled and anything not yet started in future."""
    if com.status != "open":
        return None
    now = mem.now
    score = 0.6 * com.importance
    bits = []
    if com.due is not None:
        secs = (com.due - now).total_seconds()
        hrs = secs / 3600
        if com.overdue(now):
            score += 5.0
            bits.append(f"overdue ({humanize_delta(com.due, now)})")
        elif hrs <= 12:
            score += 6.0
            bits.append(f"due today ({humanize_delta(com.due, now)})")
        elif hrs <= 36:
            score += 3.5
            bits.append(f"due soon ({humanize_delta(com.due, now)})")
        elif hrs <= 72:
            score += 1.5
            bits.append(f"due {humanize_delta(com.due, now)}")
    if com.slipping:
        score += 1.0
        bits.append("slipping")
    if com.consequence:
        score += 1.2
        bits.append("hard consequence if missed")
    # de-prioritise things with no deadline and low importance (background)
    if com.due is None and com.importance < 1.0:
        return None
    reason = "; ".join(bits) if bits else "open commitment"
    return score, reason


def _score_risk(com: Commitment, mem: Memory) -> tuple[float, str] | None:
    """Q2: commitments likely to be MISSED -> open AND (overdue OR imminent<=48h)."""
    if com.status != "open" or com.due is None:
        return None
    now = mem.now
    hrs = (com.due - now).total_seconds() / 3600
    if com.overdue(now):
        score = 5.0 + 0.5 * com.importance
        reason = f"overdue {humanize_delta(com.due, now)}"
    elif hrs <= 48:
        score = 3.0 + 0.5 * com.importance + (1.0 if com.slipping else 0.0)
        reason = f"due {humanize_delta(com.due, now)}" + (", slipping" if com.slipping else "")
    else:
        return None
    if com.consequence:
        score += 1.0
        reason += "; consequence attached"
    return score, reason


def _score_procrastination(com: Commitment, mem: Memory) -> tuple[float, str] | None:
    """Q3: things actively being avoided -> raised repeatedly or explicitly slipping,
    still open. Rank by how long it has been dragging."""
    if com.status != "open":
        return None
    explicit = com.slipping
    recurring = com.recurrence_days >= 2
    if not (explicit or recurring):
        return None
    now = mem.now
    age_days = (now - com.first_seen).total_seconds() / 86400 if com.first_seen else 0
    score = 1.5 * com.recurrence_days + 0.5 * age_days + 0.4 * com.importance
    bits = []
    if recurring:
        bits.append(f"raised on {com.recurrence_days} separate days")
    if explicit:
        bits.append("explicit 'still / again' language")
    if com.due and com.overdue(now):
        bits.append(f"now overdue {humanize_delta(com.due, now)}")
        score += 1.0
    return score, "; ".join(bits)


_SCORERS = {FOCUS: _score_focus, RISK: _score_risk, PROCRASTINATION: _score_procrastination}


def _retrieve_commitments(mem: Memory, intent: str, query: str,
                          budget_tokens: int) -> Retrieved:
    scorer = _SCORERS[intent]
    total = len(mem.commitments)
    scored: list[tuple[Commitment, float, str]] = []
    for com in mem.commitments:
        r = scorer(com, mem)
        if r is not None:
            scored.append((com, r[0], r[1]))
    scored.sort(key=lambda t: -t[1])

    # turn the chosen commitments into selected EVENTS (the evidence), budget-packed
    cands: list[Selected] = []
    chosen_coms: list[Commitment] = []
    for com, sc, why in scored:
        chosen_coms.append(com)
        # the single most informative member event = latest member at/under now
        member_evs = [mem.events[i] for i in com.members if mem.events[i].ts <= mem.now]
        if not member_evs:
            member_evs = [mem.events[com.members[0]]]
        lead = max(member_evs, key=lambda e: e.ts)
        label = _commitment_label(com)
        cands.append(Selected(idx=lead.idx, score=sc,
                              reason=f"{label}: {why}", event=lead))

    kept, used = _pack(cands, budget_tokens)
    funnel = {
        "commitments_total": total,
        "commitments_matched_intent": len(scored),
        "events_selected": len(kept),
        "approx_tokens_used": used,
        "token_budget": budget_tokens,
    }
    return Retrieved(intent=intent, query=query, selected=kept, funnel=funnel,
                     commitments=chosen_coms)


def _commitment_label(com: Commitment) -> str:
    """Human label for a commitment from its most distinctive subject words."""
    words = com.subject.split()
    return " ".join(words[:3]).title() if words else f"commit#{com.members[0]}"


# ---------------------------------------------------------------------------
# Topic query (Q4): timeline + resolved current facts
# ---------------------------------------------------------------------------
# Patterns that signal a *fact correction / supersession* inside an event.
_SUPERSEDE = [
    # money: "do not use the old $42k ... is $48.5k"
    (re.compile(r"(do not use|don'?t use|ignore).{0,40}?(\$\s?[\d,.]+k?|\binr\s?[\d,.]+)", re.I),
     re.compile(r"(updated|new|now|is)\s.{0,30}?(\$\s?[\d,.]+k?|\binr\s?[\d,.]+)", re.I),
     "amount"),
    # naming: "call it <X> in the external ..."
    (re.compile(r"\bcall it\b", re.I),
     re.compile(r"call it\s+(.+?)(?:\s+in\b|\.|$)", re.I),
     "external name"),
    # blocker: "waiting on X, not on Y anymore" / "no longer blocked"
    (re.compile(r"\b(waiting on|blocked on|no longer (blocked|on)|not on .* anymore|"
                r"clause \d+ is approved)\b", re.I),
     None, "blocker"),
]


def _resolve_topic_query(mem: Memory, query: str, budget_tokens: int) -> Retrieved:
    # map the query to the best-matching topic by anchor overlap
    qtokens = {w for w in re.findall(r"[a-z][a-z\-]+", query.lower()) if len(w) >= 3}
    qanchors = qtokens & mem.anchors
    best_cid, best_overlap = None, 0
    for cid, tc in mem.topics.items():
        lbl_tokens = set(re.split(r"[/]", tc.label))
        ov = len(qanchors & lbl_tokens)
        # also count member anchor mass
        if ov == 0:
            mem_anchor = set()
            for m in tc.members:
                mem_anchor |= (mem.events[m].subject_tokens & mem.anchors)
            ov = len(qanchors & mem_anchor)
        if ov > best_overlap:
            best_overlap, best_cid = ov, cid

    if best_cid is None:
        # fall back to lexical: any visible event sharing a query anchor
        members = [e.idx for e in mem.visible()
                   if (e.subject_tokens & qanchors)]
        label = "/".join(sorted(qanchors)) or query
    else:
        members = list(mem.topics[best_cid].members)
        label = mem.topics[best_cid].label

    # Stage 0: visible only, drop pure noise
    visible = [mem.events[i] for i in members if mem.events[i].ts <= mem.now]
    visible = [e for e in visible if e.noise < 0.6 or e.is_obligation or e.primary_deadline]
    visible.sort(key=lambda e: e.ts)

    facts = _resolve_facts(visible)

    # Stage 1/3: score by signal + recency, budget pack
    cands = []
    for e in visible:
        sc = 1.0 + e.signal + (0.5 if e.status in ("update", "done", "cancel") else 0.0)
        reason = _topic_reason(e)
        cands.append(Selected(idx=e.idx, score=sc, reason=reason, event=e))
    kept, used = _pack(cands, budget_tokens)

    funnel = {
        "topic_members_total": len(members),
        "visible_after_prefilter": len(visible),
        "events_selected": len(kept),
        "facts_resolved": len(facts),
        "approx_tokens_used": used,
        "token_budget": budget_tokens,
    }
    return Retrieved(intent=TOPIC, query=query, selected=kept, funnel=funnel,
                     facts=facts, topic_label=label)


def _topic_reason(e: Event) -> str:
    if e.status == "update":
        return "update / correction to the thread"
    if e.status == "done":
        return "marks a part complete"
    if e.status == "cancel":
        return "cancellation / schedule change"
    if e.is_obligation:
        return "open action item on the thread"
    if e.primary_deadline:
        return f"carries a deadline ({e.primary_deadline.raw})"
    return "context on the thread"


def _resolve_facts(events: list[Event]) -> list[dict]:
    """
    Walk the thread oldest->newest and surface the CURRENT value of any fact that
    was explicitly superseded, plus what it replaced. This is what makes a Q4
    summary say '$48.5k (was $42k)' and 'Unified Intelligence Engine (external
    name; UIE internal)' instead of repeating stale values.
    """
    facts: list[dict] = []

    # amounts: keep the latest money figure, note any explicitly-retired one
    money_re = re.compile(r"(\$\s?[\d,.]+k?|\binr\s?[\d,.]+)", re.I)
    retired, current, cur_ev = None, None, None
    for e in events:
        if re.search(r"\b(do not use|don'?t use|ignore)\b", e.content, re.I) and money_re.search(e.content):
            ms = money_re.findall(e.content)
            if ms:
                retired = ms[0]
                if len(ms) > 1:
                    current, cur_ev = ms[-1], e
        elif re.search(r"\b(updated|new|now)\b.{0,40}", e.content, re.I) and money_re.search(e.content):
            current, cur_ev = money_re.findall(e.content)[-1], e
    if current:
        facts.append({
            "fact": "amount",
            "current": _clean_money(current),
            "superseded": _clean_money(retired) if retired else None,
            "source_event": cur_ev.idx if cur_ev else None,
            "note": "use the current figure; the older one was explicitly retired"
                    if retired else "current figure",
        })

    # external name
    for e in events:
        m = re.search(r"call it\s+(.+?)(?:\s+in\b|\.|$)", e.content, re.I)
        if m:
            ext = m.group(1).strip().rstrip(".")
            facts.append({
                "fact": "external name",
                "current": ext,
                "superseded": None,
                "source_event": e.idx,
                "note": "use this name in external-facing material",
            })
            break

    # blocker / dependency change
    for e in events:
        if re.search(r"\b(no longer (blocked|on)|not on .* anymore|clause \d+ is approved|"
                     r"waiting on .* not on)\b", e.content, re.I):
            facts.append({
                "fact": "blocker",
                "current": _short(e.content, 120),
                "superseded": None,
                "source_event": e.idx,
                "note": "dependency state changed; prior blocker no longer applies",
            })
            break
    return facts


def _clean_money(s: str | None) -> str | None:
    if not s:
        return None
    return re.sub(r"\s+", "", s.strip())


def _short(s: str, n: int = 120) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "..."


# ---------------------------------------------------------------------------
# Generic query (ad-hoc questions): structural baseline + RAG recall
# ---------------------------------------------------------------------------
def _retrieve_generic(mem: Memory, query: str, budget_tokens: int,
                      rag_index: "EventIndex | None" = None) -> Retrieved:
    """Legacy path — agent mode uses semantic.py instead."""
    base = _retrieve_commitments(mem, FOCUS, query, budget_tokens)
    base.intent = GENERIC
    return base


def _apply_rag(r: Retrieved, mem: Memory, query: str,
               rag_index: "EventIndex | None") -> Retrieved:
    return r


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def retrieve(mem: Memory, query: str, budget_tokens: int = 1500,
             rag_index: "EventIndex | None" = None) -> Retrieved:
    intent = classify(query)
    if intent in _SCORERS:
        r = _retrieve_commitments(mem, intent, query, budget_tokens)
        return _apply_rag(r, mem, query, rag_index)
    if intent == TOPIC:
        r = _resolve_topic_query(mem, query, budget_tokens)
        return _apply_rag(r, mem, query, rag_index)
    return _retrieve_generic(mem, query, budget_tokens, rag_index)
