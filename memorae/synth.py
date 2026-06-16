"""
synth.py
--------
Turns a Retrieved bundle into a written answer.

Two synthesizers over the SAME selected context:
  * synth_deterministic  - default. Grounded, time-aware, states uncertainty,
    never invents anything not in the selected events. Runs with zero deps.
  * synth_llm            - optional polish. Only the already-selected context is
    sent to the model; it may not add facts. Enabled with MEMORAE_LLM=1 and an
    ANTHROPIC_API_KEY. Falls back to deterministic on any error.

Keeping generation separate from retrieval means the answer can never quietly
pull in an event the retriever did not choose, which is what keeps it grounded.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime

from .memory import Memory, Commitment
from .retrieval import (Retrieved, FOCUS, RISK, PROCRASTINATION, TOPIC, GENERIC,
                        _commitment_label)
from .timeutil import humanize_delta


# ---------------------------------------------------------------------------
# Deterministic synthesis
# ---------------------------------------------------------------------------
def synth_deterministic(r: Retrieved, mem: Memory) -> str:
    if r.intent == FOCUS:
        return _focus_answer(r, mem)
    if r.intent == RISK:
        return _risk_answer(r, mem)
    if r.intent == PROCRASTINATION:
        return _procrastination_answer(r, mem)
    if r.intent == TOPIC:
        return _topic_answer(r, mem)
    if r.intent == GENERIC:
        return _generic_answer(r, mem)
    return _focus_answer(r, mem)


def _line_for(com: Commitment, mem: Memory) -> str:
    now = mem.now
    label = _commitment_label(com)
    when = ""
    if com.due is not None:
        when = f" — {humanize_delta(com.due, now)} ({com.due.strftime('%a %b %d %H:%M UTC')})"
        if com.due_confidence < 0.7:
            when += " [approx]"
    tag = ""
    if com.overdue(now):
        tag = " [OVERDUE]"
    elif com.imminent(now, 12):
        tag = " [TODAY]"
    elif com.imminent(now, 48):
        tag = " [SOON]"
    return f"{label}{when}{tag}"


def _focus_answer(r: Retrieved, mem: Memory) -> str:
    coms = [c for c in r.commitments]
    if not coms:
        return "Nothing open with a near-term deadline. You are clear for today."
    coms_sorted = sorted(coms, key=lambda c: _focus_sort_key(c, mem))
    head = coms_sorted[:6]
    lines = [f"{i+1}. {_line_for(c, mem)}" for i, c in enumerate(head)]
    top = _commitment_label(coms_sorted[0])
    out = [f"Focus today, in priority order:", ""]
    out += lines
    # one-sentence steer
    first = coms_sorted[0]
    if first.due and first.imminent(mem.now, 12):
        out += ["", f"Start with {top}: it is due in the next few hours "
                f"({humanize_delta(first.due, mem.now)}). Everything else can wait until it is out the door."]
    return "\n".join(out)


def _focus_sort_key(c: Commitment, mem: Memory):
    now = mem.now
    if c.due is None:
        return (3, 0)
    if c.overdue(now):
        return (1, (c.due - now).total_seconds())     # overdue, soonest-overdue first
    return (0, (c.due - now).total_seconds())          # upcoming, soonest first


def _risk_answer(r: Retrieved, mem: Memory) -> str:
    now = mem.now
    coms = sorted(r.commitments, key=lambda c: _focus_sort_key(c, mem))
    if not coms:
        return "No open commitments are overdue or due within 48 hours."
    overdue = [c for c in coms if c.overdue(now)]
    soon = [c for c in coms if not c.overdue(now)]
    out = ["Commitments at risk of slipping:"]
    if overdue:
        out += ["", "Already overdue:"]
        out += [f"  - {_line_for(c, mem)}" + _consequence_suffix(c) for c in overdue]
    if soon:
        out += ["", "Due within 48 hours:"]
        out += [f"  - {_line_for(c, mem)}" + _consequence_suffix(c) for c in soon]
    return "\n".join(out)


def _consequence_suffix(c: Commitment) -> str:
    return "  (hard consequence if missed)" if c.consequence else ""


def _procrastination_answer(r: Retrieved, mem: Memory) -> str:
    now = mem.now
    coms = sorted(r.commitments, key=lambda c: -(c.recurrence_days * 2 +
                  ((now - c.first_seen).total_seconds() / 86400 if c.first_seen else 0)))
    if not coms:
        return "Nothing shows a repeated-avoidance pattern in the stream."
    out = ["Things you keep pushing (raised repeatedly or flagged 'still/again'):", ""]
    for c in coms[:6]:
        age = ""
        if c.first_seen:
            age = f", first raised {humanize_delta(c.first_seen, now).replace('overdue by ','').replace('in ','')} ago"
        rec = f"raised on {c.recurrence_days} days" if c.recurrence_days >= 2 else "flagged as slipping"
        overdue = f"; now {humanize_delta(c.due, now)}" if (c.due and c.overdue(now)) else ""
        out.append(f"  - {_commitment_label(c)} ({rec}{age}{overdue})")
    return "\n".join(out)


def _topic_answer(r: Retrieved, mem: Memory) -> str:
    label = (r.topic_label or r.query).replace("/", " / ")
    out = [f"Summary — {label}", ""]

    # current resolved facts first (this is the supersession payoff)
    if r.facts:
        out.append("Current facts (after corrections):")
        for f in r.facts:
            line = f"  - {f['fact']}: {f['current']}"
            if f.get("superseded"):
                line += f"  (was {f['superseded']}; do not use the old value)"
            out.append(line)
        out.append("")

    # open actions on the thread
    open_items = [s for s in r.selected if s.event.is_obligation and
                  _com_status_for_event(mem, s.event.idx) != "done"]
    if open_items:
        out.append("Open items on this thread:")
        for s in open_items[:8]:
            e = s.event
            com = _com_for_event(mem, e.idx)
            dl = ""
            # prefer the commitment's RESOLVED due (supersession-aware) over the
            # stale date in the original obligation text
            if com and com.due is not None:
                dl = f" — due {humanize_delta(com.due, mem.now)}"
            elif e.primary_deadline:
                dl = f" — due {humanize_delta(e.primary_deadline.when_utc, mem.now)}"
            out.append(f"  - {_short(e.content, 100)}{dl}")
        out.append("")

    # short chronological spine of the most important updates
    spine = [s for s in r.selected if s.event.status in ("update", "done", "cancel")][:6]
    if spine:
        out.append("Key updates, in order:")
        for s in spine:
            out.append(f"  - {s.event.ts.strftime('%b %d')}: {_short(s.event.content, 100)}")
        out.append("")

    out.append(f"Drawn from {r.funnel.get('events_selected', 0)} events on the thread "
               f"(of {r.funnel.get('topic_members_total', 0)} matched; "
               f"future-dated and noise events excluded).")
    return "\n".join(out).rstrip()


def _com_status_for_event(mem: Memory, idx: int) -> str:
    for c in mem.commitments:
        if idx in c.members:
            return c.status
    return "open"


def _com_for_event(mem: Memory, idx: int):
    for c in mem.commitments:
        if idx in c.members:
            return c
    return None


def _short(s: str, n: int = 100) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "..."


def _generic_answer(r: Retrieved, mem: Memory) -> str:
    """Deterministic fallback for ad-hoc / generic queries."""
    if not r.selected:
        return "I could not find relevant events in the stream for that question."
    out = [f"Answer — {r.query}", ""]
    if r.facts:
        out.append("Resolved facts:")
        for f in r.facts:
            line = f"  - {f['fact']}: {f['current']}"
            if f.get("superseded"):
                line += f" (was {f['superseded']})"
            out.append(line)
        out.append("")
    out.append("Relevant events:")
    for s in r.selected[:10]:
        e = s.event
        out.append(f"  - [{e.source}, {e.ts.strftime('%b %d')}]: {_short(e.content, 120)}")
    out.append("")
    out.append(f"Based on {len(r.selected)} selected event(s) from the stream.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Optional LLM polish (same selected context only)
# ---------------------------------------------------------------------------
def _llm_config() -> dict | None:
    """
    Resolve the LLM provider from the environment. Returns None if polish is off
    or no key is available (caller then uses the deterministic answer).

    Providers, in priority order:
      * OpenRouter  - set OPENROUTER_API_KEY. OpenAI-compatible chat endpoint.
                      Model defaults to moonshotai/kimi-k2-thinking; override with
                      MEMORAE_LLM_MODEL.
      * Anthropic   - set ANTHROPIC_API_KEY. Messages endpoint.
                      Model defaults to claude-sonnet-4-6.
    """
    if os.environ.get("MEMORAE_LLM") != "1":
        return None
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if or_key:
        return {
            "provider": "openrouter",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "key": or_key,
            "model": os.environ.get("MEMORAE_LLM_MODEL", "moonshotai/kimi-k2-thinking"),
        }
    an_key = os.environ.get("ANTHROPIC_API_KEY")
    if an_key:
        return {
            "provider": "anthropic",
            "url": "https://api.anthropic.com/v1/messages",
            "key": an_key,
            "model": os.environ.get("MEMORAE_LLM_MODEL", "claude-sonnet-4-6"),
        }
    return None


def _call_openrouter(cfg: dict, system: str, user: str, timeout: int) -> str:
    body = json.dumps({
        "model": cfg["model"],
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode()
    req = urllib.request.Request(
        cfg["url"], data=body,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {cfg['key']}",
            # OpenRouter likes these but does not require them:
            "http-referer": "https://memorae.local",
            "x-title": "Memorae Memory Engine",
        })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    msg = (data.get("choices") or [{}])[0].get("message", {})
    # thinking models: final answer is in content; reasoning is internal only
    return (msg.get("content") or "").strip()


def _call_anthropic(cfg: dict, system: str, user: str, timeout: int) -> str:
    body = json.dumps({
        "model": cfg["model"],
        "max_tokens": 800,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(
        cfg["url"], data=body,
        headers={"content-type": "application/json",
                 "x-api-key": cfg["key"],
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    return "".join(parts).strip()


def synth_llm(r: Retrieved, mem: Memory) -> str:
    """Polish the deterministic answer using only the selected context. Falls back
    to deterministic output if polish is off or on any error."""
    base = synth_deterministic(r, mem)
    cfg = _llm_config()
    if cfg is None:
        return base

    if r.intent == GENERIC:
        return synth_llm_generate(r, mem, base, cfg)

    ctx = [s.to_dict() for s in r.selected]
    sys = ("You rewrite a personal-assistant answer to be crisp and natural. "
           "You may ONLY use the provided context events and resolved facts. "
           "Do not invent anything. Keep it tight, time-aware, and prioritised. "
           "Preserve all dates and the resolved (current) values exactly. "
           "Return only the rewritten answer, no preamble.")
    user = json.dumps({"query": r.query, "intent": r.intent,
                       "draft": base, "context": ctx, "facts": r.facts}, default=str)
    try:
        if cfg["provider"] == "openrouter":
            out = _call_openrouter(cfg, sys, user, timeout=90)
        else:
            out = _call_anthropic(cfg, sys, user, timeout=45)
        return out or base
    except Exception as e:
        # never fail a query because polish failed; surface the reason on stderr
        import sys as _sys
        print(f"[synth] LLM polish unavailable ({type(e).__name__}: "
              f"{str(e)[:160]}); using deterministic answer.", file=_sys.stderr)
        return base


def synth_llm_generate(r: Retrieved, mem: Memory, base: str,
                       cfg: dict | None = None) -> str:
    """Full LLM answer generation for generic / ad-hoc queries."""
    if cfg is None:
        cfg = _llm_config()
    if cfg is None:
        return base

    ctx = [s.to_dict() for s in r.selected]
    now = mem.now.isoformat().replace("+00:00", "Z")
    sys = (
        "You are a personal memory assistant. Answer the user's question using "
        "ONLY the provided context events and resolved facts. Do not invent "
        "anything. Be clear, natural, and time-aware relative to the scenario "
        "'now'. Preserve all dates, amounts, and resolved (current) values "
        "exactly. If the context is thin, say what you know and what is uncertain. "
        "Return only the answer, no preamble."
    )
    user = json.dumps({
        "query": r.query,
        "intent": r.intent,
        "now": now,
        "draft": base,
        "context": ctx,
        "facts": r.facts,
    }, default=str)
    try:
        if cfg["provider"] == "openrouter":
            out = _call_openrouter(cfg, sys, user, timeout=90)
        else:
            out = _call_anthropic(cfg, sys, user, timeout=45)
        return out or base
    except Exception as e:
        import sys as _sys
        print(f"[synth] LLM generation unavailable ({type(e).__name__}: "
              f"{str(e)[:160]}); using deterministic answer.", file=_sys.stderr)
        return base


def synthesize(r: Retrieved, mem: Memory) -> str:
    return synth_llm(r, mem)
