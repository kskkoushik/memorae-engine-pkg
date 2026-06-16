#!/usr/bin/env python3
"""
run.py - command-line entry point for the Memorae memory engine.

Usage:
  python run.py                       # run the 4 assessment queries, write outputs/answers.json
  python run.py "your question"       # answer a single ad-hoc query, print JSON
  python run.py --plain               # deterministic only (skip optional LLM polish)
  python run.py --events path.json    # use a different events file
  python run.py --now 2026-04-13T03:00:00Z   # override scenario 'now'

Environment:
  MEMORAE_LLM=1 and ANTHROPIC_API_KEY=...  -> enable optional LLM polish.
  Default is fully deterministic and dependency-free.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from memorae.engine import Engine, DEFAULT_NOW

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EVENTS = os.environ.get(
    "MEMORAE_EVENTS",
    os.path.join(HERE, "memorae_mock_events.json"),
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "answers.json")


def _parse_now(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser(description="Memorae personal-memory query engine")
    ap.add_argument("query", nargs="?", help="ad-hoc query; omit to run the 4-query suite")
    ap.add_argument("--events", default=DEFAULT_EVENTS, help="path to events JSON")
    ap.add_argument("--now", default=None, help="ISO timestamp to use as 'now'")
    ap.add_argument("--plain", action="store_true", help="deterministic only (no LLM)")
    ap.add_argument("--budget", type=int, default=1500, help="context token budget")
    args = ap.parse_args()

    if not os.path.exists(args.events):
        print(f"events file not found: {args.events}", file=sys.stderr)
        return 2

    now = _parse_now(args.now) if args.now else DEFAULT_NOW
    use_llm = not args.plain
    eng = Engine.from_events_file(args.events, now=now)

    if args.query:
        rec = eng.answer(args.query, budget_tokens=args.budget, use_llm=use_llm)
        print(json.dumps(rec, indent=2, default=str))
        return 0

    # full suite
    records = eng.run_suite(use_llm=use_llm)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"now": now.isoformat().replace("+00:00", "Z"),
                   "owner_detected": eng.mem.owner,
                   "memory_stats": eng.mem.stats(),
                   "answers": records}, f, indent=2, default=str)

    # human-readable digest to stdout
    for rec in records:
        print("=" * 78)
        print("Q:", rec["query"], "   [intent:", rec["intent"] + "]")
        print("-" * 78)
        print(rec["answer"])
        print()
    print("=" * 78)
    print(f"Full structured output (context + reasoning) written to: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
