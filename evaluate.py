#!/usr/bin/env python3
"""
evaluate.py - the engine's own test harness. No pytest dependency.

Three layers, matching the assessment's ask:

  A. OFFLINE CORRECTNESS  - held-out expectations written here by hand (NOT labels
     in the dataset). Precision/recall on expected event ids, noise-leakage rate,
     freshness/future-leakage checks.
  B. REGRESSION (traps)    - one assertion per known trap: superseded deadline,
     $42k -> $48.5k, internal vs external name, blocker change, dedup, future
     non-leakage, unblock-is-not-done, no cross-topic merge.
  C. SUBJECTIVE RUBRIC     - a deterministic scorer for the open-ended queries
     (coverage / prioritisation / time-awareness / no-noise), plus a description
     of how an LLM judge would score the same axes.

Run:  python evaluate.py            (uses outputs deterministically; no network)
Exit code is non-zero if any A/B assertion fails (CI-friendly).
"""

import os
import sys
from datetime import datetime, timezone

from memorae.engine import Engine, DEFAULT_NOW
from memorae.retrieval import retrieve, classify

EVENTS = os.environ.get("MEMORAE_EVENTS",
                        "/mnt/user-data/uploads/memorae_mock_events.json")

# ----------------------------------------------------------------------------
# tiny assertion harness
# ----------------------------------------------------------------------------
class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.log: list[str] = []

    def check(self, name: str, ok: bool, detail: str = ""):
        if ok:
            self.passed += 1
            self.log.append(f"  PASS  {name}")
        else:
            self.failed += 1
            self.log.append(f"  FAIL  {name}  {detail}")

    def section(self, title: str):
        self.log.append("")
        self.log.append(title)

    def report(self) -> int:
        print("\n".join(self.log))
        print("")
        print(f"SUMMARY: {self.passed} passed, {self.failed} failed")
        return 1 if self.failed else 0


def _ids(rec) -> set[int]:
    return {c["idx"] for c in rec["selected_context"]}


# ----------------------------------------------------------------------------
# A. OFFLINE CORRECTNESS  (held-out expectations, authored here)
# ----------------------------------------------------------------------------
# Events that a correct answer MUST surface / MUST NOT surface. These are our
# judgement of ground truth, kept out of the dataset and out of preprocessing.
EXPECT = {
    "focus_today": {
        "must_include_any_of": [[0, 5, 56, 140, 142]],   # the UIE proposal (any member)
        "must_not_include": [2, 7, 20, 36, 48, 52],       # sandwich/news/delivery/music/chai/digest
    },
    "at_risk": {
        "must_include_any_of": [[0, 5, 56, 140, 142],     # UIE due today
                                [21, 67, 97, 195]],        # launch checklist overdue
        "must_not_include": [2, 7, 20, 36, 48, 52],
    },
    "procrastination": {
        "must_include_any_of": [[15, 49, 95, 145, 154]],  # export screenshots (nudged repeatedly)
        "must_not_include": [2, 7, 20, 36, 48, 52],
    },
}

NOISE_IDS = {2, 7, 20, 36, 48, 52, 64, 178}    # representative pure-noise events


def offline_correctness(eng: Engine, R: Results):
    R.section("A. OFFLINE CORRECTNESS (held-out expectations)")
    suite = {classify(q): eng.answer(q, use_llm=False) for q in [
        "What should I focus on today?",
        "What commitments am I at risk of missing?",
        "What have I been procrastinating on?",
    ]}
    for intent, exp in EXPECT.items():
        rec = suite[intent]
        ids = _ids(rec)
        for group in exp.get("must_include_any_of", []):
            R.check(f"[{intent}] includes one of {group}",
                    any(g in ids for g in group),
                    f"selected={sorted(ids)}")
        for bad in exp.get("must_not_include", []):
            R.check(f"[{intent}] excludes noise ev{bad}", bad not in ids)
        # noise-leakage rate
        leaked = ids & NOISE_IDS
        R.check(f"[{intent}] noise-leakage rate == 0",
                len(leaked) == 0, f"leaked={sorted(leaked)}")
        # freshness: nothing future-dated may be selected
        future = [c for c in rec["selected_context"]
                  if c["timestamp"] > eng.mem.now.isoformat().replace('+00:00', 'Z')]
        R.check(f"[{intent}] no future-dated events selected",
                not future, f"future={[c['idx'] for c in future]}")


# ----------------------------------------------------------------------------
# B. REGRESSION  (one per trap)
# ----------------------------------------------------------------------------
def trap_regressions(eng: Engine, R: Results):
    R.section("B. REGRESSION - trap-by-trap")
    mem = eng.mem

    # the UIE commitment (resolved)
    uie = _find_com(mem, "uie")
    R.check("UIE commitment exists", uie is not None)
    if uie:
        # 1. superseded deadline: Apr 13, NOT Apr 10
        R.check("UIE due resolves to Apr 13 (not Apr 10)",
                uie.due is not None and uie.due.month == 4 and uie.due.day == 13,
                f"due={uie.due}")
        # and it is treated as due today / imminent, not overdue-since-Apr-10
        R.check("UIE is imminent (due ~today), not stale-overdue",
                uie.imminent(mem.now, 24), f"due={uie.due}")
        # 7. unblock-is-not-done: UIE not closed by data-room / staging events
        R.check("UIE still OPEN (unblock/related events did not close it)",
                uie.status == "open", f"status={uie.status}")

    # 2/3. fact supersession on the UIE topic
    rec = eng.answer("Summarize everything related to the UIE proposal.", use_llm=False)
    facts = {f["fact"]: f for f in rec["debug"]["resolved_facts"]}
    R.check("procurement amount resolved to $48.5k",
            "amount" in facts and "48.5" in (facts["amount"]["current"] or ""),
            f"facts={facts.get('amount')}")
    R.check("old $42k flagged as superseded",
            "amount" in facts and "42" in (str(facts["amount"].get("superseded")) or ""))
    R.check("external name resolved to 'Unified Intelligence Engine'",
            "external name" in facts and
            "unified intelligence engine" in facts["external name"]["current"].lower())
    ans = rec["answer"].lower()
    R.check("Q4 answer surfaces the current $48.5k figure", "48.5" in ans)
    R.check("Q4 answer does not present $42k as current",
            "42k" not in ans or "was $42k" in ans or "(was $42" in ans)

    # 4. dedup: coffee-machine / AC chatter collapsed
    dup_collapsed = sum(len(v) for v in mem.duplicates.values())
    R.check("near-duplicate chatter collapsed (>=20 events)", dup_collapsed >= 20,
            f"collapsed={dup_collapsed}")

    # 5. future non-leakage at the memory layer
    fut = mem.stats()["events_future_hidden"]
    R.check("future events are held out of 'now'", fut >= 1, f"future={fut}")

    # 8. no cross-topic merge: insurance commitment must not contain the rubric,
    #    and the hiring-rubric commitment must not contain cardiology.
    ins = _find_com(mem, "insurance")
    if ins:
        ins_text = " ".join(mem.events[i].content.lower() for i in ins.members)
        R.check("insurance commitment did not swallow the hiring rubric",
                "rubric" not in ins_text, f"members={ins.members}")
    rub = _find_com(mem, "rubric")
    if rub:
        rub_text = " ".join(mem.events[i].content.lower() for i in rub.members)
        R.check("rubric commitment did not swallow cardiology",
                "cardiology" not in rub_text, f"members={rub.members}")


def _find_com(mem, token: str):
    cands = [c for c in mem.commitments
             if any(token in mem.events[i].subject_tokens for i in c.members)]
    # the one with the most members mentioning the token
    cands.sort(key=lambda c: -sum(token in mem.events[i].subject_tokens for i in c.members))
    return cands[0] if cands else None


# ----------------------------------------------------------------------------
# C. SUBJECTIVE RUBRIC  (deterministic scorer for open-ended queries)
# ----------------------------------------------------------------------------
def subjective_rubric(eng: Engine, R: Results):
    R.section("C. SUBJECTIVE RUBRIC (0-1 per axis; deterministic proxy)")
    print_axes = []

    def score_focus():
        rec = eng.answer("What should I focus on today?", use_llm=False)
        ids = _ids(rec)
        ans = rec["answer"]
        coverage = _frac_present(ids, [[0, 5, 56, 140, 142], [58, 173], [94, 181],
                                       [15, 49, 95, 145, 154]])
        # prioritisation: the UIE (due today) should be named first in the answer
        prioritisation = 1.0 if _names_first(ans, ["uie"]) else 0.0
        time_aware = 1.0 if ("in 6h" in ans or "today" in ans.lower()) else 0.0
        no_noise = 1.0 if not (ids & NOISE_IDS) else 0.0
        return ("focus_today", coverage, prioritisation, time_aware, no_noise)

    def score_topic():
        rec = eng.answer("Summarize everything related to the UIE proposal.", use_llm=False)
        ans = rec["answer"].lower()
        coverage = sum(k in ans for k in ["appendix", "procurement", "diagram",
                                          "failure", "review"]) / 5.0
        prioritisation = 1.0 if ("current facts" in ans and ans.find("current facts") < 200) else 0.5
        time_aware = 1.0 if ("apr 13" in ans or "in 6h" in ans) else 0.0
        no_noise = 1.0 if ("sandwich" not in ans and "playlist" not in ans) else 0.0
        return ("topic_summary", coverage, prioritisation, time_aware, no_noise)

    for name, cov, pri, ta, nn in (score_focus(), score_topic()):
        total = (cov + pri + ta + nn) / 4.0
        R.check(f"[{name}] rubric >= 0.75 (cov={cov:.2f} pri={pri:.2f} time={ta:.2f} noise={nn:.2f})",
                total >= 0.75, f"total={total:.2f}")

    print_axes.append(
        "\n  Note: this is a deterministic proxy. In production the same four axes\n"
        "  (coverage, prioritisation, time-awareness, noise-freedom) would be scored\n"
        "  by an LLM judge against a written rubric, with the deterministic checks\n"
        "  kept as guardrails so the judge can never pass an answer that leaks a\n"
        "  future event, a superseded value, or a known noise id.")
    R.log.append("".join(print_axes))


def _frac_present(ids: set[int], groups: list[list[int]]) -> float:
    if not groups:
        return 1.0
    hit = sum(1 for g in groups if any(x in ids for x in g))
    return hit / len(groups)


def _names_first(answer: str, tokens: list[str]) -> bool:
    a = answer.lower()
    positions = [a.find(t) for t in tokens if a.find(t) >= 0]
    if not positions:
        return False
    first = min(positions)
    # is it within the first listed item?
    return first < a.find("2.") if "2." in a else True


# ----------------------------------------------------------------------------
def main() -> int:
    if not os.path.exists(EVENTS):
        print(f"events file not found: {EVENTS}", file=sys.stderr)
        return 2
    eng = Engine.from_events_file(EVENTS, now=DEFAULT_NOW)
    R = Results()
    print("Memorae evaluation  |  now =", eng.mem.now.isoformat())
    print("memory:", eng.mem.stats())
    offline_correctness(eng, R)
    trap_regressions(eng, R)
    subjective_rubric(eng, R)
    return R.report()


if __name__ == "__main__":
    raise SystemExit(main())
