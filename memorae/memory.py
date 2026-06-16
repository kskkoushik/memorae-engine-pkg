"""
memory.py
---------
The memory substrate. Takes enriched events and organises them into the
structures the retriever reasons over:

  EpisodicStore     every event, indexed; the source of truth.
  Near-dup groups   collapse repeated low-signal chatter (the coffee-machine spam).
  TopicCluster      loosely-linked events about the same thing (for Q4 + context).
  Commitment        one obligation tracked across many mentions, with its
                    CURRENT status and due date resolved from the latest
                    authoritative signal (this is where supersession lives).

Design choices that matter:
  * Linking is by inverse-document-frequency tokens, so events cluster on their
    *distinctive* words ("uie", "southridge", "rubric"), not common ones.
  * State is resolved by RECENCY + explicit update markers. A later "now due
    Apr 13" beats an earlier "by Friday Apr 10"; a later "approved now" closes a
    commitment opened earlier.
  * Everything is computed at runtime. No labels are written back to the data.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from .enrich import Event
from .timeutil import humanize_delta


# ----------------------------------------------------------------------------
# IDF over the corpus -> distinctive-token signatures
# ----------------------------------------------------------------------------
def compute_idf(events: list[Event]) -> dict[str, float]:
    df: Counter = Counter()
    for e in events:
        for tok in e.topic_tokens:
            df[tok] += 1
    n = len(events)
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def compute_df(events: list[Event]) -> Counter:
    """Document frequency over SUBJECT tokens (temporal/process words stripped)."""
    df: Counter = Counter()
    for e in events:
        for tok in e.subject_tokens:
            df[tok] += 1
    return df


def anchor_tokens(df: Counter, lo: int = 2, hi: int = 45) -> set[str]:
    """
    Tokens eligible to LINK events into the same thing.
    A hapax (df==1) cannot connect two events, so it is useless for linking.
    A token in too many events (df>hi) is a theme word ("proposal", "team") that
    would glue unrelated things together. The middle band is where real topic
    anchors live: "uie", "southridge", "rubric", "cardiology".
    """
    return {t for t, c in df.items() if lo <= c <= hi}


def corpus_stoplist(events: list[Event], owner: str | None = None) -> set[str]:
    """
    Words that are frequent in THIS stream but carry no topic: the owner's own
    name, source names (slack/gmail/...), channel names (#random/#eng/...), and
    transport noise. These span every topic, so if left in the anchor set they
    fuse unrelated threads into one giant blob. Computed from the data, not hand-
    labelled per dataset.
    """
    stop: set[str] = set()
    if owner:
        stop.add(owner.lower())
    for e in events:
        if e.source:
            stop.add(e.source.lower())
        if e.channel:
            stop.add(e.channel.lower())
    stop |= {
        "ist", "utc", "gmt", "pm", "am", "http", "https", "www", "com", "org",
        "amp", "re", "fwd", "hey", "thanks", "thx", "pls", "plz", "team", "ok",
    }
    return stop


def person_names(events: list[Event]) -> set[str]:
    """
    People who author messages. A person works across many threads, so their name
    is a weak topic anchor (it would bridge UIE, hiring, school-run, etc.). We
    learn the set from message authorship and exclude it from topic linking.
    Detected from the data (the 'Name:' authoring pattern), not hand-listed.
    """
    names: set[str] = set()
    for e in events:
        if e.actor:
            w = e.actor.strip().split()[0].lower()
            if w.isalpha() and len(w) >= 2:
                names.add(w)
    return names


def email_domains(events: list[Event]) -> set[str]:
    """Second-level labels of any email domain (nina@northstar.example ->
    'northstar'). The company/domain name appears in every work email and would
    otherwise glue all work threads together."""
    doms: set[str] = set()
    for e in events:
        for m in re.findall(r"@([a-z0-9.\-]+)", e.content.lower()):
            parts = [p for p in m.split(".") if p not in ("com", "org", "net",
                                                          "example", "io", "co")]
            if parts:
                doms.add(parts[-1])
    return doms


def signature(e: Event, idf: dict[str, float], anchors: set[str] | None = None,
              k: int = 6) -> set[str]:
    """
    Subject fingerprint of an event: its most distinctive SUBJECT tokens.
    When an anchor set is supplied, restrict to anchor tokens so the fingerprint
    is built only from words that can actually link to other events.
    """
    toks = e.subject_tokens
    if anchors is not None:
        anch = toks & anchors
        if anch:
            toks = anch
    scored = sorted(toks, key=lambda t: idf.get(t, 0.0), reverse=True)
    return set(scored[:k])


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ----------------------------------------------------------------------------
# Near-duplicate collapse (the coffee-machine / air-conditioning spam)
# ----------------------------------------------------------------------------
def _norm_body(e: Event) -> str:
    body = e.content
    # strip "Name:" / "#chan Name:" prefixes so identical bodies match
    body = re.sub(r"^(#\w+\s+)?[A-Z][a-zA-Z]+(:| <| to )\s*", "", body)
    body = re.sub(r"<[^>]+>", "", body)
    return re.sub(r"[^a-z0-9 ]", " ", body.lower()).strip()


def collapse_duplicates(events: list[Event]) -> dict[int, list[int]]:
    """
    Group near-identical low-signal events. Returns {representative_idx: [dups]}.
    Real obligations are never collapsed. Detects the repeated facilities chatter
    that would otherwise flood any context window.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for e in events:
        if e.is_obligation:
            continue
        groups[_norm_body(e)].append(e.idx)
    rep_to_dups: dict[int, list[int]] = {}
    for _, idxs in groups.items():
        if len(idxs) > 1:
            rep, *dups = sorted(idxs)
            rep_to_dups[rep] = dups
            for d in dups:
                events[d].dup_of = rep
    return rep_to_dups


# ----------------------------------------------------------------------------
# Topic clusters (loose) — connected components over shared distinctive tokens
# ----------------------------------------------------------------------------
@dataclass
class TopicCluster:
    cid: str
    label: str
    members: list[int]


def _token_mean_noise(events: list[Event], anchors: set[str]) -> dict[str, float]:
    tot: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)
    for e in events:
        for t in e.subject_tokens & anchors:
            tot[t] += e.noise
            cnt[t] += 1
    return {t: tot[t] / cnt[t] for t in tot}


def build_topic_clusters(events: list[Event], idf: dict[str, float],
                         anchors: set[str], min_df: int = 3, cap: int = 30,
                         fold: float = 0.7) -> dict[str, TopicCluster]:
    """
    Build topics by SOFT anchor membership instead of transitive union-find.

    Union-find fails here: a couple of cross-cutting events ("Apr 12 UIE block
    removed, conflicts with apartment maintenance") chain every thread into one
    blob. Instead, each thread-sized anchor seeds a topic = the events that
    mention it, and an event may belong to several topics. Smaller anchor groups
    that are mostly contained in a larger one are folded in (so "sow"/"redlines"
    join "southridge", "appendix" joins "uie"). No transitive blow-up: the UIE
    conflict event lands in both the UIE and the apartment topic, and that is
    correct, not a merge of the two threads.
    """
    df = compute_df(events)
    mean_noise = _token_mean_noise(events, anchors)
    seeds = sorted(
        [t for t in anchors
         if min_df <= df[t] <= cap and not (df[t] >= 3 and mean_noise.get(t, 0.0) >= 0.45)],
        key=lambda t: (-df[t], t),               # biggest threads first
    )

    groups: list[dict] = []
    for t in seeds:
        ev_set = {e.idx for e in events if t in e.subject_tokens}
        if not ev_set:
            continue
        folded = False
        for g in groups:
            if len(ev_set & g["members"]) >= fold * len(ev_set):
                g["members"] |= ev_set
                g["tokens"].append(t)
                folded = True
                break
        if not folded:
            groups.append({"tokens": [t], "members": set(ev_set)})

    # assign each event a PRIMARY topic (for display / cluster_id): the group with
    # which it shares the highest-IDF anchor.
    clusters: dict[str, TopicCluster] = {}
    for g in groups:
        toks = g["tokens"]
        # label by the DEFINING anchors (most central = highest df first), so the
        # UIE topic reads "uie/..." not a rare folded token.
        label = "/".join(sorted(toks, key=lambda t: -df.get(t, 0))[:3])
        cid = f"topic:{label}"
        g["cid"] = cid
        clusters[cid] = TopicCluster(cid=cid, label=label, members=sorted(g["members"]))

    for e in events:
        best_cid, best_w = None, -1.0
        for g in groups:
            if e.idx in g["members"]:
                w = max((idf.get(t, 0) for t in g["tokens"] if t in e.subject_tokens),
                        default=0.0)
                if w > best_w:
                    best_w, best_cid = w, g["cid"]
        e.cluster_id = best_cid
    return clusters


def _percentile(vals: list[float], q: float) -> float:
    s = sorted(vals)
    if not s:
        return 0.0
    return s[min(len(s) - 1, int(q * len(s)))]


# ----------------------------------------------------------------------------
# Commitment ledger (tight) — one obligation across many mentions
# ----------------------------------------------------------------------------
@dataclass
class Commitment:
    key: str
    subject: str
    members: list[int]                       # obligation events, time-sorted
    related: list[int] = field(default_factory=list)  # same-topic context events

    status: str = "open"                     # open | done | cancelled
    due: datetime | None = None
    due_raw: str | None = None
    due_confidence: float = 0.0

    first_seen: datetime | None = None
    last_seen: datetime | None = None
    recurrence_days: int = 1                 # distinct days it was raised
    slipping: bool = False
    importance: float = 0.0
    consequence: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)   # human-readable resolution log

    def overdue(self, now: datetime) -> bool:
        return self.status == "open" and self.due is not None and self.due < now

    def imminent(self, now: datetime, hours: int = 36) -> bool:
        if self.status != "open" or self.due is None:
            return False
        secs = (self.due - now).total_seconds()
        return 0 <= secs <= hours * 3600


def build_commitments(
    events: list[Event],
    idf: dict[str, float],
    anchors: set[str],
    now: datetime,
) -> list[Commitment]:
    obligations = [e for e in events if e.is_obligation]
    obligations.sort(key=lambda e: e.ts)
    df = compute_df(events)

    # greedy merge by distinctive-signature overlap. Generic action/process words
    # have been stripped from the anchor set, so a single shared NOUN anchor (e.g.
    # "uie", "rubric", "southridge") is reliable evidence of the same thread and is
    # enough to merge. Rare anchors get a bonus; the threshold keeps purely
    # coincidental single-word overlaps apart.
    RARE = 8
    clusters: list[list[int]] = []
    sigs: list[set[str]] = []
    anch_sets: list[set[str]] = []
    for e in obligations:
        s = signature(e, idf, anchors=anchors, k=6)
        e_anch = e.subject_tokens & anchors
        best, best_score = -1, 0.0
        for i, cs in enumerate(sigs):
            shared_anchor = e_anch & anch_sets[i]
            if not shared_anchor:
                continue
            shared_rare = any(df.get(t, 99) <= RARE for t in shared_anchor)
            sc = jaccard(s, cs)
            score = sc + 0.25 * len(shared_anchor) + (0.25 if shared_rare else 0.0)
            if score > best_score:
                best, best_score = i, score
        if best >= 0 and best_score >= 0.25:
            clusters[best].append(e.idx)
            sigs[best] |= s
            anch_sets[best] |= e_anch
        else:
            clusters.append([e.idx])
            sigs.append(set(s))
            anch_sets.append(set(e_anch))

    commitments: list[Commitment] = []
    for members in clusters:
        members.sort(key=lambda i: events[i].ts)
        com = _resolve_commitment(members, events, idf, anchors, df, now)
        commitments.append(com)
    return commitments


def _resolve_commitment(members, events, idf, anchors, df, now) -> Commitment:
    member_evs = [events[i] for i in members]
    # subject label from the most distinctive shared tokens
    tok = Counter()
    for e in member_evs:
        for t in e.subject_tokens:
            tok[t] += idf.get(t, 0)
    subject = " ".join(w for w, _ in tok.most_common(4))

    # CORE anchors: tokens shared by >=2 members (the commitment's defining nouns),
    # or, for a single-mention commitment, its rarest anchor. Related context is
    # pulled ONLY via core anchors so a loosely-overlapping event cannot inject a
    # spurious "done"/"due" signal into an unrelated commitment.
    anchor_count: Counter = Counter()
    for e in member_evs:
        for t in (e.subject_tokens & anchors):
            anchor_count[t] += 1
    core = {t for t, c in anchor_count.items() if c >= 2}
    if not core and anchor_count:
        # rarest (most specific) anchor among the members
        rarest = min(anchor_count, key=lambda t: (df.get(t, 0), t))
        core = {rarest}

    related: list[int] = []
    if core:
        for e in events:
            if e.idx in members:
                continue
            if e.subject_tokens & core:
                related.append(e.idx)
    related.sort()

    com = Commitment(
        key=f"commit:{members[0]}",
        subject=subject,
        members=members,
        related=related,
    )
    com.first_seen = member_evs[0].ts
    visible = [e for e in member_evs if e.ts <= now]
    com.last_seen = (visible[-1] if visible else member_evs[0]).ts
    com.recurrence_days = len({e.ts.date() for e in visible}) or 1
    com.importance = max((e.importance for e in member_evs), default=0.0)
    for e in member_evs:
        com.consequence += [r for r in e.importance_reasons if "consequence" in r]
    com.consequence = sorted(set(com.consequence))

    # ----- resolve CURRENT STATUS (recency + explicit markers) -------------
    pool = sorted(
        [events[i] for i in (members + related) if events[i].ts <= now],
        key=lambda e: e.ts,
    )
    status = "open"
    for e in pool:                                   # later events override earlier
        # A calendar block being moved/removed is a SCHEDULE change, not the
        # deliverable being completed or cancelled. Never let it close a task.
        if e.source == "calendar":
            continue
        # Completion/cancellation only counts if the event is actually about THIS
        # commitment (a member, or it carries a core anchor). This stops a "done"
        # in a loosely-related thread from closing the wrong commitment.
        on_topic = (e.idx in members) or bool(e.subject_tokens & core)
        if not on_topic:
            continue
        if e.status == "done":
            status = "done"
            com.notes.append(f"closed by ev{e.idx} ({e.ts.date()}): \"{_short(e.content)}\"")
        elif e.status == "cancel":
            status = "cancelled"
            com.notes.append(f"cancelled by ev{e.idx} ({e.ts.date()}): \"{_short(e.content)}\"")
        elif e.status in ("open", "update") and e.is_obligation:
            if status in ("done", "cancelled"):
                # re-opened by a newer ask
                status = "open"
    com.status = status

    # ----- resolve CANONICAL DUE DATE (updates beat originals) -------------
    dl_candidates = [
        e for e in pool
        if e.primary_deadline and e.primary_deadline.confidence >= 0.7
    ]
    chosen = None
    updates = [e for e in dl_candidates if e.status == "update"]
    if updates:
        chosen = max(updates, key=lambda e: e.ts)           # "now due ..." wins
    if chosen is None:
        explicit = [e for e in dl_candidates if e.primary_deadline.granularity == "time"
                    or e.primary_deadline.confidence >= 0.85]
        if explicit:
            chosen = max(explicit, key=lambda e: e.ts)
    if chosen is None and dl_candidates:
        chosen = max(dl_candidates, key=lambda e: e.ts)
    if chosen is None:
        # fall back to any deadline at all (low conf), else none
        any_dl = [e for e in pool if e.primary_deadline]
        chosen = max(any_dl, key=lambda e: e.ts) if any_dl else None

    if chosen is not None:
        com.due = chosen.primary_deadline.when_utc
        com.due_raw = chosen.primary_deadline.raw
        com.due_confidence = chosen.primary_deadline.confidence
        if chosen.status == "update":
            com.notes.append(
                f"due updated by ev{chosen.idx} ({chosen.ts.date()}) -> "
                f"{com.due.isoformat()[:16]} ({com.due_raw})"
            )

    # ----- slipping? -------------------------------------------------------
    explicit_slip = any(events[i].is_slipping for i in members)
    recurring = com.recurrence_days >= 2
    aged_past_first_due = False
    first_dl = next((e.primary_deadline for e in member_evs if e.primary_deadline), None)
    if first_dl and first_dl.when_utc < now and com.status == "open":
        aged_past_first_due = True
    com.slipping = com.status == "open" and (explicit_slip or recurring or aged_past_first_due)
    return com


def _short(s: str, n: int = 70) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ----------------------------------------------------------------------------
# The assembled memory
# ----------------------------------------------------------------------------
@dataclass
class Memory:
    events: list[Event]
    idf: dict[str, float]
    now: datetime
    duplicates: dict[int, list[int]]
    topics: dict[str, TopicCluster]
    commitments: list[Commitment]
    owner: str | None = None
    anchors: set[str] = field(default_factory=set)

    def visible(self) -> list[Event]:
        """Events knowable at `now` (future events are not yet observed)."""
        return [e for e in self.events if e.ts <= self.now]

    def stats(self) -> dict:
        v = self.visible()
        return {
            "events_total": len(self.events),
            "events_visible_now": len(v),
            "events_future_hidden": len(self.events) - len(v),
            "duplicate_groups": len(self.duplicates),
            "duplicates_collapsed": sum(len(d) for d in self.duplicates.values()),
            "topic_clusters": len(self.topics),
            "commitments_tracked": len(self.commitments),
            "commitments_open": sum(1 for c in self.commitments if c.status == "open"),
        }


def build_memory(events: list[Event], now: datetime, owner: str | None = None) -> Memory:
    idf = compute_idf(events)
    df = compute_df(events)
    blocked = corpus_stoplist(events, owner) | person_names(events) | email_domains(events)
    anchors = anchor_tokens(df) - blocked
    duplicates = collapse_duplicates(events)
    topics = build_topic_clusters(events, idf, anchors)
    commitments = build_commitments(events, idf, anchors, now)
    return Memory(
        events=events, idf=idf, now=now, duplicates=duplicates,
        topics=topics, commitments=commitments, owner=owner, anchors=anchors,
    )
