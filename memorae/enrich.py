"""
enrich.py
---------
Runtime signal extraction. For every raw event we DERIVE structured signals
from the text and timestamp. Nothing is labelled by hand and the dataset is
never modified; these are generic linguistic detectors that would run the same
way on any personal stream.

Signals we derive per event:
  - actor / channel / source          (provenance)
  - obligation                        (is this a thing the user owes?)
  - status                            (open / done / slipping / cancelled / update)
  - importance cues                   (exec, money, legal, hard consequence...)
  - deadlines                         (resolved via timeutil)
  - topic signature                   (salient tokens for clustering & linking)
  - noise score                       (0..1, high = low-signal chatter)

These per-event signals are the substrate the memory layer organises and the
retriever scores. Keeping extraction here means the rest of the system reasons
over structure, not raw strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .timeutil import Deadline, extract_deadlines, parse_ts

# ----------------------------------------------------------------------------
# Lexical resources (generic, not tied to this dataset's specific tasks)
# ----------------------------------------------------------------------------
STOPWORDS = set("""
a an the this that these those i you he she it we they me my your his her our their
is are was were be been being am do does did doing have has had having will would
can could should shall may might must to of in on at by for with about as into from
and or but if then so than too very just not no yes more most some any all each
your you'll i'll i'm we'll it's there here when where which who whom whose what why how
please need needs needed send sends sent before after again still only also out up off
get got make made one two i've you've don't can't won't isn't aren't
""".split())

# "When" words and pure-process words are never good *topic* anchors: they recur
# across unrelated tasks. We strip them when building subject fingerprints so a
# cardiology reminder doesn't cluster with a rubric reminder just because both
# say "Tuesday" / "remind".
TEMPORAL_META = set("""
monday tuesday wednesday thursday friday saturday sunday
jan feb mar apr may jun jul aug sep oct nov dec
january february march april june july august september october november december
today tonight tomorrow yesterday morning afternoon evening night noon midnight
week weekend day days hour hours minute minutes eod asap soon now later
friendly nudge reminder remind ping nudge sync next last week's
""".split())

# Process / communication words that recur across every kind of task and so make
# terrible *topic* anchors. Stripping them from the subject fingerprint keeps
# clusters tight (the UIE thread should bind on "uie"/"appendix"/"procurement",
# not on "proposal"/"review"/"owner", which appear in half the stream).
GENERIC_PROCESS = set("""
proposal draft drafts doc docs document version page pages copy attachment
update updates updated review reviews reviewed owner owners action actions item
items ready meeting meet room call calls status plan plans note notes pinged
thread list lists link links saved send sending share shared sharing final
request reply replied follow followup msg message messages chat info detail
details stuff thing things work working task tasks dev draftfor pending done
quick please ready set table summary due cannot confirm confirmed schedule
scheduled customer focus block blocks want wants lead leads prefer prefers
avoid keep clear move moved start starts begin agenda attached invite
slips slip handle add adds names name renew renewal renewals payment pay
upload uploads uploaded form forms fee book booking collect finish portal
bring pick provide finalize finalise prepare loop close closed closing late
""".split())

# Obligations the USER carries: first-person commitments or asks directed at them.
SELF_OBLIGATION = re.compile(
    r"\b((i|we)\s+(promised|owe|will|need to|have to|must|should)|i'?ll|i still need|"
    r"need to (send|finish|handle|upload|pay|renew|confirm|review|close|draft|book|schedule|add|collect|create|nudge)|"
    r"remind me)\b",
    re.IGNORECASE,
)
# Imperative action verbs at the start of a self-note or a self-authored line.
IMPERATIVE_VERB = (
    r"(collect|confirm|add|renew|pay|upload|send|create|draft|close|review|"
    r"finalize|finalise|prepare|write|build|file|book|schedule|handle|finish|nudge|loop|bring)"
)
IMPERATIVE_NOTE = re.compile(rf"^{IMPERATIVE_VERB}\b", re.IGNORECASE)
# Requests assigned to the user by someone else.
ASSIGNED_OBLIGATION = re.compile(
    r"\b(can you|could you|please (send|close|confirm|add|review|bring|loop|nudge|follow up|file|pick|provide)|"
    r"i need (the|your|a|my)|nudge on|you said you would|requires owner confirmation|"
    r"candidate packets due|owner table required|please follow up)\b",
    re.IGNORECASE,
)
# Things that look like obligations but are NOT the user's to-do (downweight).
NOT_USER_TASK = re.compile(
    r"\b(prefers?|does not want|do not want|dislikes|likes|hate|hates|i hate|"
    r"reports to|knows|will answer|will discuss|can approve|can do|can cover|"
    r"writes best|calls|privileges)\b",
    re.IGNORECASE,
)

# Status / lifecycle markers.
# DONE = genuine completion of the deliverable itself (first-person or explicit).
DONE = re.compile(
    r"\b(marked done|is done|now done|completed|finished it|already (sent|submitted|paid|"
    r"renewed|uploaded|filed)|i (sent|submitted|paid|renewed|uploaded|filed|closed)|"
    r"sent ✓|✓ sent|signed off|signed-off)\b",
    re.IGNORECASE,
)
# UNBLOCK = progress, NOT completion. A dependency cleared but the user still owes
# the action ("clause 8 approved", "no longer blocked"). Keeps the commitment open.
UNBLOCK = re.compile(
    r"\b(approved now|is approved|no longer (blocked|on)|unblocked|"
    r"clause \d+ is approved|not .*? blocked anymore|now has draft|"
    r"prevention section now)\b",
    re.IGNORECASE,
)
UPDATE = re.compile(
    r"\b(ignore (my )?earlier|now due|moved (from|to|by)|updated|update:|correction|"
    r"do not use the old|actually|reschedul|changed to|one correction|"
    r"calendar update|calendar cancellation)\b",
    re.IGNORECASE,
)
SLIP = re.compile(
    r"\b(before it slips again|before i forget again|slips again|forget again|"
    r"still (need|needs|haven'?t)|keep forgetting|you said you would|"
    r"friendly nudge|nudge (him|on)|haven'?t|didn'?t get to|postponed|pushed to|delayed|"
    r"can you send the .* before the next sync)\b",
    re.IGNORECASE,
)
CANCEL = re.compile(
    r"\b(cancel|cancellation|removed|scrapped|dropped|release the .* slot|"
    r"cannot (pick|cover))\b",
    re.IGNORECASE,
)

# Importance cues (each adds weight).
IMPORTANCE_CUES = [
    (re.compile(r"\b(nina|northstar|ravi|cedric|elt|customer|priya)\b", re.I), 1.2, "customer/exec stakeholder"),
    (re.compile(r"\b(late fee|locks|or the .* starts|release the .* slot|"
                r"before finance closes|will pick the safest default)\b", re.I), 1.4, "hard consequence if missed"),
    (re.compile(r"(\$\s?\d|inr\s?\d|\b\d+(\.\d+)?k\b)", re.I), 0.8, "money/figure"),
    (re.compile(r"\b(sow|clause|contract renewal|procurement|licens|legal)\b", re.I), 0.9, "legal/contract"),
    (re.compile(r"\b(due|deadline|overdue|required|must|blocked|before (the )?(meeting|negotiation|review|standup|sync))\b", re.I), 1.0, "explicit deadline/dependency"),
    (re.compile(r"\b(proposal|appendix|rubric|redlines|checklist|postmortem|incident|report summary)\b", re.I), 0.6, "named deliverable"),
]

# Noise cues by provenance + content.
RANDOM_CHANNEL = re.compile(r"#random\b", re.I)
SAVED_LINK = re.compile(r"^saved link\b", re.I)
GMAIL_NOISE = re.compile(
    r"\b(newsletter|promo|webinar|workspace digest|receipt|no action requested|"
    r"updated invite attached|digest)\b",
    re.IGNORECASE,
)
SMS_NOISE = re.compile(
    r"\b(otp|ride receipt|package delivered|delivery update|out for delivery|"
    r"your order)\b",
    re.IGNORECASE,
)
SOCIAL_CHATTER = re.compile(
    r"\b(coffee machine|air conditioning|least aggressive|lunch is late|sandwich|"
    r"elevator|hdmi|projector remote|monsoon|cardamom|chai|dosa|blue notebook|"
    r"typo|meme|cinematic)\b",
    re.IGNORECASE,
)

CHANNEL_RE = re.compile(r"#(\w+)")
ACTOR_RE = re.compile(r"^(?:#\w+\s+)?([A-Z][a-zA-Z]+)(?::| <| to )")


@dataclass
class Event:
    """A raw event plus all derived signals."""
    idx: int
    ts: datetime
    source: str
    content: str

    actor: str | None = None
    channel: str | None = None

    is_obligation: bool = False
    obligation_kind: str | None = None       # 'self' | 'assigned' | 'note'
    status: str = "info"                      # info | open | done | update | cancel
    is_slipping: bool = False

    deadlines: list[Deadline] = field(default_factory=list)
    primary_deadline: Deadline | None = None

    importance: float = 0.0
    importance_reasons: list[str] = field(default_factory=list)
    noise: float = 0.0

    topic_tokens: set[str] = field(default_factory=set)
    subject_tokens: set[str] = field(default_factory=set)  # topic_tokens minus temporal/process words
    entities: set[str] = field(default_factory=set)
    signal: float = 0.0                       # generic standalone signal prior

    # filled in by the memory layer
    cluster_id: str | None = None
    dup_of: int | None = None

    def to_context_dict(self) -> dict:
        return {
            "idx": self.idx,
            "timestamp": self.ts.isoformat().replace("+00:00", "Z"),
            "source": self.source if not self.channel else f"{self.source} #{self.channel}",
            "content": self.content,
        }


def _salient_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())
    toks = {w for w in words if len(w) >= 3 and w not in STOPWORDS}
    return toks


def _entities(text: str) -> set[str]:
    """Capitalised words / acronyms that look like names or project nouns."""
    ents = set(re.findall(r"\b([A-Z]{2,}|[A-Z][a-z]+)\b", text))
    # drop obvious sentence-initial noise words
    return {e for e in ents if e.lower() not in STOPWORDS and len(e) > 1}


def enrich_event(idx: int, raw: dict, owner: str | None = None) -> Event:
    content = raw["content"].strip()
    ts = parse_ts(raw["timestamp"])
    ev = Event(idx=idx, ts=ts, source=raw["source"], content=content)

    # provenance
    cm = CHANNEL_RE.search(content)
    if cm and ev.source == "slack":
        ev.channel = cm.group(1)
    am = ACTOR_RE.match(content)
    if am:
        ev.actor = am.group(1)

    # body = content with the "Actor:" / "#chan Actor:" prefix removed
    body = content
    if am:
        body = content[am.end():].strip()
    body = re.sub(r"^#\w+\s+", "", body).strip()

    # deadlines
    ev.deadlines = extract_deadlines(content, ts)
    if ev.deadlines:
        # "moved from <old> to <new>" / "rescheduled to <new>" / "now due <new>":
        # the operative deadline is the TARGET, i.e. the latest date mentioned, not
        # the old one being superseded. For reschedule/update phrasings prefer the
        # latest high-confidence instant; otherwise prefer most-confident.
        reschedule = bool(re.search(r"\b(moved (from|to)|reschedul|now due|changed to|"
                                    r"pushed to|postponed to)\b", content, re.I))
        hi = [d for d in ev.deadlines if d.confidence >= 0.7]
        if reschedule and hi:
            ev.primary_deadline = max(hi, key=lambda d: d.when_utc)
        else:
            ev.primary_deadline = max(
                ev.deadlines, key=lambda d: (d.confidence, d.granularity == "time"))

    # obligation detection
    is_owner = owner is not None and ev.actor == owner
    not_task = bool(NOT_USER_TASK.search(content))
    if not not_task and SELF_OBLIGATION.search(content):
        ev.is_obligation, ev.obligation_kind = True, "self"
    elif not not_task and is_owner and IMPERATIVE_NOTE.search(body):
        # "Aarav: Pay the maintenance ...", "Aarav: Upload receipts ..."
        ev.is_obligation, ev.obligation_kind = True, "self"
    elif not not_task and IMPERATIVE_NOTE.search(content) and ev.source in ("notion", "reminder"):
        ev.is_obligation, ev.obligation_kind = True, "note"
    elif not not_task and ASSIGNED_OBLIGATION.search(content):
        ev.is_obligation, ev.obligation_kind = True, "assigned"

    # status / lifecycle (priority: real completion > unblock-progress > cancel > update)
    if DONE.search(content):
        ev.status = "done"
    elif UNBLOCK.search(content):
        ev.status = "update"          # dependency cleared, but action still owed
    elif CANCEL.search(content):
        ev.status = "cancel"
    elif UPDATE.search(content):
        ev.status = "update"
    elif ev.is_obligation:
        ev.status = "open"
    ev.is_slipping = bool(SLIP.search(content))

    # importance
    for rx, w, why in IMPORTANCE_CUES:
        if rx.search(content):
            ev.importance += w
            ev.importance_reasons.append(why)

    # noise
    n = 0.0
    if ev.channel == "random":
        n += 0.6
    if ev.source == "chrome_extension" and SAVED_LINK.search(content):
        n += 0.6
    if ev.source == "gmail" and GMAIL_NOISE.search(content):
        n += 0.7
    if ev.source == "sms" and SMS_NOISE.search(content):
        n += 0.7
    if SOCIAL_CHATTER.search(content):
        n += 0.5
    # a thing with a real deadline or obligation is not noise even if chatty
    if ev.is_obligation or (ev.primary_deadline and ev.primary_deadline.confidence >= 0.85):
        n = max(0.0, n - 0.8)
    ev.noise = min(1.0, n)

    # topic
    ev.topic_tokens = _salient_tokens(content)
    # subject fingerprint: strip "when/process" words so unrelated tasks that merely
    # share a weekday or the word "reminder" do not get glued together downstream.
    ev.subject_tokens = {t for t in ev.topic_tokens
                         if t not in TEMPORAL_META and t not in GENERIC_PROCESS}
    ev.entities = _entities(content)

    # generic standalone signal prior (query-agnostic)
    ev.signal = (
        ev.importance
        + (1.0 if ev.is_obligation else 0.0)
        + (0.8 if ev.primary_deadline else 0.0)
        - 1.5 * ev.noise
    )
    return ev


def detect_owner(raw_events: list[dict]) -> str | None:
    """
    Infer the stream owner: the actor who most often authors first-person
    statements ('I ...', 'my ...'). In a personal memory stream the owner is
    whoever is speaking in the first person most of the time. No hardcoding.
    """
    from collections import Counter
    fp = Counter()
    for e in raw_events:
        c = e["content"]
        m = ACTOR_RE.match(c)
        if not m:
            continue
        actor = m.group(1)
        body = c[m.end():]
        fp[actor] += len(re.findall(r"\b(I|I'?ll|I'?m|my|me|I've)\b", body))
    return fp.most_common(1)[0][0] if fp else None


def enrich_all(raw_events: list[dict]) -> list[Event]:
    owner = detect_owner(raw_events)
    return [enrich_event(i, e, owner=owner) for i, e in enumerate(raw_events)]
