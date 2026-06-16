"""
timeutil.py
-----------
Time understanding for the memory engine. Pure standard library.

Two jobs:
  1. Parse the ISO-8601 event timestamps (all UTC, suffix 'Z').
  2. Extract *deadline expressions* from free text and resolve them to an
     absolute UTC instant, anchored on the timestamp of the event that
     mentions them.

Why anchoring matters: "send it by Friday" means a different Friday depending
on when it was said. Relative expressions ("tonight", "tomorrow", "this week",
"Tuesday") are resolved against the *local* (IST) date of the event, because
every human in this stream operates in IST and writes deadlines in IST.

Nothing here is dataset-specific. It is a generic temporal grammar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# The people in this stream live and write in IST. Content that says a bare
# clock time ("15:00 IST", "tonight") is in IST; the stored timestamps are UTC.
IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def parse_ts(s: str) -> datetime:
    """Parse an event timestamp like '2026-04-13T03:00:00Z' into aware UTC."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass
class Deadline:
    """A resolved deadline candidate found inside an event's content."""
    when_utc: datetime          # resolved absolute instant (UTC)
    raw: str                    # the literal text span we matched
    granularity: str            # 'time' (has explicit clock) or 'day' (date only)
    confidence: float           # 0..1 how sure we are about the resolution

    def to_dict(self) -> dict:
        return {
            "when_utc": self.when_utc.isoformat(),
            "raw": self.raw,
            "granularity": self.granularity,
            "confidence": round(self.confidence, 2),
        }


# ----------------------------------------------------------------------------
# Regex building blocks
# ----------------------------------------------------------------------------
_MONTH_RE = "|".join(MONTHS.keys())
# "Apr 10", "Apr 10 15:00", "Apr 10 15:00 IST"
_ABS_RE = re.compile(
    rf"\b(?P<mon>{_MONTH_RE})[a-z]*\s+(?P<day>\d{{1,2}})"
    rf"(?:\s+(?P<hh>\d{{1,2}}):(?P<mm>\d{{2}}))?"
    rf"(?:\s*(?P<zone>IST|UTC))?",
    re.IGNORECASE,
)
_WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE
)
_TIME_RE = re.compile(r"\b(?P<hh>\d{1,2}):(?P<mm>\d{2})\b")


def _ist_local_date(anchor_utc: datetime) -> datetime:
    """Local IST midnight for the day the event was written."""
    local = anchor_utc.astimezone(IST)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def _ist_to_utc(local_ist: datetime) -> datetime:
    return local_ist.replace(tzinfo=IST).astimezone(UTC)


def extract_deadlines(text: str, anchor_utc: datetime) -> list[Deadline]:
    """
    Return every deadline-like expression resolved to UTC.

    Resolution order / precedence:
      explicit "Mon DD [HH:MM]"  > weekday name > relative keyword.
    Explicit calendar dates win because the dataset frequently pins a weekday
    *and* a date ("Friday Apr 10"); the date is authoritative, the weekday is
    flavour.
    """
    out: list[Deadline] = []
    spans_consumed: list[tuple[int, int]] = []

    # 1) Explicit month-day (optionally with time + zone) -------------------
    for m in _ABS_RE.finditer(text):
        mon = MONTHS[m.group("mon").lower()[:3]]
        day = int(m.group("day"))
        year = anchor_utc.astimezone(IST).year
        hh = m.group("hh")
        mm = m.group("mm")
        zone = (m.group("zone") or "IST").upper()
        if hh is not None:
            local = datetime(year, mon, day, int(hh), int(mm))
            gran, conf = "time", 0.95
        else:
            # date only -> treat the deadline as end of that local day
            local = datetime(year, mon, day, 23, 59)
            gran, conf = "day", 0.85
        if zone == "UTC":
            when = local.replace(tzinfo=UTC)
        else:
            when = _ist_to_utc(local)
        out.append(Deadline(when.astimezone(UTC), m.group(0), gran, conf))
        spans_consumed.append(m.span())

    def _overlaps(a: tuple[int, int]) -> bool:
        return any(not (a[1] <= b[0] or a[0] >= b[1]) for b in spans_consumed)

    # 2) Weekday names ------------------------------------------------------
    base = _ist_local_date(anchor_utc)
    for m in _WEEKDAY_RE.finditer(text):
        if _overlaps(m.span()):
            continue  # already covered by an explicit date ("Friday Apr 10")
        target = WEEKDAYS[m.group(1).lower()]
        delta = (target - base.weekday()) % 7
        # "by <weekday>" almost always means the upcoming one; same-day stays.
        day = base + timedelta(days=delta)
        # attach a clock time if one sits right after the weekday word
        tail = text[m.end(): m.end() + 12]
        tm = _TIME_RE.search(tail)
        if tm:
            local = day.replace(hour=int(tm.group("hh")), minute=int(tm.group("mm")))
            gran, conf = "time", 0.7
        else:
            local = day.replace(hour=23, minute=59)
            gran, conf = "day", 0.55
        out.append(Deadline(_ist_to_utc(local), m.group(0), gran, conf))
        spans_consumed.append(m.span())

    # 3) Relative keywords --------------------------------------------------
    low = text.lower()

    def add_relative(day_offset: int, hour: int, minute: int, raw: str,
                     gran: str, conf: float):
        local = base + timedelta(days=day_offset)
        local = local.replace(hour=hour, minute=minute)
        out.append(Deadline(_ist_to_utc(local), raw, gran, conf))

    if "tonight" in low:
        add_relative(0, 21, 0, "tonight", "time", 0.6)
    if re.search(r"\btomorrow\b", low):
        add_relative(1, 23, 59, "tomorrow", "day", 0.6)
    if re.search(r"\beod\b|end of day\b", low):
        add_relative(0, 18, 0, "EOD", "time", 0.6)
    if re.search(r"\bby noon\b|\bnoon\b", low):
        add_relative(0, 12, 0, "noon", "time", 0.6)
    if "this week" in low:
        # end of the local week (Sunday 23:59)
        delta = (6 - base.weekday()) % 7
        add_relative(delta, 23, 59, "this week", "day", 0.45)

    # De-duplicate identical instants (keep the highest-confidence raw span).
    dedup: dict[str, Deadline] = {}
    for d in out:
        key = d.when_utc.isoformat()
        if key not in dedup or d.confidence > dedup[key].confidence:
            dedup[key] = d
    return sorted(dedup.values(), key=lambda d: d.when_utc)


def humanize_delta(when_utc: datetime, now_utc: datetime) -> str:
    """'in 6h', 'overdue by 1d 4h', 'in 3d' — for human-readable reasons."""
    secs = (when_utc - now_utc).total_seconds()
    overdue = secs < 0
    secs = abs(secs)
    days = int(secs // 86400)
    hours = int((secs % 86400) // 3600)
    if days >= 1:
        body = f"{days}d {hours}h" if hours else f"{days}d"
    else:
        mins = int((secs % 3600) // 60)
        body = f"{hours}h" if hours else f"{mins}m"
    return f"overdue by {body}" if overdue else f"in {body}"
