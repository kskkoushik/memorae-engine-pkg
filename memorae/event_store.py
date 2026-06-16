from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from rapidfuzz import fuzz, process


UTC = timezone.utc


@dataclass(frozen=True)
class RawEvent:
    idx: int
    ts: datetime
    source: str
    content: str

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "timestamp": self.ts.isoformat().replace("+00:00", "Z"),
            "source": self.source,
            "content": self.content,
        }


def parse_ts(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_date_bound(value: str | None, *, end_of_day: bool = False) -> datetime | None:
    """Parse ISO date or datetime string into UTC."""
    if not value or not str(value).strip():
        return None
    s = str(value).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        dt = datetime.fromisoformat(s).replace(tzinfo=UTC)
        if end_of_day:
            return dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        return dt
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def validate_date_range(
    start: datetime | None, end: datetime | None
) -> tuple[datetime | None, datetime | None, str | None]:
    if start and end and start > end:
        return start, end, "start_date must be before or equal to end_date"
    return start, end, None


def pack_results(
    events: list[RawEvent],
    limit: int,
    *,
    sort_by_date: bool = True,
) -> dict:
    if sort_by_date:
        events = sorted(events, key=lambda e: e.ts)
    total = len(events)
    limit = max(1, int(limit))
    shown = events[:limit]
    hidden = max(0, total - len(shown))
    return {
        "total_matched": total,
        "returned": len(shown),
        "hidden_due_to_limit": hidden,
        "limit": limit,
        "events": [e.to_dict() for e in shown],
    }


class EventStore:
    """In-memory event store with fast keyword index."""

    def __init__(self, events: list[RawEvent], now: datetime, owner: str | None = None) -> None:
        self.events = events
        self.now = now
        self.owner = owner
        self._visible = [e for e in events if e.ts <= now]
        self._sources = sorted({e.source for e in events})
        self._word_index: dict[str, set[int]] = defaultdict(set)
        self._search_corpus: list[tuple[int, str]] = []
        for e in self._visible:
            self._search_corpus.append((e.idx, e.content))
            for tok in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-']{1,}", e.content.lower()):
                if len(tok) >= 2:
                    self._word_index[tok].add(e.idx)

    @classmethod
    def from_file(cls, path: str, now: datetime | None = None) -> "EventStore":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        events: list[RawEvent] = []
        for i, row in enumerate(raw):
            events.append(RawEvent(
                idx=i,
                ts=parse_ts(row["timestamp"]),
                source=row["source"].strip().lower(),
                content=row["content"].strip(),
            ))
        if now is None:
            now = datetime.now(tz=UTC)
        owner = cls._detect_owner(events)
        return cls(events, now, owner)

    @staticmethod
    def _detect_owner(events: list[RawEvent]) -> str | None:
        counts: dict[str, int] = defaultdict(int)
        actor_re = re.compile(r"^(?:#\w+\s+)?([A-Z][a-zA-Z]+):")
        for e in events:
            m = actor_re.match(e.content)
            if m:
                body = e.content[m.end():]
                if re.search(r"\b(I|I'll|I'm|my|me|I've)\b", body):
                    counts[m.group(1)] += 1
        if not counts:
            return None
        return max(counts, key=counts.get)

    def get_sources(self) -> list[str]:
        return list(self._sources)

    def source_counts(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for e in self._visible:
            out[e.source] += 1
        return dict(sorted(out.items()))

    def _by_idx(self, idx: int) -> RawEvent | None:
        if 0 <= idx < len(self.events):
            return self.events[idx]
        return None

    def filter_events(
        self,
        *,
        sources: Iterable[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        only_visible: bool = True,
    ) -> list[RawEvent]:
        pool = self._visible if only_visible else list(self.events)
        src_set = None
        if sources:
            src_set = {s.strip().lower() for s in sources if s and str(s).strip()}

        out: list[RawEvent] = []
        for e in pool:
            if src_set is not None and e.source not in src_set:
                continue
            if start and e.ts < start:
                continue
            if end and e.ts > end:
                continue
            out.append(e)
        return out

    def search_by_source(
        self,
        source_name: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
    ) -> dict:
        src = source_name.strip().lower()
        start = parse_date_bound(start_date)
        end = parse_date_bound(end_date, end_of_day=True)
        start, end, err = validate_date_range(start, end)
        if err:
            return {"error": err, "events": [], "total_matched": 0, "returned": 0,
                    "hidden_due_to_limit": 0, "limit": limit}

        matched = self.filter_events(sources=[src], start=start, end=end)
        result = pack_results(matched, limit)
        result["source"] = src
        result["date_range"] = {
            "start": start.isoformat().replace("+00:00", "Z") if start else None,
            "end": end.isoformat().replace("+00:00", "Z") if end else None,
        }
        return result

    def search_by_date(
        self,
        start_date: str,
        end_date: str,
        limit: int = 50,
    ) -> dict:
        start = parse_date_bound(start_date)
        end = parse_date_bound(end_date, end_of_day=True)
        if not start or not end:
            return {"error": "start_date and end_date are required", "events": [],
                    "total_matched": 0, "returned": 0, "hidden_due_to_limit": 0, "limit": limit}
        start, end, err = validate_date_range(start, end)
        if err:
            return {"error": err, "events": [], "total_matched": 0, "returned": 0,
                    "hidden_due_to_limit": 0, "limit": limit}

        matched = self.filter_events(start=start, end=end)
        result = pack_results(matched, limit)
        result["date_range"] = {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
        }
        return result

    def search_by_keyword(
        self,
        keyword: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        source_name: str | None = None,
    ) -> dict:
        kw = keyword.strip()
        if not kw:
            return {"error": "keyword is required", "events": [], "total_matched": 0,
                    "returned": 0, "hidden_due_to_limit": 0, "limit": limit}

        start = parse_date_bound(start_date)
        end = parse_date_bound(end_date, end_of_day=True)
        start, end, err = validate_date_range(start, end)
        if err:
            return {"error": err, "events": [], "total_matched": 0, "returned": 0,
                    "hidden_due_to_limit": 0, "limit": limit}

        sources = [source_name] if source_name and source_name.strip() else None
        pool = self.filter_events(sources=sources, start=start, end=end)
        kw_lower = kw.lower()

        # Fast path: token index for single-word keywords
        if " " not in kw_lower and kw_lower in self._word_index:
            idxs = self._word_index[kw_lower]
            matched = [e for e in pool if e.idx in idxs]
        else:
            # Phrase / multi-word: rapidfuzz partial ratio on pre-filtered pool
            if not pool:
                matched = []
            else:
                choices = {e.idx: e.content for e in pool}
                hits = process.extract(
                    kw,
                    choices,
                    scorer=fuzz.partial_ratio,
                    score_cutoff=75,
                    limit=len(choices),
                )
                hit_idxs = {int(idx) for _, idx, _ in hits}
                # Also include straightforward substring matches
                for e in pool:
                    if kw_lower in e.content.lower():
                        hit_idxs.add(e.idx)
                matched = [e for e in pool if e.idx in hit_idxs]

        result = pack_results(matched, limit)
        result["keyword"] = kw
        result["date_range"] = {
            "start": start.isoformat().replace("+00:00", "Z") if start else None,
            "end": end.isoformat().replace("+00:00", "Z") if end else None,
        }
        if source_name:
            result["source"] = source_name.strip().lower()
        return result

    def search_by_indices(
        self,
        indices: list[int],
        *,
        sources: Iterable[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 30,
        scores: dict[int, float] | None = None,
    ) -> dict:
        """Apply optional filters to a set of event indices (from RAG)."""
        src_set = None
        if sources:
            src_set = {s.strip().lower() for s in sources if s and str(s).strip()}

        matched: list[RawEvent] = []
        for idx in indices:
            e = self._by_idx(idx)
            if e is None or e.ts > self.now:
                continue
            if src_set is not None and e.source not in src_set:
                continue
            if start and e.ts < start:
                continue
            if end and e.ts > end:
                continue
            matched.append(e)

        if scores:
            matched.sort(key=lambda e: (-scores.get(e.idx, 0.0), e.ts))
        else:
            matched.sort(key=lambda e: e.ts)

        result = pack_results(matched, limit, sort_by_date=not scores)
        if scores:
            for row in result["events"]:
                row["relevance_score"] = round(scores.get(row["idx"], 0.0), 3)
        return result

    def stats(self) -> dict:
        return {
            "events_total": len(self.events),
            "events_visible_now": len(self._visible),
            "events_future_hidden": len(self.events) - len(self._visible),
            "sources": self.source_counts(),
        }
