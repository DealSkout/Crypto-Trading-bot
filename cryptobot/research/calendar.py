"""The macro calendar: scheduled events that move crypto (FOMC decisions, US CPI releases, large
token unlocks) and the high-risk window around each one.

Events live in a YAML file you can edit (macro_calendar.yaml). Times are local to the event's
timezone, so daylight saving is handled for you. The bot warns when the calendar runs out of
future events: dates for the next year have to be added by hand once they are published.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

DEFAULT_WINDOWS = {  # minutes before, minutes after
    "fomc": (60, 120),
    "cpi": (30, 90),
    "unlock": (120, 240),
    "other": (30, 60),
}


@dataclass(frozen=True)
class MacroEvent:
    name: str
    kind: str
    at: datetime  # UTC
    before_minutes: int
    after_minutes: int
    symbols: tuple[str, ...] = ()  # empty = every market (FOMC, CPI)

    @property
    def window(self) -> tuple[datetime, datetime]:
        return (
            self.at - timedelta(minutes=self.before_minutes),
            self.at + timedelta(minutes=self.after_minutes),
        )

    def affects(self, symbol: str) -> bool:
        return not self.symbols or symbol.split("/")[0].upper() in self.symbols

    def to_dict(self) -> dict[str, Any]:
        start, end = self.window
        return {
            "name": self.name,
            "kind": self.kind,
            "at": self.at.isoformat(),
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "symbols": list(self.symbols),
        }


class CalendarError(ValueError):
    pass


def parse_events(raw: dict[str, Any] | None) -> list[MacroEvent]:
    raw = raw or {}
    default_tz = raw.get("timezone", "America/New_York")
    out = []
    for i, ev in enumerate(raw.get("events") or []):
        try:
            kind = str(ev.get("kind", "other")).lower()
            tz = ZoneInfo(str(ev.get("tz", default_tz)))
            local = datetime.fromisoformat(str(ev["time"]))
            at = (local if local.tzinfo else local.replace(tzinfo=tz)).astimezone(UTC)
            before, after = DEFAULT_WINDOWS.get(kind, DEFAULT_WINDOWS["other"])
            out.append(
                MacroEvent(
                    name=str(ev.get("name", kind.upper())),
                    kind=kind,
                    at=at,
                    before_minutes=int(ev.get("before_minutes", before)),
                    after_minutes=int(ev.get("after_minutes", after)),
                    symbols=tuple(str(s).upper() for s in ev.get("symbols") or ()),
                )
            )
        except (KeyError, ValueError, TypeError) as e:
            raise CalendarError(f"macro calendar event #{i + 1} is invalid: {e}") from e
    return sorted(out, key=lambda e: e.at)


class MacroCalendar:
    def __init__(self, events: list[MacroEvent] | None = None) -> None:
        self.events = list(events or [])

    @classmethod
    def load(cls, path: str | Path) -> MacroCalendar:
        p = Path(path)
        if not p.exists():
            return cls([])
        return cls(parse_events(yaml.safe_load(p.read_text())))

    def active(self, when: datetime, symbol: str | None = None) -> MacroEvent | None:
        """The event whose high-risk window contains `when`, if any."""
        for ev in self.events:
            start, end = ev.window
            if start <= when <= end and (symbol is None or ev.affects(symbol)):
                return ev
        return None

    def upcoming(self, when: datetime, hours: float = 48) -> list[MacroEvent]:
        horizon = when + timedelta(hours=hours)
        return [ev for ev in self.events if when <= ev.at <= horizon]

    def days_of_cover(self, when: datetime) -> float:
        """How far ahead the calendar has events: a short cover means dates need adding."""
        future = [ev.at for ev in self.events if ev.at >= when]
        return (max(future) - when).total_seconds() / 86400 if future else 0.0
