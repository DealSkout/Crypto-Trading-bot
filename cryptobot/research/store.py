"""SQLite cache for research data, in the same file as the journal.

Every value has two times:
  ts        when the value refers to (the funding settlement, the index day, the headline)
  avail_ts  when the bot could first have known it (ts plus a publication lag)
Features at a decision only use rows with avail_ts <= the decision time, so backtests and
replays never see data before it was published.

One connection per thread: the research feed refreshes in a background thread while the
trading loop reads features in the main thread.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS research (
  source TEXT NOT NULL, key TEXT NOT NULL, ts TEXT NOT NULL, avail_ts TEXT NOT NULL,
  value REAL, data TEXT, fetched_at TEXT, PRIMARY KEY (source, key, ts));
CREATE INDEX IF NOT EXISTS research_avail ON research (source, key, avail_ts);
CREATE TABLE IF NOT EXISTS research_status (
  source TEXT PRIMARY KEY, last_attempt TEXT, last_ok TEXT, last_error TEXT, rows INTEGER);
CREATE TABLE IF NOT EXISTS news (
  id TEXT PRIMARY KEY, ts TEXT, avail_ts TEXT, source TEXT, title TEXT, url TEXT,
  score REAL, symbols TEXT, fetched_at TEXT);
CREATE INDEX IF NOT EXISTS news_avail ON news (avail_ts);
"""


def iso(ts: datetime | pd.Timestamp | None) -> str:
    ts = ts or datetime.now(UTC)
    return pd.Timestamp(ts).tz_convert("UTC").isoformat()


class ResearchStore:
    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._local = threading.local()
        self._shared = sqlite3.connect(path, timeout=10, check_same_thread=False)
        self._shared.executescript(SCHEMA)
        self._shared.commit()
        self._local.conn = self._shared
        self._lock = threading.Lock()

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            if self.path == ":memory:":  # one in-memory database: share it, serialised
                conn = self._shared
            else:
                conn = sqlite3.connect(self.path, timeout=10)
            self._local.conn = conn
        return conn

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        if self.path == ":memory:":
            with self._lock:
                return self.conn.execute(sql, args)
        return self.conn.execute(sql, args)

    # values ------------------------------------------------------------------------------
    def upsert(
        self,
        source: str,
        key: str,
        rows: list[tuple[datetime, datetime, float | None, dict[str, Any] | None]],
    ) -> int:
        """rows: (ts, avail_ts, value, data)."""
        now = iso(None)
        for ts, avail, value, data in rows:
            self._exec(
                "INSERT INTO research (source, key, ts, avail_ts, value, data, fetched_at)"
                " VALUES (?,?,?,?,?,?,?) ON CONFLICT(source, key, ts) DO UPDATE SET"
                " value = excluded.value, data = excluded.data, avail_ts = excluded.avail_ts",
                (
                    source,
                    key,
                    iso(ts),
                    iso(avail),
                    None if value is None else float(value),
                    json.dumps(data or {}, default=str),
                    now,
                ),
            )
        self.conn.commit()
        return len(rows)

    def asof(self, source: str, key: str, when: datetime) -> dict[str, Any] | None:
        """The latest value that was available at `when`."""
        row = self._exec(
            "SELECT ts, avail_ts, value, data FROM research WHERE source = ? AND key = ?"
            " AND avail_ts <= ? ORDER BY avail_ts DESC, ts DESC LIMIT 1",
            (source, key, iso(when)),
        ).fetchone()
        if row is None:
            return None
        return {"ts": row[0], "avail_ts": row[1], "value": row[2], "data": json.loads(row[3])}

    def series(
        self,
        source: str,
        key: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> pd.DataFrame:
        """Values available in [since, until], by availability time."""
        q, args = (
            "SELECT ts, avail_ts, value, data FROM research WHERE source = ? AND key = ?",
            [
                source,
                key,
            ],
        )
        if since is not None:
            q += " AND avail_ts >= ?"
            args.append(iso(since))
        if until is not None:
            q += " AND avail_ts <= ?"
            args.append(iso(until))
        rows = self._exec(q + " ORDER BY avail_ts", tuple(args)).fetchall()
        return pd.DataFrame(rows, columns=["ts", "avail_ts", "value", "data"])

    def coverage(self, source: str, key: str) -> tuple[str | None, str | None]:
        row = self._exec(
            "SELECT MIN(avail_ts), MAX(avail_ts) FROM research WHERE source = ? AND key = ?",
            (source, key),
        ).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def keys(self, source: str) -> list[str]:
        rows = self._exec("SELECT DISTINCT key FROM research WHERE source = ?", (source,))
        return [r[0] for r in rows.fetchall()]

    # news ------------------------------------------------------------------------------------
    def add_news(self, items: list[dict[str, Any]]) -> int:
        added = 0
        now = iso(None)
        for it in items:
            cur = self._exec(
                "INSERT OR IGNORE INTO news (id, ts, avail_ts, source, title, url, score,"
                " symbols, fetched_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    it["id"],
                    iso(it["ts"]),
                    iso(it.get("avail_ts") or it["ts"]),
                    it.get("source", ""),
                    it.get("title", "")[:500],
                    it.get("url", "")[:500],
                    it.get("score"),
                    ",".join(it.get("symbols") or []),
                    now,
                ),
            )
            added += cur.rowcount
        self.conn.commit()
        return added

    def news(self, since: datetime, until: datetime) -> pd.DataFrame:
        rows = self._exec(
            "SELECT ts, avail_ts, source, title, url, score, symbols FROM news"
            " WHERE avail_ts > ? AND avail_ts <= ? ORDER BY avail_ts",
            (iso(since), iso(until)),
        ).fetchall()
        return pd.DataFrame(
            rows, columns=["ts", "avail_ts", "source", "title", "url", "score", "symbols"]
        )

    # status ----------------------------------------------------------------------------------
    def mark(self, source: str, ok: bool, error: str = "", rows: int = 0) -> None:
        now = iso(None)
        if ok:
            self._exec(
                "INSERT INTO research_status (source, last_attempt, last_ok, last_error, rows)"
                " VALUES (?,?,?,'',?) ON CONFLICT(source) DO UPDATE SET last_attempt ="
                " excluded.last_attempt, last_ok = excluded.last_ok, last_error = '',"
                " rows = excluded.rows",
                (source, now, now, rows),
            )
        else:
            self._exec(
                "INSERT INTO research_status (source, last_attempt, last_ok, last_error, rows)"
                " VALUES (?,?,NULL,?,0) ON CONFLICT(source) DO UPDATE SET last_attempt ="
                " excluded.last_attempt, last_error = excluded.last_error",
                (source, now, error[:500]),
            )
        self.conn.commit()

    def status(self) -> dict[str, dict[str, Any]]:
        rows = self._exec(
            "SELECT source, last_attempt, last_ok, last_error, rows FROM research_status"
        ).fetchall()
        return {
            r[0]: {"last_attempt": r[1], "last_ok": r[2], "last_error": r[3], "rows": r[4]}
            for r in rows
        }

    def last_ok(self, source: str) -> datetime | None:
        info = self.status().get(source) or {}
        return pd.Timestamp(info["last_ok"]).to_pydatetime() if info.get("last_ok") else None
