"""SQLite decision log. Every signal, order, fill, skip, halt and daily summary goes here, as
do parameter changes and the coach's notes. One file, zero dependencies beyond the stdlib."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, mode TEXT, symbol TEXT, kind TEXT NOT NULL,
  message TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY, mode TEXT, strategy TEXT, symbol TEXT, side TEXT, qty REAL,
  entry_ts TEXT, entry_price REAL, exit_ts TEXT, exit_price REAL, stop REAL, target REAL,
  pnl REAL, r_multiple REAL, costs REAL, exit_reason TEXT, signal_reason TEXT,
  asset_class TEXT DEFAULT 'crypto');
CREATE TABLE IF NOT EXISTS daily (
  date TEXT, mode TEXT, start_equity REAL, end_equity REAL, pnl REAL, trades INTEGER,
  costs REAL, halted INTEGER, PRIMARY KEY (date, mode));
CREATE TABLE IF NOT EXISTS params_history (
  id INTEGER PRIMARY KEY, ts TEXT, strategy TEXT, old_params TEXT, new_params TEXT,
  stats TEXT, source TEXT, applied INTEGER);
CREATE TABLE IF NOT EXISTS coach_notes (id INTEGER PRIMARY KEY, ts TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS llm_spend (date TEXT, component TEXT, usd REAL,
  PRIMARY KEY (date, component));
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, mode TEXT, version TEXT, strategy TEXT,
  symbol TEXT, action TEXT, reason TEXT, side TEXT, signal_price REAL, stop REAL, target REAL,
  qty REAL, fill_price REAL, slippage_bps REAL, fees REAL, pnl REAL, regime TEXT, params TEXT,
  features TEXT, inputs_hash TEXT, trade_id INTEGER);
CREATE INDEX IF NOT EXISTS decisions_ts ON decisions (ts);
"""

# Columns added to `trades` after the first release: added in place on old journals.
TRADE_EXTRA_COLUMNS = {
    "version": "TEXT",
    "regime": "TEXT",
    "signal_price": "REAL",
    "entry_slippage_bps": "REAL",
    "exit_slippage_bps": "REAL",
    "fees": "REAL",
    "features": "TEXT",
    "planned_r": "REAL",  # R at the planned exit level (stop -1, target +x) before any costs
    "actual_r": "REAL",  # net P&L in units of the PLANNED risk (signal price to stop)
}

DECISION_FIELDS = (
    "version",
    "strategy",
    "symbol",
    "action",
    "reason",
    "side",
    "signal_price",
    "stop",
    "target",
    "qty",
    "fill_price",
    "slippage_bps",
    "fees",
    "pnl",
    "regime",
    "params",
    "features",
    "inputs_hash",
    "trade_id",
)


def _iso(ts: datetime | None) -> str:
    ts = ts or datetime.now(UTC)
    return ts.isoformat()


class Journal:
    def __init__(self, path: str = ":memory:", mode: str = "backtest") -> None:
        self.path = path
        self.mode = mode
        self.conn = sqlite3.connect(path, timeout=10)
        self.conn.executescript(SCHEMA)
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(trades)")}
        for name, kind in TRADE_EXTRA_COLUMNS.items():
            if name not in have:
                self.conn.execute(f"ALTER TABLE trades ADD COLUMN {name} {kind}")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # events ------------------------------------------------------------------------
    def log(
        self,
        kind: str,
        message: str = "",
        symbol: str | None = None,
        data: dict[str, Any] | None = None,
        ts: datetime | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO events (ts, mode, symbol, kind, message, data) VALUES (?,?,?,?,?,?)",
            (_iso(ts), self.mode, symbol, kind, message, json.dumps(data or {}, default=str)),
        )
        self.conn.commit()

    def events(
        self, since: datetime | None = None, kinds: list[str] | None = None, limit: int = 1000
    ) -> pd.DataFrame:
        q, args = "SELECT * FROM events WHERE 1=1", []
        if since is not None:
            q += " AND ts >= ?"
            args.append(_iso(since))
        if kinds:
            q += f" AND kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        return pd.read_sql_query(q, self.conn, params=args)

    # trades ------------------------------------------------------------------------
    def record_trade(self, t: dict[str, Any], strategy: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO trades (mode, strategy, symbol, side, qty, entry_ts, entry_price, exit_ts,"
            " exit_price, stop, target, pnl, r_multiple, costs, exit_reason, signal_reason,"
            " asset_class, version, regime, signal_price, entry_slippage_bps, exit_slippage_bps,"
            " fees, features, planned_r, actual_r)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                self.mode,
                strategy,
                t["symbol"],
                t["side"],
                t["qty"],
                str(t["entry_ts"]),
                t["entry_price"],
                str(t["exit_ts"]),
                t["exit_price"],
                t["stop"],
                t.get("target"),
                t["pnl"],
                t.get("r_multiple"),
                t.get("costs", 0.0),
                t.get("exit_reason", ""),
                t.get("signal_reason", ""),
                t.get("asset_class", "crypto"),
                t.get("version"),
                t.get("regime"),
                t.get("signal_price"),
                t.get("entry_slippage_bps"),
                t.get("exit_slippage_bps"),
                t.get("fees"),
                t.get("features"),
                t.get("planned_r"),
                t.get("actual_r"),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # decisions (the feature store) ------------------------------------------------------
    def record_decision(self, ts: datetime | None, **fields: Any) -> int:
        """One row per decision: a signal taken (enter), refused (skip) or closed (exit), with
        the features the bot knew at that moment."""
        unknown = set(fields) - set(DECISION_FIELDS)
        if unknown:
            raise ValueError(f"unknown decision fields {sorted(unknown)}")
        row = {k: fields.get(k) for k in DECISION_FIELDS}
        for k in ("params", "features"):
            if row[k] is not None and not isinstance(row[k], str):
                row[k] = json.dumps(row[k], sort_keys=True, default=str)
        cur = self.conn.execute(
            f"INSERT INTO decisions (ts, mode, {', '.join(DECISION_FIELDS)}) VALUES"
            f" ({','.join('?' * (len(DECISION_FIELDS) + 2))})",
            (_iso(ts), self.mode, *row.values()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def decisions(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        mode: str | None = None,
        actions: list[str] | None = None,
    ) -> pd.DataFrame:
        q, args = "SELECT * FROM decisions WHERE 1=1", []
        if since is not None:
            q += " AND ts >= ?"
            args.append(_iso(since))
        if until is not None:
            q += " AND ts < ?"
            args.append(_iso(until))
        if mode:
            q += " AND mode = ?"
            args.append(mode)
        if actions:
            q += f" AND action IN ({','.join('?' * len(actions))})"
            args.extend(actions)
        return pd.read_sql_query(q + " ORDER BY id", self.conn, params=args)

    def trades(self, since: datetime | None = None, mode: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM trades WHERE 1=1", []
        if since is not None:
            q += " AND entry_ts >= ?"
            args.append(_iso(since))
        if mode:
            q += " AND mode = ?"
            args.append(mode)
        return pd.read_sql_query(q + " ORDER BY id", self.conn, params=args)

    # daily -------------------------------------------------------------------------
    def record_daily(
        self,
        day: date,
        start_equity: float,
        end_equity: float,
        pnl: float,
        trades: int,
        costs: float,
        halted: bool,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO daily (date, mode, start_equity, end_equity, pnl, trades,"
            " costs, halted) VALUES (?,?,?,?,?,?,?,?)",
            (day.isoformat(), self.mode, start_equity, end_equity, pnl, trades, costs, int(halted)),
        )
        self.conn.commit()

    def daily(self, since: date | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM daily WHERE mode = ?", [self.mode]
        if since is not None:
            q += " AND date >= ?"
            args.append(since.isoformat())
        return pd.read_sql_query(q + " ORDER BY date", self.conn, params=args)

    # learning loop -----------------------------------------------------------------
    def add_params_change(
        self, strategy: str, old: dict, new: dict, stats: dict, source: str, applied: bool
    ) -> None:
        self.conn.execute(
            "INSERT INTO params_history (ts, strategy, old_params, new_params, stats, source,"
            " applied) VALUES (?,?,?,?,?,?,?)",
            (
                _iso(None),
                strategy,
                json.dumps(old, sort_keys=True),
                json.dumps(new, sort_keys=True),
                json.dumps(stats, default=str),
                source,
                int(applied),
            ),
        )
        self.conn.commit()

    def params_history(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT ts, strategy, old_params, new_params, stats, source, applied FROM"
            " params_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "ts": r[0],
                "strategy": r[1],
                "old_params": json.loads(r[2]),
                "new_params": json.loads(r[3]),
                "stats": json.loads(r[4]),
                "source": r[5],
                "applied": bool(r[6]),
            }
            for r in rows
        ]

    def last_applied_change(self) -> dict[str, Any] | None:
        for row in self.params_history(limit=100):
            if row["applied"]:
                return row
        return None

    # coach -------------------------------------------------------------------------
    def add_coach_note(self, note: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO coach_notes (ts, note) VALUES (?, ?)",
            (_iso(None), json.dumps(note, default=str)),
        )
        self.conn.commit()

    def coach_notes(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT ts, note FROM coach_notes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"ts": r[0], **json.loads(r[1])} for r in rows]

    def llm_spend_today(self, component: str, day: date | None = None) -> float:
        day = day or datetime.now(UTC).date()
        row = self.conn.execute(
            "SELECT usd FROM llm_spend WHERE date = ? AND component = ?",
            (day.isoformat(), component),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def add_llm_spend(self, component: str, usd: float, day: date | None = None) -> None:
        day = day or datetime.now(UTC).date()
        self.conn.execute(
            "INSERT INTO llm_spend (date, component, usd) VALUES (?, ?, ?)"
            " ON CONFLICT(date, component) DO UPDATE SET usd = usd + excluded.usd",
            (day.isoformat(), component, float(usd)),
        )
        self.conn.commit()

    def llm_spend_since(self, days: int, day: date | None = None) -> dict[str, float]:
        day = day or datetime.now(UTC).date()
        start = (day - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            "SELECT component, SUM(usd) FROM llm_spend WHERE date >= ? GROUP BY component", (start,)
        ).fetchall()
        return {str(r[0]): float(r[1] or 0.0) for r in rows}

    def coach_spend_today(self, day: date | None = None) -> float:
        return self.llm_spend_today("coach", day)

    # state snapshots (for the dashboard) ---------------------------------------------
    def set_state(self, key: str, value: Any, ts: datetime | None = None) -> None:
        self.conn.execute(
            "INSERT INTO state (key, value, ts) VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value, ts = excluded.ts",
            (key, json.dumps(value, default=str), _iso(ts)),
        )
        self.conn.commit()

    def get_state(self, key: str) -> Any:
        row = self.conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def all_state(self) -> dict[str, Any]:
        rows = self.conn.execute("SELECT key, value, ts FROM state").fetchall()
        return {r[0]: {"value": json.loads(r[1]), "ts": r[2]} for r in rows}

    def add_coach_spend(self, usd: float, day: date | None = None) -> None:
        self.add_llm_spend("coach", usd, day)
