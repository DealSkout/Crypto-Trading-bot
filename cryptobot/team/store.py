"""The team's shared knowledge store. One SQLite file in WAL mode holds the bot
roster, signals, fills, daily metrics, regimes, allocations, consensus trades,
challengers, version history and events, so the orchestrator and dashboard read
a single source of truth. Plain sqlite3, parameterised SQL, one connection with
a lock around writes so several threads can share a store safely."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS bots (
  bot_id TEXT PRIMARY KEY, name TEXT, repo TEXT, version TEXT, status TEXT, updated_ts TEXT);
CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, bot_id TEXT, symbol TEXT, side TEXT,
  entry REAL, stop REAL, target REAL, reason TEXT, taken INTEGER);
CREATE TABLE IF NOT EXISTS fills (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, bot_id TEXT, symbol TEXT, side TEXT,
  qty REAL, price REAL, fees REAL, internal INTEGER, order_id TEXT);
CREATE TABLE IF NOT EXISTS bot_metrics_daily (
  day TEXT, bot_id TEXT, equity REAL, pnl REAL, trades INTEGER, sharpe REAL,
  drawdown_pct REAL, regime TEXT, PRIMARY KEY (day, bot_id));
CREATE TABLE IF NOT EXISTS regimes (
  ts TEXT PRIMARY KEY, label TEXT, vol_pctile REAL, trend REAL, breadth REAL);
CREATE TABLE IF NOT EXISTS allocations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, bot_id TEXT, fraction REAL,
  reason TEXT);
CREATE TABLE IF NOT EXISTS consensus (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, symbol TEXT, side TEXT,
  n_bots INTEGER, bot_ids TEXT, outcome_pnl REAL);
CREATE TABLE IF NOT EXISTS challengers (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, bot_id TEXT, description TEXT,
  status TEXT, metrics_json TEXT);
CREATE TABLE IF NOT EXISTS bot_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, bot_id TEXT, version TEXT,
  reason TEXT, metrics_json TEXT);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, kind TEXT NOT NULL, bot_id TEXT,
  message TEXT, data_json TEXT);
"""


def _iso(ts: datetime | None) -> str:
    ts = ts or datetime.now(UTC)
    return ts.isoformat()


class TeamStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self.conn = sqlite3.connect(path, timeout=10, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._lock = threading.Lock()

    def close(self) -> None:
        self.conn.close()

    def _write(self, sql: str, args: tuple = ()) -> int | None:
        with self._lock, self.conn:
            return self.conn.execute(sql, args).lastrowid

    # bots --------------------------------------------------------------------------
    def upsert_bot(
        self,
        bot_id: str,
        name: str = "",
        repo: str = "",
        version: str = "",
        status: str = "active",
        ts: datetime | None = None,
    ) -> None:
        self._write(
            "INSERT INTO bots (bot_id, name, repo, version, status, updated_ts)"
            " VALUES (?,?,?,?,?,?) ON CONFLICT(bot_id) DO UPDATE SET name = excluded.name,"
            " repo = excluded.repo, version = excluded.version, status = excluded.status,"
            " updated_ts = excluded.updated_ts",
            (bot_id, name, repo, version, status, _iso(ts)),
        )

    def bots(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM bots ORDER BY bot_id", self.conn)

    # signals and fills ---------------------------------------------------------------
    def log_signal(
        self,
        bot_id: str,
        symbol: str,
        side: str,
        entry: float,
        stop: float,
        target: float | None = None,
        reason: str = "",
        taken: bool = False,
        ts: datetime | None = None,
    ) -> int | None:
        return self._write(
            "INSERT INTO signals (ts, bot_id, symbol, side, entry, stop, target, reason, taken)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (_iso(ts), bot_id, symbol, side, entry, stop, target, reason, int(taken)),
        )

    def log_fill(
        self,
        bot_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        fees: float = 0.0,
        internal: bool = False,
        order_id: str = "",
        ts: datetime | None = None,
    ) -> int | None:
        return self._write(
            "INSERT INTO fills (ts, bot_id, symbol, side, qty, price, fees, internal, order_id)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (_iso(ts), bot_id, symbol, side, qty, price, fees, int(internal), order_id),
        )

    # daily metrics -------------------------------------------------------------------
    def record_daily_metrics(
        self,
        day: str,
        bot_id: str,
        equity: float,
        pnl: float,
        trades: int,
        sharpe: float | None = None,
        drawdown_pct: float | None = None,
        regime: str = "",
    ) -> None:
        self._write(
            "INSERT OR REPLACE INTO bot_metrics_daily (day, bot_id, equity, pnl, trades, sharpe,"
            " drawdown_pct, regime) VALUES (?,?,?,?,?,?,?,?)",
            (day, bot_id, equity, pnl, trades, sharpe, drawdown_pct, regime),
        )

    # regimes ---------------------------------------------------------------------------
    def set_regime(
        self,
        label: str,
        vol_pctile: float | None = None,
        trend: float | None = None,
        breadth: float | None = None,
        ts: datetime | None = None,
    ) -> None:
        self._write(
            "INSERT OR REPLACE INTO regimes (ts, label, vol_pctile, trend, breadth)"
            " VALUES (?,?,?,?,?)",
            (_iso(ts), label, vol_pctile, trend, breadth),
        )

    def latest_regime(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT ts, label, vol_pctile, trend, breadth FROM regimes ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "ts": row[0],
            "label": row[1],
            "vol_pctile": row[2],
            "trend": row[3],
            "breadth": row[4],
        }

    # allocations -----------------------------------------------------------------------
    def record_allocation(
        self, bot_id: str, fraction: float, reason: str = "", ts: datetime | None = None
    ) -> None:
        self._write(
            "INSERT INTO allocations (ts, bot_id, fraction, reason) VALUES (?,?,?,?)",
            (_iso(ts), bot_id, fraction, reason),
        )

    def current_allocations(self) -> dict[str, float]:
        """The latest recorded fraction per bot."""
        rows = self.conn.execute("SELECT bot_id, fraction FROM allocations ORDER BY id").fetchall()
        return {str(r[0]): float(r[1]) for r in rows}  # later rows overwrite earlier ones

    # consensus -------------------------------------------------------------------------
    def log_consensus(
        self,
        symbol: str,
        side: str,
        n_bots: int,
        bot_ids: list[str] | str,
        outcome_pnl: float | None = None,
        ts: datetime | None = None,
    ) -> int | None:
        if not isinstance(bot_ids, str):
            bot_ids = ",".join(bot_ids)
        return self._write(
            "INSERT INTO consensus (ts, symbol, side, n_bots, bot_ids, outcome_pnl)"
            " VALUES (?,?,?,?,?,?)",
            (_iso(ts), symbol, side, n_bots, bot_ids, outcome_pnl),
        )

    # challengers -----------------------------------------------------------------------
    def add_challenger(
        self,
        bot_id: str,
        description: str,
        status: str = "proposed",
        metrics: dict[str, Any] | None = None,
        ts: datetime | None = None,
    ) -> int | None:
        return self._write(
            "INSERT INTO challengers (ts, bot_id, description, status, metrics_json)"
            " VALUES (?,?,?,?,?)",
            (_iso(ts), bot_id, description, status, json.dumps(metrics or {}, default=str)),
        )

    def update_challenger(
        self,
        challenger_id: int,
        status: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if status is not None:
            self._write("UPDATE challengers SET status = ? WHERE id = ?", (status, challenger_id))
        if metrics is not None:
            self._write(
                "UPDATE challengers SET metrics_json = ? WHERE id = ?",
                (json.dumps(metrics, default=str), challenger_id),
            )

    def challengers(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM challengers ORDER BY id", self.conn)

    # bot versions ----------------------------------------------------------------------
    def add_bot_version(
        self,
        bot_id: str,
        version: str,
        reason: str = "",
        metrics: dict[str, Any] | None = None,
        ts: datetime | None = None,
    ) -> int | None:
        return self._write(
            "INSERT INTO bot_versions (ts, bot_id, version, reason, metrics_json)"
            " VALUES (?,?,?,?,?)",
            (_iso(ts), bot_id, version, reason, json.dumps(metrics or {}, default=str)),
        )

    def versions(self, bot_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM bot_versions WHERE bot_id = ? ORDER BY id",
            self.conn,
            params=(bot_id,),
        )

    # events ----------------------------------------------------------------------------
    def log_event(
        self,
        kind: str,
        message: str = "",
        bot_id: str = "",
        data: dict[str, Any] | None = None,
        ts: datetime | None = None,
    ) -> int | None:
        return self._write(
            "INSERT INTO events (ts, kind, bot_id, message, data_json) VALUES (?,?,?,?,?)",
            (_iso(ts), kind, bot_id, message, json.dumps(data or {}, default=str)),
        )

    # read helpers ----------------------------------------------------------------------
    def events(self, limit: int = 1000) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", self.conn, params=(limit,)
        )

    def fills(self, bot_id: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM fills WHERE 1=1", []
        if bot_id is not None:
            q += " AND bot_id = ?"
            args.append(bot_id)
        return pd.read_sql_query(q + " ORDER BY id", self.conn, params=args)

    def daily_metrics(self, bot_id: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM bot_metrics_daily WHERE 1=1", []
        if bot_id is not None:
            q += " AND bot_id = ?"
            args.append(bot_id)
        return pd.read_sql_query(q + " ORDER BY day, bot_id", self.conn, params=args)
