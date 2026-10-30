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
CREATE TABLE IF NOT EXISTS real_trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT, bot_id TEXT, symbol TEXT, lot TEXT, entry_ts TEXT,
  exit_ts TEXT, qty REAL, entry_price REAL, exit_price REAL, signal_price REAL, stop REAL,
  target REAL, reason TEXT, gross_pnl REAL, fees REAL, slippage REAL, net_pnl REAL,
  planned_r REAL, actual_r REAL);
CREATE TABLE IF NOT EXISTS bot_state (
  bot_id TEXT PRIMARY KEY, state_json TEXT NOT NULL, updated_ts TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS team_state (
  key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_ts TEXT NOT NULL);
"""

# columns added after the first release; applied with ALTER TABLE when missing
EXTRA_COLUMNS = {
    "consensus": {"entry_mark": "REAL"},
    # daily returns come from equity snapshots: (end - start - transfers) / start, so capital
    # the coordinator moves in or out is never counted as profit or loss
    "bot_metrics_daily": {"start_equity": "REAL", "transfers": "REAL", "realized_pnl": "REAL"},
}


def _iso(ts: datetime | None) -> str:
    ts = ts or datetime.now(UTC)
    return ts.isoformat()


class TeamStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self.conn = sqlite3.connect(path, timeout=10, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        for table, columns in EXTRA_COLUMNS.items():
            have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for name, kind in columns.items():
                if name not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
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
        start_equity: float | None = None,
        transfers: float = 0.0,
        realized_pnl: float | None = None,
    ) -> None:
        """`equity` is the day's closing equity and `pnl` its equity change net of
        `transfers`; `realized_pnl` is what closed trades booked that day."""
        self._write(
            "INSERT OR REPLACE INTO bot_metrics_daily (day, bot_id, equity, pnl, trades, sharpe,"
            " drawdown_pct, regime, start_equity, transfers, realized_pnl)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                day,
                bot_id,
                equity,
                pnl,
                trades,
                sharpe,
                drawdown_pct,
                regime,
                start_equity,
                transfers,
                realized_pnl,
            ),
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
        entry_mark: float | None = None,
        ts: datetime | None = None,
    ) -> int | None:
        """`entry_mark` is the price when the bots agreed, so the outcome can be scored
        later without re-reading bars."""
        if not isinstance(bot_ids, str):
            bot_ids = ",".join(bot_ids)
        return self._write(
            "INSERT INTO consensus (ts, symbol, side, n_bots, bot_ids, outcome_pnl, entry_mark)"
            " VALUES (?,?,?,?,?,?,?)",
            (_iso(ts), symbol, side, n_bots, bot_ids, outcome_pnl, entry_mark),
        )

    def unscored_consensus(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, ts, symbol, side, entry_mark FROM consensus WHERE outcome_pnl IS NULL"
            " ORDER BY id"
        ).fetchall()
        return [
            {"id": r[0], "ts": r[1], "symbol": r[2], "side": r[3], "entry_mark": r[4]} for r in rows
        ]

    def score_consensus(self, consensus_id: int, outcome_pnl: float) -> None:
        self._write(
            "UPDATE consensus SET outcome_pnl = ? WHERE id = ?", (float(outcome_pnl), consensus_id)
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

    # real trades (one per closed lot on the real account) ----------------------------------
    def log_real_trade(self, r: dict[str, Any]) -> int | None:
        """gross_pnl is the P&L at the PLANNED prices (signal price to the planned exit
        level), fees and slippage are what execution cost, net_pnl is what the account got:
        net = gross - fees - slippage."""
        return self._write(
            "INSERT INTO real_trades (bot_id, symbol, lot, entry_ts, exit_ts, qty, entry_price,"
            " exit_price, signal_price, stop, target, reason, gross_pnl, fees, slippage, net_pnl,"
            " planned_r, actual_r) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["bot_id"],
                r["symbol"],
                r.get("lot", ""),
                _iso(r.get("entry_ts")) if r.get("entry_ts") else None,
                _iso(r.get("exit_ts")) if r.get("exit_ts") else None,
                r.get("qty"),
                r.get("entry_price"),
                r.get("exit_price"),
                r.get("signal_price"),
                r.get("stop"),
                r.get("target"),
                r.get("reason", ""),
                r.get("gross_pnl"),
                r.get("fees"),
                r.get("slippage"),
                r.get("net_pnl"),
                r.get("planned_r"),
                r.get("actual_r"),
            ),
        )

    def real_trades(self, bot_id: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM real_trades WHERE 1=1", []
        if bot_id is not None:
            q += " AND bot_id = ?"
            args.append(bot_id)
        return pd.read_sql_query(q + " ORDER BY id", self.conn, params=args)

    # runtime state (survives a restart) -------------------------------------------------
    def save_bot_state(
        self, bot_id: str, state: dict[str, Any], ts: datetime | None = None
    ) -> None:
        self._write(
            "INSERT INTO bot_state (bot_id, state_json, updated_ts) VALUES (?,?,?)"
            " ON CONFLICT(bot_id) DO UPDATE SET state_json = excluded.state_json,"
            " updated_ts = excluded.updated_ts",
            (bot_id, json.dumps(state, default=str), _iso(ts)),
        )

    def bot_state(self, bot_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT state_json FROM bot_state WHERE bot_id = ?", (bot_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set_team_state(self, key: str, value: Any, ts: datetime | None = None) -> None:
        self._write(
            "INSERT INTO team_state (key, value_json, updated_ts) VALUES (?,?,?)"
            " ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json,"
            " updated_ts = excluded.updated_ts",
            (key, json.dumps(value, default=str), _iso(ts)),
        )

    def team_state(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute(
            "SELECT value_json FROM team_state WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else default

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
