"""The AI risk agent's toolbox: an MCP server (Model Context Protocol) over stdio.

Point Claude Desktop, Claude Code or any MCP client at `python -m cryptobot mcp` and the AI
gets a small set of tools to keep an eye on the bot and the broker account it trades
(an Alpaca crypto paper or live account):

    bot_status        heartbeat, equity, open trades, halts, guardian stop, drawdown
    account_snapshot  cash, equity, positions and pending orders read from the broker,
                      reconciled against what the bot thinks it holds
    recent_trades     closed trades from the journal
    daily_results     one row per trading day
    alerts            halts, errors, rejected orders, capped stops, guardian actions
    risk_check        the guardian's rules evaluated now, without stopping anything
    stop_bot          pull the plug: write the STOPPED flag (the running bot flattens and exits)

What the agent can NOT do, by construction: place or cancel orders, resume a stopped bot,
or change any setting. The broker calls are reads only. `stop_bot` goes through the same
flag file as the guardian, so `python -m cryptobot resume` (a human) is the only way back.
Run with `--read-only` to remove `stop_bot` as well.

The protocol is JSON-RPC 2.0, one message per line on stdin/stdout, implemented here with the
standard library so the bot needs no extra dependency. Logs go to stderr; stdout carries
protocol messages only.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import IO, Any

import pandas as pd

from cryptobot.config import Config
from cryptobot.guardian import Guardian
from cryptobot.journal import Journal

log = logging.getLogger(__name__)

SERVER_NAME = "cryptobot-risk-agent"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
STALE_AFTER = 180  # seconds without a heartbeat before the bot counts as not running
ACCOUNT_CACHE_SECONDS = 15.0  # keep account reads cheap; the running bot shares the rate limit
ALERT_KINDS = [
    "halt",
    "flatten",
    "error",
    "order_rejected",
    "emergency_stop",
    "stop_capped",
    "guardian",
    "guardian_stop",
    "guardian_resume",
    "guardian_error",
    "decider_invalid",
    "decider_error",
    "coach_error",
]
MIN_REASON_CHARS = 10
ACCOUNT_BROKERS = ("alpaca",)

INSTRUCTIONS = (
    "You are the risk manager for Cryptobot, an automated day-trading bot that trades through"
    " Alpaca (crypto, 24/7, long-only). Your job is to protect capital, not to find trades. "
    "Use the read tools to check on the bot. Call stop_bot only on clear evidence: losses "
    "beyond the configured limits, positions on the account the bot does not know about, "
    "repeated errors or rejected orders, a bot trading while it should be halted, or results "
    "far outside what the backtest promised. A stop flattens every position and only a human "
    "can resume. You cannot place orders, resume the bot or change settings."
)

RISK_REVIEW_PROMPT = """Run a risk review of the trading bot.

1. Call bot_status. Note whether it is running, halted or already stopped, today's P&L and the
   drawdown from peak equity.
2. Call account_snapshot. Check that the positions at the broker match the bot's open trades,
   and that nothing is held that the bot does not know about.
3. Call alerts for the last 24 hours and risk_check.
4. Call daily_results and recent_trades if you need the longer picture.

Then give a short verdict: OK, WATCH (say what to watch) or STOP. Only call stop_bot if the
evidence clearly justifies flattening everything; give a specific reason with the numbers.
"""


class ToolError(Exception):
    """A tool failed in a way the agent should see (returned with isError: true)."""


def _now() -> datetime:
    return datetime.now(UTC)


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if df is None or not len(df):
        return []
    if limit is not None:
        df = df.tail(limit)
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _num(args: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    v = args.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ToolError(f"{key} must be a number")
    return float(min(max(v, lo), hi))


class RiskAgentServer:
    def __init__(
        self,
        cfg: Config,
        mode: str,
        read_only: bool = False,
        broker_factory: Callable[[], Any] | None = None,
        notifier=None,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.cfg = cfg
        self.mode = mode
        self.read_only = read_only
        self.broker_factory = broker_factory
        self.notifier = notifier
        self.clock = clock
        self._broker = None
        self._account_cache: tuple[float, dict[str, Any]] | None = None
        self.tools: dict[str, dict[str, Any]] = {t["name"]: t for t in self._tool_defs()}

    # ------------------------------------------------------------------ helpers
    def journal(self) -> Journal:
        return Journal(self.cfg.journal_path, mode=self.mode)

    def guardian(self, journal: Journal) -> Guardian:
        from cryptobot.guardian import backtest_metrics_for

        return Guardian(
            self.cfg,
            journal,
            notifier=self.notifier,
            backtest_metrics=backtest_metrics_for(self.cfg),
        )

    def _state(self, journal: Journal) -> tuple[dict, dict]:
        state = journal.all_state()
        engine = (state.get("engine") or {}).get("value") or {}
        hb = (state.get("heartbeat") or {}).get("value") or {}
        return engine, hb

    def _heartbeat_age(self, engine: dict, hb: dict) -> float | None:
        ts = hb.get("ts") or engine.get("ts")
        if not ts:
            return None
        try:
            return max(0.0, (self.clock() - datetime.fromisoformat(ts)).total_seconds())
        except ValueError:
            return None

    def _drawdown(self, journal: Journal, engine: dict) -> dict[str, Any]:
        peak = journal.get_state("peak_equity")
        equity = engine.get("equity")
        if peak is None or equity is None or float(peak) <= 0:
            return {"peak_equity": peak, "equity": equity, "drawdown_pct": None}
        dd = max(0.0, (float(peak) - float(equity)) / float(peak) * 100.0)
        return {
            "peak_equity": float(peak),
            "equity": float(equity),
            "drawdown_pct": round(dd, 3),
            "limit_pct": self.cfg.guardian.max_drawdown_pct,
        }

    # ------------------------------------------------------------------ tools
    def bot_status(self, args: dict[str, Any]) -> dict[str, Any]:
        from cryptobot.guardian import stopped_info

        journal = self.journal()
        try:
            engine, hb = self._state(journal)
            age = self._heartbeat_age(engine, hb)
            return {
                "now": self.clock().isoformat(),
                "mode": self.mode,
                "broker": self.cfg.broker,
                "strategy": self.cfg.strategy,
                "symbols": self.cfg.symbols,
                "running": age is not None and age <= STALE_AFTER,
                "heartbeat_age_seconds": None if age is None else round(age, 1),
                "loop_active": bool(hb.get("active")),
                "guardian_stop": stopped_info(self.cfg),
                "halted_today": bool(engine.get("halted")),
                "halt_reason": engine.get("halt_reason", ""),
                "equity": engine.get("equity"),
                "cash": engine.get("cash"),
                "day_pnl": engine.get("day_pnl"),
                "day_costs": engine.get("day_costs"),
                "day_trades": engine.get("day_trades"),
                "open_trades": engine.get("open_trades", []),
                "drawdown": self._drawdown(journal, engine),
                "limits": {
                    "risk_per_trade_pct": self.cfg.risk.risk_per_trade_pct,
                    "daily_loss_pct": self.cfg.risk.max_daily_loss_pct,
                    "max_positions": self.cfg.risk.max_positions,
                    "max_stop_pct": self.cfg.risk.max_stop_pct,
                    "max_drawdown_pct": self.cfg.guardian.max_drawdown_pct,
                    "max_loss_30d_pct": self.cfg.guardian.max_loss_30d_pct,
                },
            }
        finally:
            journal.close()

    def _broker_or_error(self):
        if self.cfg.broker not in ACCOUNT_BROKERS:
            raise ToolError(
                f"config.yaml has broker: {self.cfg.broker}. account_snapshot reads a real broker"
                " account; set broker: alpaca (paper mode uses your Alpaca paper account)."
            )
        if self._broker is None:
            if self.broker_factory is None:
                raise ToolError("no broker connection configured")
            try:
                self._broker = self.broker_factory()
            except Exception as e:
                raise ToolError(f"could not connect to {self.cfg.broker}: {e}") from e
        return self._broker

    def _read_account(self) -> dict[str, Any]:
        cached = self._account_cache
        if cached and time.monotonic() - cached[0] < ACCOUNT_CACHE_SECONDS:
            return cached[1]
        broker = self._broker_or_error()
        try:
            positions = broker.get_positions()
            pending = broker.open_orders()
            cash_free = broker.get_cash()
            equity = broker.get_equity()
        except Exception as e:
            raise ToolError(f"{self.cfg.broker} request failed: {e}") from e
        out = {
            "account": "live" if getattr(broker, "live", False) else "practice (demo)",
            "cash_free": cash_free,
            "equity": equity,
            "positions": [
                {"symbol": p.symbol, "side": p.side, "qty": p.qty, "avg_price": p.avg_price}
                for p in positions
            ],
            "pending_orders": [
                {"id": o.id, "symbol": o.symbol, "side": o.side, "qty": o.qty, "status": o.status}
                for o in pending
            ],
        }
        self._account_cache = (time.monotonic(), out)
        return out

    @staticmethod
    def reconcile(account_positions: list[dict], open_trades: list[dict]) -> list[str]:
        """Differences between what the broker holds and what the bot believes it holds."""
        held: dict[str, float] = {}
        for p in account_positions:
            sign = 1.0 if p["side"] == "long" else -1.0
            held[p["symbol"].upper()] = held.get(p["symbol"].upper(), 0.0) + sign * p["qty"]
        bot: dict[str, float] = {}
        for t in open_trades:
            sign = 1.0 if t["side"] == "long" else -1.0
            bot[t["symbol"].upper()] = bot.get(t["symbol"].upper(), 0.0) + sign * float(t["qty"])
        issues = []
        for sym in sorted(set(held) | set(bot)):
            a, b = held.get(sym, 0.0), bot.get(sym, 0.0)
            if abs(a - b) > max(1e-6, 0.01 * max(abs(a), abs(b))):
                if b == 0:
                    issues.append(f"{sym}: {a:g} held at the broker but the bot has no open trade")
                elif a == 0:
                    issues.append(f"{sym}: the bot thinks it holds {b:g} but the broker shows none")
                else:
                    issues.append(f"{sym}: the broker holds {a:g}, the bot thinks {b:g}")
        return issues

    def account_snapshot(self, args: dict[str, Any]) -> dict[str, Any]:
        account = self._read_account()
        journal = self.journal()
        try:
            engine, _ = self._state(journal)
        finally:
            journal.close()
        out = dict(account)
        out["mismatches"] = self.reconcile(account["positions"], engine.get("open_trades", []))
        out["note"] = (
            "Positions outside the bot (bought by hand in the app) also show up as mismatches; "
            "use a separate account for the bot to keep this check clean."
        )
        return out

    def recent_trades(self, args: dict[str, Any]) -> dict[str, Any]:
        days = _num(args, "days", 7, 1, 365)
        limit = int(_num(args, "limit", 50, 1, 500))
        journal = self.journal()
        try:
            df = journal.trades(since=self.clock() - timedelta(days=days), mode=self.mode)
            rows = _records(df, limit)
        finally:
            journal.close()
        pnl = [float(r.get("pnl") or 0.0) for r in rows]
        return {
            "days": days,
            "count": len(rows),
            "net_pnl": round(sum(pnl), 2),
            "win_rate": round(sum(1 for p in pnl if p > 0) / len(pnl), 3) if pnl else None,
            "trades": rows,
        }

    def daily_results(self, args: dict[str, Any]) -> dict[str, Any]:
        days = _num(args, "days", 30, 1, 365)
        journal = self.journal()
        try:
            df = journal.daily(since=(self.clock() - timedelta(days=days)).date())
            rows = _records(df)
        finally:
            journal.close()
        return {
            "days": days,
            "count": len(rows),
            "net_pnl": round(sum(float(r.get("pnl") or 0.0) for r in rows), 2),
            "halts": sum(1 for r in rows if r.get("halted")),
            "rows": rows,
        }

    def alerts(self, args: dict[str, Any]) -> dict[str, Any]:
        hours = _num(args, "hours", 24, 1, 24 * 30)
        journal = self.journal()
        try:
            df = journal.events(
                since=self.clock() - timedelta(hours=hours), kinds=ALERT_KINDS, limit=200
            )
            rows = _records(df)
        finally:
            journal.close()
        counts: dict[str, int] = {}
        for r in rows:
            counts[r.get("kind", "?")] = counts.get(r.get("kind", "?"), 0) + 1
        return {"hours": hours, "counts": counts, "events": rows}

    def risk_check(self, args: dict[str, Any]) -> dict[str, Any]:
        journal = self.journal()
        try:
            g = self.guardian(journal)
            findings = list(g.rule_reasons(self.clock()))
            engine, hb = self._state(journal)
            dd = self._drawdown(journal, engine)
            if dd["drawdown_pct"] is not None and dd["drawdown_pct"] >= dd["limit_pct"]:
                findings.append(
                    f"drawdown {dd['drawdown_pct']:.2f}% from peak exceeds {dd['limit_pct']}%"
                )
            age = self._heartbeat_age(engine, hb)
            warnings = []
            if engine.get("open_trades") and (age is None or age > STALE_AFTER):
                warnings.append(
                    "the bot holds open trades but its heartbeat is stale: it may have crashed"
                    " with positions open (its software stops are not being watched)"
                )
            stopped = g.stop_info()
        finally:
            journal.close()
        if args.get("include_account", True) and self.cfg.broker in ACCOUNT_BROKERS:
            try:
                acct = self._read_account()
                mism = self.reconcile(acct["positions"], engine.get("open_trades", []))
                warnings.extend(f"position mismatch: {m}" for m in mism)
            except ToolError as e:
                warnings.append(f"could not read the {self.cfg.broker} account: {e}")
        return {
            "already_stopped": stopped,
            "rule_findings": findings,
            "warnings": warnings,
            "guardian_would_stop": bool(findings),
            "note": "Nothing was stopped. Call stop_bot if you judge a stop is warranted.",
        }

    def stop_bot(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.read_only:
            raise ToolError("this server runs with --read-only; stop_bot is disabled")
        reason = str(args.get("reason") or "").strip()
        if len(reason) < MIN_REASON_CHARS:
            raise ToolError("give a specific reason (at least 10 characters) for stopping")
        journal = self.journal()
        try:
            g = self.guardian(journal)
            if g.stopped:
                return {"stopped": True, "already": True, "info": g.stop_info()}
            g.stop(reason, "risk-agent (MCP)", self.clock())
            info = g.stop_info()
        finally:
            journal.close()
        return {
            "stopped": True,
            "already": False,
            "info": info,
            "what_happens": (
                "The running bot sees the flag on its next step, flattens every position and "
                "exits. paper and live refuse to start until a human runs "
                "`python -m cryptobot resume`."
            ),
        }

    # ------------------------------------------------------------------ MCP plumbing
    def _tool_defs(self) -> list[dict[str, Any]]:
        ro = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
        defs = [
            {
                "name": "bot_status",
                "description": "Is the bot running, halted or stopped? Equity, today's P&L, "
                "open trades with stops, drawdown from peak, and the configured limits.",
                "inputSchema": {"type": "object", "properties": {}},
                "annotations": ro,
                "handler": self.bot_status,
            },
            {
                "name": "account_snapshot",
                "description": "Read the broker account (cash, equity, positions, pending "
                "orders) and list mismatches against the bot's open trades. Read-only.",
                "inputSchema": {"type": "object", "properties": {}},
                "annotations": {**ro, "openWorldHint": True},
                "handler": self.account_snapshot,
            },
            {
                "name": "recent_trades",
                "description": "Closed trades from the journal with net P&L and win rate.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "days": {"type": "number", "description": "look-back, default 7"},
                        "limit": {"type": "number", "description": "max rows, default 50"},
                    },
                },
                "annotations": ro,
                "handler": self.recent_trades,
            },
            {
                "name": "daily_results",
                "description": "One row per trading day: P&L, costs, trades, halts.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "days": {"type": "number", "description": "look-back, default 30"}
                    },
                },
                "annotations": ro,
                "handler": self.daily_results,
            },
            {
                "name": "alerts",
                "description": "Recent halts, errors, rejected orders, capped stops and "
                "guardian actions, with counts by kind.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "hours": {"type": "number", "description": "look-back, default 24"}
                    },
                },
                "annotations": ro,
                "handler": self.alerts,
            },
            {
                "name": "risk_check",
                "description": "Evaluate the guardian's stop rules now (30-day loss, losing "
                "streak, halts, errors, R versus backtest, drawdown), plus a stale-heartbeat "
                "and broker position check. Reports only; stops nothing.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_account": {
                            "type": "boolean",
                            "description": "also reconcile against the broker (default true)",
                        }
                    },
                },
                "annotations": {**ro, "openWorldHint": True},
                "handler": self.risk_check,
            },
        ]
        if not self.read_only:
            defs.append(
                {
                    "name": "stop_bot",
                    "description": "Stop the bot completely: it flattens every position and "
                    "exits, and will not trade again until a human runs `python -m cryptobot "
                    "resume`. Use only on clear evidence; give a specific reason with numbers.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "why, with the numbers that justify it",
                            }
                        },
                        "required": ["reason"],
                    },
                    "annotations": {
                        "readOnlyHint": False,
                        "destructiveHint": True,
                        "idempotentHint": True,
                        "openWorldHint": False,
                    },
                    "handler": self.stop_bot,
                }
            )
        return defs

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """Handle one JSON-RPC message; returns the response, or None for notifications."""
        method = msg.get("method")
        mid = msg.get("id")
        is_request = "id" in msg
        params = msg.get("params") or {}
        try:
            if method == "initialize":
                asked = str(params.get("protocolVersion", ""))
                result = {
                    "protocolVersion": asked
                    if asked in PROTOCOL_VERSIONS
                    else PROTOCOL_VERSIONS[0],
                    "capabilities": {"tools": {}, "prompts": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "instructions": INSTRUCTIONS,
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {
                    "tools": [
                        {k: v for k, v in t.items() if k != "handler"} for t in self.tools.values()
                    ]
                }
            elif method == "tools/call":
                result = self.call_tool(str(params.get("name", "")), params.get("arguments") or {})
            elif method == "prompts/list":
                result = {
                    "prompts": [
                        {
                            "name": "risk_review",
                            "description": "Check the bot and the broker account and give "
                            "an OK / WATCH / STOP verdict.",
                        }
                    ]
                }
            elif method == "prompts/get":
                if params.get("name") != "risk_review":
                    return _error(mid, -32602, f"unknown prompt {params.get('name')!r}")
                result = {
                    "description": "Risk review of the trading bot",
                    "messages": [
                        {"role": "user", "content": {"type": "text", "text": RISK_REVIEW_PROMPT}}
                    ],
                }
            elif not is_request:
                return None  # notifications (initialized, cancelled, ...) need no answer
            else:
                return _error(mid, -32601, f"method not found: {method}")
        except Exception as e:  # pragma: no cover - last line of defence for the protocol
            log.exception("internal error handling %s", method)
            return _error(mid, -32603, f"internal error: {e}")
        if not is_request:
            return None
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            return _tool_result({"error": f"unknown tool {name!r}"}, is_error=True)
        if not isinstance(arguments, dict):
            return _tool_result({"error": "arguments must be an object"}, is_error=True)
        try:
            out = tool["handler"](arguments)
        except ToolError as e:
            return _tool_result({"error": str(e)}, is_error=True)
        except Exception as e:
            log.exception("tool %s failed", name)
            return _tool_result({"error": f"{type(e).__name__}: {e}"}, is_error=True)
        log.info("tool %s called", name)
        return _tool_result(out)

    def serve(self, stdin: IO[str] | None = None, stdout: IO[str] | None = None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                _write(stdout, _error(None, -32700, "parse error"))
                continue
            batch = msg if isinstance(msg, list) else [msg]
            replies = []
            for m in batch:
                if not isinstance(m, dict):
                    replies.append(_error(None, -32600, "invalid request"))
                    continue
                r = self.handle(m)
                if r is not None:
                    replies.append(r)
            if replies:
                _write(stdout, replies if isinstance(msg, list) else replies[0])


def _tool_result(payload: dict[str, Any], is_error: bool = False) -> dict[str, Any]:
    text = json.dumps(payload, indent=1, default=str)
    out: dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": is_error}
    if not is_error:
        out["structuredContent"] = json.loads(text)
    return out


def _error(mid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def _write(stdout: IO[str], obj: Any) -> None:
    stdout.write(json.dumps(obj, default=str) + "\n")
    stdout.flush()
