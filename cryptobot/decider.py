"""Claude (or a local model) inside the trading loop. OFF by default.

This is the one place an LLM can make a buying or selling decision. It exists because the
owner asked for it with eyes open: Rule 2 is suspended only when `decider.provider` is set.
What does NOT change: sizing, the daily loss stop, max positions, the end-of-day flatten,
software stops and the cost model all stay in code. The model may only answer
buy / sell / close / hold with a stop and target inside bounds; every answer is validated,
clamped and journaled, and any error, timeout, refusal or over-budget call means "hold".

Roles (decider.role):
  last_resort  the rule-based strategy decides when it has a signal; the model is asked only
               when the rules have nothing, at most every `every_minutes`
  final_say    the model approves, rejects or adjusts every rule-based signal; it never
               initiates a trade on its own
  primary      the model is asked every `every_minutes` whether to buy, sell, close or hold;
               the rule-based signal is shown to it as advice
This module never imports the brokers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from cryptobot.config import DeciderConfig, RiskConfig
from cryptobot.journal import Journal
from cryptobot.strategies import Signal
from cryptobot.strategies.common import daily_atr, daily_ohlc, today_bars, vwap

log = logging.getLogger(__name__)

ACTIONS = ("buy", "sell", "close", "hold")
ROLES = ("last_resort", "final_say", "primary")

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "stop_price": {"type": "number"},
        "target_price": {"type": "number"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["action", "stop_price", "target_price", "confidence", "reason"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are the trade decider for Cryptobot, a small crypto day-trading bot
(Alpaca, USD pairs, long-only: you may buy or close, never short; trading day = UTC day). You get
one pair's bars for today, recent daily bars, the account state, the
rule-based strategy's current signal (if any) and the risk limits. Decide ONE action for right
now:
- buy: open a long at the market (only when flat)
- sell: close an open long; when flat, open a short (only if shorts are allowed)
- close: exit the open position at the market
- hold: do nothing

Rules:
- stop_price must be on the correct side of the current price and no further away than the
  stated maximum distance; it is enforced by code as a software stop.
- target_price 0 means no target; the position is flattened before the close regardless.
- confidence is 0 to 1. Anything under the stated minimum is treated as hold.
- Costs are real. A round trip costs the stated percentage of the position. Only act when the
  expected move clearly exceeds it. When in doubt, hold. Keep reason under 25 words."""


@dataclass(frozen=True)
class DecisionContext:
    symbol: str
    price: float
    open_side: str | None
    text: str


@dataclass
class Decision:
    action: str
    stop_price: float
    target_price: float
    confidence: float
    reason: str
    cost_usd: float = 0.0
    model: str = ""

    @property
    def target(self) -> float | None:
        return self.target_price if self.target_price > 0 else None


def build_context(
    symbol: str,
    bars: pd.DataFrame,
    equity: float,
    cash: float,
    open_trade: Any,
    strategy_signal: Signal | None,
    risk: RiskConfig,
    allow_short: bool,
    round_trip_pct: float,
    cfg: DeciderConfig,
    minutes_to_close: float,
    day_pnl: float,
    trades_today: int,
    strategy_name: str,
) -> DecisionContext:
    today = today_bars(bars)
    price = float(today["close"].iloc[-1])
    daily = daily_ohlc(bars).tail(11)
    prior = daily.iloc[:-1].tail(10)
    prior_close = float(prior["close"].iloc[-1]) if len(prior) else None
    atr = daily_atr(bars, 14)
    v = vwap(today)
    lines = [
        f"Symbol {symbol}; bar size {bars.attrs.get('timeframe', 5)} minutes; last bar ended"
        f" {bars.index[-1].strftime('%Y-%m-%d %H:%M %Z')}; {minutes_to_close:.0f} minutes to the"
        f" close; role {cfg.role}.",
        f"Price {price:.4f}; session VWAP {float(v.iloc[-1]):.4f}; opening range"
        f" {float(today['high'].iloc[0]):.4f}/{float(today['low'].iloc[0]):.4f};"
        + (
            f" prior close {prior_close:.4f}; today {(price / prior_close - 1) * 100:+.2f}%;"
            if prior_close
            else ""
        )
        + (f" daily ATR(14) {atr:.4f}." if atr else ""),
        "",
        "Previous sessions (date open high low close):",
    ]
    lines += [
        f"  {d.date()} {r.open:.2f} {r.high:.2f} {r.low:.2f} {r.close:.2f}"
        for d, r in prior.iterrows()
    ] or ["  none"]
    lines += ["", "Today's bars (time open high low close volume):"]
    lines += [
        f"  {t.strftime('%H:%M')} {r.open:.2f} {r.high:.2f} {r.low:.2f} {r.close:.2f}"
        f" {int(r.volume)}"
        for t, r in today.iterrows()
    ]
    if open_trade is not None:
        upnl = (
            (price - open_trade.entry_price)
            * open_trade.qty
            * (1 if open_trade.side == "long" else -1)
        )
        lines += [
            "",
            f"Open position: {open_trade.side} {open_trade.qty:g} @"
            f" {open_trade.entry_price:.4f}, stop {open_trade.stop:.4f}, target"
            f" {open_trade.target if open_trade.target is None else round(open_trade.target, 4)},"
            f" unrealised {upnl:+.2f}",
        ]
    else:
        lines += ["", "Open position: none (flat)"]
    if strategy_signal is not None:
        s = strategy_signal
        lines.append(
            f"Rule-based strategy ({strategy_name}) signal now: {s.side} entry"
            f" {s.entry:.4f} stop {s.stop:.4f} target"
            f" {s.target if s.target is None else round(s.target, 4)} ({s.reason})"
        )
    else:
        lines.append(f"Rule-based strategy ({strategy_name}) signal now: none")
    lines += [
        f"Account: equity {equity:.2f}, cash {cash:.2f}, day pnl {day_pnl:+.2f}, trades today"
        f" {trades_today}.",
        f"Risk limits (enforced in code): {risk.risk_per_trade_pct}% of equity per trade,"
        f" daily loss stop {risk.max_daily_loss_pct}%, max {risk.max_positions} positions,"
        f" flatten {risk.flatten_minutes_before_close} min before the close;"
        f" shorts {'allowed' if allow_short else 'NOT allowed'}.",
        f"Your stop may be at most {cfg.max_stop_pct}% from the price. Minimum confidence to act:"
        f" {cfg.min_confidence}. A round trip costs about {round_trip_pct:.2f}% of the position.",
    ]
    open_side = open_trade.side if open_trade is not None else None
    return DecisionContext(symbol, price, open_side, "\n".join(lines))


class Decider:
    def __init__(self, cfg: DeciderConfig, llm, journal: Journal, name: str = "claude") -> None:
        if cfg.role not in ROLES:
            raise ValueError(f"decider.role must be one of {ROLES}")
        self.cfg = cfg
        self.llm = llm
        self.journal = journal
        self.name = name
        self.last_call: dict[str, datetime] = {}
        self.calls = 0
        self.spend = 0.0
        self._cap_logged: date | None = None

    @property
    def role(self) -> str:
        return self.cfg.role

    def describe(self) -> str:
        model = self.cfg.model if self.cfg.provider == "claude" else self.cfg.ollama_model
        return f"{self.cfg.provider}:{model}:{self.cfg.role}"

    def due(self, symbol: str, ts: datetime) -> bool:
        last = self.last_call.get(symbol)
        return last is None or (ts - last) >= timedelta(minutes=self.cfg.every_minutes)

    def decide(self, ctx: DecisionContext, ts: datetime) -> Decision | None:
        """Ask the model. None means 'hold' for a reason that is not the model's (budget, error)."""
        self.last_call[ctx.symbol] = ts
        day = ts.date()
        spent = self.journal.llm_spend_today("decider", day)
        if spent >= self.cfg.max_daily_cost_usd:
            if self._cap_logged != day:
                self._cap_logged = day
                self.journal.log(
                    "decider",
                    f"daily cost cap reached ({spent:.2f} USD): holding",
                    symbol=ctx.symbol,
                    ts=ts,
                )
            return None
        try:
            res = self.llm.complete(
                SYSTEM_PROMPT, ctx.text, DECISION_SCHEMA, effort=self.cfg.effort
            )
        except Exception as e:  # timeouts, API errors, bad JSON: all mean hold
            self.journal.log("decider_error", repr(e), symbol=ctx.symbol, ts=ts)
            log.warning("decider error for %s: %r", ctx.symbol, e)
            return None
        self.calls += 1
        self.spend += res.cost_usd
        self.journal.add_llm_spend("decider", res.cost_usd, day)
        if res.refused:
            self.journal.log("decider", "model declined: holding", symbol=ctx.symbol, ts=ts)
            return None
        d = self.validate(res.data, ctx)
        d.cost_usd, d.model = res.cost_usd, res.model
        self.journal.log(
            "decider",
            f"{d.action} conf={d.confidence:.2f} stop={d.stop_price:.4f}"
            f" target={d.target_price:.4f}: {d.reason}",
            symbol=ctx.symbol,
            data={"cost_usd": res.cost_usd, "model": res.model},
            ts=ts,
        )
        return d

    def validate(self, data: dict[str, Any], ctx: DecisionContext) -> Decision:
        action = str(data.get("action", "hold")).lower()
        if action not in ACTIONS:
            action = "hold"
        try:
            conf = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
            stop = float(data.get("stop_price") or 0.0)
            target = float(data.get("target_price") or 0.0)
        except (TypeError, ValueError):
            return Decision("hold", 0.0, 0.0, 0.0, "unreadable numbers in the answer")
        reason = str(data.get("reason", ""))[:200]
        if action != "hold" and conf < self.cfg.min_confidence:
            return Decision(
                "hold",
                stop,
                target,
                conf,
                f"confidence {conf:.2f} below minimum {self.cfg.min_confidence}",
            )
        opening = (action == "buy" and ctx.open_side is None) or (
            action == "sell" and ctx.open_side is None
        )
        if opening:
            side = "long" if action == "buy" else "short"
            max_dist = ctx.price * self.cfg.max_stop_pct / 100.0
            if side == "long":
                if not (0 < stop < ctx.price):
                    return Decision("hold", stop, target, conf, "stop not below price")
                stop = max(stop, ctx.price - max_dist)
                if target and target <= ctx.price:
                    target = 0.0
            else:
                if not (stop > ctx.price):
                    return Decision("hold", stop, target, conf, "stop not above price")
                stop = min(stop, ctx.price + max_dist)
                if target and target >= ctx.price:
                    target = 0.0
        return Decision(action, stop, target, conf, reason)
