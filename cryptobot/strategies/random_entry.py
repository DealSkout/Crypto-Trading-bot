"""Benchmark: random entry time, same exits (stop, optional target, end-of-day flatten).

Deterministic given (symbol, date, seed) so reports are reproducible. If a strategy does not
beat this after costs, its entries carry no information.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from cryptobot.strategies import Signal
from cryptobot.strategies.common import bar_minutes, minute_of_day, today_bars

DEFAULT_PARAMS = {
    "seed": 0,
    "stop_pct": 0.5,
    "target_r": None,
    "earliest_minutes_after_open": 5,
    "latest_minutes_before_close": 30,
    "side": "long",
}


def _pick(symbol: str, day: str, seed: int, lo: int, hi: int) -> int:
    h = hashlib.sha256(f"{symbol}|{day}|{seed}".encode()).hexdigest()
    return lo + int(h[:8], 16) % max(1, hi - lo + 1)


def _slot(symbol: str, start, tf: int, p: dict) -> int:
    """The bar number (1-based, counted by time from 00:00 UTC) picked for this day."""
    n_session = max(1, 1440 // tf)
    lo = max(1, int(p["earliest_minutes_after_open"]) // tf)
    hi = max(lo, n_session - int(p["latest_minutes_before_close"]) // tf)
    day = str(pd.Timestamp(start).tz_convert("UTC").date())
    return _pick(symbol, day, int(p["seed"]), lo, hi)


def due(symbol: str, start, tf: int, params: dict) -> bool:
    p = {**DEFAULT_PARAMS, **(params or {})}
    return int(minute_of_day(start) // tf) + 1 == _slot(symbol, start, tf, p)


def signal(bars: pd.DataFrame, params: dict) -> Signal | None:
    p = {**DEFAULT_PARAMS, **(params or {})}
    if bars.empty:
        return None
    tf = bar_minutes(bars)
    symbol = bars.attrs.get("symbol", "?")
    if not due(symbol, bars.index[-1], tf, p):
        return None
    today = today_bars(bars)
    side = str(p["side"])
    entry = float(today["close"].iloc[-1])
    dist = entry * float(p["stop_pct"]) / 100.0
    if dist <= 0:
        return None
    target_r = p["target_r"]
    if side == "long":
        stop = entry - dist
        target = entry + float(target_r) * dist if target_r else None
    else:
        stop = entry + dist
        target = entry - float(target_r) * dist if target_r else None
    return Signal(side, entry, stop, target, reason="random entry")
