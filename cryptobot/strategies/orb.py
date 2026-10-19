"""Opening Range Breakout, after Zarattini & Aziz (2023).

Rules: take the direction of the first `range_minutes` of the session. Enter at the market when
that range completes: on the bar whose end time reaches the range end (timestamps, not a bar
count, because Alpaca leaves out 5-minute bars in which nothing traded). Stop at the opposite end
of the range, widened to at least
`atr_floor_mult` x daily ATR(`atr_days`) so tiny ranges do not produce absurd position sizes.
Target at `target_r` times the stop distance. Anything still open is flattened before the close
by risk.py (not by the strategy).
"""

from __future__ import annotations

import pandas as pd

from cryptobot.strategies import Signal
from cryptobot.strategies.common import bar_minutes, daily_atr, minute_of_day, today_bars

DEFAULT_PARAMS = {"range_minutes": 5, "target_r": 10, "atr_floor_mult": 0.1, "atr_days": 14}
PARAM_GRID = {"range_minutes": [5, 15, 30], "target_r": [3, 10]}


def due(symbol: str, start, tf: int, params: dict) -> bool:
    """True only on the bar that contains the end of the opening range."""
    range_end = float({**DEFAULT_PARAMS, **(params or {})}["range_minutes"])
    m = minute_of_day(start)
    return m < range_end <= m + tf


def signal(bars: pd.DataFrame, params: dict) -> Signal | None:
    p = {**DEFAULT_PARAMS, **(params or {})}
    if bars.empty or not due("", bars.index[-1], bar_minutes(bars), p):
        return None  # fire exactly once, on the bar that completes the opening range
    rng = today_bars(bars)
    first_open = float(rng["open"].iloc[0])
    last_close = float(rng["close"].iloc[-1])
    if last_close == first_open:
        return None
    side = "long" if last_close > first_open else "short"
    entry = last_close
    dist = entry - float(rng["low"].min()) if side == "long" else float(rng["high"].max()) - entry
    atr = daily_atr(bars, int(p["atr_days"]))
    if atr:
        dist = max(dist, float(p["atr_floor_mult"]) * atr)
    if dist <= 0:
        return None
    target_r = float(p["target_r"])
    if side == "long":
        stop, target = entry - dist, entry + target_r * dist
    else:
        stop, target = entry + dist, entry - target_r * dist
    return Signal(side, entry, stop, target, reason=f"ORB {p['range_minutes']}m {side}")
