"""Market intraday momentum, after Gao, Han, Li & Zhou (2018).

The first `opening_minutes` return of the day (vs the prior close) predicts the last
`entry_minutes_before_close` return. Enter in that direction when the final window starts and
hold into the close (the end-of-day flatten in risk.py does the exit). The stop is a plain
percentage because the paper's position has no stop at all; a bot needs one.
"""

from __future__ import annotations

import pandas as pd

from cryptobot.strategies import Signal
from cryptobot.strategies.common import (
    bar_minutes,
    minute_of_day,
    minutes_to_close,
    prior_session_close,
    today_bars,
)

DEFAULT_PARAMS = {
    "opening_minutes": 30,
    "entry_minutes_before_close": 30,
    "stop_pct": 0.5,
    "min_abs_return_pct": 0.0,
}
PARAM_GRID = {"opening_minutes": [30, 60], "stop_pct": [0.3, 0.6]}


def due(symbol: str, start, tf: int, params: dict) -> bool:
    """True only on the bar whose end opens the final entry window."""
    window = float({**DEFAULT_PARAMS, **(params or {})}["entry_minutes_before_close"])
    end_mtc = 1440.0 - (minute_of_day(start) + tf)
    return end_mtc <= window < end_mtc + tf


def signal(bars: pd.DataFrame, params: dict) -> Signal | None:
    p = {**DEFAULT_PARAMS, **(params or {})}
    today = today_bars(bars)
    tf = bar_minutes(bars)
    n_open = max(1, round(int(p["opening_minutes"]) / tf))
    if len(today) < n_open + 1:
        return None
    mtc_now = minutes_to_close(bars)
    mtc_prev = mtc_now + tf
    entry_window = float(p["entry_minutes_before_close"])
    if not (mtc_now <= entry_window < mtc_prev):  # fire once, on the bar that opens the window
        return None
    ref = prior_session_close(bars)
    if ref is None:
        ref = float(today["open"].iloc[0])
    opening_ret = float(today["close"].iloc[n_open - 1]) / ref - 1.0
    if abs(opening_ret) * 100.0 < float(p["min_abs_return_pct"]) or opening_ret == 0.0:
        return None
    side = "long" if opening_ret > 0 else "short"
    entry = float(today["close"].iloc[-1])
    stop_dist = entry * float(p["stop_pct"]) / 100.0
    stop = entry - stop_dist if side == "long" else entry + stop_dist
    return Signal(side, entry, stop, None, reason=f"intraday momentum {opening_ret * 100:+.2f}%")
