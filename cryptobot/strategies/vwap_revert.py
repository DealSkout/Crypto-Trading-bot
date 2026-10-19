"""VWAP mean reversion on liquid ETFs.

When price stretches `z_entry` standard deviations below the session VWAP (deviation measured
over the last `lookback_bars`), buy and target VWAP; stop `stop_sd` deviations below the
entry. Mirror for shorts when allowed. No entries in the first `min_minutes_after_open` or the last
`no_entry_minutes_before_close` minutes, when VWAP is least meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptobot.strategies import Signal
from cryptobot.strategies.common import (
    minute_of_day,
    minutes_since_open,
    minutes_to_close,
    today_bars,
    vwap_array,
)

DEFAULT_PARAMS = {
    "z_entry": 2.0,
    "stop_sd": 1.5,
    "lookback_bars": 20,
    "min_minutes_after_open": 30,
    "no_entry_minutes_before_close": 45,
}
PARAM_GRID = {"z_entry": [1.5, 2.0, 2.5], "stop_sd": [1.0, 2.0]}


def due(symbol: str, start, tf: int, params: dict) -> bool:
    """True inside the entry window (not too soon after 00:00 UTC, not too close to the end)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    end = minute_of_day(start) + tf
    return end >= float(p["min_minutes_after_open"]) and (1440.0 - end) >= float(
        p["no_entry_minutes_before_close"]
    )


def signal(bars: pd.DataFrame, params: dict) -> Signal | None:
    p = {**DEFAULT_PARAMS, **(params or {})}
    if minutes_since_open(bars) < float(p["min_minutes_after_open"]):
        return None
    if minutes_to_close(bars) < float(p["no_entry_minutes_before_close"]):
        return None
    today = today_bars(bars)
    lb = int(p["lookback_bars"])
    if len(today) < lb:
        return None
    arr = today.to_numpy(dtype=float)  # one block: far cheaper than four column lookups
    col = {c: i for i, c in enumerate(today.columns)}
    closes = arr[:, col["close"]]
    v = vwap_array(arr[:, col["high"]], arr[:, col["low"]], closes, arr[:, col["volume"]])
    dev = closes - v
    sd = float(np.std(dev[-lb:]))
    if not sd or sd != sd:  # zero or NaN
        return None
    z = float(dev[-1]) / sd
    entry = float(closes[-1])
    vwap_now = float(v[-1])
    stop_dist = float(p["stop_sd"]) * sd
    if z <= -float(p["z_entry"]) and vwap_now > entry:
        return Signal("long", entry, entry - stop_dist, vwap_now, reason=f"VWAP revert z={z:.2f}")
    if z >= float(p["z_entry"]) and vwap_now < entry:
        return Signal("short", entry, entry + stop_dist, vwap_now, reason=f"VWAP revert z={z:.2f}")
    return None
