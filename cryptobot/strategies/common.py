"""Small pure helpers shared by strategies. No I/O here."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from cryptobot import sessions


def market_of(bars: pd.DataFrame) -> sessions.Market:
    return sessions.market_from_tz(bars.index.tz)


def bar_minutes(bars: pd.DataFrame) -> int:
    tf = bars.attrs.get("timeframe")
    if tf:
        return int(tf)
    if len(bars) < 2:
        return 5
    diffs = np.diff(bars.index.values[-10:]).astype("timedelta64[s]").astype(float) / 60.0
    diffs = diffs[diffs > 0]
    return int(round(float(np.min(diffs)))) if len(diffs) else 5


def now_of(bars: pd.DataFrame):
    """'Now' is the END of the last completed bar."""
    return bars.index[-1] + timedelta(minutes=bar_minutes(bars))


def _today_start(bars: pd.DataFrame) -> int:
    """Position of the first bar of the last session (binary search; called on every bar)."""
    midnight = bars.index[-1].normalize()
    return int(bars.index.searchsorted(midnight))


def today_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars
    return bars.iloc[_today_start(bars) :]


def prior_session_close(bars: pd.DataFrame) -> float | None:
    if bars.empty:
        return None
    start = _today_start(bars)
    return float(bars["close"].iloc[start - 1]) if start > 0 else None


def daily_ohlc(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    g = bars.groupby(bars.index.normalize())
    return pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
        }
    )


def daily_atr(bars: pd.DataFrame, n: int = 14, exclude_today: bool = True) -> float | None:
    d = daily_ohlc(bars)
    if exclude_today and len(d):
        d = d.iloc[:-1]
    if len(d) < 2:
        return None
    prev_close = d["close"].shift(1)
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - prev_close).abs(), (d["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr = tr.dropna().tail(n)
    return float(tr.mean()) if len(tr) else None


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].clip(lower=0)
    cum_vol = vol.cumsum()
    cum_tp = (tp * vol).cumsum()
    out = cum_tp / cum_vol.replace(0, np.nan)
    return out.fillna(tp.expanding().mean())


def minutes_to_close(bars: pd.DataFrame) -> float:
    return sessions.minutes_to_close(market_of(bars), now_of(bars))


def minutes_since_open(bars: pd.DataFrame) -> float:
    return sessions.minutes_since_open(market_of(bars), now_of(bars))
