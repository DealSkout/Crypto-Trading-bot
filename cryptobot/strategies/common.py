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


def minute_of_day(ts) -> float:
    """Minutes since 00:00 UTC (the crypto session open) of a timestamp."""
    ts = pd.Timestamp(ts).tz_convert("UTC")
    return ts.hour * 60 + ts.minute + ts.second / 60.0


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


def vwap_array(high, low, close, volume) -> np.ndarray:
    """Session VWAP as a numpy array (bars with no volume carry the running typical price)."""
    tp = (np.asarray(high, float) + np.asarray(low, float) + np.asarray(close, float)) / 3.0
    vol = np.clip(np.asarray(volume, float), 0.0, None)
    cum_vol = np.cumsum(vol)
    cum_tp = np.cumsum(tp * vol)
    fallback = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(cum_vol > 0, cum_tp / np.where(cum_vol > 0, cum_vol, 1.0), fallback)
    return out


def vwap(df: pd.DataFrame) -> pd.Series:
    out = vwap_array(df["high"], df["low"], df["close"], df["volume"])
    return pd.Series(out, index=df.index)


def minutes_to_close(bars: pd.DataFrame) -> float:
    return sessions.minutes_to_close(market_of(bars), now_of(bars))


def minutes_since_open(bars: pd.DataFrame) -> float:
    return sessions.minutes_since_open(market_of(bars), now_of(bars))


def rsi(close: pd.Series, period: int = 14) -> float | None:
    """Wilder's RSI of the last bar; None until enough history exists."""
    if len(close) < period * 3:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    last_loss = float(loss.iloc[-1])
    if last_loss == 0:
        return 100.0
    rs = float(gain.iloc[-1]) / last_loss
    return 100.0 - 100.0 / (1.0 + rs)


def adx(bars: pd.DataFrame, period: int = 14) -> float | None:
    """Wilder's ADX (trend strength, 0-100) of the last bar; None until enough history.
    Shared by the trend and mean-reversion bots and by the regime detector, so every
    consumer agrees on what 'trending' means."""
    if len(bars) < period * 3:
        return None
    high, low, close = bars["high"], bars["low"], bars["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    alpha = 1.0 / period
    atr_series = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_series
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_series
    denom = (plus_di + minus_di).replace(0, pd.NA)
    dx = ((plus_di - minus_di).abs() / denom * 100).fillna(0.0)
    value = float(dx.ewm(alpha=alpha, adjust=False).mean().iloc[-1])
    return value if pd.notna(value) else None
