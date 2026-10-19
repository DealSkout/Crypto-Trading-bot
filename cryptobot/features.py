"""Features at decision time: what the bot knew when it decided, and nothing later.

Every number here is computed from the bar window the engine passes in (completed bars only,
ending with the bar that just closed) plus the context providers (regime, research feed), which
must themselves only use data published before the decision time. The engine stores the result
with every decision so any past decision can be replayed and studied.

Pure functions, numpy only: they run once per decision, not once per bar.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

FEATURE_VERSION = 1

# A context provider adds keys to the feature dict (e.g. regime, research snapshot). It gets the
# symbol, the decision time (the END of the bar that just closed) and the bar window.
ContextProvider = Callable[[str, datetime, pd.DataFrame], dict[str, Any]]


def _bar_minutes(bars: pd.DataFrame) -> int:
    tf = bars.attrs.get("timeframe")
    if tf:
        return int(tf)
    if len(bars) < 2:
        return 5
    diffs = np.diff(bars.index.values[-10:]).astype("timedelta64[s]").astype(float) / 60.0
    diffs = diffs[diffs > 0]
    return int(round(float(np.min(diffs)))) if len(diffs) else 5


def decision_time(bars: pd.DataFrame) -> datetime:
    """The moment of the decision: the end of the last completed bar."""
    return (bars.index[-1] + timedelta(minutes=_bar_minutes(bars))).to_pydatetime()


def _finite(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(float(x)) else None


def _log_return(closes: np.ndarray, stamps: np.ndarray, now_ns: int, minutes: int, tf: int):
    """Log return from the close of the bar that ended `minutes` before now to the last close."""
    target = now_ns - (minutes + tf) * 60 * 10**9  # start of the bar ending at now - minutes
    i = int(np.searchsorted(stamps, target, side="right")) - 1
    if i < 0 or closes[i] <= 0:
        return None
    return _finite(math.log(closes[-1] / closes[i]) * 100.0)


def bar_features(bars: pd.DataFrame) -> dict[str, Any]:
    """Price, volatility, trend and volume features from completed bars only."""
    if bars.empty:
        return {}
    tf = _bar_minutes(bars)
    arr = bars.to_numpy(dtype=float)
    col = {c: i for i, c in enumerate(bars.columns)}
    o, h, lo, c, v = (arr[:, col[k]] for k in ("open", "high", "low", "close", "volume"))
    stamps = bars.index.tz_convert("UTC").asi8
    now_ns = int(stamps[-1]) + tf * 60 * 10**9
    out: dict[str, Any] = {}
    for name, minutes in (("ret_1h_pct", 60), ("ret_4h_pct", 240), ("ret_24h_pct", 1440)):
        out[name] = _log_return(c, stamps, now_ns, minutes, tf)

    last_day = now_ns - 1440 * 60 * 10**9
    recent = stamps >= last_day
    rc = c[recent]
    if len(rc) > 2:
        lr = np.diff(np.log(rc))
        out["vol_24h_pct"] = _finite(float(np.std(lr, ddof=1)) * math.sqrt(1440 / tf) * 100.0)
        rv = v[recent]
        sd = float(np.std(rv, ddof=1)) if len(rv) > 2 else 0.0
        out["volume_z"] = _finite((v[-1] - float(np.mean(rv))) / sd) if sd > 0 else 0.0
    else:
        out["vol_24h_pct"] = None
        out["volume_z"] = None

    # daily bars from completed UTC days only (today's partial day is excluded)
    days = stamps // (86_400 * 10**9)
    today = days[-1]
    prior = days < today
    if prior.any():
        frame = pd.DataFrame(
            {"d": days[prior], "h": h[prior], "l": lo[prior], "c": c[prior]}
        ).groupby("d")
        dh, dl, dc = frame["h"].max().to_numpy(), frame["l"].min().to_numpy(), frame["c"].last()
        dc = dc.to_numpy()
    else:
        dh = dl = dc = np.array([])
    if len(dc) >= 8:
        dlr = np.diff(np.log(dc[-8:]))
        out["vol_7d_pct"] = _finite(float(np.std(dlr, ddof=1)) * 100.0)
    else:
        out["vol_7d_pct"] = None
    if len(dc) >= 15:
        window = dc[-15:]
        path = float(np.sum(np.abs(np.diff(window))))
        out["trend_er_14d"] = _finite(abs(window[-1] - window[0]) / path) if path > 0 else 0.0
        prev_close = dc[-15:-1]
        tr = np.maximum.reduce(
            [dh[-14:] - dl[-14:], np.abs(dh[-14:] - prev_close), np.abs(dl[-14:] - prev_close)]
        )
        out["atr_14d_pct"] = _finite(float(np.mean(tr)) / c[-1] * 100.0)
    else:
        out["trend_er_14d"] = None
        out["atr_14d_pct"] = None
    out["dist_ma20d_pct"] = (
        _finite((c[-1] / float(np.mean(dc[-20:])) - 1.0) * 100.0) if len(dc) >= 20 else None
    )

    t = days == today
    if t.any():
        tp = (h[t] + lo[t] + c[t]) / 3.0
        vol = np.clip(v[t], 0, None)
        vw = float(np.sum(tp * vol) / np.sum(vol)) if np.sum(vol) > 0 else float(np.mean(tp))
        out["vwap_dev_pct"] = _finite((c[-1] / vw - 1.0) * 100.0)
        out["today_range_pct"] = _finite((np.max(h[t]) / np.min(lo[t]) - 1.0) * 100.0)
    now = pd.Timestamp(now_ns, tz="UTC")
    out["minute_of_day"] = now.hour * 60 + now.minute
    out["day_of_week"] = int(now.dayofweek)
    out["last_close"] = float(c[-1])
    return out


def inputs_hash(bars: pd.DataFrame, strategy: str, params: dict[str, Any]) -> str:
    """Fingerprint of everything a decision depended on: the bar window, the strategy and its
    params. A replay with the same hash must reach the same decision."""
    h = hashlib.sha256()
    h.update(strategy.encode())
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    if len(bars):
        h.update(bars.index.tz_convert("UTC").asi8.tobytes())
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in bars.columns]
        h.update(np.ascontiguousarray(bars[cols].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:24]


def decision_features(
    symbol: str,
    bars: pd.DataFrame,
    providers: list[ContextProvider] | None = None,
) -> dict[str, Any]:
    """Bar features plus every context provider's keys. A provider that fails is recorded as
    such and never stops a decision."""
    feats = bar_features(bars)
    feats["feature_version"] = FEATURE_VERSION
    if bars.empty:
        return feats
    ts = decision_time(bars)
    for provider in providers or []:
        try:
            feats.update(provider(symbol, ts, bars) or {})
        except Exception as e:  # context is nice to have; a decision never waits on it
            feats.setdefault("context_errors", []).append(
                f"{getattr(provider, '__name__', '?')}: {e}"
            )
    return feats
