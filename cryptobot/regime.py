"""Market regimes: trending, ranging, high-volatility or risk-off, one label per market per UTC
day, from completed days only (a day's label never uses that day's own prices).

Simple, explainable rule thresholds (config `regime:`), in priority order:
  risk_off   most markets below their 20-day average AND the market's 20-day return is deeply
             negative, or extreme fear (Fear & Greed) while half the markets are below average
  high_vol   short-term volatility (5 days) well above its 20-day level
  trending   a high efficiency ratio over 14 days (price went somewhere, not back and forth)
  ranging    everything else
  unknown    not enough history yet

The regime is recorded with every decision and trade, and strategy results are tracked per
regime so the learning loop can require a challenger to work in more than one.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from cryptobot.config import RegimeConfig

REGIMES = ("trending", "ranging", "high_vol", "risk_off")


def daily_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """UTC daily high/low/close from intraday bars (all days present in `bars`)."""
    if bars.empty:
        return pd.DataFrame(columns=["high", "low", "close"])
    idx = bars.index.tz_convert("UTC").normalize()
    g = bars.groupby(idx)
    out = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    out.index = pd.DatetimeIndex([d.date() for d in out.index], name="day")
    return out


class RegimeDetector:
    def __init__(
        self,
        cfg: RegimeConfig,
        symbols: list[str],
        fear_greed: Callable[[datetime], float | None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.symbols = list(symbols)
        self.fear_greed = fear_greed
        self.daily: dict[str, pd.DataFrame] = {}
        self._cache: dict[tuple[str, date], dict[str, Any]] = {}

    @classmethod
    def from_bars(
        cls,
        cfg: RegimeConfig,
        bars_by_symbol: dict[str, pd.DataFrame],
        fear_greed: Callable[[datetime], float | None] | None = None,
    ) -> RegimeDetector:
        det = cls(cfg, list(bars_by_symbol), fear_greed)
        for symbol, bars in bars_by_symbol.items():
            det.daily[symbol] = daily_bars(bars)
        return det

    def update_from_bars(self, symbol: str, bars: pd.DataFrame) -> None:
        """Add the completed days in a bar window (today's partial day is left out)."""
        if bars.empty:
            return
        today = bars.index[-1].tz_convert("UTC").date()
        have = self.daily.get(symbol)
        if have is not None and len(have) and have.index[-1].date() >= today - timedelta(days=1):
            return  # already up to date
        d = daily_bars(bars)
        d = d[[ix.date() < today for ix in d.index]]
        merged = d if have is None else pd.concat([have, d])
        self.daily[symbol] = merged[~merged.index.duplicated(keep="last")].sort_index()

    # ------------------------------------------------------------------ classification
    def _history(self, symbol: str, day: date) -> pd.DataFrame:
        d = self.daily.get(symbol)
        if d is None or d.empty:
            return pd.DataFrame(columns=["high", "low", "close"])
        return d[d.index < pd.Timestamp(day)]  # completed days strictly before `day`

    def metrics(self, symbol: str, day: date) -> dict[str, Any]:
        c = self.cfg
        hist = self._history(symbol, day)
        need = max(c.vol_long_days, c.er_days, c.ma_days) + 1
        out: dict[str, Any] = {"days": int(len(hist))}
        if len(hist) < need:
            return out
        closes = hist["close"].to_numpy(dtype=float)
        lr = np.diff(np.log(closes))
        short, long_ = (
            np.std(lr[-c.vol_short_days :], ddof=1),
            np.std(lr[-c.vol_long_days :], ddof=1),
        )
        out["vol_ratio"] = float(short / long_) if long_ > 0 else 1.0
        out["vol_20d_pct"] = float(long_ * 100)
        w = closes[-(c.er_days + 1) :]
        path = float(np.sum(np.abs(np.diff(w))))
        out["efficiency"] = float(abs(w[-1] - w[0]) / path) if path > 0 else 0.0
        out["above_ma"] = bool(closes[-1] > np.mean(closes[-c.ma_days :]))
        out["ret_20d_pct"] = float((closes[-1] / closes[-(c.ma_days + 1)] - 1) * 100)
        return out

    def market(self, day: date) -> dict[str, Any]:
        """Breadth and the average 20-day return across every tracked market."""
        ms = [self.metrics(s, day) for s in self.symbols]
        ms = [m for m in ms if "above_ma" in m]
        if not ms:
            return {}
        out = {
            "breadth": float(np.mean([m["above_ma"] for m in ms])),
            "market_ret_20d_pct": float(np.mean([m["ret_20d_pct"] for m in ms])),
        }
        if self.fear_greed is not None:
            when = datetime(day.year, day.month, day.day, tzinfo=UTC)
            try:
                out["fear_greed"] = self.fear_greed(when)
            except Exception:  # research is optional
                out["fear_greed"] = None
        return out

    def classify(self, symbol: str, day: date) -> dict[str, Any]:
        key = (symbol, day)
        if key in self._cache:
            return self._cache[key]
        c = self.cfg
        m = self.metrics(symbol, day)
        info: dict[str, Any] = {"regime": "unknown", **{f"regime_{k}": v for k, v in m.items()}}
        if "vol_ratio" in m:
            mk = self.market(day)
            info.update({f"regime_{k}": v for k, v in mk.items()})
            breadth = mk.get("breadth", 1.0)
            fg = mk.get("fear_greed")
            risk_off = (
                breadth <= c.risk_off_breadth
                and mk.get("market_ret_20d_pct", 0.0) <= c.risk_off_market_ret_pct
            ) or (fg is not None and fg <= c.risk_off_fear_greed and breadth <= 0.5)
            if risk_off:
                info["regime"] = "risk_off"
            elif m["vol_ratio"] >= c.high_vol_ratio:
                info["regime"] = "high_vol"
            elif m["efficiency"] >= c.trending_efficiency:
                info["regime"] = "trending"
            else:
                info["regime"] = "ranging"
        info = {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in info.items()
        }
        self._cache[key] = info
        return info

    def context(self, symbol: str, ts: datetime, bars: pd.DataFrame | None = None) -> dict:
        """features.py context provider: the regime of `symbol` on the decision's UTC day."""
        if bars is not None and len(bars):
            self.update_from_bars(symbol, bars)
        day = pd.Timestamp(ts).tz_convert("UTC").date()
        return self.classify(symbol, day)

    def current(self, day: date) -> dict[str, str]:
        return {s: self.classify(s, day)["regime"] for s in self.symbols}


# ----------------------------------------------------------------------------- performance


def regime_performance(trades: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Results per regime (and per any extra column, e.g. strategy or version)."""
    cols = list(by or []) + ["regime"]
    if trades is None or trades.empty or "regime" not in trades.columns:
        return pd.DataFrame(
            columns=cols + ["trades", "win_rate_pct", "avg_r", "net_pnl", "profit_factor"]
        )
    t = trades.copy()
    t["regime"] = t["regime"].fillna("unknown").replace("", "unknown")
    rows = []
    for key, g in t.groupby(cols):
        key = key if isinstance(key, tuple) else (key,)
        pnl = g["pnl"].astype(float)
        wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        rows.append(
            {
                **dict(zip(cols, key, strict=True)),
                "trades": int(len(g)),
                "win_rate_pct": float((pnl > 0).mean() * 100),
                "avg_r": float(g["r_multiple"].astype(float).mean()),
                "net_pnl": float(pnl.sum()),
                "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
            }
        )
    return pd.DataFrame(rows)


def profitable_regimes(trades: pd.DataFrame, min_trades: int = 5) -> list[str]:
    """Regimes (excluding unknown) with positive net P&L over at least `min_trades` trades."""
    perf = regime_performance(trades)
    if perf.empty:
        return []
    ok = perf[
        (perf["regime"] != "unknown") & (perf["trades"] >= min_trades) & (perf["net_pnl"] > 0)
    ]
    return sorted(ok["regime"].tolist())
