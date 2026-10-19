"""Fetch and cache intraday crypto bars.

Providers:
  alpaca    Alpaca's crypto market data through alpaca-py's CryptoHistoricalDataClient
            (works without keys; the keys in .env raise the rate limit)
  csv       bring your own: data/<BASE>_<QUOTE>_<tf>m.csv with timestamp,open,high,low,close,volume
  synthetic deterministic fake data for tests and demos (clearly labelled, never for decisions)

Bars are indexed by the bar START time, timezone-aware in UTC. Crypto trades around the clock,
so every bar of every day (weekends included) is kept. The parquet cache keeps growing the
longer you run the bot, so history accumulates beyond what one request returns.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from cryptobot import sessions
from cryptobot.config import Config

log = logging.getLogger(__name__)

COLUMNS = ["open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


class DataError(Exception):
    pass


def safe_name(symbol: str) -> str:
    return symbol.upper().replace(".", "_").replace("/", "_")


def empty_bars() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS}, index=idx)


def merge(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return empty_bars()
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "ts"
    return df[COLUMNS].astype("float64")


def regular_session_only(bars: pd.DataFrame, market: sessions.Market) -> pd.DataFrame:
    """Crypto has no session to cut to: keep every bar, in the market's timezone (UTC)."""
    if bars.empty:
        return bars
    return bars.tz_convert(market.tz)


# --------------------------------------------------------------------------- cache


def cache_path(cfg: Config, symbol: str) -> Path:
    return Path(cfg.data.cache_dir) / f"{safe_name(symbol)}_{cfg.timeframe_minutes}m.parquet"


def read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_bars()
    try:
        df = pd.read_parquet(path)
    except Exception as e:  # corrupt cache is not fatal; we refetch
        log.warning("ignoring unreadable cache %s: %s", path, e)
        return empty_bars()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return merge([df.tz_convert("UTC")])


def write_cache(path: Path, bars: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.tz_convert("UTC").to_parquet(path)


# --------------------------------------------------------------------------- providers


def fetch_alpaca(symbol: str, tf_minutes: int, days: int, client=None) -> pd.DataFrame:
    """Crypto bars from Alpaca via alpaca-py. `client` is injectable for tests."""
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    if client is None:
        from alpaca.data.historical import CryptoHistoricalDataClient

        key = os.environ.get("ALPACA_API_KEY") or None
        secret = os.environ.get("ALPACA_API_SECRET") or None
        client = (
            CryptoHistoricalDataClient(key, secret)
            if key and secret
            else (CryptoHistoricalDataClient())
        )
    if not 1 <= tf_minutes <= 59:
        raise DataError("alpaca bars support 1-59 minute timeframes")
    request = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(tf_minutes, TimeFrameUnit.Minute),
        start=datetime.now(UTC) - timedelta(days=days),
    )
    df = client.get_crypto_bars(request).df
    if df is None or df.empty:
        return empty_bars()
    if isinstance(df.index, pd.MultiIndex):
        level = "symbol" if "symbol" in df.index.names else 0
        if symbol not in df.index.get_level_values(level):
            return empty_bars()
        df = df.xs(symbol, level=level)
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return merge([df[COLUMNS].astype("float64")])


def load_csv(path: Path, market: sessions.Market) -> pd.DataFrame:
    if not path.exists():
        raise DataError(f"csv file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    candidates = ("timestamp", "ts", "datetime", "date", "time")
    ts_col = next((c for c in candidates if c in df.columns), None)
    if ts_col is None or any(c not in df.columns for c in COLUMNS):
        raise DataError(f"{path} needs columns timestamp,open,high,low,close,volume")
    idx = pd.to_datetime(df[ts_col], utc=False)
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(market.tz)
    df.index = pd.DatetimeIndex(idx).tz_convert("UTC")
    return merge([df[COLUMNS].astype("float64")])


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def synthetic_bars(
    symbol: str,
    tf_minutes: int,
    days: int,
    now: datetime | None = None,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Deterministic fake intraday bars: random walk with mild autocorrelation and a U-shaped
    volume profile. Good enough to exercise every code path; never a statement about markets."""
    market = sessions.market_for(symbol)
    n_bars = max(1, market.session_minutes // tf_minutes)
    rng = np.random.default_rng(stable_seed(symbol, seed))
    end_day = (now or datetime.now(UTC)).astimezone(market.zone).date()
    session_days: list[date] = []
    d = end_day
    while len(session_days) < days:
        session_days.append(d)  # crypto trades every day
        d -= timedelta(days=1)
    session_days.reverse()

    price = 100.0 + (stable_seed(symbol) % 4000)
    daily_vol = 0.03
    bar_vol = daily_vol / np.sqrt(n_bars)
    idx: list[datetime] = []
    o_, h_, l_, c_, v_ = [], [], [], [], []
    for day in session_days:
        price *= float(np.exp(rng.normal(0, 0.004)))  # overnight gap
        prev_r = 0.0
        day_drift = rng.normal(0, bar_vol * 0.3)
        t0 = sessions.session_open(market, day)
        for i in range(n_bars):
            r = day_drift + 0.08 * prev_r + rng.normal(0, bar_vol)
            prev_r = r
            open_ = price
            close = open_ * float(np.exp(r))
            wick = abs(rng.normal(0, bar_vol * 0.6))
            high = max(open_, close) * (1 + wick)
            low = min(open_, close) * (1 - wick)
            u = (i / max(1, n_bars - 1) - 0.5) ** 2
            vol = float(rng.lognormal(mean=np.log(20_000 * (0.5 + 3 * u)), sigma=0.3))
            idx.append(t0 + timedelta(minutes=i * tf_minutes))
            o_.append(open_)
            h_.append(high)
            l_.append(low)
            c_.append(close)
            v_.append(vol)
            price = close
    df = pd.DataFrame(
        {"open": o_, "high": h_, "low": l_, "close": c_, "volume": v_},
        index=pd.DatetimeIndex(idx, name="ts"),
    )
    return df.tz_convert("UTC")


# --------------------------------------------------------------------------- entry point


def get_bars(
    cfg: Config,
    symbol: str,
    provider: str | None = None,
    lookback_days: int | None = None,
    client=None,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Return bars for `symbol`, newest last, in UTC."""
    provider = provider or cfg.data.provider
    tf = cfg.timeframe_minutes
    days = lookback_days or cfg.data.lookback_days
    market = sessions.market_for(symbol)

    if provider == "synthetic":
        bars = synthetic_bars(symbol, tf, days, now=now)
    elif provider == "csv":
        bars = load_csv(Path(cfg.data.csv_dir) / f"{safe_name(symbol)}_{tf}m.csv", market)
    elif provider == "alpaca":
        path = cache_path(cfg, symbol)
        cached = read_cache(path)
        try:
            fresh = fetch_alpaca(symbol, tf, days, client=client)
        except Exception as e:  # network, auth, rate limit: fall back to the cache
            if cached.empty:
                raise DataError(
                    f"could not fetch {symbol} from {provider} and no cache exists: {e}. "
                    "Try `--data synthetic` for a demo, or put a CSV in data/."
                ) from e
            log.warning("fetch failed for %s (%s); using cached bars only", symbol, e)
            fresh = empty_bars()
        bars = merge([cached, fresh])
        if not fresh.empty:
            write_cache(path, bars)
    else:
        raise DataError(f"unknown data provider {provider!r}")

    bars = regular_session_only(bars, market)
    bars.attrs.update({"symbol": symbol, "market": market.name, "timeframe": tf})
    return bars
