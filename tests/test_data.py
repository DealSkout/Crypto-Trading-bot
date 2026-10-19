from datetime import UTC, datetime

import pandas as pd
import pytest

from cryptobot import data
from cryptobot.data import DataError, fetch_alpaca, get_bars, synthetic_bars
from tests.conftest import NOW


def test_synthetic_bars_cover_every_day_around_the_clock(cfg):
    bars = get_bars(cfg, "BTC/USD", provider="synthetic", lookback_days=10, now=NOW)
    assert bars.attrs["symbol"] == "BTC/USD" and bars.attrs["timeframe"] == 5
    assert len(bars) == 10 * 288  # weekends included, 24h a day
    assert set(bars.index.dayofweek) == set(range(7))
    assert str(bars.index.tz) == "UTC"
    same = synthetic_bars("BTC/USD", 5, 10, now=NOW)
    assert same["close"].iloc[-1] == bars["close"].iloc[-1]  # deterministic


class FakeBarSet:
    def __init__(self, df):
        self.df = df


class FakeCryptoClient:
    def __init__(self, df=None, fail=False):
        self.df, self.fail, self.requests = df, fail, []

    def get_crypto_bars(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("network down")
        return FakeBarSet(self.df)


def alpaca_frame(symbol="BTC/USD", n=3):
    ts = pd.date_range("2026-10-15 00:00", periods=n, freq="5min", tz="UTC")
    idx = pd.MultiIndex.from_product([[symbol], ts], names=["symbol", "timestamp"])
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [2.0] * n,
            "trade_count": [5] * n,
            "vwap": [100.2 + i for i in range(n)],
        },
        index=idx,
    )


def test_fetch_alpaca_uses_the_crypto_bars_request():
    client = FakeCryptoClient(alpaca_frame())
    df = fetch_alpaca("BTC/USD", 5, 2, client=client)
    req = client.requests[0]
    assert req.symbol_or_symbols == ["BTC/USD"] and req.timeframe.amount_value == 5
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 3 and str(df.index.tz) == "UTC"
    assert fetch_alpaca("ETH/USD", 5, 2, client=FakeCryptoClient(alpaca_frame())).empty
    with pytest.raises(DataError):
        fetch_alpaca("BTC/USD", 60, 2, client=client)


def test_cache_merge_and_fallback(cfg, monkeypatch):
    client = FakeCryptoClient(alpaca_frame(n=6))
    first = get_bars(cfg, "BTC/USD", provider="alpaca", client=client, now=NOW)
    assert data.cache_path(cfg, "BTC/USD").exists() and len(first) == 6
    cached = get_bars(
        cfg, "BTC/USD", provider="alpaca", client=FakeCryptoClient(fail=True), now=NOW
    )
    assert len(cached) == 6  # falls back to the cache
    with pytest.raises(DataError):
        get_bars(cfg, "ETH/USD", provider="alpaca", client=FakeCryptoClient(fail=True), now=NOW)


def test_csv_provider(cfg, tmp_path):
    df = synthetic_bars("BTC/USD", 5, 2, now=NOW)
    out = tmp_path / "data"
    out.mkdir()
    df.rename_axis("timestamp").to_csv(out / "BTC_USD_5m.csv")
    loaded = get_bars(cfg, "BTC/USD", provider="csv")
    assert len(loaded) == len(df)
    assert loaded.index[0] == df.index[0].tz_convert("UTC")


def test_now_is_utc():
    assert NOW.tzinfo is not None and datetime.now(UTC).tzinfo is not None


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def test_incremental_fetch_only_asks_for_what_the_cache_lacks(cfg):
    client = FakeCryptoClient(alpaca_frame(n=6))
    get_bars(cfg, "BTC/USD", provider="alpaca", client=client, lookback_days=2, now=NOW)
    assert len(client.requests) == 1  # empty cache: one full request
    client2 = FakeCryptoClient(alpaca_frame(n=6))
    get_bars(cfg, "BTC/USD", provider="alpaca", client=client2, lookback_days=2, now=NOW)
    tail = client2.requests[0]
    last = pd.Timestamp("2026-10-15 00:25", tz="UTC")
    assert _utc(tail.start) == last - pd.Timedelta(minutes=15)  # small overlap only
    # a longer lookback than the cache holds also fetches the missing older history
    client3 = FakeCryptoClient(alpaca_frame(n=6))
    get_bars(cfg, "BTC/USD", provider="alpaca", client=client3, lookback_days=30, now=NOW)
    assert len(client3.requests) == 2
    head = client3.requests[1]
    assert _utc(head.end) == pd.Timestamp("2026-10-15 00:00", tz="UTC")
