"""Section 0 fixes: next-open entry fills, pessimistic stop fills, 365-day
metrics with Sortino and the first day's return, timestamp-based ORB, and
skip-log deduplication."""

import pandas as pd
import pytest

from cryptobot.backtest import compute_metrics, cost_model
from cryptobot.brokers.paper import PaperBroker
from cryptobot.engine import Engine
from cryptobot.journal import Journal
from cryptobot.risk import RiskManager
from cryptobot.strategies import orb
from tests.conftest import StubStrategy, bars_from_closes


def make_engine(cfg, strategy, cash=1000):
    cm = cost_model(cfg)
    broker = PaperBroker(cash, cm)
    risk = RiskManager(cfg.risk, cfg.allow_short, cfg.max_entries_per_day, cost_model=cm)
    return Engine(cfg, strategy, {}, broker, risk, Journal(":memory:")), broker


def feed(engine, df, upto=None):
    for i in range(1, (upto or len(df)) + 1):
        window = df.iloc[:i]
        window.attrs = dict(df.attrs)
        engine.on_bar("BTC/USD", window)


# ---------------------------------------------------------------- entry fills


def test_entry_fills_at_next_bar_open(cfg):
    # signal on bar 2 (close 101); the next bar OPENS at 101 and runs to 103
    closes = [100, 101, 103, 103, 103]
    df = bars_from_closes(closes, wick=0.0)
    engine, broker = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=5.0))
    feed(engine, df, upto=2)
    assert "BTC/USD" in engine.pending_entry and not engine.open
    feed_window = df.iloc[:3]
    feed_window.attrs = dict(df.attrs)
    engine.on_bar("BTC/USD", feed_window)
    assert "BTC/USD" in engine.open
    trade = engine.open["BTC/USD"]
    expected = broker.cost_model.fill_price(float(df["open"].iloc[2]), is_buy=True)
    assert trade.entry_price == pytest.approx(expected)


def test_gap_through_stop_abandons_the_entry(cfg):
    # stop 1% below the signal close; the next bar GAPS open 3% lower
    df = bars_from_closes([100, 100, 97, 97], wick=0.0)
    df.loc[df.index[2], "open"] = 97.0
    df.loc[df.index[2], "high"] = 97.0
    engine, _ = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0))
    journal = engine.journal
    feed(engine, df)
    assert not engine.open and engine.trades == []
    skips = journal.events(kinds=["skip"])["message"].tolist()
    assert any("gapped through the stop" in s for s in skips)


# ---------------------------------------------------------------- stop fills


def test_stop_exit_fills_through_the_stop(cfg):
    closes = [100, 100, 100, 100, 95, 95]
    df = bars_from_closes(closes, wick=0.0)
    engine, broker = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0))
    feed(engine, df)
    assert len(engine.trades) == 1
    t = engine.trades[0]
    assert t.exit_reason == "stop"
    cm = broker.cost_model
    # reference pushed through the stop by stop_gap_bps, then normal adverse fill on top
    gapped = cm.stop_exit_reference(min(float(df["open"].iloc[4]), t.stop), closing_long=True)
    assert t.exit_price == pytest.approx(cm.fill_price(gapped, is_buy=False))
    assert t.exit_price < t.stop


# ---------------------------------------------------------------- metrics


def test_metrics_use_365_days_include_first_day_and_sortino():
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=i) for i in range(3)]
    )
    equity = pd.Series([990.0, 1000.0, 1015.0], index=idx)
    m = compute_metrics(pd.DataFrame(), equity, starting_cash=1000.0)
    # day one lost 1% against starting cash; with the day-0 anchor it must count
    rets = pd.Series([990 / 1000 - 1, 1000 / 990 - 1, 1015 / 1000 - 1])
    expected_sharpe = rets.mean() / rets.std(ddof=1) * (365**0.5)
    assert m["sharpe"] == pytest.approx(float(expected_sharpe))
    assert m["trading_days"] == 3
    assert "sortino" in m  # one losing day -> finite or zero, never missing
    assert m["net_return_pct"] == pytest.approx(1.5)


# ---------------------------------------------------------------- ORB timestamps


def _orb_params():
    return {"range_minutes": 15, "target_r": 3, "atr_floor_mult": 0.0, "atr_days": 14}


def test_orb_fires_despite_missing_bar_inside_range(cfg):
    df = bars_from_closes([100, 101, 102, 103], wick=0.0)  # bars at :00 :05 :10 :15
    df = df.drop(df.index[1])  # :05 had no trades (Alpaca omits it)
    sig = orb.signal(df.iloc[:2], _orb_params())  # bars :00 and :10; range ends :15
    assert sig is not None and sig.side == "long"
    assert orb.signal(df.iloc[:3], _orb_params()) is None  # must not fire again at :15


def test_orb_fires_once_when_boundary_bar_is_missing(cfg):
    df = bars_from_closes([100, 101, 102, 103, 104], wick=0.0)  # :00 :05 :10 :15 :20
    df = df.drop(df.index[2])  # the :10 bar (ending exactly at :15) is missing
    windows = [df.iloc[:i] for i in range(1, len(df) + 1)]
    fired = [i for i, w in enumerate(windows) if orb.signal(w, _orb_params()) is not None]
    assert len(fired) == 1  # fires exactly once, on the first bar past the boundary


# ---------------------------------------------------------------- skip dedup


def test_identical_skips_are_journalled_once_per_day(cfg):
    cfg.risk.max_positions = 0  # every signal is refused for the same reason
    df = bars_from_closes([100.0] * 6, wick=0.0)

    class EveryBar(StubStrategy):
        def signal(self, bars, params):
            self.fire_at = len(bars)  # fire on every bar
            return super().signal(bars, params)

    engine, _ = make_engine(cfg, EveryBar(stop_pct=1.0))
    feed(engine, df)
    assert engine.skipped >= 3  # counted every time
    skips = engine.journal.events(kinds=["skip"])["message"].tolist()
    assert len(skips) == 1  # journalled once
