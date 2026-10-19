"""Section 0 fixes from the Step 0 audit: crypto Sharpe/Sortino, next-bar-open fills, stop
slippage, skip de-duplication, timestamp-based ORB, the resting backup stop and the startup
reconcile."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from cryptobot.backtest import (
    PERIODS_PER_YEAR,
    compute_metrics,
    cost_model,
    daily_returns,
    sharpe_sortino,
)
from cryptobot.brokers.base import Order, Position
from cryptobot.engine import Engine
from cryptobot.journal import Journal
from cryptobot.risk import RiskManager
from cryptobot.runner import Runner
from cryptobot.strategies import load_strategy
from tests.conftest import StubStrategy, bars_from_closes
from tests.test_engine import feed, make_engine
from tests.test_engine_crypto import FakeAlpaca

# ----------------------------------------------------------------------------- metrics


def test_sharpe_is_annualised_over_365_days_and_sortino_is_reported():
    rets = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001])
    sharpe, sortino = sharpe_sortino(rets)
    expected = rets.mean() / rets.std(ddof=1) * np.sqrt(365)
    assert PERIODS_PER_YEAR == 365
    assert sharpe == pytest.approx(expected)
    downside = np.sqrt(np.mean(np.minimum(rets, 0) ** 2))
    assert sortino == pytest.approx(rets.mean() / downside * np.sqrt(365))


def test_first_day_return_counts_against_starting_cash():
    idx = pd.date_range("2026-01-05", periods=2, freq="D", tz="UTC")
    rets = daily_returns(pd.Series([110.0, 121.0], index=idx), 100.0)
    assert rets.tolist() == pytest.approx([0.10, 0.10])
    m = compute_metrics(pd.DataFrame(), pd.Series([110.0, 121.0], index=idx), 100.0)
    assert m["trading_days"] == 2 and "sortino" in m


def test_drawdown_includes_a_loss_from_the_starting_cash():
    idx = pd.date_range("2026-01-05", periods=2, freq="D", tz="UTC")
    m = compute_metrics(pd.DataFrame(), pd.Series([90.0, 95.0], index=idx), 100.0)
    assert m["max_drawdown_pct"] == pytest.approx(-10.0)


# ----------------------------------------------------------------------------- fills


def test_backtest_entry_fills_at_the_next_bars_open_not_the_signal_close(cfg):
    # signal on bar 2 (close 100); bar 3 opens at 100.4 because of a gap in the open
    df = bars_from_closes([100.0, 100.0, 101.0, 101.0, 101.0], wick=0.0)
    df.iloc[2, df.columns.get_loc("open")] = 100.4
    engine, journal = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0))
    feed(engine, df)
    t = engine.open["BTC/USD"]
    cm = cost_model(cfg)
    assert t.entry_price == pytest.approx(cm.fill_price(100.4, True))
    assert t.signal_price == pytest.approx(100.0)
    assert t.entry_ts == df.index[2].to_pydatetime()
    assert t.stop == pytest.approx(99.0)  # the strategy's level, not re-based on the fill


def test_real_brokers_fill_immediately_and_the_simulator_waits_a_bar(cfg):
    df = bars_from_closes([100, 100, 101], wick=0.0)
    real = FakeAlpaca(fee_coins=0.0)  # Alpaca paper or live: the market order fills now
    cm = cost_model(cfg)
    risk = RiskManager(cfg.risk, cfg.allow_short, cfg.max_entries_per_day, cost_model=cm)
    engine = Engine(
        cfg,
        StubStrategy(fire_at=2, stop_pct=1.0),
        {},
        real,
        risk,
        Journal(":memory:"),
        mode="paper",
    )
    feed(engine, df.iloc[:2])
    assert "BTC/USD" in engine.open and not engine.pending
    sim, _ = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0), mode="paper")
    feed(sim, df.iloc[:2])
    assert "BTC/USD" in sim.pending_entry and not sim.open


def test_gap_through_the_stop_before_the_fill_skips_the_trade(cfg):
    df = bars_from_closes([100.0, 100.0, 98.0, 98.0], wick=0.0)
    df.iloc[2, df.columns.get_loc("open")] = 98.5  # below the 99 stop before we could fill
    engine, journal = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0))
    feed(engine, df)
    assert not engine.open and not engine.trades
    assert any("gapped through the stop" in m for m in journal.events(kinds=["skip"])["message"])


def test_simulated_stop_exits_pay_extra_stop_slippage(cfg):
    df = bars_from_closes([100, 100, 100, 100, 98.0, 98], wick=0.0)
    engine, _ = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0))
    feed(engine, df)
    t = engine.trades[0]
    cm = cost_model(cfg)
    assert t.exit_reason == "stop"
    expected = cm.fill_price(cm.stop_exit_reference(99.0, closing_long=True), False)
    assert t.exit_price == pytest.approx(expected)
    assert t.exit_price < cm.fill_price(99.0, False)  # worse than a fill exactly at the stop


# ----------------------------------------------------------------------------- skips


class AlwaysFires:
    NAME = "always"
    DEFAULT_PARAMS: dict = {}

    @staticmethod
    def signal(bars, params):
        from cryptobot.strategies import Signal

        entry = float(bars["close"].iloc[-1])
        return Signal("long", entry, entry * 0.99, None, reason="always")


def test_a_signal_blocked_all_day_is_logged_once(cfg):
    df = bars_from_closes([100.0] * 60, wick=0.0)
    engine, journal = make_engine(cfg, AlwaysFires(), mode="paper")
    engine.risk.max_entries_per_day = 1
    feed(engine, df.iloc[:2])  # bar 1 signals, bar 2 fills at its open
    assert "BTC/USD" in engine.open
    engine._exit("BTC/USD", "manual", 100.0, df.index[1].to_pydatetime())
    for i in range(3, len(df) + 1):  # it keeps firing for the rest of the day
        w = df.iloc[:i]
        w.attrs = dict(df.attrs)
        engine.on_bar("BTC/USD", w)
    skips = journal.events(kinds=["skip"])
    assert len(skips) == 1 and "max_entries_per_day" in skips["message"].iloc[0]
    assert engine.skipped > 1  # every refusal is counted, but journalled once


# ----------------------------------------------------------------------------- ORB


def test_orb_uses_timestamps_so_a_missing_bar_does_not_shift_the_signal():
    closes = [100, 100.5, 101, 101.5, 102, 102.5]
    df = bars_from_closes(closes, wick=0.1)
    df = df.drop(df.index[1])  # Alpaca leaves out a bar with no trades (00:05)
    orb = load_strategy("orb")
    fired = []
    for i in range(1, len(df) + 1):
        w = df.iloc[:i]
        w.attrs = dict(df.attrs)
        s = orb.signal(w, {"range_minutes": 15})
        if s:
            fired.append(w.index[-1])
    # the bar starting 00:10 (ending 00:15) completes the 15-minute range, missing bar or not
    assert len(fired) == 1 and fired[0].strftime("%H:%M") == "00:10"


def test_strategy_due_checks_agree_with_signal():
    for name in ("orb", "intraday_momentum", "vwap_revert"):
        mod = load_strategy(name)
        assert callable(getattr(mod, "due", None)), name
    orb = load_strategy("orb")
    t0 = pd.Timestamp("2026-10-05 00:00", tz="UTC")
    assert not orb.due("BTC/USD", t0, 5, {"range_minutes": 30})
    assert orb.due("BTC/USD", t0 + timedelta(minutes=25), 5, {"range_minutes": 30})


# ----------------------------------------------------------------------------- backup stop


class FakeAlpacaWithStops(FakeAlpaca):
    """Alpaca-like: resting protective stops through place_protective_stop/cancel_protective
    (the same interface as AlpacaCryptoBroker)."""

    supports_protective_stop = True

    def __init__(self, backup_fills=False):
        super().__init__(fee_coins=0.0)
        self.backup_fills = backup_fills
        self.orders: dict[str, Order] = {}
        self.protective: dict[str, str] = {}

    def place_protective_stop(self, symbol, qty, stop_price, side="sell"):
        o = self.place_order(
            symbol,
            side,
            qty,
            order_type="stop_limit",
            limit_price=stop_price * 0.995,
            stop_price=stop_price,
        )
        self.protective[symbol] = o.id
        return o

    def cancel_protective(self, symbol):
        oid = self.protective.pop(symbol, None)
        if not oid:
            return None
        o = self.get_order(oid, symbol)
        if o.status != "filled":
            self.cancel(oid)
            o = self.get_order(oid, symbol)
        return o if o.filled_qty > 0 else None

    def place_order(
        self, symbol, side, qty, order_type="market", limit_price=None, stop_price=None
    ):
        self.placed.append(
            {"side": side, "qty": qty, "type": order_type, "limit": limit_price, "stop": stop_price}
        )
        o = Order(id=f"o{len(self.placed)}", symbol=symbol, side=side, qty=qty, type=order_type)
        self.orders[o.id] = o
        return o

    def get_order(self, order_id, symbol=""):
        o = self.orders[order_id]
        if o.type == "stop_limit" and self.backup_fills and o.status != "filled":
            # the backup stop filled while the bot was not looking
            o.status, o.filled_qty, o.avg_fill_price = "filled", o.qty, self.placed[-1]["limit"]
            self.pos[o.symbol] = self.pos.get(o.symbol, 0) - o.qty
            self.cash += o.qty * o.avg_fill_price
        elif o.type == "stop_limit" and o.id in self.cancelled:
            o.status = "cancelled"
        return o

    def await_fill(self, order, timeout=30.0):
        if order.type == "stop_limit":
            return order
        return super().await_fill(order, timeout)


def live_engine(cfg, broker, strategy):
    cm = cost_model(cfg)
    risk = RiskManager(cfg.risk, cfg.allow_short, cfg.max_entries_per_day, cost_model=cm)
    journal = Journal(":memory:")
    return Engine(cfg, strategy, {}, broker, risk, journal, mode="paper"), journal


def test_a_backup_stop_rests_at_the_broker_and_is_cancelled_before_a_normal_exit(cfg):
    df = bars_from_closes([100, 100, 100, 103, 103], wick=0.0)
    b = FakeAlpacaWithStops()
    engine, journal = live_engine(cfg, b, StubStrategy(fire_at=2, stop_pct=1.0, target_pct=2.0))
    feed(engine, df)
    stops = [p for p in b.placed if p["type"] == "stop_limit"]
    assert len(stops) == 1 and stops[0]["side"] == "sell"
    buf = cfg.orders.backup_stop_buffer_pct / 100
    assert stops[0]["stop"] == pytest.approx(99.0 * (1 - buf))
    assert stops[0]["limit"] < stops[0]["stop"]
    assert b.cancelled  # cancelled before the target exit sold the coins
    assert engine.trades[0].exit_reason == "target"


def test_an_already_filled_backup_stop_is_recorded_as_the_exit(cfg):
    df = bars_from_closes([100, 100, 100, 97, 97], wick=0.0)
    b = FakeAlpacaWithStops(backup_fills=True)
    engine, _ = live_engine(cfg, b, StubStrategy(fire_at=2, stop_pct=1.0))
    feed(engine, df)
    t = engine.trades[0]
    assert t.exit_reason == "backup_stop"
    assert len([p for p in b.placed if p["side"] == "sell" and p["type"] == "market"]) == 0


def test_no_backup_stop_when_disabled(cfg):
    cfg.orders.backup_stop = False
    df = bars_from_closes([100, 100, 100], wick=0.0)
    b = FakeAlpacaWithStops()
    engine, _ = live_engine(cfg, b, StubStrategy(fire_at=2, stop_pct=1.0))
    feed(engine, df)
    assert not [p for p in b.placed if p["type"] == "stop_limit"]


# ----------------------------------------------------------------------------- startup


def test_startup_flattens_positions_left_by_a_crashed_run(cfg):
    b = FakeAlpacaWithStops()
    b.pos["BTC/USD"] = 0.5
    b.last["BTC/USD"] = 100.0
    flattened = []
    b.flatten_all = lambda: flattened.append(True) or []
    b.get_positions = lambda: [Position("BTC/USD", 0.5, "long", 100.0)]
    engine, journal = live_engine(cfg, b, StubStrategy())
    runner = Runner(cfg, "paper", engine, lambda s: pd.DataFrame(), journal)
    runner.reconcile_on_start()
    assert flattened
    assert len(journal.events(kinds=["startup_flatten"])) == 1
