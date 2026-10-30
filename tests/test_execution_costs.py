"""Items 2 and 3 in the engine: maker fees, the edge filter, resting limit entries
in simulation, the execution outbox, and planned vs actual R on every trade."""

from datetime import timedelta

import pytest

from cryptobot.backtest import cost_model
from cryptobot.brokers.paper import PaperBroker
from cryptobot.costs import CostModel
from cryptobot.engine import Engine
from cryptobot.journal import Journal
from cryptobot.risk import RiskManager
from tests.conftest import StubStrategy, bars_from_closes


def make_engine(cfg, strategy, cash=10_000, outbox=True):
    cm = cost_model(cfg)
    broker = PaperBroker(cash, cm)
    risk = RiskManager(cfg.risk, cfg.allow_short, cfg.max_entries_per_day, cost_model=cm)
    engine = Engine(cfg, strategy, {}, broker, risk, Journal(":memory:"), mode="paper")
    if outbox:
        engine.outbox = []
    return engine


def feed(engine, df, symbol="BTC/USD"):
    for i in range(1, len(df) + 1):
        window = df.iloc[:i]
        window.attrs = dict(df.attrs)
        engine.on_bar(symbol, window)


def with_low(df, i, low, open_=None):
    """Bar i trades down to `low` (and opens at `open_` if given)."""
    df = df.copy()
    df.iloc[i, df.columns.get_loc("low")] = low
    if open_ is not None:
        df.iloc[i, df.columns.get_loc("open")] = open_
    return df


# ---------------------------------------------------------------- costs
def test_maker_fee_and_round_trip():
    cm = CostModel(fee_pct=0.25, maker_fee_pct=0.15, spread_bps=10, slippage_bps=5)
    assert cm.fees(1000.0) == pytest.approx(2.5)
    assert cm.fees(1000.0, maker=True) == pytest.approx(1.5)
    taker_rt = cm.round_trip_cost_pct()
    maker_rt = cm.round_trip_cost_pct(maker_entry=True)
    assert taker_rt == pytest.approx(0.25 * 2 + (10 + 10) / 100)  # 0.70%
    assert maker_rt == pytest.approx(0.15 + 0.25 + (5 + 5) / 100)  # 0.50%: no entry spread
    assert CostModel(fee_pct=0.25).fees(1000.0, maker=True) == pytest.approx(2.5)  # unset = taker


# ---------------------------------------------------------------- edge filter
def edge_cfg(cfg, multiple=3.0):
    cfg.costs.min_edge_multiple = multiple
    cfg.costs.no_target_move_r = 2.0
    cfg.costs.maker_fee_pct = 0.15
    cfg.risk.max_stop_pct = 10.0
    return cfg


def test_edge_filter_skips_a_target_too_close_for_the_costs(cfg):
    edge_cfg(cfg)  # round trip about 0.65% with market entries, so 3x is about 1.95%
    engine = make_engine(edge_cfg(cfg), StubStrategy(fire_at=3, stop_pct=1.0, target_pct=1.0))
    feed(engine, bars_from_closes([100.0] * 6))
    assert not engine.open and not engine.pending
    skips = engine.journal.events(kinds=["skip"])
    assert len(skips) == 1
    assert skips.iloc[0]["message"].startswith("edge filter: expected move 1.00%")


def test_edge_filter_lets_a_big_enough_move_through(cfg):
    engine = make_engine(edge_cfg(cfg), StubStrategy(fire_at=3, stop_pct=2.0, target_pct=3.0))
    feed(engine, bars_from_closes([100.0] * 6))
    assert engine.open or engine.pending


def test_edge_filter_without_a_target_uses_the_stop_distance(cfg):
    # no target: expected move = 2 x stop distance = 0.8%, under 3 x 0.65%
    engine = make_engine(edge_cfg(cfg), StubStrategy(fire_at=3, stop_pct=0.4))
    feed(engine, bars_from_closes([100.0] * 6))
    skips = engine.journal.events(kinds=["skip"])
    assert "2x stop distance, no target" in skips.iloc[0]["message"]
    wide = make_engine(edge_cfg(cfg), StubStrategy(fire_at=3, stop_pct=2.0))  # 4% expected
    feed(wide, bars_from_closes([100.0] * 6))
    assert wide.open or wide.pending


def test_edge_filter_off_by_default(cfg):
    engine = make_engine(cfg, StubStrategy(fire_at=3, stop_pct=1.0, target_pct=0.2))
    feed(engine, bars_from_closes([100.0] * 6))
    assert engine.open or engine.pending


# ---------------------------------------------------------------- resting limit entries
def limit_cfg(cfg, timeout=2):
    cfg.orders.entry_type = "limit"
    cfg.orders.maker_offset_bps = 0.0
    cfg.orders.entry_timeout_bars = timeout
    cfg.costs.maker_fee_pct = 0.15
    cfg.risk.max_stop_pct = 10.0
    return cfg


def test_limit_entry_fills_only_when_the_market_trades_through_it(cfg):
    engine = make_engine(limit_cfg(cfg), StubStrategy(fire_at=3, stop_pct=2.0, target_pct=5.0))
    closes = [100.0, 100.0, 100.0, 100.5, 100.4, 100.6]
    df = bars_from_closes(closes, wick=0.0)  # bar 4 opens 100.0, low 100.0: touches the limit
    feed(engine, df.iloc[:3])
    assert "BTC/USD" in engine.resting and engine.resting["BTC/USD"].limit == pytest.approx(100.0)
    entry_event = engine.outbox[0]
    assert entry_event["type"] == "entry" and entry_event["limit"] == pytest.approx(100.0)
    assert entry_event["qty"] > 0 and entry_event["stop"] == pytest.approx(98.0)
    engine.on_bar("BTC/USD", _window(df, 4))
    lot = engine.open["BTC/USD"][0]
    assert lot.entry_price == pytest.approx(100.0)  # at the limit: no spread, no slippage
    assert lot.fees == pytest.approx(100.0 * lot.qty * 0.0015)  # maker fee
    assert lot.lot_id == entry_event["lot"]
    fill = engine.outbox[1]
    assert fill["type"] == "entry_fill" and fill["lot"] == lot.lot_id


def _window(df, n):
    w = df.iloc[:n]
    w.attrs = dict(df.attrs)
    return w


def test_limit_entry_is_cancelled_when_the_market_runs_away(cfg):
    engine = make_engine(
        limit_cfg(cfg, timeout=2), StubStrategy(fire_at=3, stop_pct=2.0, target_pct=5.0)
    )
    df = bars_from_closes([100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 104.0], wick=0.0)
    df = with_low(df, 3, 100.5, open_=100.5)  # never back down to 100
    for n in range(1, len(df) + 1):
        engine.on_bar("BTC/USD", _window(df, n))
    assert not engine.open and not engine.resting and not engine.pending
    kinds = [e["type"] for e in engine.outbox]
    assert kinds == ["entry", "entry_cancel"]
    assert "not filled within 2 bars" in engine.outbox[1]["reason"]


def test_a_gap_down_fills_the_resting_buy_at_the_open(cfg):
    engine = make_engine(limit_cfg(cfg), StubStrategy(fire_at=3, stop_pct=2.0, target_pct=5.0))
    df = bars_from_closes([100.0, 100.0, 100.0, 99.5, 99.6], wick=0.0)
    df = with_low(df, 3, 99.4, open_=99.5)
    for n in range(1, 5):
        engine.on_bar("BTC/USD", _window(df, n))
    assert engine.open["BTC/USD"][0].entry_price == pytest.approx(99.5)


def test_a_tick_fills_the_resting_limit(cfg):
    engine = make_engine(limit_cfg(cfg), StubStrategy(fire_at=3, stop_pct=2.0, target_pct=5.0))
    df = bars_from_closes([100.0, 100.0, 100.0], wick=0.0)
    for n in range(1, 4):
        engine.on_bar("BTC/USD", _window(df, n))
    t = df.index[-1] + timedelta(minutes=6)
    engine.on_tick("BTC/USD", 100.2, t)
    assert not engine.open
    engine.on_tick("BTC/USD", 99.95, t + timedelta(seconds=15))
    assert engine.open["BTC/USD"][0].entry_price == pytest.approx(99.95)


def test_market_entries_are_unchanged_by_default(cfg):
    engine = make_engine(cfg, StubStrategy(fire_at=3, stop_pct=2.0))
    df = bars_from_closes([100.0] * 5, wick=0.0)
    for n in range(1, 5):
        engine.on_bar("BTC/USD", _window(df, n))
    assert not engine.resting and engine.open
    assert engine.outbox[0]["type"] == "entry_market"


# ---------------------------------------------------------------- planned vs actual R
def test_every_closed_trade_logs_planned_and_actual_r(cfg):
    cfg.risk.max_stop_pct = 10.0
    engine = make_engine(cfg, StubStrategy(fire_at=3, stop_pct=1.0))
    closes = [100.0, 100.0, 100.0, 100.0, 100.0]
    df = bars_from_closes(closes, wick=0.0)
    df = with_low(df, 4, 98.5)  # straight through the 99 stop
    for n in range(1, 6):
        engine.on_bar("BTC/USD", _window(df, n))
    t = engine.trades[-1]
    assert t.exit_reason == "stop"
    assert t.planned_r == pytest.approx(-1.0)  # the plan: lose exactly the stop distance
    assert t.actual_r < -1.0  # slippage past the stop and the fees on top
    row = engine.journal.trades().iloc[-1]
    assert row["planned_r"] == pytest.approx(-1.0) and row["actual_r"] == pytest.approx(t.actual_r)
    exit_log = engine.journal.events(kinds=["exit"]).iloc[0]["message"]
    assert "planned -1.00R, actual" in exit_log
    assert engine.outbox[-1]["type"] == "exit" and engine.outbox[-1]["reason"] == "stop"


# ---------------------------------------------------------------- router feedback
def test_shrink_lot_and_close_lot_at_follow_the_real_account(cfg):
    engine = make_engine(limit_cfg(cfg), StubStrategy(fire_at=3, stop_pct=2.0, target_pct=5.0))
    df = bars_from_closes([100.0, 100.0, 100.0, 100.2], wick=0.0)
    for n in range(1, 5):
        engine.on_bar("BTC/USD", _window(df, n))
    lot = engine.open["BTC/USD"][0]
    full, cash_before = lot.qty, engine.broker.cash
    assert engine.shrink_lot(lot.lot_id, full / 2)
    assert lot.qty == pytest.approx(full / 2)
    assert engine.broker.cash > cash_before  # the unfilled half's cash and fee came back
    assert engine.broker.get_positions()[0].qty == pytest.approx(full / 2)
    assert engine.close_lot_at(lot.lot_id, 98.1, "broker_stop", df.index[-1])
    assert not engine.open and engine.trades[-1].exit_reason == "broker_stop"
    assert engine.trades[-1].planned_r == pytest.approx(-1.0)
    assert engine.outbox[-1]["type"] == "entry_fill"  # the broker-side close is not re-sent
    other = make_engine(limit_cfg(cfg), StubStrategy(fire_at=3, stop_pct=2.0, target_pct=5.0))
    for n in range(1, 4):
        other.on_bar("BTC/USD", _window(df, n))
    rest = other.resting["BTC/USD"]
    assert other.cancel_lot(rest.lot_id, "router refused: kill switch active")
    assert not other.resting and not other.pending
    assert other.outbox[-1]["type"] == "entry_cancel"
