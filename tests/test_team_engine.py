"""Stage 1 engine extensions: the swing session policy (multi-day holds) and
multi-lot positions per symbol (grid strategies ladder several entries)."""

import pandas as pd

from cryptobot.backtest import cost_model
from cryptobot.brokers.paper import PaperBroker
from cryptobot.engine import Engine
from cryptobot.journal import Journal
from cryptobot.risk import RiskManager
from cryptobot.strategies import Signal
from tests.conftest import StubStrategy, bars_from_closes


def make_engine(cfg, strategy, cash=1000):
    cm = cost_model(cfg)
    broker = PaperBroker(cash, cm)
    risk = RiskManager(cfg.risk, cfg.allow_short, cfg.max_entries_per_day, cost_model=cm)
    return Engine(cfg, strategy, {}, broker, risk, Journal(":memory:"))


def feed(engine, df, symbol="BTC/USD"):
    for i in range(1, len(df) + 1):
        window = df.iloc[:i]
        window.attrs = dict(df.attrs)
        engine.on_bar(symbol, window)


# ------------------------------------------------------------- swing policy


def test_swing_bot_holds_across_the_utc_day_roll(cfg):
    cfg.risk.session_policy = "swing"
    day1 = bars_from_closes([100.0] * 288, wick=0.0)  # full UTC day
    day2 = bars_from_closes([100.0] * 4, day="2026-10-06", wick=0.0)
    df = pd.concat([day1, day2])
    df.attrs = dict(day1.attrs)
    engine = make_engine(cfg, StubStrategy(fire_at=100, stop_pct=5.0))
    feed(engine, df)
    assert "BTC/USD" in engine.open  # no end-of-day flatten
    assert engine.trades == []
    assert engine.open["BTC/USD"][0].entry_ts.date().isoformat() == "2026-10-05"


def test_swing_bot_may_enter_near_midnight(cfg):
    cfg.risk.session_policy = "swing"
    df = bars_from_closes([100.0] * 288, wick=0.0)
    engine = make_engine(cfg, StubStrategy(fire_at=287, stop_pct=5.0))  # inside the old window
    feed(engine, df)
    assert "BTC/USD" in engine.open or "BTC/USD" in engine.pending_entry


def test_day_bot_still_flattens(cfg):
    cfg.risk.session_policy = "day"
    df = bars_from_closes([100.0] * 288, wick=0.0)
    engine = make_engine(cfg, StubStrategy(fire_at=100, stop_pct=5.0))
    feed(engine, df)
    assert not engine.open
    assert engine.trades and engine.trades[0].exit_reason == "eod"


# ------------------------------------------------------------- multi-lot


class LadderStrategy:
    """Signals on every bar (a grid-style ladder of long entries)."""

    NAME = "ladder"
    DEFAULT_PARAMS: dict = {}

    def signal(self, bars, params):
        entry = float(bars["close"].iloc[-1])
        return Signal("long", entry, entry * 0.90, entry * 1.5, reason=f"rung {len(bars)}")


def test_multiple_lots_accumulate_up_to_the_cap(cfg):
    cfg.max_lots_per_symbol = 3
    cfg.max_entries_per_day = 10
    cfg.risk.max_positions = 10
    cfg.risk.max_stop_pct = 15.0
    df = bars_from_closes([100.0] * 8, wick=0.0)
    engine = make_engine(cfg, LadderStrategy(), cash=10_000)
    feed(engine, df)
    assert len(engine.open["BTC/USD"]) == 3  # capped by max_lots_per_symbol
    assert engine.total_lots() == 3


def test_lots_exit_independently(cfg):
    cfg.max_lots_per_symbol = 2
    cfg.max_entries_per_day = 10
    cfg.risk.max_positions = 10
    cfg.risk.max_stop_pct = 15.0

    class TwoRungs(LadderStrategy):
        def signal(self, bars, params):
            if len(bars) > 3:
                return None
            entry = float(bars["close"].iloc[-1])
            # first rung has a tight stop, second a wide one
            stop = entry * (0.99 if len(bars) == 2 else 0.90)
            return Signal("long", entry, stop, None, reason=f"rung {len(bars)}")

    closes = [100, 100, 100, 100, 98.0, 98, 98]  # bar 5 drops 2%: only the tight stop hits
    df = bars_from_closes(closes, wick=0.0)
    engine = make_engine(cfg, TwoRungs(), cash=10_000)
    feed(engine, df)
    assert len(engine.trades) == 1 and engine.trades[0].exit_reason == "stop"
    assert len(engine.open.get("BTC/USD", [])) == 1  # the wide-stop lot survives


def test_default_single_lot_behaviour_unchanged(cfg):
    df = bars_from_closes([100.0] * 6, wick=0.0)
    engine = make_engine(cfg, LadderStrategy())
    feed(engine, df)
    assert len(engine.open.get("BTC/USD", [])) == 1  # max_lots_per_symbol defaults to 1
