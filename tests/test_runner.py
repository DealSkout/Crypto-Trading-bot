from datetime import timedelta

import pandas as pd

from cryptobot.backtest import cost_model
from cryptobot.brokers.paper import PaperBroker
from cryptobot.engine import Engine
from cryptobot.journal import Journal
from cryptobot.notify import Notifier
from cryptobot.risk import RiskManager
from cryptobot.runner import Runner
from tests.conftest import StubStrategy, bars_from_closes


def make_runner(cfg, now_holder, df):
    cfg.symbols = ["BTC/USD"]
    broker = PaperBroker(1000, cost_model(cfg))
    engine = Engine(
        cfg,
        StubStrategy(fire_at=2, stop_pct=1.0),
        {},
        broker,
        RiskManager(cfg.risk),
        Journal(":memory:", mode="paper"),
        mode="paper",
    )
    seen = []

    def fetch(symbol):
        seen.append(symbol)
        return df[df.index <= now_holder["now"]]

    runner = Runner(
        cfg,
        "paper",
        engine,
        fetch,
        engine.journal,
        Notifier(enabled=False),
        clock=lambda: now_holder["now"],
        sleep=lambda s: None,
    )
    return runner, engine, seen


def test_runner_feeds_new_bars_and_ticks(cfg):
    day1 = bars_from_closes([100, 100, 100, 100], wick=0.0)
    day2 = bars_from_closes([100, 100], day="2026-10-06", wick=0.0)
    df = pd.concat([day1, day2])
    df.attrs = dict(day1.attrs)
    now = {"now": df.index[1] + timedelta(minutes=5, seconds=1)}  # bars 0 and 1 complete
    runner, engine, seen = make_runner(cfg, now, df)
    assert runner.step() is True
    assert "BTC/USD" in engine.open and runner.last_bar["BTC/USD"] == df.index[1]
    runner.step()  # same bar -> tick only
    assert runner.last_bar["BTC/USD"] == df.index[1] and len(seen) == 2
    now["now"] = df.index[2] + timedelta(minutes=5, seconds=1)
    runner.step()
    assert runner.last_bar["BTC/USD"] == df.index[2]
    assert runner.summarised is None  # crypto never closes: no summary mid-day
    now["now"] = df.index[4] + timedelta(minutes=5, seconds=1)  # first bar of the next UTC day
    assert runner.step() is True
    assert runner.summarised == day1.index[0].date()  # the previous day was closed out once
    assert engine.risk.state.day == day2.index[0].date()
    runner.run_forever(max_steps=1)
    assert runner.steps == 1 and not engine.open  # shutdown flattened the open trade
