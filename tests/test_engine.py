from datetime import timedelta

from cryptobot.backtest import cost_model
from cryptobot.brokers.paper import PaperBroker
from cryptobot.engine import Engine
from cryptobot.journal import Journal
from cryptobot.risk import RiskManager
from tests.conftest import StubStrategy, bars_from_closes


def make_engine(cfg, strategy, cash=1000, mode="backtest"):
    cm = cost_model(cfg)
    broker = PaperBroker(cash, cm)
    risk = RiskManager(cfg.risk, cfg.allow_short, cfg.max_entries_per_day, cost_model=cm)
    journal = Journal(":memory:")
    return Engine(cfg, strategy, {}, broker, risk, journal, mode=mode), journal


def feed(engine, df):
    for i in range(1, len(df) + 1):
        window = df.iloc[:i]
        window.attrs = dict(df.attrs)
        engine.on_bar("BTC/USD", window)


def test_entry_stop_exit_and_journal(cfg):
    closes = [100, 100, 101, 100.5, 98.0, 99, 99]  # entry at bar 2, stop (1%) hit at bar 5
    df = bars_from_closes(closes, wick=0.1)
    engine, journal = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0))
    feed(engine, df)
    assert len(engine.trades) == 1
    t = engine.trades[0]
    assert t.exit_reason == "stop" and t.pnl < 0 and t.r_multiple < -1  # costs push it past -1R
    assert t.exit_price < t.stop  # slippage against us
    kinds = set(journal.events()["kind"])
    assert {"day_start", "signal", "entry", "exit"} <= kinds
    assert len(journal.trades()) == 1


def test_target_exit(cfg):
    closes = [100, 100, 100.5, 101.0, 103.0, 102]
    df = bars_from_closes(closes, wick=0.1)
    engine, _ = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0, target_pct=2.0))
    feed(engine, df)
    assert len(engine.trades) == 1 and engine.trades[0].exit_reason == "target"
    assert engine.trades[0].pnl > 0


def test_end_of_day_flatten(cfg):
    closes = [100 + 0.001 * i for i in range(288)]  # a full UTC day; never hits a 1% stop
    df = bars_from_closes(closes, wick=0.0)
    engine, journal = make_engine(cfg, StubStrategy(fire_at=280, stop_pct=1.0))
    feed(engine, df)
    assert len(engine.trades) == 1
    t = engine.trades[0]
    assert t.exit_reason == "eod"
    assert t.exit_ts.strftime("%H:%M") == "23:45"  # bar ending 23:50 = 10 min before 00:00 UTC
    assert not engine.open
    msgs = journal.events(kinds=["flatten"])["message"].tolist()
    assert "eod" in msgs


def test_no_entry_inside_flatten_window(cfg):
    closes = [100.0] * 288
    df = bars_from_closes(closes)
    engine, journal = make_engine(cfg, StubStrategy(fire_at=287, stop_pct=1.0))
    feed(engine, df)
    assert engine.trades == [] and not engine.open
    skips = journal.events(kinds=["skip"])["message"].tolist()
    assert any("close" in s for s in skips)


def test_daily_loss_halts_and_flattens(cfg):
    cfg.risk.max_daily_loss_pct = 2.0
    cfg.risk.risk_per_trade_pct = 5.0  # so a 1% move is a real loss
    cfg.risk.max_stop_pct = 10.0  # keep the 8% stop so the daily loss fires first
    closes = [100, 100, 100, 95, 94, 94, 100, 100]
    df = bars_from_closes(closes, wick=0.0)
    engine, journal = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=8.0))
    feed(engine, df)
    assert engine.risk.state.halted
    assert "daily loss" in engine.risk.state.halt_reason
    assert len(engine.trades) == 1 and engine.trades[0].exit_reason == "daily_loss"
    assert set(journal.events(kinds=["halt"])["kind"]) == {"halt"}
    # a second signal the same day is refused
    assert not engine.open and engine.trades[0].exit_ts < df.index[-1]


def test_on_tick_software_stop(cfg):
    df = bars_from_closes([100, 100, 100], wick=0.0)
    engine, _ = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0), mode="paper")
    feed(engine, df)
    assert "BTC/USD" in engine.open
    engine.on_tick("BTC/USD", 98.5, df.index[-1] + timedelta(minutes=1))
    assert not engine.open and engine.trades[0].exit_reason == "stop"


def test_max_positions_across_symbols(cfg):
    cfg.risk.max_positions = 1
    df_q = bars_from_closes([100.0] * 5, symbol="BTC/USD")
    df_s = bars_from_closes([50.0] * 5, symbol="ETH/USD")
    engine, journal = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0), cash=100_000)
    for i in range(1, 6):
        for df, sym in ((df_q, "BTC/USD"), (df_s, "ETH/USD")):
            w = df.iloc[:i]
            w.attrs = dict(df.attrs)
            engine.on_bar(sym, w)
    assert len(engine.open) == 1
    assert any("max_positions" in m for m in journal.events(kinds=["skip"])["message"])
