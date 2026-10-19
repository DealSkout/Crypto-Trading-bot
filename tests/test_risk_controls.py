"""Section 5: risk controls that override everything. Sizing by volatility targeting and
fractional Kelly (<= 0.25), never above the per-trade risk cap; automatic pauses during macro
windows and when data goes stale. (The drawdown kill switch is tested end to end in
test_guardian.py::test_runner_stops_completely_on_drawdown.)"""

from datetime import timedelta

import pytest

from cryptobot.backtest import cost_model
from cryptobot.brokers.paper import PaperBroker
from cryptobot.config import RISK_KEYS, ConfigError, validate
from cryptobot.costs import CostModel
from cryptobot.engine import Engine
from cryptobot.journal import Journal
from cryptobot.learn import ProposalError, check_proposal_params
from cryptobot.notify import Notifier
from cryptobot.research.calendar import MacroCalendar, parse_events
from cryptobot.risk import RiskManager, kelly_risk_pct
from cryptobot.runner import Runner
from tests.conftest import StubStrategy, bars_from_closes
from tests.test_engine import feed

# ----------------------------------------------------------------------------- Kelly


def test_kelly_is_fractional_floored_and_needs_enough_trades():
    rs = [2.0] * 40 + [-1.0] * 60  # win 40% at 2R: f* = 0.4 - 0.6 / 2 = 0.1
    assert kelly_risk_pct(rs, 0.25, 0.1, 30) == pytest.approx(0.25 * 0.1 * 100)
    assert kelly_risk_pct(rs[:20], 0.25, 0.1, 30) is None  # too few trades
    losing = [1.0] * 30 + [-1.0] * 70  # no edge: keep trading at the floor to learn
    assert kelly_risk_pct(losing, 0.25, 0.1, 30) == 0.1


def test_kelly_fraction_above_a_quarter_is_refused(cfg):
    cfg.risk.kelly_fraction = 0.3
    with pytest.raises(ConfigError):
        validate(cfg)


def test_nothing_automated_may_change_the_sizing_or_pause_settings():
    for key in ("sizing", "kelly_fraction", "target_daily_vol_pct", "pause_on_macro"):
        assert key in RISK_KEYS
        with pytest.raises(ProposalError):
            check_proposal_params({key: 1})


# ----------------------------------------------------------------------------- sizing


def rm(cfg, **risk):
    for k, v in risk.items():
        setattr(cfg.risk, k, v)
    return RiskManager(cfg.risk, cost_model=CostModel(0, 0, 0, 0))


def test_kelly_and_vol_target_only_ever_shrink_a_position(cfg):
    r = rm(cfg, risk_per_trade_pct=0.5, target_daily_vol_pct=1.0)
    base, _ = r.position_size(10_000, 10_000, 100, 99, lambda q: q)
    assert base == pytest.approx(50)  # 0.5% of 10k over a 1.0 stop
    big_kelly, _ = r.position_size(10_000, 10_000, 100, 99, lambda q: q, risk_pct=5.0)
    assert big_kelly == pytest.approx(base)  # the cap wins
    small_kelly, why = r.position_size(10_000, 10_000, 100, 99, lambda q: q, risk_pct=0.2)
    assert small_kelly == pytest.approx(20) and "kelly" in why
    calm, _ = r.position_size(10_000, 10_000, 100, 99, lambda q: q, daily_vol_pct=1.0)
    assert calm == pytest.approx(base)  # 50 units x 100 x 1% = 50 = 0.5% < 1% target
    wild, why = r.position_size(10_000, 10_000, 100, 99, lambda q: q, daily_vol_pct=8.0)
    assert wild == pytest.approx(10_000 * 0.01 / (100 * 0.08)) and "vol-target" in why


def test_engine_sizes_with_the_smallest_method(cfg):
    df = bars_from_closes([100.0, 100.0, 100.0], wick=0.0)

    def qty(sizing, vol):
        cfg.risk.sizing = sizing
        cm = cost_model(cfg)
        e = Engine(
            cfg,
            StubStrategy(fire_at=2, stop_pct=1.0),
            {},
            PaperBroker(100_000, cm),
            RiskManager(cfg.risk, cost_model=cm),
            Journal(":memory:", mode="paper"),
            mode="paper",
        )
        e.context_providers = [lambda s, ts, b: {"vol_7d_pct": vol}]
        feed(e, df)
        return e.open["BTC/USD"].qty

    plain = qty("risk", 9.0)
    assert qty("min", 9.0) < plain  # a 9% daily-vol market gets a smaller position
    assert qty("min", 0.5) == pytest.approx(plain)  # a calm one is limited by the risk cap


# ----------------------------------------------------------------------------- pauses


def runner_for(cfg, now, df, calendar=None, research=None):
    cfg.symbols = ["BTC/USD"]
    cm = cost_model(cfg)
    engine = Engine(
        cfg,
        StubStrategy(fire_at=2, stop_pct=1.0),
        {},
        PaperBroker(1000, cm),
        RiskManager(cfg.risk, cost_model=cm),
        Journal(":memory:", mode="paper"),
        mode="paper",
    )
    notifier = Notifier(enabled=False)
    sent = []
    notifier.send = lambda text: sent.append(text) or True
    runner = Runner(
        cfg,
        "paper",
        engine,
        lambda s: df[df.index <= now["now"]],
        engine.journal,
        notifier,
        clock=lambda: now["now"],
        sleep=lambda s: None,
        research=research,
    )
    if calendar is not None:
        runner.calendar = calendar
    return runner, engine, sent


def test_no_new_entries_inside_a_macro_window(cfg):
    df = bars_from_closes([100.0] * 6, wick=0.0)
    t = df.index[1].to_pydatetime()
    cal = MacroCalendar(
        parse_events(
            {
                "events": [
                    {
                        "name": "US CPI",
                        "kind": "cpi",
                        "time": t.strftime("%Y-%m-%d %H:%M"),
                        "tz": "UTC",
                    }
                ]
            }
        )
    )
    now = {"now": df.index[1] + timedelta(minutes=5, seconds=1)}
    runner, engine, sent = runner_for(cfg, now, df, calendar=cal)
    runner.step()
    assert not engine.open
    skip = engine.journal.decisions(actions=["skip"]).iloc[0]["reason"]
    assert skip.startswith("paused: macro window US CPI")
    assert len(engine.journal.events(kinds=["pause_start"])) == 1
    assert any("PAUSE BTC/USD" in m for m in sent)


def test_stale_prices_pause_entries_and_recovery_resumes(cfg):
    df = bars_from_closes([100.0] * 6, wick=0.0)
    now = {"now": df.index[1] + timedelta(minutes=5, seconds=1)}
    runner, engine, sent = runner_for(cfg, now, df.iloc[:2])
    now["now"] = df.index[1] + timedelta(minutes=40)  # no new bar for 35 minutes
    runner.step()
    assert "BTC/USD" in runner.stale_prices and not engine.open
    assert runner.pause_reason("BTC/USD", now["now"]).startswith("price data stale")
    assert any("PAUSE" in m for m in sent)
    # data comes back
    fresh = bars_from_closes([100.0] * 12, wick=0.0)
    runner.fetch_bars = lambda s: fresh[fresh.index <= now["now"]]
    runner.step()
    assert "BTC/USD" not in runner.stale_prices
    assert len(engine.journal.events(kinds=["pause_end"])) == 1


def test_stale_required_research_pauses_everything(cfg):
    class StaleResearch:
        calendar = MacroCalendar([])

        def stale_sources(self, now):
            return ["funding"]

        def stop(self):
            pass

    cfg.research.required_sources = ["funding"]
    df = bars_from_closes([100.0] * 6, wick=0.0)
    now = {"now": df.index[1] + timedelta(minutes=5, seconds=1)}
    runner, engine, _ = runner_for(cfg, now, df, research=StaleResearch())
    runner.step()
    assert not engine.open
    assert "research feed stale: funding" in runner.pause_reason("BTC/USD", now["now"])


def test_pauses_can_be_switched_off_by_the_human(cfg):
    cfg.risk.pause_on_stale_data = False
    cfg.risk.pause_on_macro = False
    df = bars_from_closes([100.0] * 6, wick=0.0)
    now = {"now": df.index[1] + timedelta(minutes=40)}
    runner, _, _ = runner_for(cfg, now, df.iloc[:2])
    runner.step()
    assert runner.calendar is None and runner.pause_reason("BTC/USD", now["now"]) is None


def test_open_trades_keep_their_stops_during_a_pause(cfg):
    df = bars_from_closes([100.0, 100.0, 100.0, 97.0], wick=0.0)
    now = {"now": df.index[1] + timedelta(minutes=5, seconds=1)}
    runner, engine, _ = runner_for(cfg, now, df)
    runner.step()  # the simulator queues the signal...
    now["now"] = df.index[2] + timedelta(minutes=5, seconds=1)
    runner.step()  # ...and fills it at the next bar's open
    assert "BTC/USD" in engine.open
    runner.calendar = MacroCalendar(
        parse_events(
            {
                "events": [
                    {
                        "name": "FOMC",
                        "kind": "fomc",
                        "time": now["now"].strftime("%Y-%m-%d %H:%M"),
                        "tz": "UTC",
                    }
                ]
            }
        )
    )
    now["now"] = df.index[3] + timedelta(minutes=5, seconds=1)
    runner.step()
    assert engine.trades and engine.trades[0].exit_reason == "stop"
