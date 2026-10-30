import ast
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from cryptobot import learn
from cryptobot.backtest import BacktestResult, cost_model, run_backtest
from cryptobot.brokers.paper import PaperBroker
from cryptobot.config import ConfigError, DeciderConfig, from_dict
from cryptobot.decider import Decider, DecisionContext, build_context
from cryptobot.engine import Engine
from cryptobot.journal import Journal
from cryptobot.llm import LLMResult
from cryptobot.risk import RiskManager
from tests.conftest import StubStrategy, bars_from_closes

NY = ZoneInfo("America/New_York")
HOLD = {"action": "hold", "stop_price": 0, "target_price": 0, "confidence": 0.9, "reason": "wait"}


def buy(stop=99.0, target=0, conf=0.9):
    return {
        "action": "buy",
        "stop_price": stop,
        "target_price": target,
        "confidence": conf,
        "reason": "dip into vwap",
    }


class FakeLLM:
    def __init__(self, answers=(), cost=0.02, raise_exc=None, refused=False):
        self.answers, self.cost, self.raise_exc, self.refused = (
            list(answers),
            cost,
            raise_exc,
            refused,
        )
        self.calls = []

    def complete(self, system, user, schema, effort=None):
        self.calls.append((user, effort))
        if self.raise_exc:
            raise self.raise_exc
        data = self.answers.pop(0) if self.answers else dict(HOLD)
        return LLMResult(
            data=data,
            input_tokens=100,
            output_tokens=20,
            cost_usd=self.cost,
            model="fake",
            refused=self.refused,
        )


def dcfg(**kw):
    return DeciderConfig(provider="claude", **kw)


def ctx(price=100.0, side=None):
    return DecisionContext("BTC/USD", price, side, "ctx")


def make_engine(cfg, strategy, decider, cash=1000):
    cm = cost_model(cfg)
    broker = PaperBroker(cash, cm)
    risk = RiskManager(cfg.risk, cfg.allow_short, cfg.max_entries_per_day, cost_model=cm)
    return Engine(cfg, strategy, {}, broker, risk, decider.journal, decider=decider)


def feed(engine, df, upto=None):
    for i in range(1, (upto or len(df)) + 1):
        w = df.iloc[:i]
        w.attrs = dict(df.attrs)
        engine.on_bar("BTC/USD", w)


def test_decider_never_imports_brokers():
    tree = ast.parse(Path("cryptobot/decider.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all("brokers" not in a.name for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert "brokers" not in (node.module or "")


def test_validate_clamps_and_rejects():
    d = Decider(dcfg(max_stop_pct=1.0, min_confidence=0.5), FakeLLM(), Journal(":memory:"))
    ok = d.validate(buy(stop=97.0, target=105), ctx())
    assert ok.action == "buy" and ok.stop_price == pytest.approx(99.0) and ok.target == 105
    assert d.validate(buy(stop=101.0), ctx()).action == "hold"  # stop above price
    low = d.validate(buy(stop=99.5, conf=0.3), ctx())
    assert low.action == "hold" and "confidence" in low.reason
    assert d.validate(buy(stop=99.5, target=95), ctx()).target is None  # target below price
    short = d.validate(
        {"action": "sell", "stop_price": 101.5, "target_price": 0, "confidence": 0.9, "reason": ""},
        ctx(),
    )
    assert short.action == "sell" and short.stop_price == pytest.approx(101.0)
    junk = d.validate(
        {
            "action": "dance",
            "stop_price": "x",
            "target_price": None,
            "confidence": "?",
            "reason": "",
        },
        ctx(),
    )
    assert junk.action == "hold"


def test_decide_cap_cadence_errors_and_refusal():
    j = Journal(":memory:")
    llm = FakeLLM([buy(), buy(), buy()], cost=0.6)
    d = Decider(dcfg(max_daily_cost_usd=1.0, every_minutes=30), llm, j)
    ts = datetime(2026, 10, 5, 10, 0, tzinfo=NY)
    assert d.due("BTC/USD", ts)
    assert d.decide(ctx(), ts).action == "buy"
    assert not d.due("BTC/USD", ts + timedelta(minutes=10)) and d.due(
        "BTC/USD", ts + timedelta(minutes=30)
    )
    d.decide(ctx(), ts + timedelta(minutes=30))
    assert d.decide(ctx(), ts + timedelta(minutes=60)) is None and len(llm.calls) == 2
    assert j.llm_spend_today("decider", ts.date()) == pytest.approx(1.2)
    assert any("cost cap" in m for m in j.events(kinds=["decider"])["message"])
    assert (
        Decider(dcfg(), FakeLLM(raise_exc=RuntimeError("boom")), Journal(":memory:")).decide(
            ctx(), ts
        )
        is None
    )
    assert Decider(dcfg(), FakeLLM(refused=True), Journal(":memory:")).decide(ctx(), ts) is None
    assert llm.calls[0][1] == "medium"  # effort passed through


def test_build_context_mentions_limits_and_position(cfg, bars):
    df = bars["BTC/USD"]
    c = build_context(
        "BTC/USD",
        df,
        1000.0,
        900.0,
        None,
        None,
        cfg.risk,
        False,
        0.38,
        dcfg(),
        120.0,
        0.0,
        0,
        "orb",
    )
    assert c.open_side is None and "flat" in c.text and "NOT allowed" in c.text
    assert "Today's bars" in c.text and "round trip" in c.text


def test_last_resort_uses_rules_when_they_fire(cfg):
    llm = FakeLLM([buy()])
    d = Decider(dcfg(role="last_resort"), llm, Journal(":memory:"))
    engine = make_engine(cfg, StubStrategy(fire_at=1, stop_pct=1.0), d)
    feed(engine, bars_from_closes([100.0] * 4))
    assert "BTC/USD" in engine.open and llm.calls == []
    assert engine.open["BTC/USD"][0].reason == "stub"


def test_last_resort_asks_model_when_rules_are_silent(cfg):
    llm = FakeLLM([buy(stop=99.0)])
    d = Decider(dcfg(role="last_resort", every_minutes=30), llm, Journal(":memory:"))
    engine = make_engine(cfg, StubStrategy(fire_at=999), d)
    feed(engine, bars_from_closes([100.0] * 8))
    assert len(llm.calls) == 1
    t = engine.open["BTC/USD"][0]
    assert t.stop == 99.0 and t.reason.startswith("claude:")
    ev = d.journal.events(kinds=["decider"])
    assert len(ev) == 1 and ev["message"].iloc[0].startswith("buy")


def test_final_say_rejects_then_approves(cfg):
    d = Decider(dcfg(role="final_say"), FakeLLM([HOLD]), Journal(":memory:"))
    engine = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0), d)
    feed(engine, bars_from_closes([100.0] * 3))
    assert not engine.open and len(d.llm.calls) == 1
    assert any("rejected" in m for m in d.journal.events(kinds=["skip"])["message"])

    d2 = Decider(dcfg(role="final_say"), FakeLLM([buy(stop=99.2)]), Journal(":memory:"))
    engine2 = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0), d2)
    feed(engine2, bars_from_closes([100.0] * 3))
    assert engine2.open["BTC/USD"][0].stop == pytest.approx(99.2)  # Claude's tighter stop was used

    wrong = {
        "action": "sell",
        "stop_price": 101.0,
        "target_price": 0,
        "confidence": 0.9,
        "reason": "no",
    }
    d3 = Decider(dcfg(role="final_say"), FakeLLM([wrong]), Journal(":memory:"))
    engine3 = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0), d3)
    feed(engine3, bars_from_closes([100.0] * 3))
    assert not engine3.open


def test_primary_opens_and_closes(cfg):
    llm = FakeLLM(
        [
            HOLD,
            buy(stop=99.0),
            {
                "action": "close",
                "stop_price": 0,
                "target_price": 0,
                "confidence": 0.9,
                "reason": "take it",
            },
        ]
    )
    d = Decider(dcfg(role="primary", every_minutes=5), llm, Journal(":memory:"))
    engine = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0), d)
    feed(engine, bars_from_closes([100.0] * 4))
    assert len(llm.calls) == 3
    assert len(engine.trades) == 1 and engine.trades[0].exit_reason == "llm_close"
    assert "Rule-based strategy (stub) signal now: long" in llm.calls[1][0]


def test_shorts_from_model_respect_allow_short(cfg):
    sell = {
        "action": "sell",
        "stop_price": 101.0,
        "target_price": 0,
        "confidence": 0.9,
        "reason": "fade",
    }
    d = Decider(dcfg(role="primary", every_minutes=5), FakeLLM([sell]), Journal(":memory:"))
    engine = make_engine(cfg, StubStrategy(fire_at=999), d)
    feed(engine, bars_from_closes([100.0] * 2))
    assert not engine.open and any(
        "short" in m for m in d.journal.events(kinds=["skip"])["message"]
    )


def test_backtest_with_decider_records_calls_and_gate(cfg, bars):
    llm = FakeLLM(cost=0.01)  # always hold
    d = Decider(dcfg(role="primary", every_minutes=60), llm, Journal(":memory:"))
    r = run_backtest(cfg, bars, decider=d)
    assert r.decider == "claude:claude-opus-5-5:primary"
    assert r.metrics["n_trades"] == 0 and r.metrics["llm_calls"] == len(llm.calls) > 0
    assert any("decider" in n for n in r.notes)
    path = r.save(cfg.reports_dir)
    assert BacktestResult.load(path).decider == r.decider
    ok, problems = learn.live_gate(cfg, Journal(cfg.journal_path, mode="paper"), r)
    assert any("decider" in p for p in problems)  # config has decider none, backtest used one


def test_decider_config_validation(tmp_path):
    raw = {"symbols": ["BTC/USD"], "decider": {"provider": "claude", "role": "boss"}}
    with pytest.raises(ConfigError):
        from_dict(raw)
    raw["decider"]["role"] = "primary"
    raw["decider"]["effort"] = "max"
    with pytest.raises(ConfigError):
        from_dict(raw)
    raw["decider"]["effort"] = "low"
    assert from_dict(raw).decider.role == "primary"
    assert learn.decider_description(from_dict(raw)) == "claude:claude-opus-5-5:primary"
