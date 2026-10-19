import ast
from datetime import timedelta
from pathlib import Path

from cryptobot.backtest import cost_model
from cryptobot.brokers.paper import PaperBroker
from cryptobot.engine import Engine
from cryptobot.guardian import Guardian
from cryptobot.journal import Journal
from cryptobot.llm import LLMResult
from cryptobot.main import main
from cryptobot.notify import Notifier
from cryptobot.risk import RiskManager
from cryptobot.runner import Runner
from tests.conftest import NOW, StubStrategy, bars_from_closes


class FakeLLM:
    def __init__(self, data=None, raise_exc=None, refused=False):
        self.data, self.raise_exc, self.refused, self.calls = data or {}, raise_exc, refused, 0

    def complete(self, system, user, schema, effort=None):
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc
        return LLMResult(
            data=self.data,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.01,
            model="fake",
            refused=self.refused,
        )


def make(cfg, where, llm=None, backtest=None):
    where.mkdir(parents=True, exist_ok=True)
    cfg.guardian.stop_file = str(where / "STOPPED")
    j = Journal(str(where / "j.sqlite"), mode="paper")
    return Guardian(cfg, j, llm=llm, notifier=Notifier(enabled=False), backtest_metrics=backtest), j


def trade(j, r):
    j.record_trade(
        {
            "symbol": "BTC/USD",
            "side": "long",
            "qty": 1,
            "entry_ts": NOW.isoformat(),
            "entry_price": 100,
            "exit_ts": NOW.isoformat(),
            "exit_price": 100 + r,
            "stop": 99,
            "target": None,
            "pnl": r,
            "r_multiple": r,
            "costs": 0.1,
            "exit_reason": "stop",
        }
    )


def test_guardian_never_imports_brokers():
    tree = ast.parse(Path("cryptobot/guardian.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all("brokers" not in a.name for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert "brokers" not in (node.module or "")


def test_stop_and_resume(cfg, tmp_path):
    g, j = make(cfg, tmp_path / "a")
    assert not g.stopped and g.stop_info() is None
    g.stop("too much loss", "rules", NOW)
    assert g.stopped and g.stop_info()["source"] == "rules"
    assert j.get_state("stopped")["reason"] == "too much loss"
    g.stop("second opinion", "ai", NOW)  # the first reason wins
    assert g.stop_info()["reason"] == "too much loss"
    assert "guardian_stop" in set(j.events()["kind"])
    assert g.resume("human") and not g.stopped and j.get_state("stopped") is None
    assert not g.resume("human")
    assert "guardian_resume" in set(j.events()["kind"])


def test_drawdown_check_tracks_peak(cfg, tmp_path):
    cfg.guardian.max_drawdown_pct = 10
    g, j = make(cfg, tmp_path / "b")
    assert g.check_equity(1000, NOW) is None and j.get_state("peak_equity") == 1000
    assert g.check_equity(1100, NOW) is None and j.get_state("peak_equity") == 1100
    assert g.check_equity(1000, NOW) is None  # -9.1%: fine
    reason = g.check_equity(989, NOW)  # -10.1%: stop
    assert reason and "drawdown" in reason and g.stopped


def test_rules_fire_on_bad_runs(cfg, tmp_path):
    cfg.guardian.max_consecutive_losing_days = 3
    cfg.guardian.max_loss_30d_pct = 6
    cfg.guardian.max_halts_30d = 2
    cfg.guardian.min_trades_for_judgement = 10
    g, j = make(cfg, tmp_path / "c", backtest={"avg_r": 1.5})
    day0 = NOW.date() - timedelta(days=6)
    for i in range(5):
        j.record_daily(
            day0 + timedelta(days=i), 1000 - 15 * i, 1000 - 15 * (i + 1), -15, 1, 1, i < 2
        )
    reasons = g.rule_reasons(NOW)
    assert any("consecutive" in r for r in reasons)
    assert any("halts" in r for r in reasons)
    assert any("30 days" in r for r in reasons)  # 75 lost on 1000 = 7.5% > 6%
    for _ in range(12):
        trade(j, -0.5)
    assert any("average R" in r for r in g.rule_reasons(NOW))
    out = g.daily_review(NOW)
    assert out["stopped"] and g.stopped and g.stop_info()["source"] == "rules"


def test_rules_quiet_on_a_normal_run(cfg, tmp_path):
    g, j = make(cfg, tmp_path / "d", backtest={"avg_r": 0.3})
    day0 = NOW.date() - timedelta(days=4)
    for i, pnl in enumerate([5, -3, 4, -2]):
        j.record_daily(day0 + timedelta(days=i), 1000, 1000 + pnl, pnl, 1, 0.5, False)
    for r in [1.0, -1.0, 0.5, 0.2, -0.8, 1.2, -1.0, 0.4, 0.1, -0.3, 0.9]:
        trade(j, r)
    assert g.rule_reasons(NOW) == []
    assert not g.daily_review(NOW)["stopped"]


def test_ai_can_stop_but_never_resume(cfg, tmp_path):
    cfg.guardian.min_confidence = 0.7
    stop = {"verdict": "stop", "confidence": 0.9, "reason": "losses accelerating", "notes": ""}
    g, j = make(cfg, tmp_path / "e", llm=FakeLLM(stop))
    out = g.daily_review(NOW)
    assert out["stopped"] and g.stop_info()["source"] == "ai"
    assert j.llm_spend_today("guardian", NOW.date()) == 0.01
    # a cheerful AI cannot clear a stop
    g2 = Guardian(
        cfg,
        j,
        llm=FakeLLM({"verdict": "continue", "confidence": 1.0, "reason": "fine", "notes": ""}),
    )
    assert g2.daily_review(NOW).get("already") and g2.stopped
    # low confidence, error and refusal all mean "no opinion today"
    g_err, j_err = make(cfg, tmp_path / "f_err", llm=FakeLLM(raise_exc=RuntimeError("boom")))
    assert not g_err.daily_review(NOW)["stopped"] and not g_err.stopped
    assert "guardian_error" in set(j_err.events()["kind"])
    for name, llm in (
        ("low", FakeLLM({"verdict": "stop", "confidence": 0.3, "reason": "meh", "notes": ""})),
        ("refused", FakeLLM(refused=True)),
    ):
        g3, _ = make(cfg, tmp_path / f"f_{name}", llm=llm)
        assert not g3.daily_review(NOW)["stopped"] and not g3.stopped


def test_engine_caps_the_stop_distance(cfg):
    cfg.risk.max_stop_pct = 1.0
    cm = cost_model(cfg)
    journal = Journal(":memory:")
    engine = Engine(
        cfg,
        StubStrategy(fire_at=2, stop_pct=5.0),
        {},
        PaperBroker(1000, cm),
        RiskManager(cfg.risk, cost_model=cm),
        journal,
    )
    df = bars_from_closes([100.0, 100.0, 100.0])
    for i in range(1, 4):
        w = df.iloc[:i]
        w.attrs = dict(df.attrs)
        engine.on_bar("BTC/USD", w)
    t = engine.open["BTC/USD"]
    assert abs(t.stop - 99.0) < 1e-9 and t.reason.endswith("(stop capped)")
    assert "stop_capped" in set(journal.events()["kind"])


def make_runner(cfg, where, now_holder, df, guardian):
    cfg.symbols = ["BTC/USD"]
    cm = cost_model(cfg)
    engine = Engine(
        cfg,
        StubStrategy(fire_at=2, stop_pct=1.0),
        {},
        PaperBroker(1000, cm),
        RiskManager(cfg.risk, cost_model=cm),
        guardian.journal,
        mode="paper",
    )
    runner = Runner(
        cfg,
        "paper",
        engine,
        lambda s: df[df.index <= now_holder["now"]],
        guardian.journal,
        Notifier(enabled=False),
        clock=lambda: now_holder["now"],
        sleep=lambda s: None,
        guardian=guardian,
    )
    return runner, engine


def test_runner_refuses_to_start_when_stopped(cfg, tmp_path):
    g, _ = make(cfg, tmp_path / "g")
    g.stop("bad week", "rules", NOW)
    df = bars_from_closes([100.0] * 4)
    now = {"now": df.index[1] + timedelta(minutes=5, seconds=1)}
    runner, engine = make_runner(cfg, tmp_path / "g", now, df, g)
    runner.run_forever(max_steps=3)
    assert runner.steps == 0 and not engine.open and not engine.trades


def test_runner_stops_completely_on_drawdown(cfg, tmp_path):
    cfg.guardian.max_drawdown_pct = 0.01  # any loss at all
    g, j = make(cfg, tmp_path / "h")
    df = bars_from_closes([100.0] * 6)
    now = {"now": df.index[1] + timedelta(minutes=5, seconds=1)}
    runner, engine = make_runner(cfg, tmp_path / "h", now, df, g)
    # entries fill at the next bar's open, so let each loop step advance the clock one bar
    runner.sleep = lambda s: now.__setitem__("now", now["now"] + timedelta(minutes=5))
    runner.run_forever(max_steps=5)  # bar 2 queues, bar 3 enters; costs dent equity -> stop
    assert g.stopped and runner.stop_requested and not engine.open
    assert runner.steps < 5
    kinds = set(j.events()["kind"])
    assert "guardian_stop" in kinds and "flatten" in kinds
    assert engine.trades and engine.trades[0].exit_reason == "guardian"


def test_cli_guardian_check_and_resume(cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["--config", cfg.path, "guardian", "--dry-run"]) == 0
    assert main(["--config", cfg.path, "guardian"]) == 0  # nothing fired, nothing stopped
    assert not Path("cryptobot.STOPPED").exists()
    Guardian(cfg, Journal(cfg.journal_path, mode="paper")).stop("manual test", "rules", NOW)
    assert Path(cfg.guardian.stop_file).exists()
    assert main(["--config", cfg.path, "check"]) == 1
    assert main(["--config", cfg.path, "guardian"]) == 1  # already stopped
    assert main(["--config", cfg.path, "paper"]) == 4  # refuses to trade
    assert main(["--config", cfg.path, "resume"]) == 0
    assert not Path(cfg.guardian.stop_file).exists()
    assert main(["--config", cfg.path, "resume"]) == 0
