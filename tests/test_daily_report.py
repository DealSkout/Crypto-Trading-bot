"""Section 6: the daily report (file + Telegram + email) and the baseline command."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from cryptobot import daily_report
from cryptobot.backtest import cost_model
from cryptobot.brokers.paper import PaperBroker
from cryptobot.engine import Engine
from cryptobot.journal import Journal
from cryptobot.main import main
from cryptobot.notify import Notifier
from cryptobot.regime import RegimeDetector
from cryptobot.registry import Registry
from cryptobot.risk import RiskManager
from cryptobot.runner import Runner
from tests.conftest import StubStrategy, bars_from_closes


def paper_days(cfg, journal, n_days=3):
    """A few paper days through the real engine, so the journal has daily rows and trades."""
    cm = cost_model(cfg)
    engine = Engine(
        cfg,
        StubStrategy(fire_at=2, stop_pct=1.0, target_pct=1.0),
        {},
        PaperBroker(1000, cm),
        RiskManager(cfg.risk, cost_model=cm),
        journal,
        mode="paper",
    )
    frames = []
    for d in range(n_days):
        day = (pd.Timestamp("2026-10-10") + timedelta(days=d)).strftime("%Y-%m-%d")
        closes = [100.0, 100.0, 100.5, 101.5, 101.5] if d % 2 == 0 else [100.0, 100.0, 99.5, 98.5]
        frames.append(bars_from_closes(closes, day=day))
    for df in frames:
        for i in range(1, len(df) + 1):
            w = df.iloc[:i]
            w.attrs = dict(df.attrs)
            engine.on_bar("BTC/USD", w)
    engine.end_of_data()
    journal.log("halt", "daily loss 2.1% <= -2%", ts=datetime(2026, 10, 12, 3, tzinfo=UTC))
    journal.log(
        "pause_start",
        "macro window FOMC",
        symbol="BTC/USD",
        ts=datetime(2026, 10, 12, 4, tzinfo=UTC),
    )
    return engine


def test_report_has_every_section(cfg, tmp_path):
    journal = Journal(cfg.journal_path, mode="paper")
    paper_days(cfg, journal)
    reg = Registry(cfg.journal_path)
    reg.sync_with_config("orb", {"range_minutes": 5})
    shadow = reg.add("orb", {"range_minutes": 15}, {}, "candidate", "x")
    reg.set_status(shadow.id, "shadow", "passed the gate")
    Path(cfg.reports_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.reports_dir) / "baseline.json").write_text(
        json.dumps(
            {
                "backtests": {
                    "orb": {
                        "net_return_pct": 3.2,
                        "sharpe": 1.1,
                        "sortino": 1.6,
                        "max_drawdown_pct": -4.0,
                        "win_rate_pct": 45.0,
                        "profit_factor": 1.3,
                        "n_trades": 120,
                    }
                }
            }
        )
    )
    idx = pd.date_range("2026-09-01", periods=40 * 4, freq="6h", tz="UTC")
    flat = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0}, index=idx
    )
    regimes = RegimeDetector.from_bars(cfg.regime, {s: flat for s in cfg.symbols})
    rep = daily_report.build_report(cfg, journal, date(2026, 10, 12), reg, None, regimes)
    md = daily_report.render_markdown(rep)
    for heading in (
        "## P&L",
        "## Metrics vs baseline",
        "## Regime",
        "## Champion vs challengers",
        "## Risk events",
    ):
        assert heading in md, heading
    assert "Baseline orb (backtest)" in md and "Sortino" in md
    assert f"Shadow {shadow.id}" in md
    assert "halt" in md and "pause_start" in md
    assert rep["regimes"]["BTC/USD"] in ("ranging", "trending", "high_vol", "risk_off", "unknown")
    assert rep["last_30d"]["n_trades"] >= 1


def test_report_is_written_and_sent_by_telegram_and_email(cfg, monkeypatch):
    journal = Journal(cfg.journal_path, mode="paper")
    paper_days(cfg, journal)
    rep = daily_report.build_report(cfg, journal, date(2026, 10, 12))
    sent_tg = []
    notifier = Notifier(enabled=False)
    notifier.send = lambda text: sent_tg.append(text) or True
    mails = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            self.host = host

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            pass

        def login(self, user, password):
            pass

        def send_message(self, msg):
            mails.append(msg)

    monkeypatch.setattr(daily_report.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("REPORT_EMAIL_TO", "me@example.com")
    path = daily_report.write_and_send(cfg, rep, notifier)
    assert path.exists() and path.with_suffix(".json").exists()
    assert sent_tg and sent_tg[0].startswith("Cryptobot 2026-10-12")
    assert len(sent_tg[0]) < 4000
    assert mails and mails[0]["Subject"] == "Cryptobot daily report 2026-10-12"


def test_email_is_skipped_without_smtp_settings(monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    assert daily_report.send_email("x", "y") is False


def test_runner_writes_the_report_when_the_day_closes(cfg):
    cfg.symbols = ["BTC/USD"]
    day1 = bars_from_closes([100, 100, 100, 100], wick=0.0)
    day2 = bars_from_closes([100, 100], day="2026-10-06", wick=0.0)
    df = pd.concat([day1, day2])
    df.attrs = dict(day1.attrs)
    now = {"now": df.index[4] + timedelta(minutes=5, seconds=1)}
    cm = cost_model(cfg)
    engine = Engine(
        cfg,
        StubStrategy(fire_at=2, stop_pct=1.0),
        {},
        PaperBroker(1000, cm),
        RiskManager(cfg.risk, cost_model=cm),
        Journal(cfg.journal_path, mode="paper"),
        mode="paper",
    )
    runner = Runner(
        cfg,
        "paper",
        engine,
        lambda s: df[df.index <= now["now"]],
        engine.journal,
        Notifier(enabled=False),
        clock=lambda: now["now"],
        sleep=lambda s: None,
    )
    now["now"] = df.index[1] + timedelta(minutes=5, seconds=1)
    runner.step()
    now["now"] = df.index[4] + timedelta(minutes=5, seconds=1)
    runner.step()
    assert (Path(cfg.reports_dir) / "daily" / "2026-10-05.md").exists()


def test_daily_report_cli(cfg, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    journal = Journal(cfg.journal_path, mode="paper")
    paper_days(cfg, journal)
    args = ["--config", cfg.path, "daily-report", "--date", "2026-10-12", "--no-send"]
    assert main(args) == 0
    assert (Path(cfg.reports_dir) / "daily" / "2026-10-12.md").exists()


def test_baseline_cli_saves_metrics_for_every_strategy(cfg, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg_path = cfg.path
    assert main(["--config", cfg_path, "baseline", "--data", "synthetic", "--days", "30"]) == 0
    base = json.loads((Path(cfg.reports_dir) / "baseline.json").read_text())
    assert set(base["backtests"]) >= {"orb", "vwap_revert"}
    for m in base["backtests"].values():
        for key in (
            "net_return_pct",
            "sharpe",
            "sortino",
            "max_drawdown_pct",
            "win_rate_pct",
            "profit_factor",
            "n_trades",
        ):
            assert key in m


@pytest.mark.parametrize("value,expected", [(None, "n/a"), (float("inf"), "inf"), (1.234, "1.23")])
def test_number_formatting(value, expected):
    assert daily_report._fmt("{:.2f}", value) == expected
