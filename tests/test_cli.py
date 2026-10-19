from pathlib import Path

from cryptobot.main import main


def test_backtest_cli_writes_reports(cfg, monkeypatch):
    monkeypatch.chdir(Path(cfg.path).parent)
    rc = main(["--config", cfg.path, "backtest", "--data", "synthetic", "--days", "30"])
    assert rc == 0
    reports = list(Path(cfg.reports_dir).glob("backtest_orb_*.md"))
    assert reports and (Path(cfg.reports_dir) / "latest_orb.json").exists()
    assert main(["--config", cfg.path, "report"]) == 0


def test_live_refuses_without_mode_and_confirm(cfg, monkeypatch):
    monkeypatch.delenv("CRYPTOBOT_LIVE_CONFIRM", raising=False)
    assert main(["--config", cfg.path, "live"]) == 2
    text = (
        Path(cfg.path)
        .read_text()
        .replace("mode: paper", "mode: live")
        .replace("broker: paper", "broker: alpaca")
    )
    Path(cfg.path).write_text(text)
    assert main(["--config", cfg.path, "live"]) == 2  # env var missing
    monkeypatch.setenv("CRYPTOBOT_LIVE_CONFIRM", "yes")
    assert main(["--config", cfg.path, "live"]) == 3  # Rule 6 gate not met
    assert main(["--config", cfg.path, "check"]) == 1


def test_coach_requires_provider(cfg):
    assert main(["--config", cfg.path, "coach"]) == 2


def test_bad_config_path():
    assert main(["--config", "/nonexistent.yaml", "check"]) == 2


def test_backtest_llm_flag_needs_provider(cfg):
    assert (
        main(["--config", cfg.path, "backtest", "--data", "synthetic", "--days", "12", "--llm"])
        == 2
    )
