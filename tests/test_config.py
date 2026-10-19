import pytest

from cryptobot.config import ConfigError, from_dict, live_confirmed, load_config, write_params


def test_defaults_and_validation(tmp_path):
    cfg = from_dict({"symbols": ["btcusd", "eth/usd"]})
    assert cfg.mode == "paper" and cfg.symbols == ["BTC/USD", "ETH/USD"]
    assert cfg.timeframe_minutes == 5 and cfg.broker == "alpaca"
    assert cfg.orders.time_in_force == "gtc" and cfg.orders.entry_type == "market"
    assert cfg.risk.max_total_exposure_pct == 100.0 and cfg.costs.fee_pct == 0.25
    assert cfg.risk.risk_per_trade_pct == 0.5 and cfg.coach.provider == "none"
    with pytest.raises(ConfigError):
        from_dict({"symbols": ["BTC/USD"], "mode": "yolo"})
    with pytest.raises(ConfigError):
        from_dict({"symbols": ["BTC/USD"], "mode": "live", "broker": "paper"})
    with pytest.raises(ConfigError):
        from_dict({"symbols": ["BTC/USD"], "timeframe": "5h"})
    with pytest.raises(ConfigError):
        from_dict({"symbols": ["BTC/USD"], "risk": {"nope": 1}})
    with pytest.raises(ConfigError):
        from_dict({"symbols": []})
    with pytest.raises(ConfigError):
        from_dict({"symbols": ["BTC/USD"], "risk": {"risk_per_trade_pct": 50}})


@pytest.mark.parametrize(
    "raw",
    [
        {"symbols": ["SPY"]},  # stocks belong to the stock bot
        {"symbols": ["BTC/EUR"]},  # USD pairs only
        {"symbols": ["BTC/USD"], "allow_short": True},  # long-only
        {"symbols": ["BTC/USD"], "orders": {"time_in_force": "day"}},  # crypto: gtc/ioc only
        {"symbols": ["BTC/USD"], "orders": {"entry_type": "stop"}},
        {"symbols": ["BTC/USD"], "risk": {"max_total_exposure_pct": 150}},  # no leverage
        {"symbols": ["BTC/USD"], "broker": "trading212"},
    ],
)
def test_crypto_rules_are_validated(raw):
    with pytest.raises(ConfigError):
        from_dict(raw)


def test_live_confirm_env(monkeypatch):
    monkeypatch.delenv("CRYPTOBOT_LIVE_CONFIRM", raising=False)
    assert not live_confirmed()
    monkeypatch.setenv("CRYPTOBOT_LIVE_CONFIRM", "YES")
    assert live_confirmed()
    monkeypatch.setenv("CRYPTOBOT_LIVE_CONFIRM", "true")
    assert not live_confirmed()


def test_write_params_keeps_comments(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "# keep me\nmode: paper\nsymbols: [BTC/USD]\nparams: {range_minutes: 5}\n"
        "risk:\n  max_positions: 3  # also keep\n"
    )
    write_params(p, {"range_minutes": 15, "target_r": 3})
    text = p.read_text()
    assert "# keep me" in text and "# also keep" in text
    cfg = load_config(p)
    assert cfg.params == {"range_minutes": 15, "target_r": 3}
    assert cfg.risk.max_positions == 3
