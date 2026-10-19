import math
from datetime import date

from cryptobot.config import RiskConfig
from cryptobot.costs import CostModel
from cryptobot.risk import RiskManager


def rm(**kw):
    return RiskManager(RiskConfig(**kw))


def test_position_size_risk_budget_and_rounding():
    r = rm(risk_per_trade_pct=0.5)
    qty, why = r.position_size(1000, 1000, 100, 99, lambda q: q)
    assert qty == 5 and why == "ok"
    qty, _ = r.position_size(1000, 1000, 100, 99.3, lambda q: float(math.floor(q)))
    assert qty == 7  # 5 / 0.7 = 7.14 -> 7 whole shares
    assert r.position_size(1000, 1000, 100, 100, lambda q: q)[0] == 0


def test_position_size_cash_cap_includes_costs():
    cm = CostModel(commission=1.0, fee_pct=0.15, spread_bps=2, slippage_bps=3)
    r = RiskManager(RiskConfig(risk_per_trade_pct=5), cost_model=cm)
    qty, why = r.position_size(1000, 1000, 100, 99, lambda q: q, symbol="BTC/USD")
    assert "cash-capped" in why
    fill = cm.fill_price(100, True)
    assert qty * fill * (1 + 0.0015) + 1.0 <= 1000 + 1e-9
    r2 = RiskManager(RiskConfig(risk_per_trade_pct=5), cost_model=cm, fx_for=lambda s: False)
    assert r2.position_size(1000, 1000, 100, 99, lambda q: q, symbol="VUSA.L")[0] > qty


def test_can_enter_rules():
    r = rm(max_positions=2, flatten_minutes_before_close=10)
    r.start_day(date(2026, 10, 5), 1000)
    assert r.can_enter("BTC/USD", "long", 0, 300) == (True, "ok")
    assert not r.can_enter("BTC/USD", "short", 0, 300)[0]
    assert not r.can_enter("BTC/USD", "long", 2, 300)[0]
    assert not r.can_enter("BTC/USD", "long", 0, 10)[0]
    r.record_entry("BTC/USD")
    assert not r.can_enter("BTC/USD", "long", 0, 300)[0]
    assert r.can_enter("ETH/USD", "long", 0, 300)[0]
    r.block_entries("ETH/USD")
    assert not r.can_enter("ETH/USD", "long", 0, 300)[0]
    r.halt("test")
    assert "halted" in r.can_enter("BTC/USD", "long", 0, 300)[1]


def test_daily_loss_and_flatten_window():
    r = rm(max_daily_loss_pct=2)
    r.start_day(date(2026, 10, 5), 1000)
    assert not r.daily_loss_breached(985)
    assert r.daily_loss_breached(980) and r.state.halted
    assert r.should_flatten(10) and not r.should_flatten(11)


def test_check_exit_logic():
    ce = RiskManager.check_exit
    assert ce("long", 99, 110, 100, 101, 98).reason == "stop"
    assert ce("long", 99, 110, 100, 101, 98).price == 99
    assert ce("long", 99, 110, 97, 98, 96).price == 97  # gapped through: fill at open
    assert ce("long", 99, 110, 100, 111, 99.5).reason == "target"
    assert ce("long", 99, 110, 112, 113, 111).price == 112  # gap up: fill at open
    assert ce("long", 99, 110, 100, 111, 98).reason == "stop"  # both touched -> stop first
    assert ce("long", 99, None, 100, 101, 99.5) is None
    assert ce("short", 101, 90, 100, 102, 99).reason == "stop"
    assert ce("short", 101, 90, 100, 100.5, 89).price == 90


def test_total_exposure_cap_across_the_account():
    r = RiskManager(RiskConfig(risk_per_trade_pct=5, max_total_exposure_pct=50))
    qty, why = r.position_size(1000, 1000, 100, 99, lambda q: q, symbol="BTC/USD")
    assert why == "ok (capped by max_total_exposure_pct)" and qty * 100 <= 500 + 1e-9
    qty, why = r.position_size(
        1000, 1000, 100, 99, lambda q: q, symbol="ETH/USD", open_exposure=400
    )
    assert qty * 100 <= 100 + 1e-9  # only 100 of room left
    qty, why = r.position_size(
        1000, 1000, 100, 99, lambda q: q, symbol="ETH/USD", open_exposure=500
    )
    assert qty == 0 and "total exposure cap reached" in why


def test_crypto_is_long_only():
    r = RiskManager(RiskConfig())
    r.start_day(date(2026, 10, 5), 1000)
    ok, why = r.can_enter("BTC/USD", "short", 0, 600)
    assert not ok and "long-only" in why
