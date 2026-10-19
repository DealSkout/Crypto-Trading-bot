from cryptobot.costs import CostModel


def test_fill_price_moves_against_us():
    cm = CostModel(commission=0, fee_pct=0, spread_bps=2, slippage_bps=3)
    assert cm.fill_price(100.0, True) == 100.0 * (1 + 0.0004)
    assert cm.fill_price(100.0, False) == 100.0 * (1 - 0.0004)


def test_fees_fx_and_commission():
    cm = CostModel(commission=1.0, fee_pct=0.15, spread_bps=0, slippage_bps=0)
    assert cm.fees(1000.0, fx=True) == 1.0 + 1.5
    assert cm.fees(1000.0, fx=False) == 1.0


def test_round_trip_cost_pct():
    cm = CostModel(commission=0, fee_pct=0.15, spread_bps=2, slippage_bps=3)
    # spread once (half each side) + slippage twice + fx twice
    assert abs(cm.round_trip_cost_pct(fx=True) - (0.02 + 0.06 + 0.30)) < 1e-9
    assert abs(cm.round_trip_cost_pct(fx=False) - 0.08) < 1e-9


def test_zero_cost_model_is_identity():
    cm = CostModel(0, 0, 0, 0)
    assert cm.fill_price(55.5, True) == 55.5 and cm.fees(1e6) == 0.0
