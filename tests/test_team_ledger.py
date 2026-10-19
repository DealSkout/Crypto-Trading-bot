import pytest

from cryptobot.team.ledger import VirtualAccount


def test_buy_sell_accounting():
    a = VirtualAccount("alpha", 1000.0)
    a.apply_fill("BTC/USD", "buy", 2, 100.0, 1.0)
    assert a.cash == pytest.approx(1000 - 200 - 1)
    assert a.positions["BTC/USD"] == (2, 100.0)
    a.apply_fill("BTC/USD", "buy", 2, 110.0, 1.0)
    qty, avg = a.positions["BTC/USD"]
    assert qty == pytest.approx(4) and avg == pytest.approx(105.0)
    a.apply_fill("BTC/USD", "sell", 1, 120.0, 0.5)
    assert a.cash == pytest.approx(578 + 120 - 0.5)
    assert a.realized_pnl == pytest.approx(15.0)
    assert a.fees_paid == pytest.approx(2.5)
    qty, avg = a.positions["BTC/USD"]
    assert qty == pytest.approx(3) and avg == pytest.approx(105.0)


def test_full_close_removes_position():
    a = VirtualAccount("alpha", 1000.0)
    a.apply_fill("BTC/USD", "buy", 2, 100.0, 0.0)
    a.apply_fill("BTC/USD", "sell", 2, 90.0, 0.0)
    assert "BTC/USD" not in a.positions
    assert a.realized_pnl == pytest.approx(-20.0)
    assert a.cash == pytest.approx(980.0)


def test_equity_marks_and_fallback():
    a = VirtualAccount("alpha", 200.0)
    a.apply_fill("BTC/USD", "buy", 2, 50.0, 0.0)
    a.apply_fill("ETH/USD", "buy", 1, 10.0, 0.0)
    assert a.cash == pytest.approx(90.0)
    assert a.equity({"BTC/USD": 60.0, "ETH/USD": 12.0}) == pytest.approx(90 + 120 + 12)
    # a missing mark falls back to the average price paid
    assert a.equity({"BTC/USD": 60.0}) == pytest.approx(90 + 120 + 10)
    assert a.equity() == pytest.approx(90 + 100 + 10)


def test_bad_fills_raise():
    a = VirtualAccount("alpha", 100.0)
    with pytest.raises(ValueError):
        a.apply_fill("BTC/USD", "sell", 1, 10.0, 0.0)  # nothing held
    a.apply_fill("BTC/USD", "buy", 1, 10.0, 0.0)
    with pytest.raises(ValueError):
        a.apply_fill("BTC/USD", "sell", 2, 10.0, 0.0)  # would go negative
    with pytest.raises(ValueError):
        a.apply_fill("BTC/USD", "hold", 1, 10.0, 0.0)
    with pytest.raises(ValueError):
        a.apply_fill("BTC/USD", "buy", 0, 10.0, 0.0)
    # nothing above should have corrupted the books
    assert a.positions["BTC/USD"] == (1, 10.0) and a.cash == pytest.approx(90.0)


def test_snapshot_round_trip():
    a = VirtualAccount("alpha", 500.0)
    a.apply_fill("BTC/USD", "buy", 2, 100.0, 1.0)
    a.apply_fill("BTC/USD", "sell", 1, 110.0, 1.0)
    snap = a.snapshot()
    assert isinstance(snap, dict) and snap["bot_id"] == "alpha"
    b = VirtualAccount.from_snapshot(snap)
    assert b.bot_id == a.bot_id
    assert b.cash == pytest.approx(a.cash)
    assert b.positions == a.positions
    assert b.realized_pnl == pytest.approx(a.realized_pnl)
    assert b.fees_paid == pytest.approx(a.fees_paid)
    assert b.equity({"BTC/USD": 120.0}) == pytest.approx(a.equity({"BTC/USD": 120.0}))


def test_distribution_invariant():
    """Splitting every fill between sub-accounts changes nothing in aggregate:
    the sum of the slices' cash, realized P&L and equity matches an account
    that took every fill whole, to within 1e-6."""
    fills = [
        ("BTC/USD", "buy", 4, 100.0, 2.0),
        ("ETH/USD", "buy", 10, 20.0, 1.0),
        ("BTC/USD", "sell", 2, 110.0, 1.5),
        ("ETH/USD", "buy", 6, 22.0, 0.8),
        ("BTC/USD", "sell", 2, 95.0, 1.0),
        ("ETH/USD", "sell", 16, 21.0, 2.0),
    ]
    whole = VirtualAccount("whole", 10_000.0)
    a = VirtualAccount("a", 6_000.0)
    b = VirtualAccount("b", 4_000.0)
    for symbol, side, qty, price, fees in fills:
        whole.apply_fill(symbol, side, qty, price, fees)
        a.apply_fill(symbol, side, qty / 2, price, fees / 2)
        b.apply_fill(symbol, side, qty / 2, price, fees / 2)
    marks = {"BTC/USD": 105.0, "ETH/USD": 23.0}
    assert a.cash + b.cash == pytest.approx(whole.cash, abs=1e-6)
    assert a.realized_pnl + b.realized_pnl == pytest.approx(whole.realized_pnl, abs=1e-6)
    assert a.fees_paid + b.fees_paid == pytest.approx(whole.fees_paid, abs=1e-6)
    assert a.equity(marks) + b.equity(marks) == pytest.approx(whole.equity(marks), abs=1e-6)
