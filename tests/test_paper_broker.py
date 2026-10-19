import pytest

from cryptobot.brokers.base import BrokerError
from cryptobot.brokers.paper import PaperBroker
from cryptobot.costs import CostModel


def test_buy_sell_accounting():
    cm = CostModel(commission=1.0, fee_pct=0.1, spread_bps=10, slippage_bps=0)
    b = PaperBroker(1000, cm)
    b.mark("BTC/USD", 100.0)
    o = b.place_order("BTC/USD", "buy", 5)
    assert o.is_filled and o.avg_fill_price == pytest.approx(100.05)
    assert o.fees == pytest.approx(1 + 500.25 * 0.001)
    assert b.get_cash() == pytest.approx(1000 - 500.25 - o.fees)
    assert b.get_positions()[0].qty == 5 and b.get_positions()[0].side == "long"
    b.mark("BTC/USD", 110.0)
    assert b.get_equity() == pytest.approx(b.get_cash() + 550)
    b.set_exec_price("BTC/USD", 105.0)
    s = b.place_order("BTC/USD", "sell", 5)
    assert s.avg_fill_price == pytest.approx(104.9475)
    assert b.get_positions() == [] and b.realized_pnl == pytest.approx((104.9475 - 100.05) * 5)
    assert b.total_implicit == pytest.approx(0.05 * 5 + 0.0525 * 5)


def test_rejections_and_errors():
    b = PaperBroker(100, CostModel(0, 0, 0, 0))
    with pytest.raises(BrokerError):
        b.place_order("BTC/USD", "buy", 1)
    b.mark("BTC/USD", 60.0)
    o = b.place_order("BTC/USD", "buy", 2)
    assert o.status == "rejected" and "cash" in o.note
    assert b.place_order("BTC/USD", "buy", 0).status == "rejected"


def test_flatten_all_closes_everything():
    b = PaperBroker(10_000, CostModel(0, 0, 0, 0))
    b.mark("BTC/USD", 100.0)
    b.mark("ETH/USD", 50.0)
    b.place_order("BTC/USD", "buy", 10)
    b.place_order("ETH/USD", "sell", 10)
    orders = b.flatten_all()
    assert len(orders) == 2 and b.get_positions() == []
    assert b.get_cash() == pytest.approx(10_000)
