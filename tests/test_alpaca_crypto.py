from types import SimpleNamespace as NS

import pytest
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, StopLimitOrderRequest

from cryptobot.brokers.alpaca import AlpacaCryptoBroker
from cryptobot.brokers.base import BrokerError


class FakeTradingClient:
    def __init__(self, positions=None, account=None, asset=None, fill=True, fail=None):
        self.positions = positions or []
        self.acct = account or NS(cash="1000", equity="1050", non_marginable_buying_power="900")
        self.asset = asset or NS(min_order_size="0.0001", min_trade_increment="0.0001")
        self.fill, self.fail = fill, fail
        self.submitted, self.cancelled, self.orders = [], [], {}

    def get_account(self):
        return self.acct

    def get_all_positions(self):
        return self.positions

    def get_asset(self, symbol):
        return self.asset

    def submit_order(self, order_data):
        if self.fail:
            raise RuntimeError(self.fail)
        self.submitted.append(order_data)
        oid = f"o{len(self.submitted)}"
        o = NS(
            id=oid,
            symbol=order_data.symbol,
            side=order_data.side,
            qty=str(order_data.qty),
            filled_qty="0",
            filled_avg_price=None,
            status="accepted",
            order_type="market",
            limit_price=getattr(order_data, "limit_price", None),
            asset_class="crypto",
        )
        self.orders[oid] = o
        return o

    def get_order_by_id(self, oid):
        o = self.orders[oid]
        if self.fill:
            o.status, o.filled_qty, o.filled_avg_price = "filled", o.qty, "100.5"
        return o

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)

    def get_orders(self, filter=None):
        return [
            NS(
                id="x1",
                symbol="BTC/USD",
                side="buy",
                qty="1",
                filled_qty="0",
                status="new",
                asset_class="crypto",
            ),
            NS(
                id="x2",
                symbol="SPY",
                side="buy",
                qty="1",
                filled_qty="0",
                status="new",
                asset_class="us_equity",
            ),
        ]


def broker(client, **kw):
    return AlpacaCryptoBroker("", "", client=client, sleep=lambda s: None, **kw)


def test_account_uses_cash_never_margin():
    b = broker(FakeTradingClient())
    assert b.get_cash() == 900.0 and b.get_equity() == 1050.0
    with pytest.raises(BrokerError):
        AlpacaCryptoBroker("", "")  # no keys and no client
    with pytest.raises(BrokerError):
        broker(FakeTradingClient(), time_in_force="day")  # crypto: gtc or ioc only


def test_positions_are_crypto_only_and_normalised():
    pos = [
        NS(symbol="BTCUSD", qty="0.5", avg_entry_price="60000", asset_class="crypto"),
        NS(symbol="SPY", qty="3", avg_entry_price="500", asset_class="us_equity"),
    ]
    b = broker(FakeTradingClient(positions=pos))
    assert [(p.symbol, p.qty, p.side) for p in b.get_positions()] == [("BTC/USD", 0.5, "long")]
    assert b.held("BTC/USD") == 0.5 and b.held("ETH/USD") == 0


def test_market_buy_rounds_to_the_increment_and_uses_gtc():
    c = FakeTradingClient()
    b = broker(c)
    o = b.await_fill(b.place_order("btcusd", "buy", 0.123456789))
    req = c.submitted[0]
    assert isinstance(req, MarketOrderRequest)
    assert req.symbol == "BTC/USD" and abs(req.qty - 0.1234) < 1e-12
    assert req.time_in_force.value == "gtc" and req.client_order_id.startswith("cryptobot-")
    assert o.is_filled and o.avg_fill_price == 100.5 and o.symbol == "BTC/USD"


def test_client_order_id_carries_the_order_tag():
    """The team router tags each residual with the bot id so the shared account shows who
    traded; the tag is sanitised and the whole id stays inside Alpaca's 48 characters."""
    c = FakeTradingClient()
    b = broker(c)
    b.set_order_tag("bot03")
    b.place_order("BTC/USD", "buy", 0.5)
    b.set_order_tag("bad tag/with:chars!" + "x" * 40)
    b.place_order("BTC/USD", "buy", 0.5)
    first, second = (r.client_order_id for r in c.submitted)
    assert first.startswith("bot03-") and len(first) == len("bot03-") + 20
    assert second.startswith("badtagwithcharsxxxxxxxxx-") and len(second) <= 48
    assert first != second
    b.set_order_tag("")
    b.place_order("BTC/USD", "buy", 0.5)
    assert c.submitted[-1].client_order_id.startswith("cryptobot-")


def test_limit_and_stop_limit_with_ioc():
    c = FakeTradingClient()
    b = broker(c, time_in_force="ioc")
    b.place_order("ETH/USD", "buy", 1, order_type="limit", limit_price=2500.123456789)
    b.place_order("ETH/USD", "buy", 1, order_type="stop_limit", stop_price=2600, limit_price=2610)
    lim, stop = c.submitted
    assert isinstance(lim, LimitOrderRequest) and lim.time_in_force.value == "ioc"
    assert isinstance(stop, StopLimitOrderRequest) and stop.stop_price == 2600
    with pytest.raises(BrokerError):
        b.place_order("ETH/USD", "buy", 1, order_type="stop")  # not allowed for crypto
    with pytest.raises(BrokerError):
        b.place_order("ETH/USD", "buy", 1, order_type="limit")  # missing limit price


def test_long_only_sells_never_exceed_holdings():
    pos = [NS(symbol="BTCUSD", qty="0.3", avg_entry_price="60000", asset_class="crypto")]
    c = FakeTradingClient(positions=pos)
    b = broker(c)
    b.place_order("BTC/USD", "sell", 1.0)
    assert c.submitted[0].qty == 0.3  # capped to what is held
    o = broker(FakeTradingClient()).place_order("ETH/USD", "sell", 1.0)
    assert o.status == "rejected" and "long-only" in o.note


def test_below_min_order_size_is_not_sent():
    c = FakeTradingClient(asset=NS(min_order_size="0.01", min_trade_increment="0.001"))
    o = broker(c).place_order("BTC/USD", "buy", 0.005)
    assert o.status == "rejected" and c.submitted == []


def test_errors_carry_alpacas_message_and_open_orders_are_crypto_only():
    b = broker(FakeTradingClient(fail="insufficient balance for USD"))
    with pytest.raises(BrokerError, match="insufficient balance"):
        b.place_order("BTC/USD", "buy", 1)
    orders = broker(FakeTradingClient()).open_orders()
    assert [o.id for o in orders] == ["x1"]


def test_unfilled_order_times_out_and_flatten_cancels_then_sells():
    pos = [NS(symbol="ETHUSD", qty="2", avg_entry_price="2500", asset_class="crypto")]
    c = FakeTradingClient(positions=pos)
    b = broker(c)
    out = b.flatten_all()
    assert c.cancelled == ["x1"]  # the stock order on the same account is left alone
    assert out and out[0].is_filled and c.submitted[-1].side.value == "sell"
    slow = broker(FakeTradingClient(fill=False))
    pending = slow.await_fill(slow.place_order("BTC/USD", "buy", 1), timeout=0.01)
    assert not pending.is_done


def test_protective_stop_places_cancels_and_reports_a_racing_fill():
    client = FakeTradingClient(
        positions=[NS(symbol="BTCUSD", qty="0.5", asset_class="crypto", avg_entry_price="100")],
        fill=False,
    )
    b = broker(client)
    order = b.place_protective_stop("BTC/USD", 0.5, 95.0, side="sell")
    assert order.id and isinstance(client.submitted[-1], StopLimitOrderRequest)
    req = client.submitted[-1]
    assert float(req.stop_price) == pytest.approx(95.0)
    assert float(req.limit_price) == pytest.approx(95.0 * (1 - b.PROTECTIVE_LIMIT_GAP))

    # not filled: cancel_protective cancels and returns None
    assert b.cancel_protective("BTC/USD") is None
    assert client.cancelled == [order.id]
    # idempotent once cleared
    assert b.cancel_protective("BTC/USD") is None

    # filled at the exchange before we could cancel: the fill comes back for booking
    order2 = b.place_protective_stop("BTC/USD", 0.5, 95.0, side="sell")
    client.fill = True
    filled = b.cancel_protective("BTC/USD")
    assert filled is not None and filled.id == order2.id
    assert filled.filled_qty == pytest.approx(0.5)


def test_placing_a_new_protective_stop_replaces_the_old_one():
    client = FakeTradingClient(
        positions=[NS(symbol="BTCUSD", qty="0.5", asset_class="crypto", avg_entry_price="100")],
        fill=False,
    )
    b = broker(client)
    first = b.place_protective_stop("BTC/USD", 0.5, 95.0)
    b.place_protective_stop("BTC/USD", 0.5, 96.0)
    assert client.cancelled == [first.id]  # the stale stop was cancelled first
