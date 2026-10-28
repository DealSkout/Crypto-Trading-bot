"""Item 2 and 3 on the real side: limit entries and broker-held stops through the
router's lot desk, against a venue with resting orders."""

from datetime import UTC, datetime, timedelta

import pytest

from cryptobot.brokers.venue import PaperVenue
from cryptobot.config import OrderConfig
from cryptobot.costs import CostModel
from cryptobot.team.ledger import VirtualAccount
from cryptobot.team.router import PortfolioLimits, Router

T0 = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)
CM = CostModel(commission=0.0, fee_pct=0.25, maker_fee_pct=0.15, spread_bps=0.0, slippage_bps=0.0)


def team(tmp_path, bots=("a", "b"), cash=10_000.0, venue_cls=PaperVenue):
    venue = venue_cls(cash * len(bots), CM)
    accounts = {b: VirtualAccount(b, cash) for b in bots}
    limits = PortfolioLimits(kill_switch_file=str(tmp_path / "team.KILLSWITCH"))
    orders = OrderConfig(exit_timeout_seconds=20.0, stop_limit_buffer_pct=0.5)
    router = Router(venue, accounts, limits, CM, clock=lambda: T0, order_settings=orders)
    return venue, accounts, router


def price(router, symbol, p, ts=T0):
    router.mark(symbol, p, ts)


def entry(lot="L1", qty=1.0, limit=100.0, stop=98.0, target=106.0, expires=None, symbol="BTC/USD"):
    return {"type": "entry", "lot": lot, "symbol": symbol, "side": "long", "qty": qty,
            "limit": limit, "stop": stop, "target": target, "signal_price": limit,
            "expires": expires or T0 + timedelta(minutes=15)}


def work(router, now=T0):
    return router.desk.work(now)


def identity(venue, accounts, router, marks):
    virtual = sum(a.equity(marks) for a in accounts.values())
    return venue.get_equity() - (virtual + router.internal_fees)


def test_limit_entry_rests_then_fills_and_the_stop_rests_at_the_broker(tmp_path):
    venue, accounts, router = team(tmp_path)
    price(router, "BTC/USD", 100.5)
    assert router.desk.on_event("a", entry(), T0) == []  # resting: nothing filled
    assert venue.open_orders()[0].type == "limit" and router.last_tag == "a"
    price(router, "BTC/USD", 99.9)  # trades through the limit
    reports = work(router)
    fill = [r for r in reports if r.get("side") == "buy"][0]
    assert fill["price"] == pytest.approx(100.0) and fill["fees"] == pytest.approx(0.15)  # maker
    assert accounts["a"].positions["BTC/USD"][0] == pytest.approx(1.0)
    stops = [o for o in venue.open_orders() if o.type == "stop_limit"]
    assert len(stops) == 1 and stops[0].limit_price == pytest.approx(98.0 * 0.995)
    # the stop is the exchange's job now: price falls through it
    price(router, "BTC/USD", 97.9)
    reports = work(router)
    assert any(r.get("broker_stop") for r in reports)
    assert accounts["a"].positions == {}
    result = next(r for r in work(router) + reports if r.get("lot_closed"))
    assert result["planned_r"] == pytest.approx(-1.0)
    assert result["actual_r"] < -1.0  # the fill below the stop plus both fees
    assert result["fees"] == pytest.approx(0.15 + 97.9 * 0.0025)
    assert abs(identity(venue, accounts, router, {"BTC/USD": 97.9})) < 1e-6


def test_an_entry_that_never_fills_is_cancelled_at_its_deadline(tmp_path):
    venue, accounts, router = team(tmp_path)
    price(router, "BTC/USD", 101.0)
    router.desk.on_event("a", entry(expires=T0 + timedelta(minutes=10)), T0)
    work(router, T0 + timedelta(minutes=5))
    assert venue.open_orders()
    work(router, T0 + timedelta(minutes=11))
    assert venue.open_orders() == []
    router.desk.on_event("a", {"type": "entry_cancel", "lot": "L1"}, T0)
    assert work(router, T0 + timedelta(minutes=12)) == [] and router.desk.lots == {}


def test_the_bot_is_told_when_the_broker_filled_less(tmp_path):
    venue, accounts, router = team(tmp_path)
    price(router, "BTC/USD", 101.0)
    router.desk.on_event("a", entry(expires=T0 + timedelta(minutes=10)), T0)
    # the simulation saw a low at 100 (its fill) but the exchange never traded there
    router.desk.on_event("a", {"type": "entry_fill", "lot": "L1", "qty": 1.0}, T0)
    reports = work(router, T0 + timedelta(minutes=11))
    shrink = [r for r in reports if r.get("shrink")]
    assert shrink == [{"shrink": True, "bot_id": "a", "lot": "L1", "real_qty": 0.0}]


def test_a_real_fill_the_bot_never_saw_is_unwound_at_the_market(tmp_path):
    venue, accounts, router = team(tmp_path)
    price(router, "BTC/USD", 100.5)
    router.desk.on_event("a", entry(), T0)
    price(router, "BTC/USD", 99.8)
    work(router)
    router.desk.on_event("a", {"type": "entry_cancel", "lot": "L1"}, T0)
    reports = work(router)
    sells = [r for r in reports if r.get("side") == "sell"]
    assert sells and venue.get_positions() == [] and accounts["a"].positions == {}
    assert any(r.get("lot_closed") and r["reason"] == "orphan" for r in reports + work(router))


def test_a_target_exit_rests_as_a_limit_then_goes_to_the_market(tmp_path):
    venue, accounts, router = team(tmp_path)
    price(router, "BTC/USD", 100.5)
    router.desk.on_event("a", entry(), T0)
    price(router, "BTC/USD", 100.0)
    work(router)
    router.desk.on_event("a", {"type": "entry_fill", "lot": "L1", "qty": 1.0}, T0)
    work(router)
    price(router, "BTC/USD", 106.2)
    router.desk.on_event("a", {"type": "exit", "lot": "L1", "qty": 1.0, "price": 106.0,
                               "reason": "target"}, T0)
    work(router)
    limits = [o for o in venue.open_orders() if o.type == "limit" and o.side == "sell"]
    stops = [o for o in venue.open_orders() if o.type == "stop_limit"]
    assert limits and not stops  # the stop was cancelled before the exit order went in
    price(router, "BTC/USD", 106.1)  # drifts away; the limit at 106.2 does not fill
    reports = work(router, T0 + timedelta(seconds=25))  # past exit_timeout_seconds
    assert any(r.get("side") == "sell" for r in reports)
    assert venue.get_positions() == [] and venue.open_orders() == []
    result = next(r for r in reports + work(router, T0 + timedelta(seconds=26))
                  if r.get("lot_closed"))
    assert result["reason"] == "target" and result["planned_r"] == pytest.approx(3.0)


def test_a_stop_out_the_broker_could_not_fill_goes_to_the_market(tmp_path):
    venue, accounts, router = team(tmp_path)
    price(router, "BTC/USD", 100.5)
    router.desk.on_event("a", entry(), T0)
    price(router, "BTC/USD", 100.0)
    work(router)
    router.desk.on_event("a", {"type": "entry_fill", "lot": "L1", "qty": 1.0}, T0)
    work(router)
    price(router, "BTC/USD", 96.0)  # gapped through the stop AND its limit: the stop-limit rests
    work(router)
    assert venue.get_positions()  # still held: a stop-limit cannot fill below its limit
    router.desk.on_event("a", {"type": "exit", "lot": "L1", "qty": 1.0, "price": 96.0,
                               "reason": "stop"}, T0)
    reports = work(router)
    assert venue.get_positions() == [] and venue.open_orders() == []
    assert [r for r in reports if r.get("side") == "sell"][0]["price"] == pytest.approx(96.0)


def test_the_kill_switch_refuses_new_entries(tmp_path):
    venue, accounts, router = team(tmp_path)
    price(router, "BTC/USD", 100.5)
    (tmp_path / "team.KILLSWITCH").write_text("x")
    reports = router.desk.on_event("a", entry(), T0)
    assert reports[0]["rejected"] and reports[0]["cancel_entry"]
    assert "kill switch" in reports[0]["reason"] and venue.open_orders() == []


def test_opposing_bots_still_net_internally(tmp_path):
    venue, accounts, router = team(tmp_path)
    price(router, "BTC/USD", 100.5)
    router.desk.on_event("b", entry(lot="B1", limit=100.0, stop=98.0), T0)
    price(router, "BTC/USD", 100.0)
    work(router)
    router.desk.on_event("b", {"type": "entry_fill", "lot": "B1", "qty": 1.0}, T0)
    work(router)
    orders_before = len(venue._done)
    price(router, "BTC/USD", 101.0)
    router.desk.on_event("a", entry(lot="A1", limit=101.0, stop=99.0, target=106.0), T0)
    router.desk.on_event("b", {"type": "exit", "lot": "B1", "qty": 1.0, "price": 101.0,
                               "reason": "rule"}, T0)
    reports = work(router)
    cross = [r for r in reports if r.get("internal")]
    assert cross and cross[0]["bots"] == ["a", "b"] and cross[0]["price"] == pytest.approx(101.0)
    assert accounts["a"].positions["BTC/USD"][0] == pytest.approx(1.0)
    assert "BTC/USD" not in accounts["b"].positions
    assert venue.get_positions()[0].qty == pytest.approx(1.0)  # the coins never left
    assert abs(identity(venue, accounts, router, {"BTC/USD": 101.0})) < 1e-6
    assert len(venue._done) <= orders_before + 2  # only cancels, no sell or buy for the cross


class FeeInCoinsVenue(PaperVenue):
    """Alpaca crypto: no fee on the order, a buy delivers coins minus the fee."""

    reports_fees = False

    def _fill(self, oid, price, maker):
        super()._fill(oid, price, maker)
        order = self._done[oid]
        if order.side == "buy":
            self.cash += order.fees
            self._qty[order.symbol] -= order.fees / price
        order.fees = 0.0


def test_the_broker_stop_covers_exactly_the_coins_received(tmp_path):
    venue, accounts, router = team(tmp_path, venue_cls=FeeInCoinsVenue)
    price(router, "BTC/USD", 100.5)
    router.desk.on_event("a", entry(), T0)
    price(router, "BTC/USD", 100.0)
    work(router)
    router.desk.on_event("a", {"type": "entry_fill", "lot": "L1", "qty": 1.0}, T0)
    work(router)
    held = venue.get_positions()[0].qty
    assert held == pytest.approx(1.0 - 0.0015)
    assert accounts["a"].positions["BTC/USD"][0] == pytest.approx(held)
    stop = [o for o in venue.open_orders() if o.type == "stop_limit"][0]
    assert stop.qty == pytest.approx(held)
