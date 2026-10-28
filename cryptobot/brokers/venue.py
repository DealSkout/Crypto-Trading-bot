"""A simulated venue with resting orders, for the team's local real side and tests.

PaperBroker fills every order on the spot. The team router now leaves orders
working at the broker (maker limit entries, stop-limit protective stops, exit
limits), so its local stand-in needs an order book of one: orders rest until
mark() moves the price through them, exactly as they would at Alpaca.

  limit buy      fills at its limit once price <= limit (maker); if the price was
                 already strictly below it when placed it fills at once (taker)
  limit sell     the mirror image
  stop_limit sell  triggers once price <= stop: fills at the price if that is
                 still >= the limit (taker), otherwise rests as a limit sell
  market         fills at once, through PaperBroker's spread and slippage
"""

from __future__ import annotations

import itertools
from dataclasses import replace

from cryptobot.brokers.base import BrokerError, Order
from cryptobot.brokers.paper import PaperBroker


class PaperVenue(PaperBroker):
    name = "paper"
    delayed_entry = False
    supports_limit = True
    supports_protective_stop = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._book: dict[str, dict] = {}  # order id -> resting order state
        self._ids = itertools.count(1)
        self._done: dict[str, Order] = {}

    # ------------------------------------------------------------------ placing
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        if order_type == "market":
            order = super().place_order(symbol, side, qty)
            self._done[order.id] = order
            return order
        if order_type not in ("limit", "stop_limit"):
            raise BrokerError(f"unsupported order type {order_type!r}")
        if not limit_price or (order_type == "stop_limit" and not stop_price):
            raise BrokerError("limit and stop_limit orders need their prices")
        if side == "sell" and qty > self._qty.get(symbol, 0.0) + 1e-9:
            qty = max(0.0, self._qty.get(symbol, 0.0))
        oid = f"venue-{next(self._ids)}"
        self._book[oid] = {
            "id": oid, "symbol": symbol, "side": side, "qty": float(qty), "type": order_type,
            "limit": float(limit_price), "stop": float(stop_price or 0.0), "triggered": False,
        }
        price = self._prices.get(symbol)
        if price is not None:
            self._work(oid, price, at_placement=True)
        return self.get_order(oid)

    # ------------------------------------------------------------------ the book
    def mark(self, symbol: str, price: float, ts=None) -> None:
        super().mark(symbol, price, ts)
        for oid in [o for o, r in self._book.items() if r["symbol"] == symbol]:
            self._work(oid, float(price))

    def _work(self, oid: str, price: float, at_placement: bool = False) -> None:
        r = self._book.get(oid)
        if r is None:
            return
        if r["type"] == "stop_limit" and not r["triggered"]:
            if (r["side"] == "sell" and price > r["stop"]) or (
                r["side"] == "buy" and price < r["stop"]
            ):
                return
            r["triggered"] = True
            at_placement = True  # once triggered it is a limit order hitting the book: taker
        limit = r["limit"]
        if at_placement and not r["triggered"]:
            # a new limit AT the last price rests at the touch; only a strictly better
            # price makes it marketable on arrival
            crosses = price < limit if r["side"] == "buy" else price > limit
        else:
            crosses = price <= limit if r["side"] == "buy" else price >= limit
        if not crosses:
            return
        taker = at_placement
        fill = price if taker else limit
        self._fill(oid, fill, maker=not taker)

    def _fill(self, oid: str, price: float, maker: bool) -> None:
        r = self._book.pop(oid)
        self.set_exec_price(r["symbol"], price)
        # a resting limit fills at its price as a maker; a marketable or triggered one at
        # the touch as a taker (no extra slippage: the price is the trade price)
        order = PaperBroker.place_order(
            self, r["symbol"], r["side"], r["qty"], order_type="limit", limit_price=price
        )
        if not maker and order.is_filled:  # charge the taker rate instead of the maker one
            extra = self.cost_model.fees(price * order.filled_qty) - order.fees
            self.cash -= extra
            self.total_fees += extra
            order.fees += extra
        order.id = oid
        order.type = r["type"]
        self._done[oid] = order

    # ------------------------------------------------------------------ queries
    def get_order(self, order_id: str, symbol: str = "") -> Order:
        if order_id in self._done:
            return replace(self._done[order_id])
        r = self._book.get(order_id)
        if r is None:
            raise BrokerError(f"unknown order {order_id}")
        return Order(id=order_id, symbol=r["symbol"], side=r["side"], qty=r["qty"],
                     type=r["type"], status="new", limit_price=r["limit"])

    def cancel(self, order_id: str) -> None:
        r = self._book.pop(order_id, None)
        if r is not None:
            self._done[order_id] = Order(
                id=order_id, symbol=r["symbol"], side=r["side"], qty=r["qty"], type=r["type"],
                status="cancelled", limit_price=r["limit"], note="cancelled",
            )

    def open_orders(self) -> list[Order]:
        return [self.get_order(oid) for oid in self._book]

    def await_fill(self, order: Order, timeout: float = 30.0) -> Order:
        return self.get_order(order.id) if order.id else order
