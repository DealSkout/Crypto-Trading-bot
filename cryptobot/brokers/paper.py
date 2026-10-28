"""Simulated broker: instant market fills at reference price +/- half-spread and slippage,
plus explicit fees from the cost model. Used by the backtester and by paper mode when no demo
account is configured. Shorts are allowed here only so the risk layer can be tested; the risk
layer decides whether shorts are permitted at all."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from cryptobot.brokers.base import Broker, BrokerError, Order, Position
from cryptobot.costs import CostModel


class PaperBroker(Broker):
    name = "paper"
    supports_fractional = True
    qty_decimals = 4
    delayed_entry = True  # the engine fills queued entries at the next bar's open

    def __init__(
        self,
        cash: float,
        cost_model: CostModel,
        fx_for: Callable[[str], bool] | None = None,
    ) -> None:
        self.cash = float(cash)
        self.cost_model = cost_model
        self._fx_for = fx_for or (lambda symbol: True)
        self._prices: dict[str, float] = {}
        self._exec: dict[str, float] = {}
        self._qty: dict[str, float] = {}  # signed quantity per symbol
        self._avg: dict[str, float] = {}
        self.orders: list[Order] = []
        self.total_fees = 0.0
        self.total_implicit = 0.0
        self.realized_pnl = 0.0
        self.now: datetime | None = None
        self._n = 0

    # prices -----------------------------------------------------------------------------
    def mark(self, symbol: str, price: float, ts: datetime | None = None) -> None:
        self._prices[symbol] = float(price)
        if ts is not None:
            self.now = ts

    def set_exec_price(self, symbol: str, price: float) -> None:
        self._exec[symbol] = float(price)

    def last_price(self, symbol: str) -> float | None:
        return self._prices.get(symbol)

    # account ----------------------------------------------------------------------------
    def get_cash(self) -> float:
        return self.cash

    def get_equity(self) -> float:
        value = 0.0
        for symbol, qty in self._qty.items():
            if qty:
                value += qty * self._prices.get(symbol, self._avg.get(symbol, 0.0))
        return self.cash + value

    def get_positions(self) -> list[Position]:
        out = []
        for symbol, qty in self._qty.items():
            if abs(qty) > 1e-12:
                side = "long" if qty > 0 else "short"
                out.append(Position(symbol, abs(qty), side, self._avg.get(symbol, 0.0)))
        return out

    def open_orders(self) -> list[Order]:
        return []  # market orders fill instantly

    def cancel(self, order_id: str) -> None:
        return None

    # orders -----------------------------------------------------------------------------
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> Order:
        if side not in ("buy", "sell"):
            raise BrokerError(f"side must be buy or sell, got {side!r}")
        if order_type not in ("market", "limit"):
            raise BrokerError("the paper broker simulates market and limit orders")
        if order_type == "limit" and not limit_price:
            raise BrokerError("a limit order needs limit_price")
        qty = self.round_qty(qty)
        self._n += 1
        order = Order(id=f"paper-{self._n}", symbol=symbol, side=side, qty=qty, created_at=self.now)
        if qty <= 0:
            order.status, order.note = "rejected", "quantity rounds to zero"
            self.orders.append(order)
            return order
        is_buy = side == "buy"
        maker = order_type == "limit"
        if maker:
            # the caller (the engine) has checked the market traded through the limit: a
            # resting limit fills at its own price as a maker, no spread or slippage
            ref = price = float(limit_price)
            self._exec.pop(symbol, None)
        else:
            ref = self._exec.pop(symbol, None)
            if ref is None:
                ref = self._prices.get(symbol)
            if ref is None:
                raise BrokerError(f"no price known for {symbol}; call mark() first")
            price = self.cost_model.fill_price(ref, is_buy)
        notional = price * qty
        fees = self.cost_model.fees(notional, fx=self._fx_for(symbol), maker=maker)
        if is_buy and self.cash + 1e-9 < notional + fees:
            order.status, order.note = "rejected", "insufficient cash"
            self.orders.append(order)
            return order

        old = self._qty.get(symbol, 0.0)
        avg = self._avg.get(symbol, price)
        signed = qty if is_buy else -qty
        new = old + signed
        if abs(new) < 1e-12:
            new = 0.0
        if old and (old > 0) != (signed > 0):  # reducing, closing or flipping
            closed = min(abs(old), qty)
            self.realized_pnl += (price - avg) * closed * (1 if old > 0 else -1)
            new_avg = price if (new and (new > 0) != (old > 0)) else avg
        elif old == 0:
            new_avg = price
        else:  # adding to a position
            new_avg = (abs(old) * avg + qty * price) / abs(new)
        self._qty[symbol] = new
        if new == 0:
            self._avg.pop(symbol, None)
        else:
            self._avg[symbol] = new_avg

        self.cash += (-notional - fees) if is_buy else (notional - fees)
        implicit = abs(price - ref) * qty
        self.total_fees += fees
        self.total_implicit += implicit
        order.status = "filled"
        order.type = order_type
        order.filled_qty = qty
        order.avg_fill_price = price
        order.fees = fees
        order.implicit_cost = implicit
        self.orders.append(order)
        return order

    def adjust(self, symbol: str, qty_delta: float, price: float) -> None:
        """Book a correction without fees or P&L: the team router found the real account
        filled less than this simulation did, so part of a simulated entry is undone at
        its own entry price."""
        new = self._qty.get(symbol, 0.0) + qty_delta
        if abs(new) < 1e-12:
            new = 0.0
        self._qty[symbol] = new
        if new == 0:
            self._avg.pop(symbol, None)
        self.cash -= qty_delta * price
