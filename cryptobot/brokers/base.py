"""The broker interface every adapter implements, plus the plain data types they exchange."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


class BrokerError(Exception):
    pass


@dataclass
class Position:
    symbol: str
    qty: float  # absolute size
    side: str  # "long" | "short"
    avg_price: float

    @property
    def signed_qty(self) -> float:
        return self.qty if self.side == "long" else -self.qty


@dataclass
class Order:
    id: str
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    type: str = "market"
    status: str = "new"  # new | filled | partially_filled | cancelled | rejected
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    fees: float = 0.0  # explicit fees the broker reported (commission, FX fee)
    implicit_cost: float = 0.0  # spread + slippage versus the reference price (simulator only)
    created_at: datetime | None = None
    limit_price: float | None = None
    note: str = ""

    @property
    def is_filled(self) -> bool:
        return self.status == "filled"

    @property
    def is_done(self) -> bool:
        return self.status in ("filled", "cancelled", "rejected")


class Broker(ABC):
    name = "base"
    supports_fractional = False
    qty_decimals = 0
    supports_native_brackets = False
    # prefix for every client order id this adapter sends; the team router sets it to the
    # bot id before each order so the real account shows who traded (see set_order_tag)
    order_tag = "cryptobot"

    @abstractmethod
    def get_cash(self) -> float: ...

    @abstractmethod
    def get_equity(self) -> float: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> Order: ...

    @abstractmethod
    def cancel(self, order_id: str) -> None: ...

    @abstractmethod
    def open_orders(self) -> list[Order]: ...

    # Optional hooks -------------------------------------------------------------------

    def mark(self, symbol: str, price: float, ts: datetime | None = None) -> None:  # noqa: B027
        """Simulators learn the latest price from the engine; real brokers ignore this."""

    def set_exec_price(self, symbol: str, price: float) -> None:  # noqa: B027
        """Simulators: the next order in `symbol` fills against this reference (e.g. a stop)."""

    def await_fill(self, order: Order, timeout: float = 30.0) -> Order:
        """Block until a market order is done. Simulators fill instantly."""
        return order

    def get_order(self, order_id: str, symbol: str = "") -> Order | None:
        """The current state of an order, or None when the adapter cannot look orders up."""
        return None

    def set_order_tag(self, tag: str) -> None:
        """Who the next orders are for: letters, digits, dash and underscore only, so the tag is
        safe inside any broker's client order id."""
        clean = "".join(ch for ch in str(tag) if ch.isalnum() or ch in "-_")
        self.order_tag = (clean or "cryptobot")[:24]

    def round_qty(self, qty: float) -> float:
        if qty <= 0:
            return 0.0
        if self.supports_fractional:
            factor = 10**self.qty_decimals
            return math.floor(qty * factor + 1e-9) / factor
        return float(math.floor(qty + 1e-9))

    def round_qty_for(self, symbol: str, qty: float) -> float:
        """Per-symbol rounding (crypto pairs have their own increments); default: round_qty."""
        return self.round_qty(qty)

    def flatten_all(self) -> list[Order]:
        """Cancel every open order, then close every position at the market."""
        orders: list[Order] = []
        for o in self.open_orders():
            try:
                self.cancel(o.id)
            except BrokerError:
                pass
        for p in self.get_positions():
            side = "sell" if p.side == "long" else "buy"
            orders.append(self.await_fill(self.place_order(p.symbol, side, p.qty)))
        return orders

    def close(self) -> None:  # noqa: B027
        """Release connections. Safe to call more than once."""
