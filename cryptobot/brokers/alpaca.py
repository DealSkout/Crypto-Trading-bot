"""Alpaca crypto adapter (paper and live) built on alpaca-py's TradingClient.

Alpaca crypto rules this adapter follows (https://docs.alpaca.markets/docs/crypto-trading):
  * pairs are written "BTC/USD"; positions come back as "BTCUSD" and are normalised
  * order types: market, limit and stop_limit only; time in force: gtc or ioc only
  * long-only and no margin: crypto is not marginable and cannot be shorted, so a sell
    never exceeds what is held (a sell with nothing held is refused locally) and sizing uses
    cash / non-marginable buying power, never margin
  * quantities are fractional, rounded down to the asset's min_trade_increment, and
    orders below the asset's min_order_size are not sent
  * paper and live accounts have SEPARATE keys
Stops and targets stay SOFTWARE stops checked by the engine on every bar and price poll.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from cryptobot.brokers.base import Broker, BrokerError, Order, Position
from cryptobot.sessions import normalize

log = logging.getLogger(__name__)

ORDER_TYPES = ("market", "limit", "stop_limit")
TIME_IN_FORCE = ("gtc", "ioc")
DEFAULT_INCREMENT = 1e-6

STATUS_MAP = {
    "filled": "filled",
    "partially_filled": "partially_filled",
    "canceled": "cancelled",
    "expired": "cancelled",
    "done_for_day": "cancelled",
    "replaced": "cancelled",
    "pending_cancel": "cancelled",
    "rejected": "rejected",
    "suspended": "rejected",
}


def _val(x: Any) -> str:
    """Enum or plain value -> lowercase string."""
    return str(getattr(x, "value", x) or "").lower()


def _num(x: Any) -> float:
    try:
        return float(x) if x not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


class AlpacaCryptoBroker(Broker):
    name = "alpaca"
    supports_fractional = True
    qty_decimals = 9
    supports_limit = True
    supports_protective_stop = True  # resting stop_limit survives a crashed process
    delayed_entry = False  # live market orders fill moments after the bar closes anyway

    # the protective limit sits this far through the stop so a fast move still fills
    PROTECTIVE_LIMIT_GAP = 0.005
    # crypto market orders usually fill well inside a second: check often, not once a second
    FILL_POLL_SECONDS = 0.2

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        live: bool = False,
        time_in_force: str = "gtc",
        client=None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if time_in_force not in TIME_IN_FORCE:
            raise BrokerError(f"time_in_force must be one of {TIME_IN_FORCE} for crypto")
        self.live = live
        self.time_in_force = time_in_force
        self.sleep = sleep
        if client is None:
            if not api_key or not api_secret:
                raise BrokerError("ALPACA_API_KEY and ALPACA_API_SECRET must both be set in .env")
            from alpaca.trading.client import TradingClient

            client = TradingClient(api_key, api_secret, paper=not live)
        self.client = client
        self._assets: dict[str, tuple[float, float]] = {}
        self._protective: dict[str, str] = {}  # symbol -> resting stop-limit order id

    # plumbing -----------------------------------------------------------------------------
    def _call(self, what: str, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BrokerError:
            raise
        except Exception as e:  # alpaca-py raises APIError for 4xx/5xx; keep its message
            msg = getattr(e, "message", None) or str(e)
            raise BrokerError(f"Alpaca {what} failed: {msg}") from e

    @staticmethod
    def _to_order(o: Any) -> Order:
        filled = _num(getattr(o, "filled_qty", 0))
        avg = getattr(o, "filled_avg_price", None)
        return Order(
            id=str(getattr(o, "id", "")),
            symbol=normalize(str(getattr(o, "symbol", ""))),
            side=_val(getattr(o, "side", "buy")) or "buy",
            qty=abs(_num(getattr(o, "qty", 0)) or filled),
            type=_val(getattr(o, "order_type", None) or getattr(o, "type", "market")),
            status=STATUS_MAP.get(_val(getattr(o, "status", "")), "new"),
            filled_qty=abs(filled),
            avg_fill_price=_num(avg) if avg not in (None, "") else None,
            created_at=datetime.now(UTC),
            limit_price=_num(getattr(o, "limit_price", None)) or None,
            note=_val(getattr(o, "status", "")),
        )

    # account ------------------------------------------------------------------------------
    def account(self):
        return self._call("account", self.client.get_account)

    def get_cash(self) -> float:
        """What the bot may spend: the smaller of cash and non-marginable buying power."""
        a = self.account()
        cash = _num(getattr(a, "cash", 0))
        nmbp = getattr(a, "non_marginable_buying_power", None)
        return max(0.0, min(cash, _num(nmbp)) if nmbp not in (None, "") else cash)

    def get_equity(self) -> float:
        return _num(getattr(self.account(), "equity", 0))

    def get_positions(self) -> list[Position]:
        out = []
        for p in self._call("positions", self.client.get_all_positions) or []:
            if _val(getattr(p, "asset_class", "crypto")) not in ("crypto", ""):
                continue  # only the crypto side of the account is ours
            qty = _num(getattr(p, "qty", 0))
            if not qty:
                continue
            out.append(
                Position(
                    symbol=normalize(str(p.symbol)),
                    qty=abs(qty),
                    side="long",
                    avg_price=_num(getattr(p, "avg_entry_price", 0)),
                )
            )
        return out

    # sizing -------------------------------------------------------------------------------
    def asset_limits(self, symbol: str) -> tuple[float, float]:
        """(min_order_size, min_trade_increment) for a pair, cached."""
        symbol = normalize(symbol)
        if symbol not in self._assets:
            try:
                a = self.client.get_asset(symbol)
                self._assets[symbol] = (
                    _num(getattr(a, "min_order_size", 0)),
                    _num(getattr(a, "min_trade_increment", 0)) or DEFAULT_INCREMENT,
                )
            except Exception as e:
                log.warning("asset lookup for %s failed (%s); using defaults", symbol, e)
                self._assets[symbol] = (0.0, DEFAULT_INCREMENT)
        return self._assets[symbol]

    def round_qty_for(self, symbol: str, qty: float) -> float:
        min_size, inc = self.asset_limits(symbol)
        if qty <= 0:
            return 0.0
        steps = math.floor(qty / inc + 1e-9)
        q = round(steps * inc, 12)
        return q if q >= min_size else 0.0

    def held(self, symbol: str) -> float:
        symbol = normalize(symbol)
        return sum(p.qty for p in self.get_positions() if p.symbol == symbol)

    # orders -------------------------------------------------------------------------------
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
        )

        symbol = normalize(symbol)
        if order_type not in ORDER_TYPES:
            raise BrokerError(f"crypto orders must be one of {ORDER_TYPES}, got {order_type!r}")
        if side == "sell":
            held = self.held(symbol)
            if held <= 0:
                return Order(
                    id="",
                    symbol=symbol,
                    side=side,
                    qty=0,
                    status="rejected",
                    note="long-only: nothing held to sell (no shorting)",
                )
            qty = min(qty, held)  # never sell more than we own
        qty = self.round_qty_for(symbol, qty) if side == "buy" else qty
        if qty <= 0:
            return Order(
                id="",
                symbol=symbol,
                side=side,
                qty=0,
                status="rejected",
                note="quantity below the asset's minimum order size",
            )
        common = {
            "symbol": symbol,
            "qty": qty,
            "side": OrderSide.BUY if side == "buy" else OrderSide.SELL,
            "time_in_force": TimeInForce(self.time_in_force),
            # Alpaca allows 48 characters: a 24-character tag plus the dash and 20 hex fits
            "client_order_id": f"{self.order_tag}-{uuid.uuid4().hex[:20]}",
        }
        if order_type == "market":
            req = MarketOrderRequest(**common)
        elif order_type == "limit":
            if not limit_price:
                raise BrokerError("a limit order needs limit_price")
            req = LimitOrderRequest(**common, limit_price=round(limit_price, 8))
        else:
            if not limit_price or not stop_price:
                raise BrokerError("a stop_limit order needs stop_price and limit_price")
            req = StopLimitOrderRequest(
                **common, stop_price=round(stop_price, 8), limit_price=round(limit_price, 8)
            )
        return self._to_order(self._call("order", self.client.submit_order, order_data=req))

    def get_order(self, order_id: str, symbol: str = "") -> Order:
        return self._to_order(self._call("order lookup", self.client.get_order_by_id, order_id))

    # protective stop ----------------------------------------------------------------------
    def place_protective_stop(
        self, symbol: str, qty: float, stop_price: float, side: str = "sell"
    ) -> Order:
        """A resting stop_limit at the exchange as a backstop for the software stop, so a
        crashed process still has a stop. At most one per symbol."""
        symbol = normalize(symbol)
        self.cancel_protective(symbol)
        gap = self.PROTECTIVE_LIMIT_GAP
        limit = stop_price * (1 - gap) if side == "sell" else stop_price * (1 + gap)
        order = self.place_order(
            symbol, side, qty, order_type="stop_limit", limit_price=limit, stop_price=stop_price
        )
        if order.id and order.status not in ("rejected", "cancelled"):
            self._protective[symbol] = order.id
        return order

    def cancel_protective(self, symbol: str) -> Order | None:
        """Cancels the symbol's resting stop. Returns the Order if it already (partly)
        filled at the exchange, so the engine can book that exit instead of selling twice."""
        symbol = normalize(symbol)
        order_id = self._protective.pop(symbol, None)
        if not order_id:
            return None
        current = self.get_order(order_id, symbol)
        if not current.is_done:
            try:
                self.cancel(order_id)
            except BrokerError:
                pass  # a cancel can race the fill; the re-check below decides
            current = self.get_order(order_id, symbol)
        return current if current.filled_qty > 0 else None

    def await_fill(self, order: Order, timeout: float = 30.0) -> Order:
        if order.is_done or not order.id:
            return order
        deadline = time.monotonic() + timeout
        current = order
        while time.monotonic() < deadline:
            current = self.get_order(order.id, order.symbol)
            if current.is_done:
                return current
            self.sleep(self.FILL_POLL_SECONDS)
        return current

    def cancel(self, order_id: str) -> None:
        self._call("cancel", self.client.cancel_order_by_id, order_id)

    def open_orders(self) -> list[Order]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        rows = self._call("open orders", self.client.get_orders, filter=req) or []
        return [
            self._to_order(o)
            for o in rows
            if _val(getattr(o, "asset_class", "crypto")) in ("crypto", "")
        ]
