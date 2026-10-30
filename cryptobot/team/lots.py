"""The lot desk: every position's life on the one real account.

Each bot runs its strategy against its own simulated book and tells the desk
what it did through the engine's outbox. The desk mirrors that on the real
account, one "lot" (one entry and its exit) at a time, with orders that work
at the broker instead of being decided after the fact:

  entry        a maker LIMIT buy at the bot's price, tagged with the bot id,
               cancelled when the bot's entry window closes
  protection   once the entry has filled, a STOP-LIMIT sell rests at Alpaca at
               the bot's stop (limit orders.stop_limit_buffer_pct below it), so
               the stop is honoured by the exchange, not by a bar-close check.
               Alpaca crypto has no plain stop order; if price gaps through the
               limit the stop cannot fill, and the bot's own stop-out then sends
               a MARKET sell: the one place market orders remain
  exit         stop-outs and forced exits (shutdown, drawdown, daily loss,
               regime, orphan) go to the market; any other exit rests as a
               limit at the last price for orders.exit_timeout_seconds, then
               goes to the market so a position is never left open

The real account is the truth. Fills are booked into the bot's virtual account
as they happen; when the broker filled less of an entry than the simulation,
or the broker's stop closed a position first, the desk reports it so the
supervisor can make the bot's book follow. Opposing orders still net: a bot
closing a position while another bot has an entry working in the same symbol
at or above the mark crosses internally at the mark, no exchange involved
(both still pay fees).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from cryptobot.brokers.base import BrokerError, Order

log = logging.getLogger(__name__)

STOP_REASONS = ("stop", "backup_stop", "broker_stop")
FORCED_REASONS = ("shutdown", "drawdown", "daily_loss", "regime", "orphan", "end_of_data")
EPS = 1e-9


@dataclass
class Lot:
    lot_id: str
    bot_id: str
    symbol: str
    qty: float  # what the bot asked for
    stop: float
    target: float | None
    signal_price: float
    limit: float | None = None  # None: a market entry
    expires: datetime | None = None
    opened_ts: datetime | None = None
    sim: str = "resting"  # resting | open | cancelled | closed (the bot's side)
    sim_qty: float = 0.0
    exit_reason: str = ""
    exit_ref: float = 0.0  # the bot's exit price (the planned level for non-stop exits)
    entry_order: str = ""
    entry_done: bool = False
    bought: float = 0.0  # gross quantity filled at the broker
    held: float = 0.0  # coins this lot holds in its virtual account (after any fee in kind)
    cost: float = 0.0  # cash paid for `bought`
    fees: float = 0.0
    sold: float = 0.0
    proceeds: float = 0.0
    stop_order: str = ""
    exit_order: str = ""
    exit_deadline: datetime | None = None
    shrink_sent: bool = False
    closed: bool = False
    closed_ts: datetime | None = None
    seen: dict[str, tuple[float, float]] = field(default_factory=dict)  # order -> (qty, notional)


class LotDesk:
    def __init__(self, router) -> None:
        self.r = router
        self.lots: dict[str, Lot] = {}

    # ------------------------------------------------------------------ settings
    @property
    def orders(self):
        return self.r.order_settings

    def _place(
        self,
        lot: Lot,
        side: str,
        qty: float,
        order_type: str = "market",
        limit: float | None = None,
        stop: float | None = None,
    ) -> Order | None:
        self.r.broker.set_order_tag(lot.bot_id)
        self.r.last_tag = lot.bot_id
        try:
            return self.r.broker.place_order(
                lot.symbol, side, qty, order_type=order_type, limit_price=limit, stop_price=stop
            )
        except TypeError:  # an adapter without stop_price (PaperBroker): limit/market only
            return self.r.broker.place_order(
                lot.symbol, side, qty, order_type=order_type, limit_price=limit
            )
        except BrokerError as e:
            log.warning("%s %s %s order failed: %s", lot.bot_id, lot.symbol, side, e)
            return None

    # ------------------------------------------------------------------ events
    def on_event(self, bot_id: str, ev: dict[str, Any], now: datetime) -> list[dict]:
        """One outbox event from a bot. Returns reports for the supervisor."""
        kind = ev.get("type")
        if kind in ("entry", "entry_market"):
            return self._open(bot_id, ev, now, market=kind == "entry_market")
        lot = self.lots.get(ev.get("lot", ""))
        if lot is None:
            return []
        if kind == "entry_fill":
            lot.sim, lot.sim_qty = "open", float(ev["qty"])
        elif kind == "entry_cancel":
            lot.sim = "cancelled"
        elif kind == "exit":
            if lot.sim == "resting":
                lot.sim_qty = float(ev.get("qty", 0.0))
            lot.sim = "closed"
            lot.exit_reason = str(ev.get("reason", ""))
            lot.exit_ref = float(ev.get("price") or 0.0)
        return []

    def _open(self, bot_id: str, ev: dict, now: datetime, market: bool) -> list[dict]:
        lot = Lot(
            lot_id=ev["lot"],
            bot_id=bot_id,
            symbol=ev["symbol"],
            qty=float(ev["qty"]),
            stop=float(ev["stop"]),
            target=ev.get("target"),
            signal_price=float(ev.get("signal_price") or ev.get("limit") or ev.get("price")),
            limit=None if market else float(ev["limit"]),
            expires=ev.get("expires"),
            opened_ts=now,
            sim="open" if market else "resting",
            sim_qty=float(ev["qty"]) if market else 0.0,
        )
        reason = self.r.entry_precheck(
            bot_id,
            lot.symbol,
            lot.qty,
            lot.limit or float(ev.get("price") or 0.0),
            now,
            working=self.working_notional(),
        )
        if reason:
            lot.closed = True
            return [
                {
                    "rejected": True,
                    "bot_id": bot_id,
                    "symbol": lot.symbol,
                    "lot": lot.lot_id,
                    "reason": reason,
                    "cancel_entry": not market,
                    "unwind_sim": market,
                }
            ]
        if market:
            order = self._place(lot, "buy", lot.qty)
        else:
            order = self._place(lot, "buy", lot.qty, "limit", limit=lot.limit)
        if order is None or not order.id or order.status in ("rejected", "cancelled"):
            note = (order.note if order is not None else "") or "the broker refused the order"
            lot.closed = True
            return [
                {
                    "rejected": True,
                    "bot_id": bot_id,
                    "symbol": lot.symbol,
                    "lot": lot.lot_id,
                    "reason": note,
                    "cancel_entry": not market,
                    "unwind_sim": market,
                }
            ]
        lot.entry_order = order.id
        self.lots[lot.lot_id] = lot
        return self._absorb(lot, order, now)

    def working_notional(self) -> float:
        return sum(
            (lot.qty - lot.bought) * (lot.limit or lot.signal_price)
            for lot in self.lots.values()
            if lot.entry_order and not lot.entry_done
        )

    # ------------------------------------------------------------------ fills
    def _absorb(self, lot: Lot, order: Order, now: datetime) -> list[dict]:
        """Book whatever part of `order` filled since we last looked."""
        prev_qty, prev_notional = lot.seen.get(order.id, (0.0, 0.0))
        filled = float(order.filled_qty or 0.0)
        if filled <= prev_qty + EPS:
            return []
        avg = float(order.avg_fill_price or 0.0)
        notional = filled * avg
        qty = filled - prev_qty
        price = (notional - prev_notional) / qty if qty > 0 else avg
        lot.seen[order.id] = (filled, notional)
        # a resting limit (entry or exit) fills as a maker; market and stop-limit as takers
        fee = self._fee(order, qty, price, maker=order.type == "limit")
        account = self.r.accounts[lot.bot_id]
        reports = []
        if order.side == "buy":
            received = qty
            if not getattr(self.r.broker, "reports_fees", True) and price > 0:
                received = qty - fee / price  # Alpaca keeps its fee in coins
            account.apply_fill(lot.symbol, "buy", received, price, fee)
            lot.bought += qty
            lot.held += received
            lot.cost += qty * price
            lot.fees += fee
        else:
            qty = min(qty, lot.held)
            if qty <= EPS:
                return []
            account.apply_fill(lot.symbol, "sell", qty, price, fee)
            lot.held -= qty
            lot.sold += qty
            lot.proceeds += qty * price
            lot.fees += fee
            if lot.held <= EPS:
                lot.held = 0.0
        reports.append(
            {
                "internal": False,
                "bot_id": lot.bot_id,
                "symbol": lot.symbol,
                "side": order.side,
                "qty": qty,
                "price": price,
                "fees": fee,
                "order_id": order.id,
                "lot": lot.lot_id,
            }
        )
        return reports

    def _fee(self, order: Order, qty: float, price: float, maker: bool) -> float:
        reported = float(order.fees or 0.0)
        if getattr(self.r.broker, "reports_fees", True) and reported:
            return reported * qty / max(float(order.filled_qty or qty), EPS)  # this fill's share
        return self.r.cost_model.fees(qty * price, maker=maker)

    def _poll(self, lot: Lot, order_id: str, now: datetime) -> tuple[Order | None, list[dict]]:
        if not order_id:
            return None, []
        try:
            order = self.r.broker.get_order(order_id, lot.symbol)
        except BrokerError as e:
            log.warning("order lookup %s failed: %s", order_id, e)
            return None, []
        if order is None:
            return None, []
        return order, self._absorb(lot, order, now)

    def _cancel(self, lot: Lot, order_id: str, now: datetime) -> tuple[Order | None, list[dict]]:
        """Cancel, then look again: a cancel can race a fill, and the fill wins."""
        try:
            self.r.broker.cancel(order_id)
        except BrokerError:
            pass
        return self._poll(lot, order_id, now)

    # ------------------------------------------------------------------ the tick
    def work(self, now: datetime, symbols: set[str] | None = None) -> list[dict]:
        """Poll and act on every open lot (or only those in `symbols`)."""
        reports: list[dict] = []
        for lot in list(self.lots.values()):
            if lot.closed or (symbols is not None and lot.symbol not in symbols):
                continue
            try:
                reports += self._work_lot(lot, now)
            except Exception as e:  # one lot's trouble never stops the others
                log.exception("lot %s failed: %s", lot.lot_id, e)
        for lot_id in [k for k, v in self.lots.items() if v.closed]:
            self.lots.pop(lot_id)
        return reports

    def _work_lot(self, lot: Lot, now: datetime) -> list[dict]:
        reports: list[dict] = []
        # 1. the entry order
        if lot.entry_order and not lot.entry_done:
            order, got = self._poll(lot, lot.entry_order, now)
            reports += got
            if order is not None and order.is_done:
                lot.entry_done = True
            elif lot.expires is not None and now >= lot.expires:
                _, got = self._cancel(lot, lot.entry_order, now)
                reports += got
                lot.entry_done = True
        # 2. the bot's side settled the entry: shrink its book to what really filled
        if (
            lot.sim == "open"
            and lot.entry_done
            and not lot.shrink_sent
            and (lot.bought < lot.sim_qty - max(EPS, lot.sim_qty * 1e-6))
        ):
            lot.shrink_sent = True
            reports.append(
                {"shrink": True, "bot_id": lot.bot_id, "lot": lot.lot_id, "real_qty": lot.bought}
            )
        # 3. protection: a resting stop-limit for whatever the lot holds
        # (real coins are protected as soon as they are bought, whatever the simulation is
        # doing, unless the bot is already leaving the position)
        if (
            lot.held > EPS
            and lot.entry_done
            and not lot.stop_order
            and lot.sim in ("resting", "open")
        ):
            buffer = self.orders.stop_limit_buffer_pct / 100.0
            order = self._place(
                lot, "sell", lot.held, "stop_limit", limit=lot.stop * (1 - buffer), stop=lot.stop
            )
            if order is not None and order.id and order.status not in ("rejected", "cancelled"):
                lot.stop_order = order.id
                reports += self._absorb(lot, order, now)
            else:
                reports.append(
                    {
                        "warning": True,
                        "bot_id": lot.bot_id,
                        "lot": lot.lot_id,
                        "reason": "could not place the broker stop; software stop only",
                    }
                )
        # 4. the broker's stop
        if lot.stop_order:
            order, got = self._poll(lot, lot.stop_order, now)
            reports += got
            if order is not None and order.is_done:
                lot.stop_order = "" if order.status != "filled" else lot.stop_order
                if lot.held <= EPS and lot.sim in ("resting", "open"):
                    reports.append(
                        {
                            "broker_stop": True,
                            "bot_id": lot.bot_id,
                            "lot": lot.lot_id,
                            "price": lot.proceeds / lot.sold if lot.sold else lot.stop,
                        }
                    )
                    lot.sim = "closed"
                    lot.exit_reason = "broker_stop"
                    lot.exit_ref = lot.stop
        # 5. closing: the bot exited, or its entry was cancelled with coins already bought
        if lot.sim in ("closed", "cancelled"):
            reports += self._close(lot, now)
        if (
            lot.sim in ("closed", "cancelled")
            and lot.held <= EPS
            and lot.entry_done
            and not lot.exit_order
            and not self._stop_working(lot)
        ):
            lot.closed = True
            lot.closed_ts = now
            if lot.bought > EPS:
                reports.append(self.result(lot))
        return reports

    def _stop_working(self, lot: Lot) -> bool:
        if not lot.stop_order:
            return False
        try:
            return not self.r.broker.get_order(lot.stop_order, lot.symbol).is_done
        except BrokerError:
            return False

    def _close(self, lot: Lot, now: datetime) -> list[dict]:
        reports: list[dict] = []
        if lot.entry_order and not lot.entry_done:  # still buying: stop buying
            _, got = self._cancel(lot, lot.entry_order, now)
            reports += got
            lot.entry_done = True
        if lot.held <= EPS and not lot.exit_order:
            if lot.stop_order:
                _, got = self._cancel(lot, lot.stop_order, now)
                reports += got
                lot.stop_order = ""
            return reports
        reason = lot.exit_reason or ("orphan" if lot.sim == "cancelled" else "")
        stop_out = reason in STOP_REASONS
        if lot.stop_order:
            order, got = self._poll(lot, lot.stop_order, now)
            reports += got
            if order is not None and not order.is_done:
                _, got = self._cancel(lot, lot.stop_order, now)
                reports += got
            lot.stop_order = ""
            if lot.held <= EPS:
                return reports
        reports += self._cross(lot, now)
        if lot.held <= EPS:
            return reports
        if lot.exit_order:
            order, got = self._poll(lot, lot.exit_order, now)
            reports += got
            if order is not None and order.is_done:
                lot.exit_order = ""
            elif lot.exit_deadline is not None and now >= lot.exit_deadline:
                _, got = self._cancel(lot, lot.exit_order, now)
                reports += got
                lot.exit_order = ""
                if lot.held > EPS:
                    reports += self._market_out(lot, now)
            return reports
        if stop_out or reason in FORCED_REASONS:
            return reports + self._market_out(lot, now)
        mark = self.r.mark_price(lot.symbol) or lot.exit_ref
        order = self._place(lot, "sell", lot.held, "limit", limit=mark)
        if order is None or not order.id or order.status in ("rejected", "cancelled"):
            return reports + self._market_out(lot, now)
        lot.exit_order = order.id
        lot.exit_deadline = now + timedelta(seconds=self.orders.exit_timeout_seconds)
        reports += self._absorb(lot, order, now)
        if order.is_done:
            lot.exit_order = ""
        return reports

    def _market_out(self, lot: Lot, now: datetime) -> list[dict]:
        order = self._place(lot, "sell", lot.held)
        if order is None:
            return [
                {
                    "warning": True,
                    "bot_id": lot.bot_id,
                    "lot": lot.lot_id,
                    "reason": "market exit failed; retrying next tick",
                }
            ]
        if order.id and not order.is_done:
            order = self.r.broker.await_fill(order, timeout=10.0)
        return self._absorb(lot, order, now)

    # ------------------------------------------------------------------ netting
    def _cross(self, seller: Lot, now: datetime) -> list[dict]:
        """A closing lot crosses internally with another bot's working entry in the same
        symbol when that entry's limit is at or above the mark: both get the mark price,
        both pay fees, no order reaches the exchange for the crossed quantity."""
        mark = self.r.mark_price(seller.symbol)
        if mark is None:
            return []
        reports: list[dict] = []
        for buyer in self.lots.values():
            if seller.held <= EPS:
                break
            if (
                buyer is seller
                or buyer.bot_id == seller.bot_id
                or buyer.symbol != seller.symbol
                or buyer.entry_done
                or not buyer.entry_order
                or buyer.limit is None
                or buyer.limit < mark
            ):
                continue
            _, got = self._cancel(buyer, buyer.entry_order, now)  # its own fills come first
            reports += got
            buyer.entry_done = True
            want = buyer.qty - buyer.bought
            qty = min(seller.held, want)
            if qty <= EPS:
                continue
            buy_fee = self.r.cost_model.fees(qty * mark, maker=True)
            sell_fee = self.r.cost_model.fees(qty * mark)
            self.r.accounts[buyer.bot_id].apply_fill(buyer.symbol, "buy", qty, mark, buy_fee)
            self.r.accounts[seller.bot_id].apply_fill(seller.symbol, "sell", qty, mark, sell_fee)
            self.r.internal_fees += buy_fee + sell_fee
            buyer.bought += qty
            buyer.held += qty
            buyer.cost += qty * mark
            buyer.fees += buy_fee
            seller.held -= qty
            seller.sold += qty
            seller.proceeds += qty * mark
            seller.fees += sell_fee
            reports.append(
                {
                    "internal": True,
                    "bots": [buyer.bot_id, seller.bot_id],
                    "symbol": seller.symbol,
                    "qty": qty,
                    "price": mark,
                    "fees": buy_fee + sell_fee,
                    "lots": [buyer.lot_id, seller.lot_id],
                }
            )
        return reports

    # ------------------------------------------------------------------ results
    def result(self, lot: Lot) -> dict[str, Any]:
        """The real trade: what the plan expected and what the account actually got."""
        qty = lot.bought
        entry = lot.cost / qty if qty else 0.0
        exit_ = lot.proceeds / lot.sold if lot.sold else entry
        planned_risk = abs(lot.signal_price - lot.stop)
        reason = lot.exit_reason or ("orphan" if lot.sim == "cancelled" else "")
        if reason in STOP_REASONS:
            planned_exit = lot.stop
        elif reason == "target" and lot.target is not None:
            planned_exit = float(lot.target)
        else:
            planned_exit = lot.exit_ref or exit_
        gross_planned = (planned_exit - lot.signal_price) * lot.sold
        net = lot.proceeds - lot.cost * (lot.sold / qty if qty else 0.0) - lot.fees
        slippage = gross_planned - (net + lot.fees)
        return {
            "lot_closed": True,
            "bot_id": lot.bot_id,
            "symbol": lot.symbol,
            "lot": lot.lot_id,
            "entry_ts": lot.opened_ts,
            "exit_ts": lot.closed_ts,
            "qty": lot.sold,
            "entry_price": entry,
            "exit_price": exit_,
            "stop": lot.stop,
            "target": lot.target,
            "signal_price": lot.signal_price,
            "reason": reason,
            "gross_pnl": gross_planned,
            "fees": lot.fees,
            "slippage": slippage,
            "net_pnl": net,
            "planned_r": (planned_exit - lot.signal_price) / planned_risk if planned_risk else None,
            "actual_r": net / (planned_risk * lot.sold) if planned_risk and lot.sold else None,
        }

    def cancel_all(self, now: datetime) -> list[dict]:
        """Shutdown: stop buying. Positions are closed by the bots' own forced exits."""
        reports: list[dict] = []
        for lot in self.lots.values():
            if lot.entry_order and not lot.entry_done:
                _, got = self._cancel(lot, lot.entry_order, now)
                reports += got
                lot.entry_done = True
        return reports
