"""The order router. Bots never talk to the real broker directly: they submit
intents here, and once per tick flush() nets opposing intents internally,
enforces the portfolio-wide limits, and sends only the residual orders to the
real broker, one order per bot so each keeps its bot tag.

Fees on an internal cross are deducted from both virtual accounts but no cash
leaves the real account, so the router accumulates them in `internal_fees`.
Reconciliation: real account equity = sum of virtual equities + internal_fees."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from cryptobot.brokers.base import Broker, BrokerError
from cryptobot.costs import CostModel
from cryptobot.team.ledger import VirtualAccount


@dataclass
class PortfolioLimits:
    """Caps applied across the whole team, not per bot."""

    max_gross_exposure_pct: float = 100.0
    max_symbol_exposure_pct: float = 50.0
    kill_switch_file: str = "team.KILLSWITCH"
    stale_after_seconds: float = 600.0


class RouterRejection(Exception):
    """Never raised to callers: flush() returns rejections in its report list.
    Kept as a named type so callers who want to escalate a rejection can."""


class Router:
    """Synchronous, deterministic order router for one shared account."""

    def __init__(
        self,
        broker: Broker,
        accounts: dict[str, VirtualAccount],
        limits: PortfolioLimits,
        cost_model: CostModel,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.broker = broker
        self.accounts = accounts
        self.limits = limits
        self.cost_model = cost_model
        self._clock = clock or (lambda: datetime.now(UTC))
        self._marks: dict[str, tuple[float, datetime]] = {}  # symbol -> (price, ts)
        self._queue: list[dict] = []
        self.internal_fees = 0.0  # fees charged on internal crosses; stays in the real account
        self.last_tag: str | None = None  # bot tag set right before each broker order

    # marks ----------------------------------------------------------------------------
    def mark(self, symbol: str, price: float, ts: datetime) -> None:
        """Remember the latest price and when we saw it; forwarded to the broker
        so simulators know prices too (real brokers ignore it)."""
        self._marks[symbol] = (float(price), ts)
        self.broker.mark(symbol, float(price), ts)

    # intents --------------------------------------------------------------------------
    def submit(self, bot_id: str, symbol: str, side: str, qty: float, ref_price: float) -> None:
        """Queue an intent for the current tick. Nothing executes until flush()."""
        if bot_id not in self.accounts:
            raise ValueError(f"unknown bot {bot_id!r}")
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be buy or sell, got {side!r}")
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        self._queue.append(
            {
                "bot_id": bot_id,
                "symbol": symbol,
                "side": side,
                "qty": float(qty),
                "ref_price": float(ref_price),
            }
        )

    # execution ------------------------------------------------------------------------
    def flush(self, ts: datetime | None = None) -> list[dict]:
        """Execute the queued batch. Returns one report dict per outcome:
        {"internal": True, ...} for a cross, {"internal": False, ...} for a real
        fill, and {"rejected": True, ...} for anything turned away."""
        ts = ts or self._clock()
        killed = os.path.exists(self.limits.kill_switch_file)
        reports: list[dict] = []

        queue, self._queue = self._queue, []
        live: list[dict] = []
        pending_sells: dict[tuple[str, str], float] = {}
        for intent in queue:
            reason = self._precheck(intent, ts, killed, pending_sells)
            if reason:
                reports.append(self._rejection(intent, reason))
            else:
                live.append(dict(intent))

        # 1. Net opposing intents per symbol, oldest first, at the current mark.
        residual: list[dict] = []
        for symbol in self._symbols_in_order(live):
            buys = [i for i in live if i["symbol"] == symbol and i["side"] == "buy"]
            sells = [i for i in live if i["symbol"] == symbol and i["side"] == "sell"]
            reports.extend(self._cross(symbol, buys, sells))
            residual.extend(i for i in buys + sells if i["qty"] > 1e-12)

        # 2. Limits, then the real broker, one residual order at a time.
        mark_prices = {s: p for s, (p, _) in self._marks.items()}
        total_equity = sum(a.equity(mark_prices) for a in self.accounts.values())
        gross, team_qty = self._current_exposure(mark_prices)
        for intent in residual:
            bot_id, symbol, side, qty = (
                intent["bot_id"],
                intent["symbol"],
                intent["side"],
                intent["qty"],
            )
            if side == "buy":
                price = mark_prices[symbol]  # precheck guarantees a fresh mark for buys
                cap = total_equity * self.limits.max_gross_exposure_pct / 100.0
                if gross + qty * price > cap + 1e-9:
                    reports.append(self._rejection(intent, "gross exposure cap"))
                    continue
                cap = total_equity * self.limits.max_symbol_exposure_pct / 100.0
                if (team_qty.get(symbol, 0.0) + qty) * price > cap + 1e-9:
                    reports.append(self._rejection(intent, "symbol exposure cap"))
                    continue

            self.last_tag = bot_id
            self.broker.set_order_tag(bot_id)
            try:
                order = self.broker.await_fill(self.broker.place_order(symbol, side, qty))
                if not order.is_done and order.id:
                    order = self._cancel_unfilled(order)
            except BrokerError as exc:
                reports.append(self._rejection(intent, str(exc)))
                continue
            if order.filled_qty <= 0 and not order.is_filled:
                note = order.note or f"broker returned status {order.status}"
                reports.append(self._rejection(intent, note))
                continue

            fill_qty = order.filled_qty or qty
            fill_price = (
                order.avg_fill_price if order.avg_fill_price is not None else intent["ref_price"]
            )
            self.accounts[bot_id].apply_fill(symbol, side, fill_qty, fill_price, order.fees)
            signed = fill_qty if side == "buy" else -fill_qty
            gross += signed * mark_prices.get(symbol, fill_price)
            team_qty[symbol] = team_qty.get(symbol, 0.0) + signed
            reports.append(
                {
                    "internal": False,
                    "bot_id": bot_id,
                    "symbol": symbol,
                    "side": side,
                    "qty": fill_qty,
                    "price": fill_price,
                    "fees": order.fees,
                    "order_id": order.id,
                }
            )
        return reports

    # helpers --------------------------------------------------------------------------
    def _cancel_unfilled(self, order):
        """An order still open after the fill wait is cancelled so it cannot fill later
        behind the books' back. Whatever filled before the cancel landed is kept, so the
        caller books a partial fill instead of pretending nothing happened."""
        try:
            self.broker.cancel(order.id)
        except BrokerError:
            pass  # the cancel can race a fill; the lookup below decides
        current = self.broker.get_order(order.id, order.symbol)
        if current is None:
            current = order
        if current.filled_qty > 0 and not current.is_filled:
            current.note = current.note or "partial fill; the rest was cancelled"
        elif current.filled_qty <= 0:
            current.note = current.note or "not filled in time; cancelled"
        return current

    def _precheck(
        self,
        intent: dict,
        ts: datetime,
        killed: bool,
        pending_sells: dict[tuple[str, str], float],
    ) -> str:
        """Empty string when the intent may proceed, else the rejection reason.
        Sells that reduce an existing long are always allowed; buys are blocked
        by the kill switch, a missing mark, or a stale mark."""
        bot_id, symbol, side, qty = (
            intent["bot_id"],
            intent["symbol"],
            intent["side"],
            intent["qty"],
        )
        if side == "sell":
            held = self.accounts[bot_id].positions.get(symbol, (0.0, 0.0))[0]
            already = pending_sells.get((bot_id, symbol), 0.0)
            if qty + already > held + 1e-9:
                return "sell exceeds position"
            pending_sells[(bot_id, symbol)] = already + qty
            return ""
        if killed:
            return "kill switch active"
        marked = self._marks.get(symbol)
        if marked is None:
            return "no mark for symbol"
        if (ts - marked[1]).total_seconds() > self.limits.stale_after_seconds:
            return "stale mark"
        return ""

    def _cross(self, symbol: str, buys: list[dict], sells: list[dict]) -> list[dict]:
        """Pair opposing intents at the current mark. No exchange is touched, so
        no spread or slippage, but each side still pays the cost model's fees."""
        reports: list[dict] = []
        bi = si = 0
        while bi < len(buys) and si < len(sells):
            buy, sell = buys[bi], sells[si]
            qty = min(buy["qty"], sell["qty"])
            price = self._marks[symbol][0]
            fee = self.cost_model.fees(qty * price)
            self.accounts[buy["bot_id"]].apply_fill(symbol, "buy", qty, price, fee)
            self.accounts[sell["bot_id"]].apply_fill(symbol, "sell", qty, price, fee)
            self.internal_fees += 2 * fee
            reports.append(
                {
                    "internal": True,
                    "bots": [buy["bot_id"], sell["bot_id"]],
                    "symbol": symbol,
                    "qty": qty,
                    "price": price,
                    "fees": 2 * fee,
                }
            )
            buy["qty"] -= qty
            sell["qty"] -= qty
            if buy["qty"] <= 1e-12:
                bi += 1
            if sell["qty"] <= 1e-12:
                si += 1
        return reports

    def _current_exposure(self, mark_prices: dict[str, float]) -> tuple[float, dict[str, float]]:
        """Gross exposure across all bots plus the team's signed qty per symbol."""
        gross = 0.0
        team_qty: dict[str, float] = {}
        for account in self.accounts.values():
            for symbol, (qty, avg_price) in account.positions.items():
                gross += abs(qty) * mark_prices.get(symbol, avg_price)
                team_qty[symbol] = team_qty.get(symbol, 0.0) + qty
        return gross, team_qty

    @staticmethod
    def _symbols_in_order(intents: list[dict]) -> list[str]:
        seen: list[str] = []
        for intent in intents:
            if intent["symbol"] not in seen:
                seen.append(intent["symbol"])
        return seen

    @staticmethod
    def _rejection(intent: dict, reason: str) -> dict:
        return {
            "rejected": True,
            "bot_id": intent["bot_id"],
            "symbol": intent["symbol"],
            "reason": reason,
        }
