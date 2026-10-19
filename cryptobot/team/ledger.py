"""Virtual sub-accounts. Seven bots share one real paper account, and each bot
gets a slice of it here: its own cash, positions, realized P&L and fees. The
router books every fill into these so the team's books always reconcile against
the real account."""

from __future__ import annotations


class VirtualAccount:
    """One bot's slice of the shared account.

    Long-only is not enforced here (the risk layer owns that rule), but a sell
    that would push a position below zero raises ValueError so a router bug
    surfaces immediately instead of quietly corrupting the books.
    """

    def __init__(
        self,
        bot_id: str,
        cash: float,
        positions: dict[str, tuple[float, float]] | None = None,
        realized_pnl: float = 0.0,
        fees_paid: float = 0.0,
    ) -> None:
        self.bot_id = bot_id
        self.cash = float(cash)
        # symbol -> (qty, average price paid)
        self.positions: dict[str, tuple[float, float]] = dict(positions or {})
        self.realized_pnl = float(realized_pnl)
        self.fees_paid = float(fees_paid)

    def apply_fill(self, symbol: str, side: str, qty: float, price: float, fees: float) -> None:
        """Book one fill: cash moves, the position's qty/average updates, and a
        reduce realises P&L against the average price paid."""
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be buy or sell, got {side!r}")
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        old_qty, old_avg = self.positions.get(symbol, (0.0, 0.0))
        if side == "buy":
            self.cash -= qty * price + fees
            new_qty = old_qty + qty
            new_avg = (old_qty * old_avg + qty * price) / new_qty
            self.positions[symbol] = (new_qty, new_avg)
        else:
            if qty > old_qty + 1e-9:
                raise ValueError(f"{self.bot_id}: sell of {qty} {symbol} exceeds held {old_qty}")
            self.cash += qty * price - fees
            self.realized_pnl += (price - old_avg) * qty
            new_qty = old_qty - qty
            if new_qty < 1e-9:
                self.positions.pop(symbol, None)
            else:
                self.positions[symbol] = (new_qty, old_avg)
        self.fees_paid += fees

    def equity(self, marks: dict[str, float] | None = None) -> float:
        """Cash plus positions valued at the given marks. A symbol with no mark
        is valued at its average price paid."""
        marks = marks or {}
        value = 0.0
        for symbol, (qty, avg_price) in self.positions.items():
            value += qty * marks.get(symbol, avg_price)
        return self.cash + value

    def snapshot(self) -> dict:
        """A plain dict for the store; positions become lists so it is JSON-safe."""
        return {
            "bot_id": self.bot_id,
            "cash": self.cash,
            "positions": {s: [q, a] for s, (q, a) in self.positions.items()},
            "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid,
        }

    @classmethod
    def from_snapshot(cls, d: dict) -> VirtualAccount:
        return cls(
            bot_id=d["bot_id"],
            cash=float(d["cash"]),
            positions={
                s: (float(q), float(a)) for s, (q, a) in dict(d.get("positions") or {}).items()
            },
            realized_pnl=float(d.get("realized_pnl", 0.0)),
            fees_paid=float(d.get("fees_paid", 0.0)),
        )
