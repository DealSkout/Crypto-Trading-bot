"""The cost model. Rule 3: every fill goes through here, so no result is ever gross-only."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission: float = 0.0  # flat, per order, in the quote currency (USD)
    fee_pct: float = 0.25  # exchange fee, % of notional per side (Alpaca crypto taker fee)
    spread_bps: float = 5.0  # full spread; half is paid on each side
    slippage_bps: float = 5.0  # per side

    def fill_price(self, reference: float, is_buy: bool) -> float:
        """Price actually paid/received: reference moved against us by half-spread + slippage."""
        adverse = (self.spread_bps / 2 + self.slippage_bps) / 10_000.0
        return reference * (1 + adverse) if is_buy else reference * (1 - adverse)

    def fees(self, notional: float, fx: bool = True) -> float:
        """Explicit fees on one side: commission plus the percentage fee. `fx` is kept for
        interface compatibility; crypto fees always apply."""
        fee = abs(notional) * self.fee_pct / 100.0 if fx else 0.0
        return self.commission + fee

    def round_trip_cost(self, notional: float, fx: bool = True) -> float:
        """Estimated total cost of opening and closing a position of this size."""
        implicit = abs(notional) * (self.spread_bps + 2 * self.slippage_bps) / 10_000.0
        return implicit + 2 * self.fees(notional, fx=fx)

    def round_trip_cost_pct(self, fx: bool = True) -> float:
        return 100.0 * self.round_trip_cost(1.0, fx=fx)
