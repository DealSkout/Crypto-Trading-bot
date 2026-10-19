"""The cost model. Rule 3: every fill goes through here, so no result is ever gross-only."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission: float = 0.0  # flat, per order, in the quote currency (USD)
    fee_pct: float = 0.25  # exchange fee, % of notional per side (Alpaca crypto taker fee)
    spread_bps: float = 5.0  # full spread; half is paid on each side
    slippage_bps: float = 5.0  # per side
    # Extra adverse move on SIMULATED stop exits, on top of normal slippage. Live, the stop
    # is a software stop polled every 30 s and closed with a market order, so fast moves fill
    # beyond the stop price; this keeps backtests from assuming a perfect stop fill.
    stop_gap_bps: float = 10.0

    def fill_price(self, reference: float, is_buy: bool) -> float:
        """Price actually paid/received: reference moved against us by half-spread + slippage."""
        adverse = (self.spread_bps / 2 + self.slippage_bps) / 10_000.0
        return reference * (1 + adverse) if is_buy else reference * (1 - adverse)

    def stop_exit_reference(self, reference: float, closing_long: bool) -> float:
        """Reference price for a stop exit: pushed further through the stop before the normal
        fill_price adjustment is applied on top."""
        gap = self.stop_gap_bps / 10_000.0
        return reference * (1 - gap) if closing_long else reference * (1 + gap)

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
