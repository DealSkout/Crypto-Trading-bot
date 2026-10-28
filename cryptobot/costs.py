"""The cost model. Rule 3: every fill goes through here, so no result is ever gross-only."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission: float = 0.0  # flat, per order, in the quote currency (USD)
    fee_pct: float = 0.25  # taker fee, % of notional per side (Alpaca crypto, lowest tier)
    spread_bps: float = 5.0  # full spread; half is paid on each side
    slippage_bps: float = 5.0  # per side
    # Extra adverse move on SIMULATED stop exits, on top of normal slippage. Live, the stop
    # is a software stop polled every 30 s and closed with a market order, so fast moves fill
    # beyond the stop price; this keeps backtests from assuming a perfect stop fill.
    stop_gap_bps: float = 10.0
    # entry filter: skip a trade whose expected move is below this multiple of the round-trip
    # cost (0 = off). A trade without a target is expected to move no_target_move_r times its
    # stop distance.
    maker_fee_pct: float | None = None  # fee on a resting limit order that fills; None = taker
    min_edge_multiple: float = 0.0
    no_target_move_r: float = 2.0

    def fill_price(self, reference: float, is_buy: bool) -> float:
        """Price actually paid/received: reference moved against us by half-spread + slippage."""
        adverse = (self.spread_bps / 2 + self.slippage_bps) / 10_000.0
        return reference * (1 + adverse) if is_buy else reference * (1 - adverse)

    def stop_exit_reference(self, reference: float, closing_long: bool) -> float:
        """Reference price for a stop exit: pushed further through the stop before the normal
        fill_price adjustment is applied on top."""
        gap = self.stop_gap_bps / 10_000.0
        return reference * (1 - gap) if closing_long else reference * (1 + gap)

    @property
    def maker_pct(self) -> float:
        return self.fee_pct if self.maker_fee_pct is None else self.maker_fee_pct

    def fees(self, notional: float, fx: bool = True, maker: bool = False) -> float:
        """Explicit fees on one side: commission plus the percentage fee (the maker rate for a
        resting limit order that filled, else the taker rate). `fx` is kept for interface
        compatibility; crypto fees always apply."""
        pct = self.maker_pct if maker else self.fee_pct
        fee = abs(notional) * pct / 100.0 if fx else 0.0
        return self.commission + fee

    def round_trip_cost(self, notional: float, fx: bool = True, maker_entry: bool = False) -> float:
        """Estimated total cost of opening and closing a position of this size: the spread,
        the entry and exit fees, and slippage on each market side. A resting limit entry pays
        the maker fee and no half-spread or slippage; the exit is costed as a market order."""
        n = abs(notional)
        if maker_entry:
            implicit = n * (self.spread_bps / 2 + self.slippage_bps) / 10_000.0
        else:
            implicit = n * (self.spread_bps + 2 * self.slippage_bps) / 10_000.0
        return implicit + self.fees(n, fx=fx, maker=maker_entry) + self.fees(n, fx=fx)

    def round_trip_cost_pct(self, fx: bool = True, maker_entry: bool = False) -> float:
        return 100.0 * self.round_trip_cost(1.0, fx=fx, maker_entry=maker_entry)
