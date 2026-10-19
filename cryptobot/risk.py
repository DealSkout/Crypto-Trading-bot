"""Rule 4: risk limits live here, in code, not in any strategy.

Sizing, max open positions, the daily loss stop, the end-of-day flatten window and the
software stop/target checks all go through RiskManager. Strategies only propose.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from cryptobot.config import RiskConfig
from cryptobot.costs import CostModel


@dataclass
class RiskState:
    day: date | None = None
    day_start_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    blocked_symbols: set[str] = field(default_factory=set)
    entries_today: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ExitDecision:
    reason: str  # "stop" | "target"
    price: float  # reference price the exit order is sent against


def kelly_risk_pct(r_multiples, fraction: float, floor_pct: float, min_trades: int) -> float | None:
    """Fractional Kelly as a % of equity to risk per trade, from closed trades' R multiples.
    f* = W - (1 - W) / B (W = win rate, B = average win / average loss, in R). None when there
    are fewer than `min_trades` trades. With no measured edge it returns `floor_pct` so the bot
    keeps trading small enough to learn. The caller caps it at risk_per_trade_pct."""
    r = [float(x) for x in r_multiples if x == x]
    if len(r) < min_trades:
        return None
    wins = [x for x in r if x > 0]
    losses = [-x for x in r if x < 0]
    if not wins:
        return floor_pct
    w = len(wins) / len(r)
    if not losses:
        return max(floor_pct, fraction * w * 100.0)
    b = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
    f = w - (1.0 - w) / b
    return max(floor_pct, fraction * f * 100.0)


class RiskManager:
    def __init__(
        self,
        risk: RiskConfig,
        allow_short: bool = False,
        max_entries_per_day: int = 1,
        cost_model: CostModel | None = None,
        fx_for=None,
    ) -> None:
        self.cfg = risk
        self.allow_short = allow_short
        self.max_entries_per_day = max(1, int(max_entries_per_day))
        self.cost_model = cost_model or CostModel(0.0, 0.0, 0.0, 0.0)
        self._fx_for = fx_for or (lambda symbol: True)
        self.state = RiskState()

    # day lifecycle -----------------------------------------------------------------
    def start_day(self, day: date, equity: float) -> None:
        self.state = RiskState(day=day, day_start_equity=float(equity))

    def halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason

    def block_entries(self, symbol: str) -> None:
        self.state.blocked_symbols.add(symbol)

    # entries -----------------------------------------------------------------------
    def position_size(
        self,
        equity: float,
        cash: float,
        entry: float,
        stop: float,
        round_qty: Callable[[float], float],
        symbol: str = "",
        open_exposure: float = 0.0,
        risk_pct: float | None = None,
        daily_vol_pct: float | None = None,
    ) -> tuple[float, str]:
        """Units such that (entry - stop) * units <= the risk budget, capped by what cash can
        buy after spread, slippage and fees (no leverage) and by the account-wide exposure cap
        (all open positions together <= max_total_exposure_pct of equity), rounded down to what
        the broker can trade.

        The risk budget is risk_per_trade_pct of equity, or less: `risk_pct` (e.g. quarter
        Kelly) can only lower it. With vol targeting (`daily_vol_pct`, the asset's typical daily
        move in %), the position is also kept small enough that a typical day moves it by at
        most target_daily_vol_pct of equity. Nothing here can make a position bigger."""
        per_unit = abs(entry - stop)
        if per_unit <= 0 or entry <= 0:
            return 0.0, "stop distance is zero"
        budget = self.cfg.risk_per_trade_pct
        notes = []
        if risk_pct is not None and risk_pct < budget:
            budget = max(0.0, float(risk_pct))
            notes.append(f"kelly {budget:.2f}%")
        risk_amount = equity * budget / 100.0
        qty = risk_amount / per_unit
        if daily_vol_pct is not None and daily_vol_pct > 0:
            vol_qty = (
                equity * self.cfg.target_daily_vol_pct / 100.0 / (entry * daily_vol_pct / 100.0)
            )
            if vol_qty < qty:
                qty = vol_qty
                notes.append(f"vol-target at {daily_vol_pct:.2f}% daily vol")
        if qty <= 0:
            return 0.0, "risk budget is zero"
        cm = self.cost_model
        fill = cm.fill_price(entry, True)
        fx = self._fx_for(symbol) if symbol else True
        per_unit_cost = fill * (1.0 + (cm.fee_pct / 100.0 if fx else 0.0))
        affordable = max(0.0, cash - cm.commission) / per_unit_cost
        room = equity * self.cfg.max_total_exposure_pct / 100.0 - max(0.0, open_exposure)
        if room <= 0:
            return 0.0, (
                f"total exposure cap reached ({self.cfg.max_total_exposure_pct}% of equity"
                " already in open positions)"
            )
        exposure_limit = room / per_unit_cost
        reason = f"ok ({', '.join(notes)})" if notes else "ok"
        if exposure_limit < min(qty, affordable):
            reason = "ok (capped by max_total_exposure_pct)"
        elif affordable < qty:
            reason = "ok (cash-capped: no leverage)"
        qty = round_qty(min(qty, affordable, exposure_limit))
        if qty <= 0:
            return 0.0, "position rounds to zero (risk budget, cash or exposure room too small)"
        return qty, reason

    def can_enter(
        self, symbol: str, side: str, n_open_positions: int, minutes_to_close: float
    ) -> tuple[bool, str]:
        s = self.state
        if s.halted:
            return False, f"halted: {s.halt_reason}"
        if symbol in s.blocked_symbols:
            return False, "entries closed for the day after the end-of-day flatten"
        if (
            self.cfg.session_policy != "swing"
            and minutes_to_close <= self.cfg.flatten_minutes_before_close
        ):
            return False, "too close to the close"
        if side == "short" and not self.allow_short:
            return False, "long-only: crypto cannot be shorted (no margin)"
        if n_open_positions >= self.cfg.max_positions:
            return False, f"max_positions ({self.cfg.max_positions}) reached"
        if s.entries_today.get(symbol, 0) >= self.max_entries_per_day:
            return False, f"max_entries_per_day ({self.max_entries_per_day}) reached"
        return True, "ok"

    def record_entry(self, symbol: str) -> None:
        self.state.entries_today[symbol] = self.state.entries_today.get(symbol, 0) + 1

    # exits -------------------------------------------------------------------------
    def daily_loss_breached(self, equity: float) -> bool:
        start = self.state.day_start_equity
        if start <= 0:
            return False
        loss_pct = (equity - start) / start * 100.0
        if loss_pct <= -self.cfg.max_daily_loss_pct:
            self.halt(f"daily loss {loss_pct:.2f}% <= -{self.cfg.max_daily_loss_pct}%")
            return True
        return False

    def should_flatten(self, minutes_to_close: float) -> bool:
        if self.cfg.session_policy == "swing":
            return False  # swing bots hold across the UTC day roll
        return minutes_to_close <= self.cfg.flatten_minutes_before_close

    @staticmethod
    def check_exit(
        side: str,
        stop: float,
        target: float | None,
        bar_open: float,
        bar_high: float,
        bar_low: float,
    ) -> ExitDecision | None:
        """Software stop/target on one bar (or one tick, with open=high=low=price).
        If both are touched in the same bar we assume the stop hit first. Gaps through a level
        fill at the open, never at the level."""
        if side == "long":
            if bar_low <= stop:
                return ExitDecision("stop", min(stop, bar_open))
            if target is not None and bar_high >= target:
                return ExitDecision("target", max(target, bar_open))
        else:
            if bar_high >= stop:
                return ExitDecision("stop", max(stop, bar_open))
            if target is not None and bar_low <= target:
                return ExitDecision("target", min(target, bar_open))
        return None
