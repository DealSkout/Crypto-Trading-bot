"""The bar-by-bar engine. Backtest, paper and live all run this exact code path; only the
broker and the source of bars differ.

Per completed bar, in this order:
  1. mark the latest price
  2. software stop / target check on the open trade (Rule 4)
  3. daily loss stop: flatten and halt for the day
  4. end-of-day flatten window: flatten and block new entries
  5. ask the strategy for a signal; size it; send a market order
Everything is written to the journal.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from types import ModuleType
from typing import Any

import pandas as pd

from cryptobot import sessions
from cryptobot.brokers.base import Broker, BrokerError, Order
from cryptobot.config import Config
from cryptobot.decider import Decider, build_context
from cryptobot.features import decision_features, inputs_hash
from cryptobot.journal import Journal
from cryptobot.risk import RiskManager
from cryptobot.strategies import Signal, due_check, strategy_name

log = logging.getLogger(__name__)


class EngineError(Exception):
    pass


@dataclass
class OpenTrade:
    symbol: str
    side: str
    qty: float
    entry_price: float
    entry_ts: datetime
    stop: float
    target: float | None
    fees: float
    implicit_cost: float
    risk_per_unit: float
    reason: str
    signal_price: float = 0.0
    backup_order_id: str = ""
    regime: str = ""
    features: str = ""  # JSON of the features at the entry decision
    entry_slippage_bps: float = 0.0
    lot_id: str = ""  # one id per position, shared with the team router


@dataclass
class TradeRecord:
    symbol: str
    side: str
    qty: float
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    stop: float
    target: float | None
    pnl: float
    r_multiple: float
    costs: float
    exit_reason: str
    signal_reason: str
    version: str = ""
    regime: str = ""
    signal_price: float = 0.0
    entry_slippage_bps: float = 0.0
    exit_slippage_bps: float = 0.0
    fees: float = 0.0
    features: str = ""
    planned_r: float | None = None  # R at the planned exit level, before any costs
    actual_r: float | None = None  # net P&L in units of the planned risk

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RestingEntry:
    """A simulated maker limit entry waiting for the market to trade through it."""

    lot_id: str
    signal: Signal
    qty: float
    limit: float
    placed_ts: datetime
    expires_ts: datetime  # bar start time from which it is cancelled if still unfilled


class Engine:
    def __init__(
        self,
        cfg: Config,
        strategy: ModuleType,
        params: dict[str, Any],
        broker: Broker,
        risk: RiskManager,
        journal: Journal,
        mode: str = "backtest",
        notifier=None,
        decider: Decider | None = None,
    ) -> None:
        self.cfg = cfg
        self.decider = decider
        self.strategy = strategy
        self.strategy_name = strategy_name(strategy)
        self._due_fn = due_check(strategy)
        self.params = dict(params)
        self.broker = broker
        self.risk = risk
        self.journal = journal
        self.mode = mode
        self.notifier = notifier
        self.open: dict[str, list[OpenTrade]] = {}  # symbol -> open lots, oldest first
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.last_prices: dict[str, float] = {}
        self.day_pnl = 0.0
        self.day_costs = 0.0
        self.day_trades = 0
        self.errors = 0
        self.capped_entries = 0
        self.skipped = 0
        # Simulated brokers fill a signal at the NEXT bar's open (a live order goes out after
        # the bar closes, so it never gets the signal bar's close). Real brokers (Alpaca paper
        # or live) fill the market order immediately, which is moments after the close anyway.
        self.fill_next_open = bool(getattr(broker, "delayed_entry", False))
        self.pending: dict[str, tuple[Signal, datetime]] = {}
        # resting maker entries (simulated brokers, orders.entry_type "limit"); each also
        # holds a slot in `pending` so every "is anything waiting" check stays one lookup
        self.limit_entries = self.fill_next_open and cfg.orders.entry_type == "limit"
        self.resting: dict[str, RestingEntry] = {}
        # execution events for the team router (None = nobody listens: backtests, solo bot)
        self.outbox: list[dict[str, Any]] | None = None
        self.dedupe_skips = True  # False: journal every refused entry, not once a day
        self._skip_seen: set[tuple[str, str]] = set()
        # the feature store: every decision is journalled with what was known at that moment
        self.version = "config"  # the model version trading (the registry sets this)
        self.context_providers: list = []  # regime, research snapshot: see features.py
        self.record_decisions = True
        # model filters from the registry: skip rules on decision features and per-regime size
        # weights (capped at 1: a filter can only make a position smaller, never bigger)
        self.filters: dict[str, Any] = {}
        self._size_mult: dict[str, float] = {}
        # pause checks: (symbol, ts) -> reason or None. Any reason blocks NEW entries.
        self.pauses: list = []
        self._source: dict[str, pd.DataFrame] = {}  # the bar window behind the current signal
        self._info: dict[str, tuple[int, dict[str, Any]]] = {}

    # ----------------------------------------------------------------------- helpers
    def _say(self, text: str) -> None:
        log.info(text)
        if self.notifier is not None:
            try:
                self.notifier.send(text)
            except Exception as e:  # notifications must never break trading
                log.warning("notify failed: %s", e)

    @property
    def pending_entry(self) -> dict[str, tuple[Signal, datetime]]:
        """Entries queued for the next bar's open (simulated brokers)."""
        return self.pending

    def equity(self) -> float:
        return float(self.broker.get_equity())

    def lots(self, symbol: str) -> list[OpenTrade]:
        return self.open.get(symbol, [])

    def total_lots(self) -> int:
        return sum(len(lots) for lots in self.open.values())

    def _all_lots(self):
        for lots in self.open.values():
            yield from lots

    def open_exposure(self) -> float:
        """Market value of every open position (the account-wide exposure cap uses this)."""
        return sum(t.qty * self.last_prices.get(t.symbol, t.entry_price) for t in self._all_lots())

    def _start_day(self, day, ts: datetime) -> None:
        prev = self.risk.state.day
        if prev is not None:
            self.journal.record_daily(
                prev,
                self.risk.state.day_start_equity,
                self.equity(),
                self.day_pnl,
                self.day_trades,
                self.day_costs,
                self.risk.state.halted,
            )
        self.risk.start_day(day, self.equity())
        self._skip_seen.clear()
        for symbol in list(self.resting):
            self._cancel_resting(symbol, "a new UTC day started", ts)
        self.pending.clear()  # an overnight signal is stale; strategies re-signal
        self.day_pnl = self.day_costs = 0.0
        self.day_trades = 0
        self.journal.log("day_start", f"{day} equity={self.equity():.2f}", ts=ts)

    # ----------------------------------------------------------------------- main hooks
    def on_bar(self, symbol: str, bars: pd.DataFrame) -> None:
        """Process the bar that just closed (the last row of `bars`)."""
        self.process_bar(
            symbol,
            bars.index[-1].to_pydatetime(),
            int(bars.attrs.get("timeframe") or self.cfg.timeframe_minutes),
            float(bars["open"].iat[-1]),
            float(bars["high"].iat[-1]),
            float(bars["low"].iat[-1]),
            float(bars["close"].iat[-1]),
            lambda: bars,
        )

    def process_bar(
        self,
        symbol: str,
        ts: datetime,
        tf: int,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        price: float,
        window: Callable[[], pd.DataFrame],
    ) -> None:
        """The per-bar steps. `window()` returns the history DataFrame ending at this bar; it
        is only built when something needs it (backtests skip it on most bars)."""
        market = sessions.market_for(symbol)
        bar_end = ts + timedelta(minutes=tf)
        day = sessions.session_date(market, ts)
        if day != self.risk.state.day:
            self._start_day(day, ts)

        if symbol in self.resting:
            self._work_resting(symbol, bar_open, bar_low, ts, sessions.minutes_to_close(market, ts))
        elif symbol in self.pending:
            self._fill_pending(symbol, bar_open, ts, sessions.minutes_to_close(market, ts))

        self.broker.mark(symbol, price, ts)
        self.last_prices[symbol] = price

        for trade in list(self.lots(symbol)):
            decision = RiskManager.check_exit(
                trade.side, trade.stop, trade.target, bar_open, bar_high, bar_low
            )
            if decision is not None:
                ref = decision.price
                if decision.reason == "stop" and self.fill_next_open:  # simulated fills only
                    ref = self.risk.cost_model.stop_exit_reference(
                        ref, closing_long=trade.side == "long"
                    )
                self._exit(symbol, decision.reason, ref, ts, trade=trade)
            elif (
                self.decider is not None
                and self.decider.role == "primary"
                and not self.risk.state.halted
                and self.decider.due(symbol, ts)
            ):
                self._llm_manage_open(symbol, window(), ts, mtc=None)

        if not self.risk.state.halted and self.risk.daily_loss_breached(self.equity()):
            self.journal.log("halt", self.risk.state.halt_reason, ts=ts)
            self._say(
                f"DAILY LOSS STOP: {self.risk.state.halt_reason}. Flattening, done for today."
            )
            for sym in list(self.resting):
                self._cancel_resting(sym, "halted by the daily loss stop", ts)
            self.pending.clear()
            self._flatten("daily_loss", ts)

        mtc = sessions.minutes_to_close(market, bar_end)
        if self.risk.should_flatten(mtc):
            same_market = [s for s in self.open if sessions.market_for(s) is market]
            if same_market:
                self._flatten("eod", ts, symbols=same_market)
            self.risk.block_entries(symbol)

        if (
            len(self.lots(symbol)) < self.cfg.max_lots_per_symbol
            and symbol not in self.pending
            and not self.risk.state.halted
            and (self.decider is not None or self._due(symbol, ts, tf))
        ):
            self._maybe_enter(symbol, window(), ts, mtc)

        self.equity_curve.append((ts, self.equity()))
        if self.mode != "backtest":
            self.snapshot(ts)

    def _due(self, symbol: str, ts: datetime, tf: int) -> bool:
        if self._due_fn is None:
            return True
        try:
            return bool(self._due_fn(symbol, ts, tf, self.params))
        except Exception:  # a broken pre-check must never hide a signal
            return True

    def snapshot(self, ts: datetime | None = None) -> dict[str, Any]:
        """Write the engine's current state to the journal (the dashboard reads it)."""
        s = self.risk.state
        open_trades = []
        for t in self._all_lots():
            last = self.last_prices.get(t.symbol, t.entry_price)
            direction = 1.0 if t.side == "long" else -1.0
            open_trades.append(
                {
                    "symbol": t.symbol,
                    "side": t.side,
                    "qty": t.qty,
                    "entry_price": t.entry_price,
                    "entry_ts": t.entry_ts.isoformat(),
                    "stop": t.stop,
                    "target": t.target,
                    "reason": t.reason,
                    "last_price": last,
                    "unrealized": (last - t.entry_price) * t.qty * direction,
                    "risk": t.risk_per_unit * t.qty,
                }
            )
        state = {
            "ts": (ts or datetime.now(UTC)).isoformat(),
            "day": str(s.day),
            "equity": self.equity(),
            "cash": float(self.broker.get_cash()),
            "day_pnl": self.day_pnl,
            "day_costs": self.day_costs,
            "day_trades": self.day_trades,
            "halted": s.halted,
            "halt_reason": s.halt_reason,
            "open_trades": open_trades,
            "last_prices": dict(self.last_prices),
            "strategy": self.strategy_name,
            "params": self.params,
            "mode": self.mode,
            "broker": self.broker.name,
            "decider": (
                {
                    "calls": self.decider.calls,
                    "spend": self.decider.spend,
                    "describe": self.decider.describe(),
                }
                if self.decider
                else None
            ),
        }
        try:
            self.journal.set_state("engine", state, ts=ts)
        except Exception as e:  # a dashboard snapshot must never stop trading
            log.warning("state snapshot failed: %s", e)
        return state

    def on_tick(self, symbol: str, price: float, ts: datetime) -> None:
        """Between bars (live/paper): check software stops against the latest price, and fill
        a resting limit entry the moment the price trades at or through it."""
        self.broker.mark(symbol, price, ts)
        self.last_prices[symbol] = price
        resting = self.resting.get(symbol)
        if resting is not None and price <= resting.limit:
            mtc = sessions.minutes_to_close(sessions.market_for(symbol), ts)
            self._work_resting(symbol, price, price, ts, mtc, tick=True)
        for trade in list(self.lots(symbol)):
            decision = RiskManager.check_exit(
                trade.side, trade.stop, trade.target, price, price, price
            )
            if decision is not None:
                self._exit(symbol, decision.reason, decision.price, ts, trade=trade)

    def end_of_data(self, ts: datetime | None = None) -> None:
        self.resting.clear()
        self.pending.clear()  # a signal on the very last bar has no next bar to fill on
        if self.open:
            last = ts or max(t.entry_ts for t in self._all_lots())
            self._flatten("end_of_data", last)
        if self.risk.state.day is not None:
            self.journal.record_daily(
                self.risk.state.day,
                self.risk.state.day_start_equity,
                self.equity(),
                self.day_pnl,
                self.day_trades,
                self.day_costs,
                self.risk.state.halted,
            )

    # ----------------------------------------------------------------------- entries
    def _strategy_signal(self, symbol: str, bars: pd.DataFrame, ts: datetime) -> Signal | None:
        try:
            sig = self.strategy.signal(bars, self.params)
        except Exception as e:
            self.errors += 1
            self.journal.log("error", f"strategy raised: {e!r}", symbol=symbol, ts=ts)
            log.exception("strategy %s raised", self.strategy_name)
            return None
        if sig is None:
            return None
        if not isinstance(sig, Signal):
            self.journal.log("error", "strategy returned a non-Signal", symbol=symbol, ts=ts)
            return None
        return sig

    # ----------------------------------------------------------------------- feature store
    def _decision_info(self, symbol: str) -> dict[str, Any]:
        """Features, regime and input fingerprint for the symbol's current signal, computed
        once per signal and only when a decision is actually recorded."""
        bars = self._source.get(symbol)
        if bars is None:
            return {"features": {}, "regime": "", "inputs_hash": ""}
        cached = self._info.get(symbol)
        if cached is not None and cached[0] == id(bars):
            return cached[1]
        try:
            feats = decision_features(symbol, bars, self.context_providers)
            digest = inputs_hash(bars, self.strategy_name, self.params)
        except Exception as e:  # the journal must never stop trading
            log.warning("feature computation failed for %s: %s", symbol, e)
            feats, digest = {"error": repr(e)}, ""
        if len(bars):  # the bar the decision was made on (a queued entry fills one bar later)
            feats["signal_bar"] = bars.index[-1].tz_convert("UTC").isoformat()
        info = {"features": feats, "regime": str(feats.get("regime", "")), "inputs_hash": digest}
        self._info[symbol] = (id(bars), info)
        return info

    def _record_decision(self, symbol: str, action: str, ts: datetime, **fields: Any) -> int:
        if not self.record_decisions:
            return 0
        info = self._decision_info(symbol)
        try:
            return self.journal.record_decision(
                ts,
                version=self.version,
                strategy=self.strategy_name,
                symbol=symbol,
                action=action,
                regime=info["regime"],
                params=self.params,
                features=info["features"],
                inputs_hash=info["inputs_hash"],
                **fields,
            )
        except Exception as e:  # pragma: no cover - bookkeeping must never stop trading
            log.warning("decision journal write failed: %s", e)
            return 0

    def _skip(self, symbol: str, why: str, sig: Signal, ts: datetime) -> None:
        """Log a refused signal once per symbol, reason and day (a strategy that keeps firing
        while blocked would otherwise write a skip on every bar)."""
        self.skipped += 1  # every refusal counts...
        key = (symbol, why)
        if self.dedupe_skips and key in self._skip_seen:
            return  # ...but is journalled once per symbol, reason and day
        self._skip_seen.add(key)
        self.journal.log("skip", why, symbol=symbol, data={"signal": asdict(sig)}, ts=ts)
        self._record_decision(
            symbol,
            "skip",
            ts,
            reason=why,
            side=sig.side,
            signal_price=sig.entry,
            stop=sig.stop,
            target=sig.target,
        )

    def _try_enter(
        self,
        symbol: str,
        sig: Signal,
        ts: datetime,
        mtc: float,
        defer: bool = True,
        signal_price: float | None = None,
    ) -> bool:
        sig = self._cap_stop(symbol, sig, ts)
        n_open = self.total_lots() + len(self.pending)
        ok, why = self.risk.can_enter(symbol, sig.side, n_open, mtc)
        if not ok:
            self._skip(symbol, why, sig, ts)
            return False
        if signal_price is None:  # at signal time (a pending fill was filtered already)
            ok, why, mult = self._apply_filters(symbol)
            if not ok:
                self._skip(symbol, why, sig, ts)
                return False
            edge = self._edge_check(sig)
            if edge:
                self._skip(symbol, edge, sig, ts)
                return False
            self._size_mult[symbol] = mult
        for pause in self.pauses:  # macro windows, stale data: no NEW entries
            try:
                paused = pause(symbol, ts)
            except Exception as e:  # pragma: no cover - a broken check pauses, never trades
                paused = f"pause check failed: {e}"
            if paused:
                self._skip(symbol, f"paused: {paused}", sig, ts)
                return False
        if defer and self.limit_entries:
            return self._place_resting(symbol, sig, ts)
        if defer and self.fill_next_open:
            self.pending[symbol] = (sig, ts)
            self.journal.log(
                "signal",
                f"{sig.reason} (fills at the next bar's open)",
                symbol=symbol,
                data=asdict(sig),
                ts=ts,
            )
            return True
        mult = min(1.0, max(0.0, self._size_mult.pop(symbol, 1.0)))
        risk_pct, daily_vol = self._sizing_inputs(symbol)
        qty, why = self.risk.position_size(
            self.equity(),
            self.broker.get_cash(),
            sig.entry,
            sig.stop,
            lambda q: self.broker.round_qty_for(symbol, q * mult),
            symbol=symbol,
            open_exposure=self.open_exposure(),
            risk_pct=risk_pct,
            daily_vol_pct=daily_vol,
        )
        if qty <= 0:
            self._skip(symbol, why, sig, ts)
            return False
        if "capped" in why:
            self.capped_entries += 1
        self._enter(symbol, sig, qty, ts, signal_price=signal_price)
        return symbol in self.open

    def _edge_check(self, sig: Signal) -> str:
        """Empty when the trade's expected move pays for its costs at least
        costs.min_edge_multiple times over; else the reason it is skipped."""
        cm = getattr(self.risk, "cost_model", None)
        mult = float(getattr(cm, "min_edge_multiple", 0.0) or 0.0) if cm is not None else 0.0
        if mult <= 0 or sig.entry <= 0:
            return ""
        if sig.target is not None:
            move = abs(sig.target - sig.entry)
            basis = "target"
        else:
            move = cm.no_target_move_r * sig.risk_per_unit
            basis = f"{cm.no_target_move_r:g}x stop distance, no target"
        move_pct = 100.0 * move / sig.entry
        maker = self.cfg.orders.entry_type == "limit"
        cost_pct = cm.round_trip_cost_pct(maker_entry=maker)
        if move_pct < mult * cost_pct:
            return (
                f"edge filter: expected move {move_pct:.2f}% ({basis}) is under"
                f" {mult:g}x the round-trip cost {cost_pct:.2f}%"
            )
        return ""

    def _emit(self, event: dict[str, Any]) -> None:
        if self.outbox is not None:
            self.outbox.append(event)

    def _place_resting(self, symbol: str, sig: Signal, ts: datetime) -> bool:
        """Size the trade now and rest a maker limit buy at the signal price less
        orders.maker_offset_bps; it fills only if the market trades down to it."""
        orders = self.cfg.orders
        sign = 1.0 if sig.side == "long" else -1.0
        limit = sig.entry * (1 - sign * orders.maker_offset_bps / 10_000.0)
        beyond = limit <= sig.stop if sig.side == "long" else limit >= sig.stop
        if beyond:
            self._skip(symbol, "maker limit would sit beyond the stop", sig, ts)
            return False
        rest_sig = Signal(sig.side, limit, sig.stop, sig.target, reason=sig.reason)
        mult = min(1.0, max(0.0, self._size_mult.pop(symbol, 1.0)))
        risk_pct, daily_vol = self._sizing_inputs(symbol)
        qty, why = self.risk.position_size(
            self.equity(),
            self.broker.get_cash(),
            limit,
            sig.stop,
            lambda q: self.broker.round_qty_for(symbol, q * mult),
            symbol=symbol,
            open_exposure=self.open_exposure(),
            risk_pct=risk_pct,
            daily_vol_pct=daily_vol,
        )
        if qty <= 0:
            self._skip(symbol, why, sig, ts)
            return False
        tf = timedelta(minutes=int(self.cfg.timeframe_minutes))
        resting = RestingEntry(
            lot_id=uuid.uuid4().hex[:12],
            signal=Signal(sig.side, sig.entry, sig.stop, sig.target, reason=sig.reason),
            qty=qty,
            limit=limit,
            placed_ts=ts,
            expires_ts=ts + tf * (1 + orders.entry_timeout_bars),
        )
        self.resting[symbol] = resting
        self.pending[symbol] = (rest_sig, ts)
        self.journal.log(
            "signal",
            f"{sig.reason} (maker limit {limit:.4f} for {qty}, {orders.entry_timeout_bars} bars)",
            symbol=symbol,
            data=asdict(sig),
            ts=ts,
        )
        self._emit(
            {
                "type": "entry",
                "lot": resting.lot_id,
                "symbol": symbol,
                "side": sig.side,
                "qty": qty,
                "limit": limit,
                "stop": sig.stop,
                "target": sig.target,
                "signal_price": sig.entry,
                # wall-clock deadline for the real order: the bars it may rest after this one
                "expires": ts + tf * (1 + orders.entry_timeout_bars),
            }
        )
        return True

    def _cancel_resting(self, symbol: str, why: str, ts: datetime) -> None:
        resting = self.resting.pop(symbol, None)
        self.pending.pop(symbol, None)
        if resting is None:
            return
        self._skip(symbol, f"entry limit cancelled: {why}", resting.signal, ts)
        self._emit({"type": "entry_cancel", "lot": resting.lot_id, "symbol": symbol, "reason": why})

    def _work_resting(
        self,
        symbol: str,
        bar_open: float,
        bar_low: float,
        ts: datetime,
        mtc: float,
        tick: bool = False,
    ) -> None:
        """Fill the resting limit if this bar (or tick) traded at or through it; cancel it
        once its time is up."""
        resting = self.resting[symbol]
        sig = resting.signal
        if self.risk.state.halted:
            self._cancel_resting(symbol, f"halted: {self.risk.state.halt_reason}", ts)
            return
        if bar_low <= resting.limit:
            price = min(resting.limit, bar_open)  # a gap down fills a resting buy at the open
            if price <= sig.stop:
                self._cancel_resting(symbol, "price gapped through the stop before the fill", ts)
                return
            self.resting.pop(symbol)
            self.pending.pop(symbol, None)
            fill = Signal(sig.side, price, sig.stop, sig.target, reason=sig.reason)
            ok, why = self.risk.can_enter(symbol, sig.side, self.total_lots(), mtc)
            if not ok:
                self._skip(symbol, why, sig, ts)
                self._emit(
                    {"type": "entry_cancel", "lot": resting.lot_id, "symbol": symbol, "reason": why}
                )
                return
            self._enter(
                symbol,
                fill,
                resting.qty,
                ts,
                signal_price=sig.entry,
                lot_id=resting.lot_id,
                maker_price=price,
            )
            return
        if not tick and ts >= resting.expires_ts:
            self._cancel_resting(
                symbol, f"not filled within {self.cfg.orders.entry_timeout_bars} bars", ts
            )

    def cancel_lot(self, lot_id: str, why: str, ts: datetime | None = None) -> bool:
        """The router refused or could not place this entry: drop the resting order."""
        for symbol, resting in list(self.resting.items()):
            if resting.lot_id == lot_id:
                self._cancel_resting(symbol, why, ts or datetime.now(UTC))
                return True
        return False

    def shrink_lot(self, lot_id: str, real_qty: float, ts: datetime | None = None) -> bool:
        """The real account filled less of this entry than the simulation did: undo the
        difference at the entry price (no fee, no P&L), removing the lot if nothing filled."""
        for symbol, lots in list(self.open.items()):
            for trade in list(lots):
                if trade.lot_id != lot_id:
                    continue
                delta = trade.qty - max(0.0, real_qty)
                if delta <= 1e-12:
                    return False
                self.broker.adjust(
                    symbol, -delta if trade.side == "long" else delta, trade.entry_price
                )
                share = delta / trade.qty
                self.broker.cash += trade.fees * share  # the fee on the part that never filled
                trade.fees -= trade.fees * share
                trade.qty -= delta
                self.journal.log(
                    "entry_shortfall",
                    f"the broker filled {max(0.0, real_qty)} of {trade.qty + delta}; the rest"
                    " of the simulated entry was undone",
                    symbol=symbol,
                    ts=ts,
                )
                if trade.qty <= 1e-12:
                    lots.remove(trade)
                    if not lots:
                        del self.open[symbol]
                return True
        return False

    def close_lot_at(self, lot_id: str, price: float, reason: str, ts: datetime) -> bool:
        """The real position closed at the broker (its stop filled): close the simulated lot
        at that same price so the bot's books follow the account."""
        for symbol, lots in list(self.open.items()):
            for trade in list(lots):
                if trade.lot_id == lot_id:
                    self._exit(symbol, reason, price, ts, trade=trade, notify=False)
                    return True
        return False

    def _sizing_inputs(self, symbol: str) -> tuple[float | None, float | None]:
        """(quarter-Kelly risk %, the asset's daily vol %) for the configured sizing method.
        Either can only make a position smaller (risk.position_size caps everything)."""
        method = self.cfg.risk.sizing
        r = self.cfg.risk
        risk_pct = daily_vol = None
        if method in ("kelly", "min"):
            from cryptobot.risk import kelly_risk_pct

            try:
                hist = self.journal.trades(mode=self.journal.mode)
                if "version" in hist.columns and (hist["version"] == self.version).any():
                    hist = hist[hist["version"] == self.version]  # this model's own record
                recent = hist.tail(r.kelly_lookback_trades)
                risk_pct = kelly_risk_pct(
                    recent["r_multiple"].tolist(),
                    r.kelly_fraction,
                    r.kelly_floor_pct,
                    r.kelly_min_trades,
                )
            except Exception as e:  # pragma: no cover - fall back to the plain risk cap
                log.warning("kelly sizing unavailable: %s", e)
        if method in ("vol_target", "min"):
            feats = self._decision_info(symbol)["features"]
            daily_vol = feats.get("vol_7d_pct") or feats.get("atr_14d_pct")
        return risk_pct, daily_vol

    OPS = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
    }

    def _apply_filters(self, symbol: str) -> tuple[bool, str, float]:
        """(enter?, reason, size multiplier <= 1) from the model's filters and the features
        at this decision. A rule whose feature is missing does not fire."""
        if not self.filters:
            return True, "ok", 1.0
        feats = self._decision_info(symbol)["features"]
        for rule in self.filters.get("skip_if", []):
            value = feats.get(rule.get("feature"))
            op = self.OPS.get(rule.get("op"))
            if value is None or op is None:
                continue
            try:
                if op(float(value), float(rule["value"])):
                    return False, f"filter: {rule['feature']} {rule['op']} {rule['value']}", 0.0
            except (TypeError, ValueError):
                continue
        regime = str(feats.get("regime", ""))
        weight = float(self.filters.get("regime_weights", {}).get(regime, 1.0))
        weight = min(1.0, max(0.0, weight))
        if weight <= 0:
            return False, f"filter: no entries in the {regime} regime", 0.0
        return True, "ok", weight

    def set_model(self, strategy, params: dict[str, Any], filters: dict, version: str) -> None:
        """Switch to another model version (between trades: the caller checks it is flat)."""
        self.strategy = strategy
        self.strategy_name = strategy_name(strategy)
        self._due_fn = due_check(strategy)
        self.params = dict(params)
        self.filters = dict(filters or {})
        self.version = version
        self.journal.log("model", f"now trading {version} {self.strategy_name} {self.params}")

    def _fill_pending(self, symbol: str, bar_open: float, ts: datetime, mtc: float) -> None:
        """A backtest signal from the previous bar fills at this bar's open, if it still can:
        the strategy's stop and target levels stay where they were."""
        sig, _sig_ts = self.pending.pop(symbol)
        if self.risk.state.halted:
            self._skip(symbol, f"halted before the fill: {self.risk.state.halt_reason}", sig, ts)
            return
        long = sig.side == "long"
        if (long and bar_open <= sig.stop) or (not long and bar_open >= sig.stop):
            self._skip(symbol, "price gapped through the stop before the fill", sig, ts)
            return
        if sig.target is not None and (
            (long and bar_open >= sig.target) or (not long and bar_open <= sig.target)
        ):
            self._skip(symbol, "price gapped past the target before the fill", sig, ts)
            return
        self.broker.mark(symbol, bar_open, ts)
        self.last_prices[symbol] = bar_open
        fill_sig = Signal(sig.side, bar_open, sig.stop, sig.target, reason=sig.reason)
        self._try_enter(symbol, fill_sig, ts, mtc, defer=False, signal_price=sig.entry)

    def _cap_stop(self, symbol: str, sig: Signal, ts: datetime) -> Signal:
        """Rule 4: no stop may sit further than risk.max_stop_pct from the entry, whatever the
        strategy or the model asked for."""
        cap = sig.entry * self.cfg.risk.max_stop_pct / 100.0
        if sig.risk_per_unit <= cap:
            return sig
        stop = sig.entry - cap if sig.side == "long" else sig.entry + cap
        self.journal.log(
            "stop_capped",
            f"stop {sig.stop:.4f} tightened to {stop:.4f} ({self.cfg.risk.max_stop_pct}% cap)",
            symbol=symbol,
            ts=ts,
        )
        return Signal(sig.side, sig.entry, stop, sig.target, reason=sig.reason + " (stop capped)")

    def _maybe_enter(self, symbol: str, bars: pd.DataFrame, ts: datetime, mtc: float) -> None:
        self._source[symbol] = bars  # what any decision on this bar was based on
        sig = self._strategy_signal(symbol, bars, ts)
        if self.decider is None:
            if sig is not None:
                self._try_enter(symbol, sig, ts, mtc)
            return
        self._llm_entry(symbol, bars, ts, mtc, sig)

    # ----------------------------------------------------------------------- LLM decider
    def _decision_context(self, symbol: str, bars: pd.DataFrame, sig: Signal | None, mtc: float):
        cm = getattr(self.risk, "cost_model", None)
        rt = cm.round_trip_cost_pct(fx=self.risk._fx_for(symbol)) if cm is not None else 0.0
        lots = self.lots(symbol)
        return build_context(
            symbol,
            bars,
            self.equity(),
            self.broker.get_cash(),
            lots[0] if lots else None,
            sig,
            self.cfg.risk,
            self.cfg.allow_short,
            rt,
            self.decider.cfg,
            mtc,
            self.day_pnl,
            self.day_trades,
            self.strategy_name,
        )

    def _llm_entry(
        self, symbol: str, bars: pd.DataFrame, ts: datetime, mtc: float, sig: Signal | None
    ) -> None:
        role = self.decider.role
        if role == "last_resort" and sig is not None:
            self._try_enter(symbol, sig, ts, mtc)  # the rules had an opinion: no model call
            return
        if role == "final_say" and sig is None:
            return
        if role != "final_say" and not self.decider.due(symbol, ts):
            return
        ok, _why = self.risk.can_enter(symbol, "long", self.total_lots(), mtc)
        if not ok:
            return  # do not pay for a decision we could not act on
        decision = self.decider.decide(self._decision_context(symbol, bars, sig, mtc), ts)
        if decision is None or decision.action in ("hold", "close"):
            if sig is not None and role == "final_say":
                self.skipped += 1
                self.journal.log(
                    "skip",
                    f"{self.decider.name} rejected the signal:"
                    f" {decision.reason if decision else 'no answer'}",
                    symbol=symbol,
                    data={"signal": asdict(sig)},
                    ts=ts,
                )
            return
        side = "long" if decision.action == "buy" else "short"
        if role == "final_say" and sig is not None and side != sig.side:
            self.skipped += 1
            self.journal.log(
                "skip",
                f"{self.decider.name} answered {decision.action} to a"
                f" {sig.side} signal: treated as rejection",
                symbol=symbol,
                ts=ts,
            )
            return
        entry = float(bars["close"].iloc[-1])
        reason = f"{self.decider.name}: {decision.reason}"
        try:
            llm_sig = Signal(side, entry, decision.stop_price, decision.target, reason=reason)
        except ValueError as e:
            if sig is not None and role == "final_say":
                llm_sig = Signal(sig.side, sig.entry, sig.stop, sig.target, reason=reason)
            else:
                self.journal.log("decider_invalid", str(e), symbol=symbol, ts=ts)
                return
        self._try_enter(symbol, llm_sig, ts, mtc)

    def _llm_manage_open(self, symbol: str, bars: pd.DataFrame, ts: datetime, mtc) -> None:
        lots = self.lots(symbol)
        if not lots:
            return
        trade = lots[0]
        market = sessions.market_for(symbol)
        tf = int(bars.attrs.get("timeframe") or self.cfg.timeframe_minutes)
        mtc = sessions.minutes_to_close(market, ts + timedelta(minutes=tf)) if mtc is None else mtc
        decision = self.decider.decide(self._decision_context(symbol, bars, None, mtc), ts)
        if decision is None:
            return
        closing = (
            decision.action == "close"
            or (decision.action == "sell" and trade.side == "long")
            or (decision.action == "buy" and trade.side == "short")
        )
        if closing:
            price = self.last_prices.get(symbol, trade.entry_price)
            self._exit(symbol, "llm_close", price, ts, trade=trade)

    def _enter(
        self,
        symbol: str,
        sig: Signal,
        qty: float,
        ts: datetime,
        signal_price: float | None = None,
        lot_id: str = "",
        maker_price: float | None = None,
    ) -> None:
        side = "buy" if sig.side == "long" else "sell"
        if signal_price is None:
            self.journal.log("signal", sig.reason, symbol=symbol, data=asdict(sig), ts=ts)
        orders = self.cfg.orders
        try:
            if maker_price is not None:  # a simulated resting limit the market traded through
                order = self.broker.place_order(
                    symbol, side, qty, order_type="limit", limit_price=maker_price
                )
            elif orders.entry_type == "limit" and getattr(self.broker, "supports_limit", False):
                last = self.last_prices.get(symbol, sig.entry)
                offset = orders.limit_offset_bps / 10_000.0
                limit = last * (1 + offset) if side == "buy" else last * (1 - offset)
                order = self.broker.place_order(
                    symbol, side, qty, order_type="limit", limit_price=limit
                )
            else:
                order = self.broker.place_order(symbol, side, qty)
            order = self.broker.await_fill(order, timeout=orders.fill_timeout_seconds)
            if not order.is_done and order.id:
                if order.filled_qty > 0:
                    order.status = "partially_filled"
                else:  # a resting limit that did not fill in time: cancel, no trade
                    self.broker.cancel(order.id)
                    order.status = "cancelled"
                    order.note = "entry limit not filled in time; cancelled"
        except BrokerError as e:
            self.errors += 1
            self.journal.log("order_rejected", str(e), symbol=symbol, ts=ts)
            if self.mode == "live":
                raise EngineError(f"entry order failed in live mode: {e}") from e
            return
        if order.status in ("rejected", "cancelled") or order.filled_qty <= 0:
            self.journal.log("order_rejected", order.note or order.status, symbol=symbol, ts=ts)
            return
        if order.status == "partially_filled":
            self.journal.log(
                "partial_fill", f"filled {order.filled_qty} of {qty}", symbol=symbol, ts=ts
            )
            try:
                self.broker.cancel(order.id)
            except BrokerError:
                pass
        qty = order.filled_qty
        price = float(order.avg_fill_price or sig.entry)
        if self.broker.name != "paper":
            # Alpaca takes the crypto fee out of the coins received and reports no fee on the
            # order: book it from the cost model and track what is actually held.
            if not order.fees:
                order.fees = self.risk.cost_model.fees(qty * price)
            held = getattr(self.broker, "held", None)
            if held is not None:
                try:
                    have = float(held(symbol))
                    if 0 < have < qty:
                        qty = have
                except BrokerError:
                    pass
        self.open.setdefault(symbol, []).append(
            OpenTrade(
                symbol=symbol,
                side=sig.side,
                qty=qty,
                entry_price=price,
                entry_ts=ts,
                stop=sig.stop,
                target=sig.target,
                fees=order.fees,
                implicit_cost=order.implicit_cost,
                risk_per_unit=sig.risk_per_unit,
                reason=sig.reason,
                signal_price=float(signal_price if signal_price is not None else sig.entry),
                lot_id=lot_id or uuid.uuid4().hex[:12],
            )
        )
        trade = self.open[symbol][-1]
        ref = trade.signal_price
        sgn = 1.0 if sig.side == "long" else -1.0
        trade.entry_slippage_bps = (price - ref) / ref * 10_000.0 * sgn if ref > 0 else 0.0
        info = self._decision_info(symbol)
        trade.regime = info["regime"]
        trade.features = json.dumps(info["features"], sort_keys=True, default=str)
        self._record_decision(
            symbol,
            "enter",
            ts,
            reason=sig.reason,
            side=sig.side,
            signal_price=ref,
            stop=sig.stop,
            target=sig.target,
            qty=qty,
            fill_price=price,
            slippage_bps=trade.entry_slippage_bps,
            fees=order.fees,
        )
        if len(self.open[symbol]) == 1:
            # the broker tracks ONE resting stop per symbol, so only the first lot gets it
            trade.backup_order_id = self._place_backup_stop(symbol, trade, ts)
        self.risk.record_entry(symbol)
        self._emit(
            {
                "type": "entry_fill" if maker_price is not None else "entry_market",
                "lot": trade.lot_id,
                "symbol": symbol,
                "side": sig.side,
                "qty": qty,
                "price": price,
                "stop": sig.stop,
                "target": sig.target,
                "signal_price": trade.signal_price,
            }
        )
        self.journal.log(
            "entry",
            f"{sig.side} {qty} {symbol} @ {price:.4f} stop {sig.stop:.4f}"
            f" target {sig.target if sig.target is None else round(sig.target, 4)}",
            symbol=symbol,
            data={"qty": qty, "price": price, "order_id": order.id},
            ts=ts,
        )
        if self.mode != "backtest":
            self._say(f"ENTRY {sig.side} {qty} {symbol} @ {price:.2f} (stop {sig.stop:.2f})")

    # ----------------------------------------------------------------------- exits
    def _place_backup_stop(self, symbol: str, trade: OpenTrade, ts: datetime) -> str:
        """A resting stop_limit at the broker at (or backup_stop_buffer_pct beyond) the software
        stop: protection that survives a crashed or frozen bot. Cancelled, or booked if it
        already filled, before every normal exit."""
        orders = self.cfg.orders
        if not orders.backup_stop or not getattr(self.broker, "supports_protective_stop", False):
            return ""
        buf = orders.backup_stop_buffer_pct / 100.0
        long = trade.side == "long"
        stop_px = trade.stop * (1 - buf) if long else trade.stop * (1 + buf)
        try:
            order = self.broker.place_protective_stop(
                symbol, trade.qty, stop_px, side="sell" if long else "buy"
            )
        except BrokerError as e:
            self.journal.log("backup_stop_failed", str(e), symbol=symbol, ts=ts)
            return ""
        if not order.id or order.status in ("rejected", "cancelled"):
            self.journal.log("backup_stop_failed", order.note or order.status, symbol=symbol, ts=ts)
            return ""
        self.journal.log(
            "backup_stop",
            f"resting stop_limit at {stop_px:.4f} for {trade.qty}",
            symbol=symbol,
            data={"order_id": order.id},
            ts=ts,
        )
        return order.id

    def _cancel_backup(self, symbol: str, trade: OpenTrade) -> Order | None:
        """Cancel the resting backup stop. Returns it if it had already (partly) filled."""
        oid, trade.backup_order_id = trade.backup_order_id, ""
        if not oid:
            return None
        try:
            return self.broker.cancel_protective(symbol)
        except BrokerError as e:
            self.journal.log("error", f"backup stop cancel failed: {e}", symbol=symbol)
            return None

    def _exit(
        self,
        symbol: str,
        reason: str,
        ref_price: float,
        ts: datetime,
        trade: OpenTrade | None = None,
        notify: bool = True,
    ) -> None:
        lots = self.lots(symbol)
        if not lots:
            return
        trade = lots[0] if trade is None else trade
        if trade not in lots:
            return  # already closed (e.g. flattened earlier in the same pass)
        side = "sell" if trade.side == "long" else "buy"
        backup = self._cancel_backup(symbol, trade)
        if backup is not None and backup.filled_qty >= trade.qty * 0.999:
            order, reason = backup, "backup_stop"
        else:
            self.broker.set_exec_price(symbol, ref_price)
            remaining = trade.qty - (backup.filled_qty if backup is not None else 0.0)
            try:
                order = self.broker.place_order(symbol, side, remaining)
                order = self.broker.await_fill(order)
            except BrokerError as e:
                self.errors += 1
                self.journal.log("order_rejected", f"exit failed: {e}", symbol=symbol, ts=ts)
                if self.mode == "live":
                    raise EngineError(f"exit order failed in live mode: {e}") from e
                return
            if order.status == "rejected" or order.filled_qty <= 0:
                self.errors += 1
                self.journal.log(
                    "order_rejected", f"exit {order.note or order.status}", symbol=symbol, ts=ts
                )
                if self.mode == "live":
                    raise EngineError(f"exit order rejected in live mode: {order.note}")
                return
            if backup is not None:  # part went at the backup stop, the rest at the market
                total = backup.filled_qty + order.filled_qty
                avg = (
                    backup.filled_qty * float(backup.avg_fill_price or ref_price)
                    + order.filled_qty * float(order.avg_fill_price or ref_price)
                ) / total
                order.avg_fill_price, order.filled_qty = avg, total
                order.fees += backup.fees
        exit_price = float(order.avg_fill_price or ref_price)
        if self.broker.name != "paper" and not order.fees:
            order.fees = self.risk.cost_model.fees(trade.qty * exit_price)
        direction = 1.0 if trade.side == "long" else -1.0
        gross = (exit_price - trade.entry_price) * trade.qty * direction
        explicit = trade.fees + order.fees
        implicit = trade.implicit_cost + order.implicit_cost
        pnl = gross - explicit  # implicit costs are already inside the fill prices
        risk_amount = trade.risk_per_unit * trade.qty
        r_multiple = pnl / risk_amount if risk_amount > 0 else 0.0
        # planned vs actual R, both on the PLANNED risk (signal price to stop): the gap is
        # what slippage past the stop and the costs took
        planned_risk = abs(trade.signal_price - trade.stop) if trade.signal_price else 0.0
        if reason in ("stop", "backup_stop", "broker_stop"):
            planned_exit = trade.stop
        elif reason == "target" and trade.target is not None:
            planned_exit = trade.target
        else:
            planned_exit = ref_price
        planned_r = (
            (planned_exit - trade.signal_price) * direction / planned_risk
            if planned_risk > 0
            else None
        )
        actual_r = pnl / (planned_risk * trade.qty) if planned_risk > 0 else None
        rec = TradeRecord(
            symbol=symbol,
            side=trade.side,
            qty=trade.qty,
            entry_ts=trade.entry_ts,
            entry_price=trade.entry_price,
            exit_ts=ts,
            exit_price=exit_price,
            stop=trade.stop,
            target=trade.target,
            pnl=pnl,
            r_multiple=r_multiple,
            costs=explicit + implicit,
            exit_reason=reason,
            signal_reason=trade.reason,
            version=self.version,
            regime=trade.regime,
            signal_price=trade.signal_price,
            entry_slippage_bps=trade.entry_slippage_bps,
            exit_slippage_bps=(
                (ref_price - exit_price) / ref_price * 10_000.0 * direction
                if ref_price > 0
                else 0.0
            ),
            fees=explicit,
            features=trade.features,
            planned_r=planned_r,
            actual_r=actual_r,
        )
        self.trades.append(rec)
        trade_id = self.journal.record_trade(rec.to_dict(), strategy=self.strategy_name)
        if self.record_decisions:
            try:
                self.journal.record_decision(
                    ts,
                    version=self.version,
                    strategy=self.strategy_name,
                    symbol=symbol,
                    action="exit",
                    reason=reason,
                    side=trade.side,
                    signal_price=ref_price,
                    stop=trade.stop,
                    target=trade.target,
                    qty=trade.qty,
                    fill_price=exit_price,
                    slippage_bps=rec.exit_slippage_bps,
                    fees=order.fees,
                    pnl=pnl,
                    regime=trade.regime,
                    params=self.params,
                    trade_id=trade_id,
                )
            except Exception as e:  # pragma: no cover
                log.warning("decision journal write failed: %s", e)
        r_note = (
            f"; planned {planned_r:+.2f}R, actual {actual_r:+.2f}R"
            if planned_r is not None and actual_r is not None
            else ""
        )
        self.journal.log(
            "exit",
            f"{reason}: {trade.side} {trade.qty} {symbol} @ {exit_price:.4f}"
            f" pnl {pnl:+.2f} ({r_multiple:+.2f}R){r_note}",
            symbol=symbol,
            data={
                "pnl": pnl,
                "r": r_multiple,
                "costs": rec.costs,
                "planned_r": planned_r,
                "actual_r": actual_r,
            },
            ts=ts,
        )
        if notify:
            self._emit(
                {
                    "type": "exit",
                    "lot": trade.lot_id,
                    "symbol": symbol,
                    "side": trade.side,
                    "qty": trade.qty,
                    "price": exit_price,
                    "reason": reason,
                }
            )
        self.day_pnl += pnl
        self.day_costs += rec.costs
        self.day_trades += 1
        lots.remove(trade)
        if not lots:
            del self.open[symbol]
        if self.mode != "backtest":
            self._say(f"EXIT ({reason}) {symbol} @ {exit_price:.2f} pnl {pnl:+.2f}")

    def _flatten(self, reason: str, ts: datetime, symbols: list[str] | None = None) -> None:
        for symbol in list(symbols or self.open):
            for trade in list(self.lots(symbol)):
                self._exit(
                    symbol,
                    reason,
                    self.last_prices.get(symbol, trade.entry_price),
                    ts,
                    trade=trade,
                )
        if symbols is None and self.mode != "backtest":
            try:
                self.broker.flatten_all()  # anything the broker holds that we do not know about
            except BrokerError as e:
                self.journal.log("error", f"flatten_all failed: {e}", ts=ts)
                if self.mode == "live":
                    raise EngineError(f"flatten failed in live mode: {e}") from e
        self.journal.log("flatten", reason, ts=ts)

    def emergency_stop(self, why: str) -> None:
        """Live-mode panic button: cancel everything, flatten, tell the human."""
        self.journal.log("emergency_stop", why)
        try:
            self.broker.flatten_all()
        finally:
            self.open.clear()
            self._say(f"EMERGENCY STOP: {why}. All orders cancelled and positions flattened.")

    # ----------------------------------------------------------------------- reporting
    def daily_summary(self) -> str:
        s = self.risk.state
        eq = self.equity()
        lines = [
            f"Cryptobot {self.mode} summary for {s.day}",
            f"equity {eq:.2f} (day pnl {self.day_pnl:+.2f}, costs {self.day_costs:.2f})",
            f"trades today: {self.day_trades}; open: {len(self.open)}",
            f"strategy {self.strategy_name} params {self.params}",
        ]
        if self.decider is not None:
            lines.append(
                f"decider {self.decider.describe()}: {self.decider.calls} calls,"
                f" {self.decider.spend:.2f} USD"
            )
        if s.halted:
            lines.append(f"HALTED: {s.halt_reason}")
        return "\n".join(lines)
