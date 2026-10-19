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

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from types import ModuleType
from typing import Any

import pandas as pd

from cryptobot import sessions
from cryptobot.brokers.base import Broker, BrokerError
from cryptobot.config import Config
from cryptobot.decider import Decider, build_context
from cryptobot.journal import Journal
from cryptobot.risk import RiskManager
from cryptobot.strategies import Signal, strategy_name

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        self.params = dict(params)
        self.broker = broker
        self.risk = risk
        self.journal = journal
        self.mode = mode
        self.notifier = notifier
        self.open: dict[str, OpenTrade] = {}
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.last_prices: dict[str, float] = {}
        # Simulated brokers fill entries at the NEXT bar's open (live market orders fill
        # moments after the bar closes anyway, so this keeps backtests honest).
        self.pending_entry: dict[str, tuple[Signal, datetime]] = {}
        self._skips_logged: set[tuple[str, Any, str]] = set()
        self.day_pnl = 0.0
        self.day_costs = 0.0
        self.day_trades = 0
        self.errors = 0
        self.capped_entries = 0
        self.skipped = 0

    # ----------------------------------------------------------------------- helpers
    def _say(self, text: str) -> None:
        log.info(text)
        if self.notifier is not None:
            try:
                self.notifier.send(text)
            except Exception as e:  # notifications must never break trading
                log.warning("notify failed: %s", e)

    def equity(self) -> float:
        return float(self.broker.get_equity())

    def open_exposure(self) -> float:
        """Market value of every open position (the account-wide exposure cap uses this)."""
        return sum(
            t.qty * self.last_prices.get(t.symbol, t.entry_price) for t in self.open.values()
        )

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
        self.day_pnl = self.day_costs = 0.0
        self.day_trades = 0
        self.pending_entry.clear()  # an overnight signal is stale; strategies re-signal
        self._skips_logged.clear()
        self.journal.log("day_start", f"{day} equity={self.equity():.2f}", ts=ts)

    # ----------------------------------------------------------------------- main hooks
    def on_bar(self, symbol: str, bars: pd.DataFrame) -> None:
        bar = bars.iloc[-1]
        ts = bars.index[-1].to_pydatetime()
        market = sessions.market_from_tz(bars.index.tz)
        tf = int(bars.attrs.get("timeframe") or self.cfg.timeframe_minutes)
        bar_end = ts + timedelta(minutes=tf)
        day = sessions.session_date(market, ts)
        if day != self.risk.state.day:
            self._start_day(day, ts)

        price = float(bar["close"])
        self.broker.mark(symbol, price, ts)
        self.last_prices[symbol] = price

        mtc = sessions.minutes_to_close(market, bar_end)
        self._fill_pending(symbol, bar, ts, mtc)

        trade = self.open.get(symbol)
        if trade is not None:
            decision = RiskManager.check_exit(
                trade.side,
                trade.stop,
                trade.target,
                float(bar["open"]),
                float(bar["high"]),
                float(bar["low"]),
            )
            if decision is not None:
                self._exit(symbol, decision.reason, decision.price, ts)
            elif (
                self.decider is not None
                and self.decider.role == "primary"
                and not self.risk.state.halted
                and self.decider.due(symbol, ts)
            ):
                self._llm_manage_open(symbol, bars, ts, mtc=None)

        if not self.risk.state.halted and self.risk.daily_loss_breached(self.equity()):
            self.journal.log("halt", self.risk.state.halt_reason, ts=ts)
            self._say(
                f"DAILY LOSS STOP: {self.risk.state.halt_reason}. Flattening, done for today."
            )
            self._flatten("daily_loss", ts)

        if self.risk.should_flatten(mtc):
            same_market = [s for s in self.open if sessions.market_for(s) is market]
            if same_market:
                self._flatten("eod", ts, symbols=same_market)
            self.risk.block_entries(symbol)

        if symbol not in self.open and not self.risk.state.halted:
            self._maybe_enter(symbol, bars, ts, mtc)

        self.equity_curve.append((ts, self.equity()))
        if self.mode != "backtest":
            self.snapshot(ts)

    def snapshot(self, ts: datetime | None = None) -> dict[str, Any]:
        """Write the engine's current state to the journal (the dashboard reads it)."""
        s = self.risk.state
        open_trades = []
        for t in self.open.values():
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
        """Between bars (live/paper): check software stops against the latest price."""
        self.broker.mark(symbol, price, ts)
        self.last_prices[symbol] = price
        trade = self.open.get(symbol)
        if trade is None:
            return
        decision = RiskManager.check_exit(trade.side, trade.stop, trade.target, price, price, price)
        if decision is not None:
            self._exit(symbol, decision.reason, decision.price, ts)

    def end_of_data(self, ts: datetime | None = None) -> None:
        if self.open:
            last = ts or max(t.entry_ts for t in self.open.values())
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

    def _log_skip(self, symbol: str, why: str, sig: Signal | None, ts: datetime) -> None:
        """Counts every skip, but journals each (symbol, day, reason) only once so a filter
        that blocks all day cannot flood the journal with thousands of identical rows."""
        self.skipped += 1
        key = (symbol, self.risk.state.day, why)
        if key in self._skips_logged:
            return
        self._skips_logged.add(key)
        data = {"signal": asdict(sig)} if sig is not None else None
        self.journal.log("skip", why, symbol=symbol, data=data, ts=ts)

    def _try_enter(self, symbol: str, sig: Signal, ts: datetime, mtc: float) -> bool:
        sig = self._cap_stop(symbol, sig, ts)
        ok, why = self.risk.can_enter(symbol, sig.side, len(self.open), mtc)
        if not ok:
            self._log_skip(symbol, why, sig, ts)
            return False
        qty, why = self.risk.position_size(
            self.equity(),
            self.broker.get_cash(),
            sig.entry,
            sig.stop,
            lambda q: self.broker.round_qty_for(symbol, q),
            symbol=symbol,
            open_exposure=self.open_exposure(),
        )
        if qty <= 0:
            self._log_skip(symbol, why, sig, ts)
            return False
        if "capped" in why:
            self.capped_entries += 1
        self._enter(symbol, sig, qty, ts)
        return symbol in self.open

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
        sig = self._strategy_signal(symbol, bars, ts)
        if self.decider is None:
            if sig is None:
                return
            if getattr(self.broker, "delayed_entry", False):
                # Simulated fills happen at the NEXT bar's open. Pre-check entry rules now so
                # an impossible signal (near the close, halted, position open) is reported at
                # signal time instead of silently queueing.
                ok, why = self.risk.can_enter(symbol, sig.side, len(self.open), mtc)
                if not ok:
                    self._log_skip(symbol, why, sig, ts)
                    return
                self.pending_entry[symbol] = (sig, ts)
                self.journal.log(
                    "signal_queued",
                    f"{sig.reason} (fills at the next bar's open)",
                    symbol=symbol,
                    data=asdict(sig),
                    ts=ts,
                )
                return
            self._try_enter(symbol, sig, ts, mtc)
            return
        self._llm_entry(symbol, bars, ts, mtc, sig)

    def _fill_pending(self, symbol: str, bar, ts: datetime, mtc: float) -> None:
        """Entry queued on the previous bar fills at THIS bar's open (simulated brokers)."""
        pending = self.pending_entry.pop(symbol, None)
        if pending is None:
            return
        sig, _sig_ts = pending
        if self.risk.state.halted:
            self._log_skip(symbol, "halted before the queued entry could fill", sig, ts)
            return
        open_px = float(bar["open"])
        if (sig.side == "long" and open_px <= sig.stop) or (
            sig.side == "short" and open_px >= sig.stop
        ):
            self._log_skip(symbol, "gapped through the stop before entry; abandoned", sig, ts)
            return
        if sig.target is not None and (
            (sig.side == "long" and open_px >= sig.target)
            or (sig.side == "short" and open_px <= sig.target)
        ):
            self._log_skip(symbol, "gapped through the target before entry; abandoned", sig, ts)
            return
        filled_sig = Signal(sig.side, open_px, sig.stop, sig.target, reason=sig.reason)
        self.broker.set_exec_price(symbol, open_px)
        self._try_enter(symbol, filled_sig, ts, mtc)

    # ----------------------------------------------------------------------- LLM decider
    def _decision_context(self, symbol: str, bars: pd.DataFrame, sig: Signal | None, mtc: float):
        cm = getattr(self.risk, "cost_model", None)
        rt = cm.round_trip_cost_pct(fx=self.risk._fx_for(symbol)) if cm is not None else 0.0
        return build_context(
            symbol,
            bars,
            self.equity(),
            self.broker.get_cash(),
            self.open.get(symbol),
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
        ok, _why = self.risk.can_enter(symbol, "long", len(self.open), mtc)
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
        trade = self.open.get(symbol)
        if trade is None:
            return
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
            self._exit(symbol, "llm_close", self.last_prices.get(symbol, trade.entry_price), ts)

    def _enter(self, symbol: str, sig: Signal, qty: float, ts: datetime) -> None:
        side = "buy" if sig.side == "long" else "sell"
        self.journal.log("signal", sig.reason, symbol=symbol, data=asdict(sig), ts=ts)
        orders = self.cfg.orders
        try:
            if orders.entry_type == "limit" and getattr(self.broker, "supports_limit", False):
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
        self.open[symbol] = OpenTrade(
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
        )
        self.risk.record_entry(symbol)
        if getattr(self.broker, "supports_protective_stop", False):
            try:
                # resting stop-limit at the exchange: survives a crashed process
                self.broker.place_protective_stop(
                    symbol, qty, sig.stop, side="sell" if sig.side == "long" else "buy"
                )
            except BrokerError as e:
                self.journal.log(
                    "error", f"protective stop not placed: {e}", symbol=symbol, ts=ts
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
    def _exit(self, symbol: str, reason: str, ref_price: float, ts: datetime) -> None:
        trade = self.open.get(symbol)
        if trade is None:
            return
        side = "sell" if trade.side == "long" else "buy"
        if reason == "stop" and getattr(self.broker, "delayed_entry", False):
            # software stops fill late in fast moves; push the simulated fill through the stop
            cm = getattr(self.risk, "cost_model", None)
            if cm is not None:
                ref_price = cm.stop_exit_reference(ref_price, closing_long=trade.side == "long")
        order = None
        if getattr(self.broker, "supports_protective_stop", False):
            try:
                # if the resting stop-limit already executed at the exchange, use that fill
                order = self.broker.cancel_protective(symbol)
            except BrokerError as e:
                self.journal.log(
                    "error", f"protective stop cancel failed: {e}", symbol=symbol, ts=ts
                )
        if order is not None and order.filled_qty > 0:
            reason = "stop_resting"
        else:
            self.broker.set_exec_price(symbol, ref_price)
            try:
                order = self.broker.place_order(symbol, side, trade.qty)
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
        )
        self.trades.append(rec)
        self.journal.record_trade(rec.to_dict(), strategy=self.strategy_name)
        self.journal.log(
            "exit",
            f"{reason}: {trade.side} {trade.qty} {symbol} @ {exit_price:.4f}"
            f" pnl {pnl:+.2f} ({r_multiple:+.2f}R)",
            symbol=symbol,
            data={"pnl": pnl, "r": r_multiple, "costs": rec.costs},
            ts=ts,
        )
        self.day_pnl += pnl
        self.day_costs += rec.costs
        self.day_trades += 1
        del self.open[symbol]
        if self.mode != "backtest":
            self._say(f"EXIT ({reason}) {symbol} @ {exit_price:.2f} pnl {pnl:+.2f}")

    def _flatten(self, reason: str, ts: datetime, symbols: list[str] | None = None) -> None:
        for symbol in list(symbols or self.open):
            if symbol in self.open:
                self._exit(
                    symbol, reason, self.last_prices.get(symbol, self.open[symbol].entry_price), ts
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
