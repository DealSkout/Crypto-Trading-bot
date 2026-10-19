"""The long-running paper/live loop, 24/7. Polls bars, feeds completed bars to the Engine,
checks software stops between bars, and when the UTC day rolls over runs the end-of-day jobs
(daily summary, guardian review, trade coach) for the day that just closed. On any unexpected
error in live mode it cancels everything, flattens, alerts and stops."""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from cryptobot import sessions
from cryptobot.brokers.base import BrokerError
from cryptobot.config import Config
from cryptobot.engine import Engine, EngineError
from cryptobot.journal import Journal
from cryptobot.notify import Notifier

log = logging.getLogger(__name__)

HISTORY_SESSIONS = 25


class Runner:
    def __init__(
        self,
        cfg: Config,
        mode: str,
        engine: Engine,
        fetch_bars: Callable[[str], pd.DataFrame],
        journal: Journal,
        notifier: Notifier | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        flatten_on_exit: bool = True,
        guardian=None,
        research=None,
        regimes=None,
        registry=None,
        shadows=None,
    ) -> None:
        self.cfg = cfg
        self.guardian = guardian
        self.research = research
        self.regimes = regimes
        self.registry = registry
        self.shadows = shadows
        # automatic pauses of new entries (risk.pause_on_macro / pause_on_stale_data)
        self.stale_prices: dict[str, str] = {}
        self.active_pauses: dict[str, str] = {}
        self.calendar = None
        if cfg.risk.pause_on_macro:
            if research is not None:
                self.calendar = research.calendar
            else:
                from cryptobot.research.calendar import MacroCalendar

                self.calendar = MacroCalendar.load(cfg.research.calendar_file)
        engine.pauses.append(self.pause_reason)
        self.mode = mode
        self.engine = engine
        self.fetch_bars = fetch_bars
        self.journal = journal
        self.notifier = notifier
        self.clock = clock or (lambda: datetime.now(UTC))
        self.sleep = sleep
        self.flatten_on_exit = flatten_on_exit
        self.stop_requested = False
        self.last_bar: dict[str, pd.Timestamp] = {}
        self.summarised: date | None = None
        self.seen_day: date | None = None
        self.consecutive_errors = 0
        self.steps = 0

    # ------------------------------------------------------------------ lifecycle
    def request_stop(self, *_args) -> None:
        self.stop_requested = True

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.request_stop)
            except ValueError:  # not in the main thread
                pass

    def say(self, text: str) -> None:
        log.info(text)
        if self.notifier is not None:
            self.notifier.send(text)

    def guardian_stop(self, reason: str) -> None:
        """Flatten everything and leave the loop; the flag keeps us from restarting."""
        self.say(
            f"GUARDIAN STOP: {reason}. Flattening and exiting; run `python -m cryptobot resume`"
            " after reviewing."
        )
        try:
            self.engine._flatten("guardian", self.clock())
        finally:
            self.stop_requested = True

    def run_forever(self, max_steps: int | None = None) -> None:
        if self.guardian is not None and self.guardian.stopped:
            info = self.guardian.stop_info() or {}
            self.say(
                f"Cryptobot refuses to start: stopped by the guardian ({info.get('source')}) on"
                f" {str(info.get('ts', ''))[:16]}: {info.get('reason')}. Run"
                " `python -m cryptobot resume` after reviewing."
            )
            return
        self.reconcile_on_start()
        if self.guardian is not None:  # the starting equity is the first peak
            self.guardian.check_equity(self.engine.equity(), self.clock())
        self.say(
            f"Cryptobot {self.mode} started: {self.cfg.strategy} {self.engine.params} on"
            f" {self.cfg.symbols} ({self.cfg.timeframe}), broker {self.engine.broker.name}"
        )
        try:
            while not self.stop_requested and (max_steps is None or self.steps < max_steps):
                self.steps += 1
                try:
                    active = self.step()
                    self.consecutive_errors = 0
                    if self.guardian is not None and not self.stop_requested:
                        if self.guardian.stopped:
                            info = self.guardian.stop_info() or {}
                            self.guardian_stop(info.get("reason", ""))
                        else:
                            reason = self.guardian.check_equity(self.engine.equity(), self.clock())
                            if reason:
                                self.guardian_stop(reason)
                    if self.stop_requested:
                        break
                except (EngineError, BrokerError, Exception) as e:
                    self.consecutive_errors += 1
                    self.journal.log("error", repr(e))
                    log.exception("loop error")
                    if self.mode == "live":
                        self.engine.emergency_stop(repr(e))
                        raise
                    wait = min(300.0, 5.0 * 2**self.consecutive_errors)
                    self.say(f"paper loop error ({e!r}); retrying in {wait:.0f}s")
                    self.sleep(wait)
                    continue
                self.sleep(self.cfg.poll_seconds if active else 60.0)
        finally:
            self.shutdown()

    # ------------------------------------------------------------------ pauses
    def pause_reason(self, symbol: str, ts: datetime) -> str | None:
        """Why new entries are paused for `symbol` right now, or None."""
        when = self.clock()
        if self.calendar is not None:
            ev = self.calendar.active(when, symbol)
            if ev is not None:
                start, end = ev.window
                return f"macro window {ev.name} ({start:%H:%M}-{end:%H:%M} UTC)"
        if self.cfg.risk.pause_on_stale_data:
            if symbol in self.stale_prices:
                return self.stale_prices[symbol]
            if self.research is not None and self.cfg.research.required_sources:
                stale = self.research.stale_sources(when)
                if stale:
                    return f"research feed stale: {', '.join(stale)}"
        return None

    def track_pauses(self, now: datetime) -> None:
        """Journal and announce each pause when it starts and ends (risk events)."""
        for symbol in self.cfg.symbols:
            reason = self.pause_reason(symbol, now)
            before = self.active_pauses.get(symbol)
            if reason and not before:
                self.active_pauses[symbol] = reason
                self.journal.log("pause_start", reason, symbol=symbol, ts=now)
                self.say(f"PAUSE {symbol}: no new entries ({reason})")
            elif before and not reason:
                del self.active_pauses[symbol]
                self.journal.log("pause_end", before, symbol=symbol, ts=now)
                self.say(f"RESUMED {symbol}: entries allowed again (was: {before})")
            elif reason and before != reason:
                self.active_pauses[symbol] = reason

    def check_stale(self, symbol: str, completed: pd.DataFrame, now: datetime) -> None:
        tf = timedelta(minutes=self.cfg.timeframe_minutes)
        limit = tf * self.cfg.risk.stale_bars
        last_end = completed.index[-1].to_pydatetime() + tf if len(completed) else None
        if last_end is None or now - last_end > limit:
            age = "no bars" if last_end is None else f"last bar ended {last_end:%H:%M} UTC"
            self.stale_prices[symbol] = f"price data stale ({age})"
        else:
            self.stale_prices.pop(symbol, None)

    def reconcile_on_start(self) -> None:
        """The bot day-trades and always starts flat. A position or order already at the
        broker means the last run died without flattening (a backup stop may still be resting
        there): cancel and close it all, and say so."""
        broker = self.engine.broker
        if broker.name == "paper":
            return
        try:
            positions, orders = broker.get_positions(), broker.open_orders()
        except BrokerError as e:
            self.journal.log("error", f"startup reconcile could not read the account: {e}")
            return
        if not positions and not orders:
            return
        held = ", ".join(f"{p.symbol} {p.qty:g}" for p in positions) or "none"
        try:
            broker.flatten_all()
        except BrokerError as e:
            self.journal.log("error", f"startup flatten failed: {e}")
            if self.mode == "live":
                raise
        self.journal.log(
            "startup_flatten",
            f"left over from the last run: positions {held}; {len(orders)} open order(s)",
        )
        self.say(
            f"Startup: found positions ({held}) and {len(orders)} open order(s) left over from"
            " the last run. Cancelled and flattened them; the bot starts flat."
        )

    def shutdown(self) -> None:
        if self.research is not None:
            self.research.stop()
        if self.flatten_on_exit and self.engine.open:
            try:
                self.engine._flatten("shutdown", self.clock())
            except Exception as e:  # pragma: no cover - last resort
                log.exception("flatten on shutdown failed: %s", e)
        try:
            self.engine.broker.close()
        except Exception:  # pragma: no cover
            pass
        self.say(f"Cryptobot {self.mode} stopped. {self.engine.daily_summary()}")

    # ------------------------------------------------------------------ one iteration
    def step(self) -> bool:
        """Process every symbol once. Returns True if any market is in session."""
        now = self.clock()
        tf = timedelta(minutes=self.cfg.timeframe_minutes)
        active = False
        for symbol in self.cfg.symbols:
            market = sessions.market_for(symbol)
            if not sessions.is_in_session(market, now):
                continue
            active = True
            bars = self.fetch_bars(symbol)
            completed = bars[(bars.index + tf) <= now] if not bars.empty else bars
            self.check_stale(symbol, completed, now)
            if completed.empty:
                continue
            last = completed.index[-1]
            if self.last_bar.get(symbol) != last:
                self.last_bar[symbol] = last
                n = HISTORY_SESSIONS * (market.session_minutes // self.cfg.timeframe_minutes)
                window = completed.tail(n)
                window.attrs = dict(bars.attrs)
                if self.regimes is not None:  # breadth needs every market's completed days
                    self.regimes.update_from_bars(symbol, window)
                self.engine.on_bar(symbol, window)
                if self.shadows is not None:
                    self.shadows.on_bar(symbol, window)
            else:
                price = float(bars["close"].iloc[-1])
                self.engine.on_tick(symbol, price, now)
                if self.shadows is not None:
                    self.shadows.on_tick(symbol, price, now)
        self.check_day_rollover(now)
        self.track_pauses(now)
        try:
            self.journal.set_state(
                "heartbeat",
                {"ts": now.isoformat(), "active": active, "steps": self.steps, "mode": self.mode},
                ts=now,
            )
        except Exception as e:  # never let bookkeeping stop the loop
            log.warning("heartbeat write failed: %s", e)
        return active

    def check_day_rollover(self, now: datetime) -> None:
        """When the engine starts a new UTC day, close out the previous one. The engine has
        already flattened before midnight and written the previous day's row."""
        day = self.engine.risk.state.day
        if day is None:
            return
        if self.seen_day is None:
            self.seen_day = day
            return
        if day != self.seen_day:
            closed, self.seen_day = self.seen_day, day
            self.end_of_day(closed, now)

    def end_of_day(self, day: date, now: datetime) -> None:
        if self.summarised == day:
            return
        self.summarised = day
        self.say(f"Day {day} (UTC) closed. Equity {self.engine.equity():,.2f}.")
        if self.guardian is not None and not self.guardian.stopped:
            outcome = self.guardian.daily_review(now)
            if outcome.get("stopped"):
                self.guardian_stop((self.guardian.stop_info() or {}).get("reason", ""))
        self.run_trade_coach(day)
        self.run_digest(day, now)
        self.run_shadows(now)
        self.apply_champion()
        self.export_store()
        self.daily_report(day)

    def daily_report(self, day) -> None:
        """The day's report: file, Telegram, email. Never allowed to break the loop."""
        if self.mode == "backtest":
            return
        try:
            from cryptobot.daily_report import build_report, write_and_send

            rep = build_report(
                self.cfg, self.engine.journal, day, self.registry, self.research, self.regimes
            )
            write_and_send(self.cfg, rep, self.notifier)
        except Exception as e:  # pragma: no cover - reporting must never stop trading
            log.warning("daily report failed: %s", e)

    def run_digest(self, day, now: datetime) -> dict | None:
        """The day's research digest (at most one LLM call). Never allowed to break the loop."""
        if self.research is None or self.mode == "backtest":
            return None
        try:
            from cryptobot.research.digest import make_digest_llm, run_daily_digest

            llm = make_digest_llm(self.cfg) if self.cfg.research.digest else None
            regimes = self.regimes.current(now.date()) if self.regimes is not None else None
            return run_daily_digest(
                self.cfg,
                self.engine.journal,
                self.research,
                day,
                llm=llm,
                now=now,
                regimes=regimes,
            )
        except Exception as e:  # pragma: no cover - research must never stop trading
            log.warning("research digest failed: %s", e)
            return None

    def run_shadows(self, now: datetime) -> None:
        """Judge shadow runs that have run long enough (paper promotion only)."""
        if self.shadows is None:
            return
        try:
            for version, verdict, reason in self.shadows.evaluate(now):
                if verdict != "wait":
                    word = "PROMOTED to paper champion" if verdict == "promote" else "rejected"
                    self.say(f"Shadow {version} {word}: {reason}")
            self.shadows.sync()
        except Exception as e:  # pragma: no cover - learning must never stop trading
            log.warning("shadow evaluation failed: %s", e)

    def apply_champion(self) -> None:
        """Paper only: if the registry's champion changed (shadow promotion or a human
        rollback), trade it from now on. Only between trades."""
        if self.registry is None or self.mode != "paper" or self.engine.open:
            return
        try:
            from cryptobot.strategies import load_strategy

            champ = self.registry.champion()
            if champ is None or champ.id == self.engine.version:
                return
            self.engine.set_model(
                load_strategy(champ.strategy), champ.params, champ.filters, champ.id
            )
            self.say(f"Paper bot now trades {champ.describe()} ({champ.reason})")
        except Exception as e:  # pragma: no cover
            log.warning("champion switch failed: %s", e)

    def export_store(self) -> None:
        """Refresh the Parquet copy of the feature store. Never allowed to break the loop."""
        try:
            from cryptobot.store import export_store

            export_store(self.engine.journal, self.cfg.store_dir)
        except Exception as e:  # pragma: no cover - bookkeeping only
            log.warning("feature store export failed: %s", e)

    def run_trade_coach(self, day) -> None:
        """The daily trade review, after the close. Never allowed to break the loop."""
        if not self.cfg.coach.enabled or self.mode == "backtest":
            return
        try:
            from cryptobot.guardian import backtest_metrics_for
            from cryptobot.trade_coach import make_review_llm, run_daily_review

            out = run_daily_review(
                self.cfg,
                self.engine.journal,
                day,
                llm=make_review_llm(self.cfg),
                backtest=backtest_metrics_for(self.cfg),
            )
            if out.get("status") == "done":
                self.say(f"Coach: {out['daily_summary']}")
        except Exception as e:  # pragma: no cover - the coach must never stop trading
            log.warning("daily coach failed: %s", e)
