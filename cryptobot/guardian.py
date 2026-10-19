"""The guardian: a watchdog that can stop the bot COMPLETELY.

Two layers, both of which may only ever pull the plug, never loosen a limit or restart trading:

  * deterministic rules (always on): drawdown from peak equity, net loss over the last 30
    days, consecutive losing days, repeated daily-loss halts, and paper results far outside
    the backtest's expected range
  * an optional AI reviewer (Claude or a local model) that reads the same numbers after the
    close and may order a stop when it judges the bot to be doing badly or behaving oddly

A stop flattens every position, writes a STOPPED flag file next to the journal, alerts you,
and makes the trading loop exit. The flag survives restarts: `paper` and `live` refuse to run
while it exists, and only `python -m cryptobot resume` (a human) removes it.

This module never imports the brokers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from cryptobot.config import Config, GuardianConfig
from cryptobot.journal import Journal

log = logging.getLogger(__name__)

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["continue", "stop"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reason", "notes"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are the independent risk guardian for Cryptobot, a small crypto day-
trading bot (Alpaca, USD pairs such as BTC/USD, long-only, no margin; its trading day is the
UTC day and it flattens before midnight UTC). After each day you receive its recent results and
the backtest it was approved on. Decide ONE thing: may it keep trading tomorrow, or must it be
STOPPED COMPLETELY (every position flattened, no trading until a human reviews it)?

You cannot loosen any limit, change any setting, or restart a stopped bot. Your only power is
to stop. Stop when losses are accelerating, when results are clearly outside the backtest's
expected range, when costs dominate the result, when the bot behaves erratically (many
rejected orders, errors, repeated daily halts), or when anything looks like a bug. When the
sample is tiny and nothing is alarming, continue. Be concrete: cite the numbers. Keep reason
under 40 words."""


@dataclass
class Verdict:
    verdict: str
    confidence: float
    reason: str
    notes: str = ""
    model: str = ""
    cost_usd: float = 0.0


class Guardian:
    def __init__(
        self,
        cfg: Config,
        journal: Journal,
        llm=None,
        notifier=None,
        backtest_metrics: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.g: GuardianConfig = cfg.guardian
        self.journal = journal
        self.llm = llm
        self.notifier = notifier
        self.backtest = backtest_metrics or {}
        self.stop_file = Path(self.g.stop_file)

    # ------------------------------------------------------------------ the flag
    @property
    def stopped(self) -> bool:
        return self.stop_file.exists()

    def stop_info(self) -> dict[str, Any] | None:
        if not self.stopped:
            return None
        try:
            return json.loads(self.stop_file.read_text())
        except (OSError, json.JSONDecodeError):
            return {"reason": "stopped (unreadable flag file)", "source": "unknown"}

    def stop(self, reason: str, source: str, now: datetime | None = None) -> None:
        """Pull the plug. Idempotent; the first reason wins."""
        now = now or datetime.now(UTC)
        if self.stopped:
            return
        info = {
            "reason": reason,
            "source": source,
            "ts": now.isoformat(),
            "resume": "python -m cryptobot resume",
        }
        self.stop_file.parent.mkdir(parents=True, exist_ok=True)
        self.stop_file.write_text(json.dumps(info, indent=2))
        self.journal.set_state("stopped", info, ts=now)
        self.journal.log("guardian_stop", f"{source}: {reason}", ts=now)
        log.error("GUARDIAN STOP (%s): %s", source, reason)
        if self.notifier is not None:
            try:
                self.notifier.send(
                    f"🛑 Cryptobot STOPPED by the guardian ({source}): {reason}\n"
                    "All positions flattened. It will not trade again until you run "
                    "`python -m cryptobot resume`."
                )
            except Exception as e:  # pragma: no cover - alerts must not break the stop
                log.warning("notify failed: %s", e)

    def resume(self, who: str = "human") -> bool:
        """Only a human calls this. Returns True if a flag was removed."""
        if not self.stopped:
            return False
        info = self.stop_info() or {}
        self.stop_file.unlink(missing_ok=True)
        self.journal.set_state("stopped", None)
        self.journal.log("guardian_resume", f"resumed by {who} after: {info.get('reason', '')}")
        return True

    # ------------------------------------------------------------------ live checks
    def check_equity(self, equity: float, now: datetime | None = None) -> str | None:
        """Called on every bar: drawdown from peak equity. Returns the stop reason if it fires."""
        now = now or datetime.now(UTC)
        stored = self.journal.get_state("peak_equity")
        if stored is None:
            peak = float(equity)
            self.journal.set_state("peak_equity", peak, ts=now)
        else:
            peak = float(stored)
        if equity > peak:
            peak = float(equity)
            self.journal.set_state("peak_equity", peak, ts=now)
        if peak > 0:
            dd = (equity / peak - 1.0) * 100.0
            if dd <= -self.g.max_drawdown_pct:
                reason = (
                    f"drawdown {dd:.2f}% from peak equity {peak:.2f} exceeds"
                    f" {self.g.max_drawdown_pct}%"
                )
                self.stop(reason, "rule:max_drawdown_pct", now)
                return reason
        return None

    def rule_reasons(self, now: datetime | None = None) -> list[str]:
        """The deterministic daily checks. Empty list means nothing fired."""
        now = now or datetime.now(UTC)
        reasons: list[str] = []
        daily = self.journal.daily(since=(now - timedelta(days=30)).date())
        if len(daily):
            start = float(daily["start_equity"].iloc[0])
            pnl = float(daily["pnl"].sum())
            if start > 0 and pnl / start * 100.0 <= -self.g.max_loss_30d_pct:
                reasons.append(
                    f"net loss {pnl:+.2f} ({pnl / start * 100:.2f}%) over the last"
                    f" 30 days exceeds {self.g.max_loss_30d_pct}%"
                )
            losing = 0
            for v in reversed(daily["pnl"].tolist()):
                if v < 0:
                    losing += 1
                else:
                    break
            if losing >= self.g.max_consecutive_losing_days:
                reasons.append(
                    f"{losing} consecutive losing days (limit {self.g.max_consecutive_losing_days})"
                )
            halts = int(daily["halted"].sum())
            if halts >= self.g.max_halts_30d:
                reasons.append(
                    f"{halts} daily-loss halts in 30 days (limit {self.g.max_halts_30d})"
                )
        trades = self.journal.trades(mode=self.journal.mode)
        if len(trades) >= self.g.min_trades_for_judgement and self.backtest:
            expected = self.backtest.get("avg_r")
            if expected is not None:
                actual = float(trades["r_multiple"].astype(float).tail(50).mean())
                if actual < float(expected) - self.g.expected_range_r:
                    reasons.append(
                        f"average R {actual:.2f} over the last {min(50, len(trades))}"
                        f" trades is more than {self.g.expected_range_r}R below the"
                        f" backtest's {float(expected):.2f}"
                    )
        errors = self.journal.events(
            since=now - timedelta(days=1), kinds=["error", "order_rejected", "emergency_stop"]
        )
        if len(errors) >= self.g.max_errors_per_day:
            reasons.append(
                f"{len(errors)} errors or rejected orders in the last day (limit"
                f" {self.g.max_errors_per_day})"
            )
        return reasons

    # ------------------------------------------------------------------ the AI reviewer
    def build_context(self, now: datetime | None = None) -> str:
        now = now or datetime.now(UTC)
        daily = self.journal.daily(since=(now - timedelta(days=30)).date())
        trades = self.journal.trades(mode=self.journal.mode)
        lines = [
            f"Date {now.date()}; mode {self.journal.mode}; strategy {self.cfg.strategy}"
            f" {self.cfg.params}; symbols {self.cfg.symbols}",
            f"Risk limits (fixed): {self.cfg.risk.__dict__}",
            f"Guardian rules (fixed): drawdown {self.g.max_drawdown_pct}%, 30-day loss"
            f" {self.g.max_loss_30d_pct}%, {self.g.max_consecutive_losing_days} losing days,"
            f" {self.g.max_halts_30d} halts",
            "",
        ]
        if self.backtest:
            lines.append(
                "Backtest it was approved on (out of sample, after costs): "
                + json.dumps(
                    {
                        k: round(v, 3) if isinstance(v, float) else v
                        for k, v in self.backtest.items()
                    }
                )
            )
        peak = self.journal.get_state("peak_equity")
        lines.append(f"Peak equity: {peak}")
        lines.append("")
        lines.append(
            "Daily results, last 30 days (date, start equity, end equity, pnl, trades,"
            " costs, halted):"
        )
        if len(daily):
            for _, d in daily.iterrows():
                lines.append(
                    f"  {d['date']} {float(d['start_equity']):.2f} {float(d['end_equity']):.2f}"
                    f" {float(d['pnl']):+.2f} {int(d['trades'])} {float(d['costs']):.2f}"
                    f"{' HALTED' if int(d['halted']) else ''}"
                )
        else:
            lines.append("  none yet")
        lines.append("")
        if len(trades):
            pnl = trades["pnl"].astype(float)
            lines.append(
                f"Trades: {len(trades)} total, win rate {(pnl > 0).mean() * 100:.0f}%,"
                f" avg R {trades['r_multiple'].astype(float).mean():+.2f}, total pnl"
                f" {pnl.sum():+.2f}, costs {trades['costs'].astype(float).sum():.2f}"
            )
            lines.append("Last 20 trades (entry, symbol, side, exit reason, pnl, R):")
            for _, t in trades.tail(20).iterrows():
                lines.append(
                    f"  {str(t['entry_ts'])[:16]} {t['symbol']} {t['side']} {t['exit_reason']}"
                    f" {float(t['pnl']):+.2f} {float(t['r_multiple']):+.2f}R"
                )
        else:
            lines.append("Trades: none yet")
        events = self.journal.events(
            since=now - timedelta(days=3),
            kinds=["error", "order_rejected", "halt", "skip", "decider_error", "emergency_stop"],
        )
        if len(events):
            counts = events["message"].value_counts().head(8)
            lines += ["", "Recent problems (count: message):"]
            lines += [f"  {int(n)}: {m}" for m, n in counts.items()]
        return "\n".join(lines)

    def ai_review(self, now: datetime | None = None) -> Verdict | None:
        """Ask the model. None means no opinion today (not configured, error, refusal)."""
        if self.llm is None:
            return None
        now = now or datetime.now(UTC)
        spent = self.journal.llm_spend_today("guardian", now.date())
        if spent >= self.g.max_daily_cost_usd:
            self.journal.log("guardian", "AI review skipped: daily cost cap reached", ts=now)
            return None
        try:
            res = self.llm.complete(SYSTEM_PROMPT, self.build_context(now), VERDICT_SCHEMA)
        except Exception as e:  # the rules still stand; the AI just has no say today
            self.journal.log("guardian_error", repr(e), ts=now)
            log.warning("guardian AI review failed: %r", e)
            return None
        self.journal.add_llm_spend("guardian", res.cost_usd, now.date())
        if res.refused:
            self.journal.log("guardian", "AI reviewer declined; no opinion today", ts=now)
            return None
        data = res.data or {}
        try:
            conf = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
        verdict = str(data.get("verdict", "continue")).lower()
        v = Verdict(
            verdict if verdict in ("continue", "stop") else "continue",
            conf,
            str(data.get("reason", ""))[:300],
            str(data.get("notes", ""))[:1000],
            res.model,
            res.cost_usd,
        )
        self.journal.log(
            "guardian",
            f"AI verdict {v.verdict} (confidence {v.confidence:.2f}): {v.reason}",
            data={"cost_usd": v.cost_usd},
            ts=now,
        )
        return v

    # ------------------------------------------------------------------ the daily review
    def daily_review(self, now: datetime | None = None) -> dict[str, Any]:
        """Run after the close. Deterministic rules first; then the AI reviewer, which may
        only add a stop. Returns what happened."""
        now = now or datetime.now(UTC)
        if self.stopped:
            return {"stopped": True, "already": True, "info": self.stop_info()}
        reasons = self.rule_reasons(now)
        if reasons:
            self.stop("; ".join(reasons), "rules", now)
            return {"stopped": True, "reasons": reasons, "verdict": None}
        v = self.ai_review(now)
        if v is not None and v.verdict == "stop" and v.confidence >= self.g.min_confidence:
            self.stop(f"AI reviewer ({v.model}): {v.reason}", "ai", now)
            return {"stopped": True, "reasons": [], "verdict": v}
        self.journal.log(
            "guardian",
            "review passed: nothing fired" + (f"; AI said continue ({v.reason})" if v else ""),
            ts=now,
        )
        return {"stopped": False, "reasons": [], "verdict": v}


def backtest_metrics_for(cfg: Config) -> dict[str, Any]:
    """The saved out-of-sample metrics the strategy was approved on, if any."""
    path = Path(cfg.reports_dir) / f"latest_{cfg.strategy}.json"
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text()).get("metrics", {}))
    except (OSError, json.JSONDecodeError):
        return {}


def make_guardian(cfg: Config, journal: Journal, notifier=None, llm=None) -> Guardian:
    """Build the guardian with the AI reviewer if configured and reachable."""
    if llm is None and cfg.guardian.provider != "none":
        from cryptobot.llm import LLMError, make_llm

        try:
            llm = make_llm(cfg, component="guardian")
        except LLMError as e:
            log.warning("guardian AI reviewer disabled: %s (the rules still apply)", e)
            llm = None
    return Guardian(
        cfg, journal, llm=llm, notifier=notifier, backtest_metrics=backtest_metrics_for(cfg)
    )


def stopped_info(cfg: Config) -> dict[str, Any] | None:
    """Cheap check for the CLI and dashboard without building a Guardian."""
    p = Path(cfg.guardian.stop_file)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {
            "reason": "stopped (unreadable flag file)",
            "source": "unknown",
            "ts": datetime.fromtimestamp(p.stat().st_mtime, tz=UTC).isoformat(),
        }


__all__ = ["Guardian", "Verdict", "backtest_metrics_for", "make_guardian", "stopped_info", "date"]
