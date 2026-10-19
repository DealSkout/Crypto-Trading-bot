"""The daily report: one page per UTC day, written after the day closes.

  1. P&L: the day, the last 7 and 30 days
  2. metrics over the last 30 days against the baseline (net return after fees, Sharpe,
     Sortino, max drawdown, win rate, profit factor, trades)
  3. today's regime per market
  4. the research digest
  5. champion vs challengers: the model trading, shadow runs and the last learning run
  6. risk events: halts, guardian stops, pauses, stale data, errors, rejected orders

Delivered to reports/daily/YYYY-MM-DD.md, Telegram (a short version) and email when SMTP is
configured in .env. Python only: no LLM call here.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
from datetime import UTC, date, datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cryptobot.backtest import PERIODS_PER_YEAR, sharpe_sortino
from cryptobot.config import Config
from cryptobot.journal import Journal

log = logging.getLogger(__name__)

RISK_EVENT_KINDS = [
    "halt",
    "guardian_stop",
    "guardian_resume",
    "pause_start",
    "pause_end",
    "emergency_stop",
    "startup_flatten",
    "order_rejected",
    "error",
    "backup_stop_failed",
]

METRIC_ROWS = [
    ("Net return after fees", "net_return_pct", "{:+.2f}%"),
    ("Sharpe (365-day)", "sharpe", "{:.2f}"),
    ("Sortino (365-day)", "sortino", "{:.2f}"),
    ("Max drawdown", "max_drawdown_pct", "{:.2f}%"),
    ("Win rate", "win_rate_pct", "{:.1f}%"),
    ("Profit factor", "profit_factor", "{:.2f}"),
    ("Trades", "n_trades", "{:d}"),
]


def _fmt(fmt: str, v) -> str:
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a" if v is None else ("inf" if v > 0 else "-inf")
        return fmt.format(int(v) if "d}" in fmt else float(v))
    except (TypeError, ValueError):
        return str(v)


def journal_metrics(journal: Journal, mode: str, since: date, until: date) -> dict[str, Any]:
    """Metrics from the journal's daily rows and trades in [since, until]."""
    daily = journal.daily(since=since)
    daily = daily[daily["date"] <= until.isoformat()] if len(daily) else daily
    trades = journal.trades(mode=mode)
    if len(trades):
        d = pd.to_datetime(trades["exit_ts"], utc=True, format="mixed").dt.date
        trades = trades[(d >= since) & (d <= until)]
    out: dict[str, Any] = {"days": int(len(daily)), "n_trades": int(len(trades))}
    if len(daily):
        start = float(daily["start_equity"].iloc[0])
        end = float(daily["end_equity"].iloc[-1])
        rets = (daily["end_equity"] / daily["start_equity"] - 1.0).astype(float)
        sharpe, sortino = sharpe_sortino(rets, PERIODS_PER_YEAR)
        curve = pd.concat([pd.Series([start]), daily["end_equity"].astype(float)])
        dd = float(((curve / curve.cummax()) - 1).min() * 100)
        out.update(
            net_return_pct=(end / start - 1) * 100 if start > 0 else 0.0,
            pnl=float(daily["pnl"].sum()),
            costs=float(daily["costs"].sum()),
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown_pct=dd,
            halts=int(daily["halted"].sum()),
            end_equity=end,
        )
    if len(trades):
        pnl = trades["pnl"].astype(float)
        wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        out.update(
            win_rate_pct=float((pnl > 0).mean() * 100),
            profit_factor=float(wins / losses) if losses > 0 else float("inf"),
            avg_r=float(trades["r_multiple"].astype(float).mean()),
        )
    return out


def load_baseline(cfg: Config) -> dict[str, Any] | None:
    path = Path(cfg.reports_dir) / "baseline.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def latest_learning_run(cfg: Config) -> dict[str, Any] | None:
    runs = sorted((Path(cfg.reports_dir) / "learning").glob("*.json"))
    if not runs:
        return None
    try:
        return json.loads(runs[-1].read_text())
    except (OSError, json.JSONDecodeError):
        return None


def shadow_standings(journal_path: str, registry) -> list[dict[str, Any]]:
    if registry is None:
        return []
    champ = registry.champion()
    trades = Journal(journal_path, mode="shadow").trades(mode="shadow")
    out = []
    for m in registry.shadows():
        since = m.shadow_start
        mine = trades[(trades["version"] == m.id) & (trades["entry_ts"] >= since)]
        mirror = (
            trades[(trades["version"] == champ.id) & (trades["entry_ts"] >= since)]
            if champ is not None
            else trades.iloc[0:0]
        )
        started = pd.Timestamp(since).to_pydatetime() if since else datetime.now(UTC)
        out.append(
            {
                "version": m.id,
                "model": m.describe(),
                "days": (datetime.now(UTC) - started).days,
                "trades": int(len(mine)),
                "pnl": float(mine["pnl"].sum()) if len(mine) else 0.0,
                "champion_pnl": float(mirror["pnl"].sum()) if len(mirror) else 0.0,
            }
        )
    return out


def build_report(
    cfg: Config,
    journal: Journal,
    day: date,
    registry=None,
    research=None,
    regimes=None,
) -> dict[str, Any]:
    mode = journal.mode
    rep: dict[str, Any] = {
        "day": str(day),
        "mode": mode,
        "generated": datetime.now(UTC).isoformat(),
    }
    daily = journal.daily()
    row = daily[daily["date"] == day.isoformat()] if len(daily) else daily
    rep["today"] = (
        {
            "pnl": float(row["pnl"].iloc[0]),
            "trades": int(row["trades"].iloc[0]),
            "costs": float(row["costs"].iloc[0]),
            "start_equity": float(row["start_equity"].iloc[0]),
            "end_equity": float(row["end_equity"].iloc[0]),
            "halted": bool(row["halted"].iloc[0]),
        }
        if len(row)
        else {"pnl": 0.0, "trades": 0, "costs": 0.0, "halted": False}
    )
    rep["last_7d"] = journal_metrics(journal, mode, day - timedelta(days=6), day)
    rep["last_30d"] = journal_metrics(journal, mode, day - timedelta(days=29), day)
    base = load_baseline(cfg)
    if registry is not None and registry.champion() is not None:
        champ = registry.champion()
        rep["champion"] = {"version": champ.id, "model": champ.describe(), "reason": champ.reason}
        holdout = champ.metrics.get("holdout")
        if holdout:
            rep["champion"]["holdout"] = holdout
        live = registry.live_version()
        rep["live_version"] = live.describe() if live is not None else None
    rep["baseline"] = base
    if regimes is not None:
        nxt = day + timedelta(days=1)
        rep["regimes"] = {s: regimes.classify(s, nxt)["regime"] for s in cfg.symbols}
    if research is not None:
        dg = research.store.series("digest", "daily")
        if len(dg):
            rec = json.loads(dg.iloc[-1]["data"])
            rep["digest"] = {"day": rec.get("day"), "source": rec.get("source"), **rec["digest"]}
        rep["research_staleness_min"] = {
            k: None if v is None else round(v) for k, v in research.staleness_minutes().items()
        }
        rep["calendar_days_of_cover"] = round(research.calendar.days_of_cover(datetime.now(UTC)))
        rep["macro_next_72h"] = [
            e.to_dict() for e in research.calendar.upcoming(datetime.now(UTC), hours=72)
        ]
    rep["shadows"] = shadow_standings(cfg.journal_path, registry) if registry else []
    rep["last_learning_run"] = latest_learning_run(cfg)
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    ev = journal.events(since=start, kinds=RISK_EVENT_KINDS, limit=500)
    ev = ev[ev["ts"] < (start + timedelta(days=1)).isoformat()] if len(ev) else ev
    rep["risk_events"] = [
        {"ts": r["ts"][:16], "kind": r["kind"], "symbol": r["symbol"], "message": r["message"]}
        for _, r in ev.iloc[::-1].iterrows()
    ]
    skips = journal.decisions(since=start, until=start + timedelta(days=1), actions=["skip"])
    rep["skips_by_reason"] = (
        skips["reason"].str.split(":").str[0].value_counts().to_dict() if len(skips) else {}
    )
    return rep


def _baseline_rows(rep: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    cols, data = ["Last 30 days (" + rep["mode"] + ")"], [rep["last_30d"]]
    base = rep.get("baseline") or {}
    for name, metrics in (base.get("backtests") or {}).items():
        cols.append(f"Baseline {name} (backtest)")
        data.append(metrics)
    holdout = (rep.get("champion") or {}).get("holdout")
    if holdout:
        cols.append("Champion holdout")
        data.append(
            {
                "net_return_pct": holdout.get("net_return_pct"),
                "sharpe": holdout.get("sharpe"),
                "max_drawdown_pct": holdout.get("max_drawdown_pct"),
                "n_trades": holdout.get("trades"),
            }
        )
    return cols, data


def render_markdown(rep: dict[str, Any]) -> str:
    t = rep["today"]
    out = [f"# Cryptobot daily report {rep['day']} (UTC, {rep['mode']})", ""]
    out += [
        "## P&L",
        "",
        f"- Today: {t['pnl']:+.2f} USD from {t['trades']} trade(s), costs {t['costs']:.2f}"
        + (" — daily loss halt hit" if t.get("halted") else ""),
        f"- Last 7 days: {rep['last_7d'].get('pnl', 0.0):+.2f} USD,"
        f" {rep['last_7d'].get('n_trades', 0)} trades",
        f"- Last 30 days: {rep['last_30d'].get('pnl', 0.0):+.2f} USD,"
        f" {rep['last_30d'].get('n_trades', 0)} trades",
    ]
    if "end_equity" in t:
        out.append(f"- Equity: {t['end_equity']:,.2f}")
    cols, data = _baseline_rows(rep)
    out += ["", "## Metrics vs baseline", "", "| Metric | " + " | ".join(cols) + " |"]
    out.append("|---|" + "---|" * len(cols))
    for label, key, fmt in METRIC_ROWS:
        out.append(f"| {label} | " + " | ".join(_fmt(fmt, d.get(key)) for d in data) + " |")
    if not rep.get("baseline"):
        out.append("\n_No baseline yet: run `python -m cryptobot baseline` once._")
    if rep.get("regimes"):
        out += ["", "## Regime (for the next UTC day)", ""]
        out += [f"- {s}: {r}" for s, r in rep["regimes"].items()]
    if rep.get("digest"):
        d = rep["digest"]
        out += [
            "",
            f"## Research digest ({d.get('source')})",
            "",
            f"Risk: **{d['risk_level']}**",
            "",
        ]
        out.append(d["summary"])
        if d.get("watch"):
            out += ["", "Watch: " + "; ".join(d["watch"])]
    if rep.get("macro_next_72h"):
        out += ["", "Macro windows in the next 72h (no new entries inside them):"]
        out += [
            f"- {e['name']}: {e['window_start'][:16]} to {e['window_end'][11:16]} UTC"
            for e in rep["macro_next_72h"]
        ]
    if rep.get("calendar_days_of_cover") is not None and rep["calendar_days_of_cover"] < 45:
        out.append(
            f"\n**Macro calendar has only {rep['calendar_days_of_cover']} days of events left:"
            " add the next dates to macro_calendar.yaml.**"
        )
    out += ["", "## Champion vs challengers", ""]
    champ = rep.get("champion")
    out.append(f"- Paper champion: {champ['model']} ({champ['reason']})" if champ else "- n/a")
    out.append(f"- Live version (human-approved): {rep.get('live_version') or 'none'}")
    for s in rep.get("shadows", []):
        out.append(
            f"- Shadow {s['version']} day {s['days']}: {s['trades']} trades, P&L {s['pnl']:+.2f}"
            f" vs champion mirror {s['champion_pnl']:+.2f}"
        )
    run = rep.get("last_learning_run")
    if run:
        verdict = "accepted" if run["accepted"] else "rejected"
        out.append(
            f"- Last learning run {run['ts'][:10]}: best {run.get('best', 'none')} {verdict}"
        )
        out += [f"  - {r}" for r in run.get("reasons", [])[:4]]
    out += ["", "## Risk events", ""]
    if rep["risk_events"]:
        out += [
            f"- {e['ts']} {e['kind']} {e['symbol'] or ''}: {e['message']}"
            for e in rep["risk_events"][:40]
        ]
    else:
        out.append("- none")
    if rep["skips_by_reason"]:
        out.append(
            "- Signals skipped: "
            + ", ".join(f"{k} x{v}" for k, v in sorted(rep["skips_by_reason"].items()))
        )
    stale = {
        k: v for k, v in (rep.get("research_staleness_min") or {}).items() if v is None or v > 720
    }
    if stale:
        out.append(f"- Research sources stale or never fetched: {sorted(stale)}")
    return "\n".join(out) + "\n"


def telegram_text(rep: dict[str, Any]) -> str:
    t = rep["today"]
    m30 = rep["last_30d"]
    lines = [
        f"Cryptobot {rep['day']} ({rep['mode']}): {t['pnl']:+.2f} USD, {t['trades']} trades",
        f"30d: {m30.get('net_return_pct', 0.0):+.2f}%, Sharpe {_fmt('{:.2f}', m30.get('sharpe'))},"
        f" max DD {_fmt('{:.2f}%', m30.get('max_drawdown_pct'))}",
    ]
    if rep.get("regimes"):
        lines.append(
            "Regimes: " + ", ".join(f"{s.split('/')[0]} {r}" for s, r in rep["regimes"].items())
        )
    if rep.get("digest"):
        lines.append(f"Digest risk: {rep['digest']['risk_level']}")
    if rep.get("shadows"):
        lines.append(
            "Shadow: " + ", ".join(f"{s['version']} day {s['days']}" for s in rep["shadows"])
        )
    n_risk = len([e for e in rep["risk_events"] if e["kind"] not in ("pause_end",)])
    if n_risk:
        lines.append(f"Risk events: {n_risk} (see the report)")
    return "\n".join(lines)


def send_email(subject: str, body: str) -> bool:
    """Email via SMTP settings in .env (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
    REPORT_EMAIL_TO, optional REPORT_EMAIL_FROM). Never raises."""
    host, to = os.environ.get("SMTP_HOST", ""), os.environ.get("REPORT_EMAIL_TO", "")
    if not host or not to:
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("REPORT_EMAIL_FROM") or os.environ.get("SMTP_USER", to)
    msg["To"] = to
    msg.set_content(body)
    try:
        with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", "587")), timeout=20) as s:
            s.starttls()
            user = os.environ.get("SMTP_USER")
            if user:
                s.login(user, os.environ.get("SMTP_PASSWORD", ""))
            s.send_message(msg)
        return True
    except (OSError, smtplib.SMTPException) as e:
        log.warning("report email failed: %s", e)
        return False


def write_and_send(
    cfg: Config,
    rep: dict[str, Any],
    notifier=None,
    email: bool = True,
) -> Path:
    out = Path(cfg.reports_dir) / "daily"
    out.mkdir(parents=True, exist_ok=True)
    text = render_markdown(rep)
    path = out / f"{rep['day']}.md"
    path.write_text(text)
    (out / f"{rep['day']}.json").write_text(json.dumps(rep, indent=2, default=str))
    if notifier is not None:
        try:
            notifier.send(telegram_text(rep))
        except Exception as e:  # pragma: no cover - notifier never raises, belt and braces
            log.warning("telegram report failed: %s", e)
    if email:
        send_email(f"Cryptobot daily report {rep['day']}", text)
    return path
