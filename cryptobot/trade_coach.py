"""The daily trade coach: a disciplined pro trader and risk manager reviewing every trade.

Runs once a day after the UTC day closes (the runner calls it; `python -m cryptobot review`
runs it by hand). Python computes every number first. Claude then gets ONE compact summary
per day (Haiku by default, max_tokens capped, cost logged) and returns, per trade, a verdict,
the mistake (if any) and the lesson, plus a daily summary and proposed changes.

Hard rules, enforced here in code whatever the model says:
  * proposals are only SAVED to coach/proposals/<date>.json; nothing is ever applied
  * no parameter proposals until coach.min_trades_for_params closed trades exist (30)
  * never a proposal touching the risk limits (risk per trade, daily loss cap, drawdown kill
    switch, position limits, stop cap) or the paper/live mode; such proposals are dropped
    and the reason is recorded
  * coach.enabled: false switches the whole coach off

Outputs: coach/reviews/<date>.json (stats + verdicts + cost), coach/proposals/<date>.json
(only when there are proposals) and a short entry appended to coach/journal.md.
Without an API key, or on any error or refusal, the review still runs on the Python stats
alone and says so. This module never imports cryptobot.brokers.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from cryptobot.config import RISK_KEYS, Config
from cryptobot.journal import Journal
from cryptobot.llm import LLMError

log = logging.getLogger(__name__)

COMPONENT = "coach_review"  # name used in the journal's llm_spend table
STATE_KEY = "coach_review_last_day"

VERDICTS = ["good", "acceptable", "mistake"]
CHANGE_TYPES = ["params", "filter", "process"]

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "trades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "trade_id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": VERDICTS},
                    "mistake": {"type": "string"},
                    "lesson": {"type": "string"},
                },
                "required": ["trade_id", "verdict", "mistake", "lesson"],
                "additionalProperties": False,
            },
        },
        "daily_summary": {"type": "string"},
        "proposed_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": CHANGE_TYPES},
                    "description": {"type": "string"},
                    "params": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string"},
                                "value": {"type": "string"},
                            },
                            "required": ["key", "value"],
                            "additionalProperties": False,
                        },
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["type", "description", "params", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["trades", "daily_summary", "proposed_changes"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are the trading coach for Cryptobot, a small rule-based crypto
day-trading bot (Alpaca USD pairs, 24/7, long-only, no margin; the trading day is the UTC day).
Think like a disciplined professional trader and risk manager reviewing the day that just ended.

You get one compact summary: today's closed trades with numbers already computed, today's
stats, rolling stats and what the backtest expects. Judge each trade on PROCESS, not outcome:
a losing trade that followed the plan with sensible risk is "good" or "acceptable"; a winning
trade that broke the plan is a "mistake". Typical mistakes: entering against the setup, a
stop too tight for the volatility, costs eating most of the edge, exiting early or late,
overtrading, ignoring the day's conditions.

For every trade give: verdict (good | acceptable | mistake), mistake (one short sentence, or
"" if none) and lesson (one short, concrete sentence). Then a daily_summary of 2-4 sentences
and at most 3 proposed_changes.

Rules for proposals (they are only saved for a human; nothing is applied):
- Never propose changing risk limits: risk per trade, the daily loss cap, the drawdown kill
  switch, max positions, the stop-distance cap, position sizing or leverage.
- Never propose leaving paper trading or going live.
- type "params" only when the summary says parameter proposals are allowed, and only for the
  strategy parameters listed. Give values as strings.
- type "filter" (a condition to skip trades) or "process" (how the bot or its operator works)
  otherwise. Leave params empty for those.
- Few trades prove nothing. Prefer "no change" (an empty list) over guessing.
Be brief: short sentences, no preamble."""

# Proposal text that touches protected settings is dropped, whatever the type.
PROTECTED_PATTERNS = [
    r"risk[_ ]?per[_ ]?trade",
    r"daily[_ ]?loss",
    r"draw[_ ]?down",
    r"kill[_ ]?switch",
    r"max[_ ]?positions",
    r"max[_ ]?stop",
    r"stop[- ]?(distance|cap)",
    r"position[_ ]?siz",
    r"leverage",
    r"margin",
    r"go(ing)?[_ ]live",
    r"live[_ ]trading",
    r"\bmode\b",
    r"paper[- ]trading",
    r"guardian",
    r"flatten",
]


# --------------------------------------------------------------------------- stats in Python
def _minutes(a: Any, b: Any) -> float | None:
    try:
        return round((pd.Timestamp(b) - pd.Timestamp(a)).total_seconds() / 60.0, 1)
    except (TypeError, ValueError):
        return None


def trade_facts(row: dict[str, Any]) -> dict[str, Any]:
    """Everything worth knowing about one closed trade, computed without any model."""
    entry, stop = float(row.get("entry_price") or 0), float(row.get("stop") or 0)
    target = row.get("target")
    exit_price = float(row.get("exit_price") or 0)
    qty = float(row.get("qty") or 0)
    risk_per_unit = abs(entry - stop)
    risk = risk_per_unit * qty
    costs = float(row.get("costs") or 0)
    pnl = float(row.get("pnl") or 0)
    planned_r = None
    if target not in (None, "") and risk_per_unit > 0:
        planned_r = round(abs(float(target) - entry) / risk_per_unit, 2)
    hold = _minutes(row.get("entry_ts"), row.get("exit_ts"))
    flags = []
    if risk > 0 and costs / risk >= 0.25:
        flags.append(f"costs were {costs / risk:.0%} of the money at risk")
    if hold is not None and hold <= 10:
        flags.append(f"held only {hold:g} min")
    reason = str(row.get("exit_reason") or "")
    if reason in ("eod", "flatten", "guardian", "shutdown", "halt"):
        flags.append(f"closed by the {reason} rule, not by stop or target")
    if entry and risk_per_unit / entry * 100 < 0.15:
        flags.append("stop very tight (<0.15% from entry)")
    if "capped" in str(row.get("signal_reason") or ""):
        flags.append("stop was capped by max_stop_pct")
    return {
        "trade_id": int(row["id"]),
        "asset_class": row.get("asset_class") or "crypto",
        "symbol": row.get("symbol"),
        "side": row.get("side"),
        "entry": round(entry, 4),
        "exit": round(exit_price, 4),
        "stop": round(stop, 4),
        "target": None if target in (None, "") else round(float(target), 4),
        "planned_r": planned_r,
        "r_multiple": round(float(row.get("r_multiple") or 0), 2),
        "pnl": round(pnl, 2),
        "costs": round(costs, 2),
        "hold_minutes": hold,
        "entry_time": str(row.get("entry_ts", ""))[11:16],
        "exit_reason": reason,
        "signal": str(row.get("signal_reason") or "")[:120],
        "flags": flags,
    }


def day_stats(facts: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [f["pnl"] for f in facts]
    rs = [f["r_multiple"] for f in facts]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "trades": len(facts),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(facts), 3) if facts else None,
        "net_pnl": round(sum(pnls), 2),
        "costs": round(sum(f["costs"] for f in facts), 2),
        "avg_r": round(sum(rs) / len(rs), 2) if rs else None,
        "best_r": max(rs) if rs else None,
        "worst_r": min(rs) if rs else None,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2)
        if losses and sum(losses) < 0
        else None,
    }


def rolling_stats(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or not len(trades):
        return {"closed_trades": 0}
    r = trades["r_multiple"].astype(float)
    pnl = trades["pnl"].astype(float)
    last = trades.tail(30)
    return {
        "closed_trades": int(len(trades)),
        "win_rate": round(float((pnl > 0).mean()), 3),
        "avg_r": round(float(r.mean()), 2),
        "avg_r_last_30": round(float(last["r_multiple"].astype(float).mean()), 2),
        "net_pnl": round(float(pnl.sum()), 2),
        "costs": round(float(trades["costs"].astype(float).sum()), 2),
    }


def _day_of(ts: Any) -> date | None:
    try:
        t = pd.Timestamp(ts)
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC").date()


def closed_trades_on(journal: Journal, day: date, mode: str) -> list[dict[str, Any]]:
    since = datetime(day.year, day.month, day.day, tzinfo=UTC) - timedelta(days=1)
    df = journal.trades(since=since, mode=mode)
    rows = json.loads(df.to_json(orient="records", date_format="iso")) if len(df) else []
    return [r for r in rows if _day_of(r.get("exit_ts") or r.get("entry_ts")) == day]


def build_summary(
    cfg: Config, journal: Journal, day: date, backtest: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The compact, numbers-only picture of the day that the model sees."""
    from cryptobot.strategies import default_params, load_strategy

    mode = journal.mode
    facts = [trade_facts(r) for r in closed_trades_on(journal, day, mode)]
    all_trades = journal.trades(mode=mode)
    rolling = rolling_stats(all_trades)
    since = datetime(day.year, day.month, day.day, tzinfo=UTC)
    events = journal.events(since=since, limit=5000)
    rows = json.loads(events.to_json(orient="records")) if len(events) else []
    counts: dict[str, int] = {}
    skips: dict[str, int] = {}
    for e in (r for r in rows if _day_of(r.get("ts")) == day):
        kind = str(e.get("kind"))
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "skip":
            msg = re.sub(r"[\d.]+", "#", str(e.get("message", "")))[:60]
            skips[msg] = skips.get(msg, 0) + 1
    daily = journal.daily(since=day)
    halted = bool(len(daily) and int(daily["halted"].iloc[0] or 0))
    try:
        module = load_strategy(cfg.strategy)
        tunable = sorted({**default_params(module), **cfg.params})
    except Exception:  # a broken strategy file must not stop the review
        tunable = sorted(cfg.params)
    closed = int(rolling.get("closed_trades", 0))
    return {
        "date": day.isoformat(),
        "mode": mode,
        "strategy": cfg.strategy,
        "params": cfg.params,
        "tunable_params": [k for k in tunable if k not in RISK_KEYS],
        "risk_limits_read_only": {
            "risk_per_trade_pct": cfg.risk.risk_per_trade_pct,
            "max_daily_loss_pct": cfg.risk.max_daily_loss_pct,
            "max_positions": cfg.risk.max_positions,
            "max_stop_pct": cfg.risk.max_stop_pct,
            "drawdown_kill_switch_pct": cfg.guardian.max_drawdown_pct,
        },
        "trades": facts,
        "today": {
            **day_stats(facts),
            "halted_by_daily_loss": halted,
            "errors": counts.get("error", 0),
            "rejected_orders": counts.get("order_rejected", 0),
            "skips": dict(sorted(skips.items(), key=lambda kv: -kv[1])[:5]),
        },
        "rolling": rolling,
        "backtest_expects": {
            k: backtest[k]
            for k in ("avg_r", "win_rate", "sharpe", "trades")
            if backtest and k in backtest
        },
        "param_proposals_allowed": closed >= cfg.coach.min_trades_for_params,
        "param_proposals_need": cfg.coach.min_trades_for_params,
    }


# --------------------------------------------------------------------------- enforcement
def _touches_protected(text: str) -> str | None:
    low = text.lower()
    for pat in PROTECTED_PATTERNS:
        if re.search(pat, low):
            return pat
    return None


def screen_proposals(
    proposals: list[dict[str, Any]], summary: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the model's proposals into (kept, dropped-with-reason). Code has the last word."""
    kept, dropped = [], []
    tunable = set(summary.get("tunable_params", []))
    for p in proposals[:3]:
        kind = p.get("type")
        params = [x for x in (p.get("params") or []) if isinstance(x, dict)]
        text = " ".join(
            [str(p.get("description", "")), str(p.get("rationale", ""))]
            + [f"{x.get('key')} {x.get('value')}" for x in params]
        )
        reason = None
        if kind not in CHANGE_TYPES:
            reason = f"unknown type {kind!r}"
        elif any(str(x.get("key")) in RISK_KEYS for x in params):
            reason = "touches a risk limit"
        elif (hit := _touches_protected(text)) is not None:
            reason = f"touches a protected setting ({hit})"
        elif kind == "params":
            if not summary.get("param_proposals_allowed"):
                reason = (
                    f"parameter proposals need {summary.get('param_proposals_need')}+ closed"
                    f" trades; there are {summary['rolling'].get('closed_trades', 0)}"
                )
            elif not params:
                reason = "params proposal without parameters"
            elif bad := [x["key"] for x in params if str(x.get("key")) not in tunable]:
                reason = f"not a strategy parameter: {bad}"
        elif params:
            p = {**p, "params": []}  # filters and process changes carry no parameter edits
        if reason:
            dropped.append({**p, "dropped": reason})
        else:
            kept.append({**p, "status": "proposed: needs human review, never auto-applied"})
    return kept, dropped


def merge_verdicts(facts: list[dict[str, Any]], ai: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {int(t.get("trade_id", -1)): t for t in ai if isinstance(t, dict)}
    out = []
    for f in facts:
        v = by_id.get(f["trade_id"], {})
        verdict = v.get("verdict") if v.get("verdict") in VERDICTS else "not reviewed"
        out.append(
            {
                **f,
                "verdict": verdict,
                "mistake": str(v.get("mistake", "")).strip(),
                "lesson": str(v.get("lesson", "")).strip(),
            }
        )
    return out


# --------------------------------------------------------------------------- the run
def make_review_llm(cfg: Config):
    """The once-a-day Claude client (Haiku by default). None when no key is configured."""
    from cryptobot.llm import ClaudeLLM

    try:
        return ClaudeLLM(
            model=cfg.coach.review_model,
            max_tokens=cfg.coach.review_max_tokens,
            timeout=120.0,
        )
    except LLMError as e:
        log.info("daily coach runs on stats only: %s", e)
        return None


def run_daily_review(
    cfg: Config,
    journal: Journal,
    day: date | None = None,
    llm=None,
    backtest: dict[str, Any] | None = None,
    force: bool = False,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Review every trade closed on `day`. Returns the review record (also written to disk)."""
    if not cfg.coach.enabled:
        return {"status": "off", "reason": "coach.enabled is false"}
    day = day or datetime.now(UTC).date()
    if not force and journal.get_state(STATE_KEY) == day.isoformat():
        return {"status": "skipped", "reason": f"{day} already reviewed (use --force to redo)"}
    base = Path(out_dir or cfg.coach.dir)
    summary = build_summary(cfg, journal, day, backtest)
    facts = summary["trades"]
    record: dict[str, Any] = {
        "date": day.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": summary,
        "model": None,
        "cost_usd": 0.0,
        "ai": "not used",
    }
    ai: dict[str, Any] = {}
    if not facts:
        record["ai"] = "no closed trades today, no call made"
    elif llm is None:
        record["ai"] = "no API key: stats only (set ANTHROPIC_API_KEY to get verdicts)"
    elif journal.llm_spend_today(COMPONENT, datetime.now(UTC).date()) >= (
        cfg.coach.max_daily_cost_usd
    ):  # spend is booked on the day the call is made
        record["ai"] = "daily coach budget already spent"
    else:
        try:
            res = llm.complete(
                SYSTEM_PROMPT,
                json.dumps(summary, separators=(",", ":"), default=str),
                REVIEW_SCHEMA,
                effort="low",
            )
            journal.add_llm_spend(COMPONENT, res.cost_usd)
            record.update(
                {
                    "model": res.model,
                    "cost_usd": round(res.cost_usd, 6),
                    "tokens": {"input": res.input_tokens, "output": res.output_tokens},
                }
            )
            if res.refused:
                record["ai"] = "the model declined; stats only"
            else:
                ai = res.data
                record["ai"] = "reviewed"
        except LLMError as e:
            record["ai"] = f"model error, stats only: {e}"
        except Exception as e:  # network, auth, anything: never let the coach break the bot
            log.warning("daily coach call failed: %s", e)
            record["ai"] = f"call failed, stats only: {type(e).__name__}"
    record["trades"] = merge_verdicts(facts, ai.get("trades") or [])
    record["daily_summary"] = str(ai.get("daily_summary") or "").strip() or _plain_summary(summary)
    kept, dropped = screen_proposals(list(ai.get("proposed_changes") or []), summary)
    record["proposed_changes"] = kept
    record["dropped_proposals"] = dropped
    _write_outputs(base, record)
    journal.set_state(STATE_KEY, day.isoformat())
    journal.log(
        "coach_review",
        f"{len(facts)} trades reviewed ({record['ai']}); {len(kept)} proposals saved"
        + (f", {len(dropped)} dropped" if dropped else "")
        + f"; cost ${record['cost_usd']:.4f}",
        data={"cost_usd": record["cost_usd"], "model": record["model"]},
    )
    record["status"] = "done"
    return record


def _plain_summary(summary: dict[str, Any]) -> str:
    t = summary["today"]
    if not t["trades"]:
        return "No closed trades today."
    return (
        f"{t['trades']} trades, {t['wins']} won, net {t['net_pnl']:+.2f} after"
        f" {t['costs']:.2f} costs, average {t['avg_r']:+.2f}R."
    )


def _write_outputs(base: Path, record: dict[str, Any]) -> None:
    day = record["date"]
    (base / "reviews").mkdir(parents=True, exist_ok=True)
    (base / "reviews" / f"{day}.json").write_text(json.dumps(record, indent=2, default=str))
    if record["proposed_changes"] or record["dropped_proposals"]:
        (base / "proposals").mkdir(parents=True, exist_ok=True)
        (base / "proposals" / f"{day}.json").write_text(
            json.dumps(
                {
                    "date": day,
                    "note": "Proposals only. Nothing here is applied automatically; review and"
                    " change config.yaml yourself if you agree.",
                    "proposed_changes": record["proposed_changes"],
                    "dropped_by_code": record["dropped_proposals"],
                },
                indent=2,
            )
        )
    t = record["summary"]["today"]
    lines = [
        f"## {day}",
        "",
        f"- {t['trades']} trades, {t['wins']} won, net {t['net_pnl']:+.2f}, costs"
        f" {t['costs']:.2f}" + (f", avg {t['avg_r']:+.2f}R" if t["avg_r"] is not None else ""),
    ]
    verdicts = [x["verdict"] for x in record["trades"]]
    if verdicts:
        lines.append("- verdicts: " + ", ".join(f"{v} {verdicts.count(v)}" for v in set(verdicts)))
    lines.append(f"- summary: {record['daily_summary']}")
    for x in record["trades"]:
        if x["lesson"]:
            lines.append(f"- lesson ({x['symbol']} #{x['trade_id']}): {x['lesson']}")
    if record["proposed_changes"]:
        lines.append(
            f"- {len(record['proposed_changes'])} proposal(s) saved to"
            f" coach/proposals/{day}.json (not applied)"
        )
    lines.append(f"- coach: {record['ai']}; cost ${record['cost_usd']:.4f}")
    with (base / "journal.md").open("a", encoding="utf-8") as fh:
        if fh.tell() == 0:
            fh.write(
                "# Coach journal\n\nOne entry per trading day. Proposals are never applied"
                " automatically.\n\n"
            )
        fh.write("\n".join(lines) + "\n\n")
