"""Layer 2 of the learning loop: the offline AI coach.

Runs once a day after the close, as its own process (`python -m cryptobot coach`). It reads the
decision log, the latest backtest and its own past notes, writes a plain-English review and
proposes changes as structured JSON. Every proposal goes through learn.gate. Nothing is ever
applied: parameter proposals that pass are recorded for a human, and code proposals are
written to strategies/proposed/ and backtested in a subprocess with no secrets.

This module never imports cryptobot.brokers and never sees broker keys (tests enforce it).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cryptobot import learn
from cryptobot.backtest import BacktestResult
from cryptobot.config import RISK_KEYS, Config
from cryptobot.journal import Journal
from cryptobot.llm import LLMError, LLMResult
from cryptobot.strategies import default_params, load_strategy

log = logging.getLogger(__name__)

MAX_PROPOSALS = 3
PROPOSED_DIR = Path("cryptobot/strategies/proposed")

PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "review": {"type": "string"},
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["params", "filter", "strategy"]},
                    "name": {"type": "string"},
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
                    "code": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "expected_effect": {"type": "string"},
                },
                "required": ["type", "name", "params", "code", "hypothesis", "expected_effect"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["review", "proposals"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are the offline coach for Cryptobot, a small deterministic crypto
day-trading bot (Alpaca USD pairs, 24/7, long-only, trading day = the UTC day).
You run after the UTC trading day has ended. You never trade and nothing you say is executed
directly.

Your job: read the bot's recent decisions, trades and backtest numbers, write a short honest
review (what worked, what did not, whether the results are within the backtest's expected
range, whether anything looks like a bug), and propose at most three changes.

Rules:
- A proposal is either `params` (new values for the strategy's existing parameters), `filter`
  (a Python strategy file that wraps the current logic with an extra condition) or `strategy`
  (a new Python strategy file). Code must define `signal(bars, params)` returning a
  `cryptobot.strategies.Signal` or None, with no I/O and no imports beyond pandas, numpy and
  cryptobot.strategies.*.
- Never propose changes to risk settings (risk per trade, daily loss, max positions, flatten
  time). They belong to the human.
- Every proposal needs a falsifiable hypothesis and the effect you expect in numbers.
- Each proposal is backtested out of sample after costs and must beat the current settings on
  Sharpe. Most proposals should fail that test; propose nothing rather than something weak.
- If the sample is too small to conclude anything, say so in the review and propose nothing.
- Be concrete and brief. Put numbers in the review."""


def _trades_summary(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "No trades in the window."
    pnl = trades["pnl"].astype(float)
    lines = [
        f"{len(trades)} trades, total pnl {pnl.sum():+.2f}, win rate {(pnl > 0).mean() * 100:.0f}%,"
        f" avg R {trades['r_multiple'].astype(float).mean():+.2f}, costs"
        f" {trades['costs'].astype(float).sum():.2f}",
        "By exit reason: "
        + ", ".join(
            f"{k}: n={len(g)} pnl={g['pnl'].astype(float).sum():+.2f}"
            for k, g in trades.groupby("exit_reason")
        ),
        "By symbol: "
        + ", ".join(
            f"{k}: n={len(g)} pnl={g['pnl'].astype(float).sum():+.2f}"
            for k, g in trades.groupby("symbol")
        ),
        "Last trades (entry, symbol, side, qty, entry px, exit px, reason, pnl, R):",
    ]
    for _, t in trades.tail(30).iterrows():
        lines.append(
            f"  {str(t['entry_ts'])[:16]} {t['symbol']} {t['side']} {float(t['qty']):.4g}"
            f" {float(t['entry_price']):.2f} {float(t['exit_price']):.2f} {t['exit_reason']}"
            f" {float(t['pnl']):+.2f} {float(t['r_multiple']):+.2f}R"
        )
    return "\n".join(lines)


def build_context(
    cfg: Config,
    journal: Journal,
    latest: BacktestResult | None,
    lookback_days: int,
    now: datetime | None = None,
) -> str:
    now = now or datetime.now(UTC)
    since = now - timedelta(days=lookback_days)
    module = load_strategy(cfg.strategy)
    params = {**default_params(module), **cfg.params}
    parts = [
        f"Date: {now.date()}  Mode: {journal.mode}",
        f"Strategy: {cfg.strategy}  params: {json.dumps(params)}",
        f"Symbols: {cfg.symbols}  timeframe: {cfg.timeframe}  allow_short: {cfg.allow_short}",
        f"Costs: {json.dumps(cfg.costs.__dict__)}",
        f"Risk (read-only, not yours to change): {json.dumps(cfg.risk.__dict__)}",
        "",
        f"== Trades in the last {lookback_days} days ==",
        _trades_summary(journal.trades(since=since)),
        "",
        "== Daily results ==",
    ]
    daily = journal.daily(since=since.date())
    if len(daily):
        for _, d in daily.tail(lookback_days).iterrows():
            parts.append(
                f"  {d['date']}: pnl {float(d['pnl']):+.2f} trades {int(d['trades'])} costs"
                f" {float(d['costs']):.2f}{' HALTED' if int(d['halted']) else ''}"
            )
    else:
        parts.append("  none")
    events = journal.events(since=since, kinds=["skip", "order_rejected", "error", "halt"])
    if len(events):
        counts = events["message"].value_counts().head(10)
        parts += ["", "== Skipped signals / errors (count: reason) =="]
        parts += [f"  {int(n)}: {msg}" for msg, n in counts.items()]
    parts += ["", "== Latest backtest (out of sample, after costs) =="]
    if latest is not None:
        parts.append(f"  window {latest.start} to {latest.end}; params {latest.params}")
        parts.append(
            "  strategy: "
            + json.dumps(
                {k: round(v, 3) if isinstance(v, float) else v for k, v in latest.metrics.items()}
            )
        )
        for name, m in latest.benchmarks.items():
            parts.append(
                f"  {name}: "
                + json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})
            )
    else:
        parts.append("  none saved")
    history = journal.params_history(limit=10)
    if history:
        parts += ["", "== Parameter changes (newest first) =="]
        for h in history:
            parts.append(
                f"  {h['ts'][:10]} {h['source']} applied={h['applied']}"
                f" {h['old_params']} -> {h['new_params']}: {h['stats'].get('reason', '')}"
            )
    notes = journal.coach_notes(limit=5)
    if notes:
        parts += ["", "== Your previous notes (newest first) =="]
        for n in notes:
            parts.append(f"  {n['ts'][:10]}: {str(n.get('review', ''))[:400]}")
            for o in n.get("outcomes", [])[:3]:
                parts.append(
                    f"    - {o.get('name')} [{o.get('type')}] -> {o.get('verdict')}:"
                    f" {o.get('reason')}"
                )
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        source = "(source unavailable)"
    parts += ["", f"== Current strategy source ({cfg.strategy}) ==", source]
    return "\n".join(parts)


def parse_params(pairs: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in pairs or []:
        key = str(p.get("key", "")).strip()
        if not key:
            continue
        try:
            out[key] = yaml.safe_load(str(p.get("value", "")))
        except yaml.YAMLError:
            out[key] = str(p.get("value", ""))
    return out


def scrubbed_env() -> dict[str, str]:
    """Environment for the code-proposal subprocess: no keys, secrets or tokens."""
    bad = ("_KEY", "_SECRET", "_TOKEN", "PASSWORD")
    return {
        k: v
        for k, v in os.environ.items()
        if not k.upper().endswith(bad) and not k.upper().startswith("ANTHROPIC")
    }


def gate_code_proposal(cfg: Config, path: Path, data_provider: str) -> tuple[bool, str, dict]:
    """Backtest a proposed strategy file in a subprocess with no secrets, then apply the gate
    against the current strategy's saved out-of-sample metrics."""
    cmd = [
        sys.executable,
        "-m",
        "cryptobot",
        "--config",
        cfg.path,
        "backtest",
        "--strategy-file",
        str(path),
        "--data",
        data_provider,
        "--json",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=scrubbed_env(), check=False
        )
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"could not run the proposal: {e}", {}
    if proc.returncode != 0:
        return False, f"proposal backtest failed: {proc.stderr.strip()[-500:]}", {}
    try:
        js = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        return False, f"proposal backtest printed no result: {e}", {}
    challenger = js.get("metrics", {})
    latest_path = Path(cfg.reports_dir) / f"latest_{cfg.strategy}.json"
    champion = (
        json.loads(latest_path.read_text()).get("metrics", {}) if latest_path.exists() else {}
    )
    ok, reason = learn.gate(
        champion, challenger, cfg.learning.min_oos_trades, cfg.learning.min_sharpe_gain
    )
    return ok, reason, {"champion": champion, "challenger": challenger}


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")[:40] or "proposal"


def run_coach(
    cfg: Config,
    journal: Journal,
    llm,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
    latest: BacktestResult | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
    code_gate: Callable[[Config, Path, str], tuple[bool, str, dict]] | None = None,
    data_provider: str | None = None,
    proposed_dir: Path | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    day = now.date()
    spent = journal.coach_spend_today(day)
    if spent >= cfg.coach.max_daily_cost_usd:
        journal.log("coach", f"skipped: daily cost cap reached ({spent:.2f} USD)")
        return {"status": "skipped", "reason": "daily cost cap reached", "cost_usd": spent}

    context = build_context(cfg, journal, latest, cfg.coach.lookback_days, now=now)
    try:
        res: LLMResult = llm.complete(SYSTEM_PROMPT, context, PROPOSAL_SCHEMA)
    except LLMError as e:
        journal.log("coach_error", str(e))
        return {"status": "error", "reason": str(e)}
    except Exception as e:  # any provider failure means "no proposal today"
        journal.log("coach_error", repr(e))
        return {"status": "error", "reason": repr(e)}
    journal.add_coach_spend(res.cost_usd, day)
    if res.refused:
        journal.log("coach", "model declined to answer; no proposal today")
        return {"status": "refused", "cost_usd": res.cost_usd}

    review = str(res.data.get("review", "")).strip()
    proposals = list(res.data.get("proposals", []))[:MAX_PROPOSALS]
    outcomes: list[dict[str, Any]] = []
    provider = data_provider or cfg.data.provider
    out_dir = proposed_dir or PROPOSED_DIR
    for p in proposals:
        kind = str(p.get("type", ""))
        name = str(p.get("name", kind))
        outcome = {
            "type": kind,
            "name": name,
            "hypothesis": p.get("hypothesis", ""),
            "expected_effect": p.get("expected_effect", ""),
            "applied": False,
        }
        if kind == "params":
            new = parse_params(p.get("params", []))
            try:
                learn.check_proposal_params(new)
                if not new:
                    raise learn.ProposalError("empty params proposal")
                if bars_by_symbol is None:
                    raise learn.ProposalError("no bars available to run the gate")
                ok, reason, stats = learn.evaluate_candidate(cfg, bars_by_symbol, new)
            except learn.ProposalError as e:
                ok, reason, stats = False, str(e), {}
            merged = {**cfg.params, **new}
            outcome.update(
                {"params": new, "verdict": "accepted" if ok else "rejected", "reason": reason}
            )
            if ok and not dry_run:
                # proposals only: recorded in params_history, never written to config.yaml
                learn.apply_params(
                    cfg, journal, dict(cfg.params), merged, stats, "coach", applied=False
                )
                outcome["applied"] = False
                outcome["reason"] += " (passed the gate; proposal only, apply it yourself)"
        elif kind in ("filter", "strategy"):
            code = str(p.get("code", ""))
            if "def signal" not in code:
                outcome.update({"verdict": "rejected", "reason": "no signal() in proposed code"})
            elif re.search(
                r"^\s*(import|from)\s+(?!pandas|numpy|cryptobot\.strategies)", code, re.M
            ):
                outcome.update(
                    {
                        "verdict": "rejected",
                        "reason": "proposed code imports something other than pandas,"
                        " numpy or cryptobot.strategies",
                    }
                )
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{day.isoformat()}_{_safe_name(name)}.py"
                path.write_text(code)
                outcome["path"] = str(path)
                gate_fn = code_gate or gate_code_proposal
                ok, reason, stats = gate_fn(cfg, path, provider)
                outcome.update(
                    {
                        "verdict": "accepted" if ok else "rejected",
                        "reason": reason
                        + (
                            " (code is never applied automatically: review it and open a PR)"
                            if ok
                            else ""
                        ),
                    }
                )
        else:
            outcome.update({"verdict": "rejected", "reason": f"unknown proposal type {kind!r}"})
        outcomes.append(outcome)

    report_dir = Path(cfg.reports_dir) / "coach"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{day.isoformat()}.md"
    lines = [
        f"# Coach review {day.isoformat()} ({res.model}, {res.cost_usd:.4f} USD)",
        "",
        review,
        "",
        "## Proposals",
        "",
    ]
    if not outcomes:
        lines.append("_none_")
    for o in outcomes:
        lines += [
            f"### {o['name']} ({o['type']}) -> {o['verdict']}",
            f"- hypothesis: {o['hypothesis']}",
            f"- expected: {o['expected_effect']}",
            f"- gate: {o['reason']}",
            f"- applied: {o['applied']}",
        ]
        if o.get("params"):
            lines.append(f"- params: {o['params']}")
        if o.get("path"):
            lines.append(f"- code: {o['path']}")
        lines.append("")
    report_path.write_text("\n".join(lines))
    journal.add_coach_note(
        {
            "review": review[:2000],
            "outcomes": outcomes,
            "cost_usd": res.cost_usd,
            "model": res.model,
        }
    )
    accepted = sum(1 for o in outcomes if o.get("verdict") == "accepted")
    journal.log(
        "coach", f"{len(outcomes)} proposals, {accepted} passed the gate, {res.cost_usd:.4f} USD"
    )
    return {
        "status": "ok",
        "review": review,
        "outcomes": outcomes,
        "cost_usd": res.cost_usd,
        "report": str(report_path),
        "model": res.model,
    }


assert not (set(RISK_KEYS) & {"range_minutes"})  # sanity: risk keys never collide with params
