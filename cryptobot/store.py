"""The feature store's two jobs beyond the journal: export to Parquet for analysis, and replay a
past day to prove every decision can be reproduced exactly from stored inputs.

Replay re-runs the strategy on the cached bars at each journalled decision time and compares:
  - the signal (side, entry reference, stop, target),
  - the bar features,
  - the fingerprint of the inputs (bar window + strategy + params).
A hash mismatch means the stored bars changed since (Alpaca revised a bar, or a bar was still
forming when the bot read it); a signal mismatch with matching inputs would be a bug.
LLM-decider decisions are not deterministic and are reported as such.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from cryptobot import sessions
from cryptobot.backtest import HISTORY_SESSIONS
from cryptobot.config import Config
from cryptobot.features import bar_features, inputs_hash
from cryptobot.journal import Journal
from cryptobot.strategies import Signal, load_strategy

FEATURE_TOLERANCE = 1e-6


def _expand(df: pd.DataFrame, column: str = "features", prefix: str = "f_") -> pd.DataFrame:
    """JSON feature column -> one numeric/text column per feature (prefixed)."""
    if df.empty or column not in df.columns:
        return df
    parsed = [json.loads(x) if isinstance(x, str) and x else {} for x in df[column]]
    flat = pd.DataFrame(
        [
            {f"{prefix}{k}": v for k, v in p.items() if not isinstance(v, dict | list)}
            for p in parsed
        ],
        index=df.index,
    )
    return pd.concat([df, flat], axis=1)


def export_store(journal: Journal, out_dir: str | Path) -> dict[str, Path]:
    """Write decisions, trades and daily results to Parquet (features expanded to columns)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "decisions": _expand(journal.decisions()),
        "trades": _expand(journal.trades()),
        "daily": journal.daily(),
    }
    paths = {}
    for name, df in tables.items():
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        paths[name] = path
    return paths


def _cap_stop(sig: Signal, max_stop_pct: float) -> Signal:
    cap = sig.entry * max_stop_pct / 100.0
    if sig.risk_per_unit <= cap:
        return sig
    stop = sig.entry - cap if sig.side == "long" else sig.entry + cap
    return Signal(sig.side, sig.entry, stop, sig.target, reason=sig.reason + " (stop capped)")


def _close(a, b) -> bool:
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)):
        return (a is None or (isinstance(a, float) and math.isnan(a))) and (
            b is None or (isinstance(b, float) and math.isnan(b))
        )
    return abs(float(a) - float(b)) <= FEATURE_TOLERANCE * max(1.0, abs(float(a)))


def window_at(bars: pd.DataFrame, ts: pd.Timestamp, tf: int) -> pd.DataFrame:
    """The window the live loop handed the engine when the bar starting at `ts` closed."""
    n = HISTORY_SESSIONS * (sessions.CRYPTO.session_minutes // tf)
    w = bars[bars.index <= ts].tail(n)
    w.attrs = dict(bars.attrs)
    return w


def replay_decisions(
    cfg: Config,
    decisions: pd.DataFrame,
    bars_by_symbol: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Re-run every journalled enter/skip decision from the stored bars and compare."""
    out = []
    tf = cfg.timeframe_minutes
    for _, d in decisions.iterrows():
        if d["action"] not in ("enter", "skip"):
            continue
        row: dict[str, Any] = {"ts": d["ts"], "symbol": d["symbol"], "action": d["action"]}
        reason = str(d.get("reason") or "")
        if reason.startswith(("claude:", "ollama:")):
            row["status"] = "llm (not deterministic)"
            out.append(row)
            continue
        bars = bars_by_symbol.get(d["symbol"])
        if bars is None or bars.empty:
            row["status"] = "no stored bars"
            out.append(row)
            continue
        ts = pd.Timestamp(d["ts"]).tz_convert("UTC")
        window = window_at(bars, ts, tf)
        if window.empty or window.index[-1] != ts:
            row["status"] = "bar missing from the store"
            out.append(row)
            continue
        # exactly the params the engine held (strategies apply their own defaults)
        params = json.loads(d["params"]) if isinstance(d["params"], str) else {}
        module = load_strategy(str(d["strategy"]))
        sig = module.signal(window, params)
        if sig is not None:
            sig = _cap_stop(sig, cfg.risk.max_stop_pct)
        row["hash_match"] = inputs_hash(window, str(d["strategy"]), params) == d["inputs_hash"]
        row["signal_match"] = sig is not None and (
            sig.side == d["side"]
            and _close(sig.entry, d["signal_price"])
            and _close(sig.stop, d["stop"])
            and _close(sig.target, d["target"])
        )
        stored = json.loads(d["features"]) if isinstance(d["features"], str) else {}
        fresh = bar_features(window)
        diffs = [k for k, v in fresh.items() if k in stored and not _close(v, stored[k])]
        row["features_match"] = not diffs
        if diffs:
            row["feature_diffs"] = diffs
        row["status"] = (
            "exact"
            if row["hash_match"] and row["signal_match"] and row["features_match"]
            else ("inputs changed" if not row["hash_match"] else "MISMATCH")
        )
        out.append(row)
    return out


def replay_day(
    cfg: Config,
    journal: Journal,
    day: date,
    bars_by_symbol: dict[str, pd.DataFrame],
    mode: str | None = None,
) -> list[dict[str, Any]]:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    decisions = journal.decisions(
        since=start, until=start + timedelta(days=1), mode=mode, actions=["enter", "skip"]
    )
    return replay_decisions(cfg, decisions, bars_by_symbol)
