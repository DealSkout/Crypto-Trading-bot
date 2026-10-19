"""The weekly self-learning loop: generate challengers, validate them without fooling ourselves,
and only adopt what is statistically better. Paper only: it can start a shadow run and, after
two clean weeks, make a model the PAPER champion. It can never touch live trading or risk.

Pipeline (python -m cryptobot learn, weekly via the systemd timer):
  1. challengers   parameter variations (each strategy's grid around the champion), feature
                   filters (skip rules on research features that have enough history) and
                   regime size weights (only ever <= 1: the loop can shrink positions, never grow
                   them past the human's risk limits)
  2. simulate      every candidate once over the full history, with the realistic fill model
                   (next-bar-open entries, spread, slippage, stop slippage, fees), in parallel
  3. validate      on the first 70% of days (selection window):
                     - purged walk-forward: pick the best on the past, score it on the next fold
                     - combinatorially purged cross-validation -> probability of backtest
                       overfitting (PBO)
                   then on the last 30% (holdout), challenger vs champion, all must hold:
                     - >= min_oos_trades trades
                     - Sharpe beats the champion's by min_sharpe_gain
                     - the deflated Sharpe ratio of the daily return DIFFERENCE is >= dsr_min.
                       The holdout is reused week after week, so every learning run in the
                       last trial_memory_days counts as one more look at it (the multiple-
                       testing correction); the spread of the challengers' results in the
                       selection window sets how lucky the best of a search can look
                     - max drawdown no worse than the champion's
                     - profitable in >= min_profitable_regimes regimes
                     - positive net return after costs, PBO <= max_pbo, walk-forward not worse
  4. shadow        a winner runs in simulation next to a mirror of the champion on live bars for
                   shadow_days; only then can it become the paper champion (see ShadowRunner)
Every candidate, result and reason is kept in the registry and in reports/learning/.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import multiprocessing as mp
import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from cryptobot import backtest
from cryptobot.config import Config, LearningConfig
from cryptobot.registry import Model, Registry
from cryptobot.strategies import default_params, load_strategy, param_grid

log = logging.getLogger(__name__)

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()
PERIODS = backtest.PERIODS_PER_YEAR


# ============================================================================ candidates


@dataclass
class Candidate:
    strategy: str
    params: dict[str, Any]
    filters: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    @classmethod
    def from_model(cls, m: Model) -> Candidate:
        return cls(m.strategy, dict(m.params), dict(m.filters), label=f"champion {m.id}")

    def key(self) -> str:
        return json.dumps([self.strategy, self.params, self.filters], sort_keys=True, default=str)


# Skip rules the loop may try, by the research feature they need. A rule is only tried when the
# research store holds that feature for the whole backtest period.
FILTER_RULES: list[tuple[str, dict[str, Any]]] = [
    ("fear_greed", {"feature": "fear_greed", "op": "<=", "value": 20}),
    ("fear_greed", {"feature": "fear_greed", "op": ">=", "value": 80}),
    ("funding", {"feature": "funding_rate_pct", "op": ">=", "value": 0.03}),
    ("funding", {"feature": "funding_rate_pct", "op": "<=", "value": -0.01}),
]
REGIME_WEIGHTS: list[dict[str, float]] = [
    {"risk_off": 0.0},
    {"risk_off": 0.0, "high_vol": 0.5},
    {"high_vol": 0.0},
    {"ranging": 0.0},
    {"trending": 0.0},
]


def _label(strategy: str, params: dict, filters: dict) -> str:
    p = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    parts = [f"{strategy} {p}"]
    for rule in filters.get("skip_if", []):
        parts.append(f"skip {rule['feature']}{rule['op']}{rule['value']}")
    if filters.get("regime_weights"):
        parts.append(
            "weights " + ",".join(f"{k}={v}" for k, v in sorted(filters["regime_weights"].items()))
        )
    return " | ".join(parts)


def generate_challengers(
    cfg: Config, champion: Model, available: set[str] | None = None
) -> list[Candidate]:
    """Deterministic list of challengers, champion-related ideas first, capped at
    learning.max_challengers."""
    lc = cfg.learning
    available = available or set()
    out: list[Candidate] = []
    # 1. the champion's strategy: every grid point around its current params, same filters
    strategies = [champion.strategy] + [s for s in lc.strategies if s != champion.strategy]
    for name in strategies:
        module = load_strategy(name)
        base = dict(champion.params) if name == champion.strategy else {**default_params(module)}
        grid = param_grid(module)
        keys = sorted(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            params = {**base, **dict(zip(keys, combo, strict=True))}
            filters = dict(champion.filters) if name == champion.strategy else {}
            out.append(Candidate(name, params, filters, _label(name, params, filters)))
        if name == champion.strategy:
            # 2. feature filters and regime weights on top of the champion as it is
            for need, rule in FILTER_RULES:
                if need not in available:
                    continue
                f = json.loads(json.dumps(champion.filters))
                f.setdefault("skip_if", [])
                if rule in f["skip_if"]:
                    continue
                f["skip_if"] = [*f["skip_if"], rule]
                out.append(
                    Candidate(name, dict(champion.params), f, _label(name, champion.params, f))
                )
            for weights in REGIME_WEIGHTS:
                f = json.loads(json.dumps(champion.filters))
                f["regime_weights"] = {**f.get("regime_weights", {}), **weights}
                out.append(
                    Candidate(name, dict(champion.params), f, _label(name, champion.params, f))
                )
    seen = {Candidate.from_model(champion).key()}
    unique = []
    for c in out:
        if c.key() not in seen:
            seen.add(c.key())
            unique.append(c)
    return unique[: lc.max_challengers]


def research_coverage(cfg: Config, research_path: str | None, start: datetime) -> set[str]:
    """Which research features have history back to `start` (so filters can be backtested)."""
    if not research_path or not Path(research_path).exists():
        return set()
    from cryptobot.research.feeds import base_of
    from cryptobot.research.store import ResearchStore

    store = ResearchStore(research_path)
    out = set()
    first, _ = store.coverage("fear_greed", "index")
    if first and pd.Timestamp(first) <= pd.Timestamp(start):
        out.add("fear_greed")
    ok = True
    for symbol in cfg.symbols:
        base = base_of(symbol)
        firsts = [store.coverage("funding", f"{base}@{ex}")[0] for ex in cfg.research.exchanges]
        firsts = [f for f in firsts if f]
        if not firsts or pd.Timestamp(min(firsts)) > pd.Timestamp(start) + timedelta(days=1):
            ok = False
    if ok:
        out.add("funding")
    return out


# ============================================================================ simulation

_SHARED: dict[str, Any] = {}


def _providers(cfg: Config, bars: dict[str, pd.DataFrame], research_path: str | None) -> list:
    from cryptobot.regime import RegimeDetector

    providers = []
    fg = None
    if research_path and Path(research_path).exists():
        from cryptobot.research.feed import ResearchFeed
        from cryptobot.research.store import ResearchStore

        feed = ResearchFeed(cfg, ResearchStore(research_path))
        providers.append(feed.context)

        def fg(when, _store=feed.store):
            row = _store.asof("fear_greed", "index", when)
            return row["value"] if row else None

    regimes = RegimeDetector.from_bars(cfg.regime, bars, fear_greed=fg)
    return [regimes.context, *providers]


def simulate(
    cfg: Config, cand: Candidate, bars: dict[str, pd.DataFrame], research_path: str | None
) -> dict[str, Any]:
    """One continuous run over every day: daily returns and trades (with regimes)."""
    days = backtest.all_session_days(bars)
    module = load_strategy(cand.strategy)
    _, trades, equity = backtest.run_period(
        cfg,
        module,
        cand.params,
        bars,
        days,
        context_providers=_providers(cfg, bars, research_path),
        filters=cand.filters,
    )
    rets = backtest.daily_returns(equity, cfg.starting_cash)
    rets.index = [pd.Timestamp(i).date() for i in rets.index]
    keep = [c for c in ("entry_ts", "pnl", "r_multiple", "regime", "symbol") if c in trades]
    return {
        "daily": rets,
        "trades": trades[keep].copy() if len(trades) else pd.DataFrame(columns=keep),
    }


def _worker(i: int) -> tuple[int, dict[str, Any]]:
    cfg, bars, path, cands = (_SHARED[k] for k in ("cfg", "bars", "path", "cands"))
    return i, simulate(cfg, cands[i], bars, path)


def simulate_all(
    cfg: Config,
    cands: list[Candidate],
    bars: dict[str, pd.DataFrame],
    research_path: str | None,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    workers = workers if workers is not None else cfg.learning.workers
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)
    if workers == 1 or len(cands) == 1 or "fork" not in mp.get_all_start_methods():
        return [simulate(cfg, c, bars, research_path) for c in cands]
    _SHARED.update(cfg=cfg, bars=bars, path=research_path, cands=cands)
    out: list[dict[str, Any] | None] = [None] * len(cands)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
            for i, res in ex.map(_worker, range(len(cands))):
                out[i] = res
    finally:
        _SHARED.clear()
    return out  # type: ignore[return-value]


# ============================================================================ statistics


def sharpe(x: np.ndarray) -> float:
    """Per-period Sharpe (not annualised); 0 when undefined."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x) / sd) if sd > 0 else 0.0


def max_drawdown(x: np.ndarray) -> float:
    """Max drawdown (%, negative) of compounding these returns from 1."""
    curve = np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(x, dtype=float))])
    peak = np.maximum.accumulate(curve)
    return float(np.min(curve / peak - 1.0) * 100.0)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe of n_trials skill-less strategies (Bailey & Lopez de Prado)."""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    a = _N.inv_cdf(1.0 - 1.0 / n_trials)
    b = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(x: np.ndarray, n_trials: int, sr_variance: float) -> dict[str, float]:
    """Probability that the true Sharpe of `x` beats the best a skill-less search over
    `n_trials` would find by luck (deflated Sharpe ratio). Accounts for skew and fat tails."""
    x = np.asarray(x, dtype=float)
    t = len(x)
    sr = sharpe(x)
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    if t < 3 or float(np.std(x)) == 0:
        return {"dsr": 0.0, "sr": sr, "sr0": sr0}
    z = (x - x.mean()) / x.std()
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2))
    stat = (sr - sr0) * math.sqrt(t - 1) / denom
    return {"dsr": float(_N.cdf(stat)), "sr": sr, "sr0": sr0}


def _block_sharpes(r: np.ndarray, idx: np.ndarray) -> np.ndarray:
    sub = r[idx]
    sd = sub.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, sub.mean(axis=0) / sd, 0.0)


def pbo_cscv(r: np.ndarray, n_blocks: int = 8, embargo: int = 1) -> float | None:
    """Probability of backtest overfitting by combinatorially symmetric, purged cross-validation:
    how often the in-sample best ranks below the median out of sample. None if too little data
    or fewer than 2 candidates."""
    t, k = r.shape
    if k < 2 or t < n_blocks * 5:
        return None
    blocks = np.array_split(np.arange(t), n_blocks)
    logits = []
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        oos = np.concatenate([blocks[i] for i in range(n_blocks) if i not in combo])
        oos_set = set(oos.tolist())
        is_idx = []
        for i in combo:
            for j in blocks[i]:  # purge: drop in-sample days next to an out-of-sample day
                if all((j + d) not in oos_set for d in range(-embargo, embargo + 1)):
                    is_idx.append(j)
        is_idx = np.array(is_idx, dtype=int)
        if len(is_idx) < 5:
            continue
        best = int(np.argmax(_block_sharpes(r, is_idx)))
        oos_sr = _block_sharpes(r, oos)
        rank = int(np.sum(oos_sr < oos_sr[best])) + 1  # 1 = worst
        omega = rank / (k + 1)
        logits.append(math.log(omega / (1.0 - omega)))
    if not logits:
        return None
    return float(np.mean(np.array(logits) <= 0.0))


def walk_forward(
    r: np.ndarray, champion: int = 0, folds: int = 4, embargo: int = 1
) -> dict[str, Any]:
    """Anchored walk-forward of the selection rule ("take the best Sharpe so far"): each fold's
    pick is made only from earlier days (minus an embargo) and scored on the fold."""
    t = len(r)
    size = t // (folds + 1)
    if size < 5:
        return {"wf_sharpe": 0.0, "champion_sharpe": 0.0, "picks": []}
    wf, ch, picks = [], [], []
    for j in range(1, folds + 1):
        start = j * size
        end = (j + 1) * size if j < folds else t
        train = r[: max(2, start - embargo)]
        best = int(np.argmax(_block_sharpes(train, np.arange(len(train)))))
        picks.append(best)
        wf.append(r[start:end, best])
        ch.append(r[start:end, champion])
    wf_r, ch_r = np.concatenate(wf), np.concatenate(ch)
    return {
        "wf_sharpe": sharpe(wf_r) * math.sqrt(PERIODS),
        "champion_sharpe": sharpe(ch_r) * math.sqrt(PERIODS),
        "picks": picks,
    }


# ============================================================================ the gate


@dataclass
class GateResult:
    accepted: bool
    reasons: list[str]
    best: int | None
    stats: dict[str, Any]


def _trades_in(trades: pd.DataFrame, days: set[date]) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["pnl", "r_multiple", "regime"])
    d = pd.to_datetime(trades["entry_ts"], utc=True).dt.date
    return trades[d.isin(days)]


def evaluate(
    r: np.ndarray,
    trades: list[pd.DataFrame],
    days: list[date],
    lc: LearningConfig,
    n_trials: int,
    champion: int = 0,
) -> GateResult:
    """The champion/challenger decision on a matrix of daily returns (days x candidates)."""
    from cryptobot.regime import profitable_regimes

    t, k = r.shape
    stats: dict[str, Any] = {"candidates": k - 1, "days": t, "holdout_looks": n_trials}
    if k < 2:
        return GateResult(False, ["no challengers"], None, stats)
    cut = int(round(t * (1.0 - lc.holdout_fraction)))
    sel, hold = r[:cut], r[cut:]
    sel_days, hold_days = set(days[:cut]), set(days[cut:])
    stats["selection_days"], stats["holdout_days"] = len(sel), len(hold)

    # selection: best challenger on the selection window with enough trades there
    sel_trades = [len(_trades_in(tr, sel_days)) for tr in trades]
    min_sel = max(1, int(lc.min_oos_trades * (1 - lc.holdout_fraction) / lc.holdout_fraction))
    scores = np.array([sharpe(sel[:, j]) for j in range(k)])
    eligible = [j for j in range(k) if j != champion and sel_trades[j] >= min_sel]
    if not eligible:
        stats["selection_trades"] = sel_trades
        return GateResult(
            False,
            [f"no challenger had {min_sel}+ trades in the selection window"],
            None,
            stats,
        )
    best = max(eligible, key=lambda j: scores[j])
    stats["best"] = best
    stats["selection_sharpe"] = {
        "challenger": scores[best] * math.sqrt(PERIODS),
        "champion": scores[champion] * math.sqrt(PERIODS),
    }

    # overfitting checks on the selection window
    stats["pbo"] = pbo_cscv(sel, lc.cv_blocks, lc.embargo_days)
    wf = walk_forward(sel, champion, lc.wf_folds, lc.embargo_days)
    stats["walk_forward"] = wf

    # holdout: challenger vs champion
    h_tr_c = _trades_in(trades[best], hold_days)
    h_tr_h = _trades_in(trades[champion], hold_days)
    diff = hold[:, best] - hold[:, champion]
    sel_diffs = [sharpe(sel[:, j] - sel[:, champion]) for j in range(k) if j != champion]
    sr_var = float(np.var(sel_diffs, ddof=1)) if len(sel_diffs) > 1 else 0.0
    dsr = deflated_sharpe(diff, max(1, n_trials), sr_var)
    regimes = profitable_regimes(h_tr_c, lc.min_regime_trades)
    holdout = {
        "challenger": {
            "sharpe": sharpe(hold[:, best]) * math.sqrt(PERIODS),
            "net_return_pct": float((np.prod(1 + hold[:, best]) - 1) * 100),
            "max_drawdown_pct": max_drawdown(hold[:, best]),
            "trades": int(len(h_tr_c)),
            "avg_r": float(h_tr_c["r_multiple"].astype(float).mean()) if len(h_tr_c) else 0.0,
            "profitable_regimes": regimes,
        },
        "champion": {
            "sharpe": sharpe(hold[:, champion]) * math.sqrt(PERIODS),
            "net_return_pct": float((np.prod(1 + hold[:, champion]) - 1) * 100),
            "max_drawdown_pct": max_drawdown(hold[:, champion]),
            "trades": int(len(h_tr_h)),
        },
        "deflated_sharpe": dsr,
    }
    stats["holdout"] = holdout
    c, h = holdout["challenger"], holdout["champion"]

    reasons = []
    if c["trades"] < lc.min_oos_trades:
        reasons.append(f"{c['trades']} holdout trades, needs {lc.min_oos_trades}")
    if c["net_return_pct"] <= 0:
        reasons.append(f"loses money in the holdout after costs ({c['net_return_pct']:+.2f}%)")
    if c["sharpe"] < h["sharpe"] + lc.min_sharpe_gain:
        reasons.append(
            f"holdout Sharpe {c['sharpe']:.2f} < champion {h['sharpe']:.2f} + {lc.min_sharpe_gain}"
        )
    if dsr["dsr"] < lc.dsr_min:
        reasons.append(
            f"deflated Sharpe {dsr['dsr']:.2f} < {lc.dsr_min} after {max(1, n_trials)} holdout"
            " look(s) (the gain could be luck)"
        )
    if c["max_drawdown_pct"] < h["max_drawdown_pct"] - 1e-9:
        reasons.append(
            f"holdout max drawdown {c['max_drawdown_pct']:.2f}% worse than champion"
            f" {h['max_drawdown_pct']:.2f}%"
        )
    if len(regimes) < lc.min_profitable_regimes:
        reasons.append(
            f"profitable in {len(regimes)} regime(s) {regimes}, needs {lc.min_profitable_regimes}"
        )
    if stats["pbo"] is not None and stats["pbo"] > lc.max_pbo:
        reasons.append(f"probability of backtest overfitting {stats['pbo']:.2f} > {lc.max_pbo}")
    if wf["wf_sharpe"] < wf["champion_sharpe"]:
        reasons.append(
            f"walk-forward selection Sharpe {wf['wf_sharpe']:.2f} < keeping the champion"
            f" {wf['champion_sharpe']:.2f}"
        )
    return GateResult(not reasons, reasons, best, stats)


# ============================================================================ the loop


def returns_matrix(sims: list[dict[str, Any]], days: list[date]) -> np.ndarray:
    cols = []
    for s in sims:
        d = s["daily"]
        cols.append(np.array([float(d.get(day, 0.0)) for day in days]))
    return np.column_stack(cols) if cols else np.zeros((len(days), 0))


def run_learning(
    cfg: Config,
    bars: dict[str, pd.DataFrame],
    registry: Registry,
    research_path: str | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
    simulate_fn=None,
    workers: int | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """One learning run. Returns the report (also written to reports/learning/)."""
    now = now or datetime.now(UTC)
    lc = cfg.learning
    champion = registry.champion()
    if champion is None:
        module = load_strategy(cfg.strategy)
        champion = registry.sync_with_config(cfg.strategy, {**default_params(module), **cfg.params})
    days = backtest.all_session_days(bars)
    start = datetime(days[0].year, days[0].month, days[0].day, tzinfo=UTC)
    available = research_coverage(cfg, research_path, start)
    cands = [Candidate.from_model(champion), *generate_challengers(cfg, champion, available)]
    log.info("learning: %d challengers vs champion %s", len(cands) - 1, champion.id)
    if simulate_fn is None:
        sims = simulate_all(cfg, cands, bars, research_path, workers)
    else:
        sims = [simulate_fn(c) for c in cands]
    r = returns_matrix(sims, days)
    # each learning run looks at the (overlapping) holdout once: those looks are the trials the
    # deflated Sharpe corrects for. Picking the best of many challengers is covered by PBO and
    # the walk-forward check on the selection window.
    looks = registry.runs_since(now - timedelta(days=lc.trial_memory_days)) + 1
    result = evaluate(r, [s["trades"] for s in sims], days, lc, looks)
    run_id = uuid.uuid4().hex[:8]
    report: dict[str, Any] = {
        "run_id": run_id,
        "ts": now.isoformat(),
        "champion": champion.describe(),
        "challengers": [c.label for c in cands[1:]],
        "available_research_features": sorted(available),
        "accepted": result.accepted,
        "reasons": result.reasons,
        "stats": result.stats,
        "dry_run": dry_run,
    }
    if result.best is not None:
        report["best"] = cands[result.best].label
    action = "none"
    if not dry_run:
        registry.record_trials(run_id, len(cands) - 1)
        if result.best is not None:
            best = cands[result.best]
            metrics = {
                "holdout": result.stats.get("holdout", {}).get("challenger", {}),
                "gate": result.stats,
                "run_id": run_id,
            }
            if result.accepted and registry.shadows():
                action = "accepted, but a shadow run is already in progress: kept as candidate"
                registry.add(
                    best.strategy,
                    best.params,
                    best.filters,
                    "candidate",
                    "passed the gate; waiting for the current shadow run",
                    metrics,
                    parent=champion.id,
                    source="learning",
                )
            elif result.accepted:
                m = registry.add(
                    best.strategy,
                    best.params,
                    best.filters,
                    "candidate",
                    "passed the gate",
                    metrics,
                    parent=champion.id,
                    source="learning",
                )
                registry.set_status(
                    m.id, "shadow", f"passed the gate in run {run_id}; shadowing", metrics
                )
                action = f"{m.id} starts a {lc.shadow_days}-day shadow run"
            else:
                m = registry.add(
                    best.strategy,
                    best.params,
                    best.filters,
                    "rejected",
                    "; ".join(result.reasons),
                    metrics,
                    parent=champion.id,
                    source="learning",
                )
                if m.status == "candidate":
                    registry.set_status(m.id, "rejected", "; ".join(result.reasons), metrics)
                action = f"best challenger rejected ({m.id})"
    report["action"] = action
    write_report(cfg, report, out_dir, now)
    return report


def write_report(
    cfg: Config, report: dict[str, Any], out_dir: str | Path | None, now: datetime
) -> Path:
    out = Path(out_dir or Path(cfg.reports_dir) / "learning")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{now.date()}_{report['run_id']}"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str))
    st = report["stats"]
    lines = [
        f"# Learning run {report['run_id']} ({now.date()})",
        "",
        f"- Champion: {report['champion']}",
        f"- Challengers tried: {len(report['challengers'])}; holdout looks counted for the"
        f" deflated Sharpe (runs in the last {cfg.learning.trial_memory_days} days):"
        f" {st.get('holdout_looks')}",
        "- Research features with enough history:"
        f" {report['available_research_features'] or 'none'}",
        f"- Best challenger: {report.get('best', 'none')}",
        f"- Verdict: **{'ACCEPTED' if report['accepted'] else 'REJECTED'}**;"
        f" action: {report['action']}",
    ]
    if report["reasons"]:
        lines += ["", "## Why not", ""] + [f"- {r}" for r in report["reasons"]]
    if "holdout" in st:
        h = st["holdout"]
        lines += [
            "",
            f"## Holdout (last {cfg.learning.holdout_fraction:.0%} of days, after costs)",
            "",
            "| | Challenger | Champion |",
            "|---|---|---|",
            f"| Sharpe | {h['challenger']['sharpe']:.2f} | {h['champion']['sharpe']:.2f} |",
            f"| Net return | {h['challenger']['net_return_pct']:+.2f}% |"
            f" {h['champion']['net_return_pct']:+.2f}% |",
            f"| Max drawdown | {h['challenger']['max_drawdown_pct']:.2f}% |"
            f" {h['champion']['max_drawdown_pct']:.2f}% |",
            f"| Trades | {h['challenger']['trades']} | {h['champion']['trades']} |",
            "",
            f"- Deflated Sharpe of the difference: {h['deflated_sharpe']['dsr']:.3f}"
            f" (needs {cfg.learning.dsr_min})",
            f"- Profitable regimes: {h['challenger']['profitable_regimes']}",
        ]
    if st.get("pbo") is not None:
        lines.append(f"- Probability of backtest overfitting: {st['pbo']:.2f}")
    if "walk_forward" in st:
        wf = st["walk_forward"]
        lines.append(
            f"- Walk-forward selection Sharpe {wf['wf_sharpe']:.2f} vs champion"
            f" {wf['champion_sharpe']:.2f}"
        )
    lines += [
        "",
        "Live trading is never changed by this loop. Promotion to live needs"
        " `python -m cryptobot approve-live <version>` by a human.",
    ]
    path = out / f"{stem}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ============================================================================ shadow runs


def _drawdown_from_pnl(pnl: pd.Series, start_equity: float) -> float:
    curve = start_equity + pnl.cumsum()
    curve = pd.concat([pd.Series([start_equity]), curve.reset_index(drop=True)])
    return float(((curve / curve.cummax()) - 1).min() * 100)


def shadow_verdict(
    model: Model,
    shadow_trades: pd.DataFrame,
    mirror_trades: pd.DataFrame,
    lc: LearningConfig,
    start_equity: float,
    now: datetime,
) -> tuple[str, str]:
    """('wait' | 'promote' | 'reject', reason) for a model in its shadow run."""
    started = pd.Timestamp(model.shadow_start).to_pydatetime() if model.shadow_start else now
    age = (now - started).total_seconds() / 86400
    if age < lc.shadow_days:
        return "wait", f"day {age:.0f} of {lc.shadow_days}"
    n = len(shadow_trades)
    if n < lc.min_shadow_trades:
        if age >= lc.shadow_max_days:
            return "reject", f"only {n} trades in {age:.0f} shadow days"
        return "wait", f"only {n} trades so far (needs {lc.min_shadow_trades})"
    s_pnl = float(shadow_trades["pnl"].sum())
    m_pnl = float(mirror_trades["pnl"].sum()) if len(mirror_trades) else 0.0
    s_dd = _drawdown_from_pnl(shadow_trades["pnl"].astype(float), start_equity)
    m_dd = (
        _drawdown_from_pnl(mirror_trades["pnl"].astype(float), start_equity)
        if len(mirror_trades)
        else 0.0
    )
    avg_r = float(shadow_trades["r_multiple"].astype(float).mean())
    expected = float(model.metrics.get("holdout", {}).get("avg_r", 0.0))
    if avg_r < expected - 0.5:
        return "reject", f"shadow avg R {avg_r:+.2f} is below the backtest's {expected:+.2f} - 0.5"
    if s_pnl < m_pnl:
        return "reject", f"shadow P&L {s_pnl:+.2f} lost to the champion's {m_pnl:+.2f}"
    if s_dd < m_dd - 1.0:
        return "reject", f"shadow drawdown {s_dd:.2f}% deeper than the champion's {m_dd:.2f}%"
    return "promote", (
        f"{age:.0f} shadow days, {n} trades: P&L {s_pnl:+.2f} vs champion {m_pnl:+.2f},"
        f" drawdown {s_dd:.2f}% vs {m_dd:.2f}%, avg R {avg_r:+.2f}"
    )


class ShadowRunner:
    """Runs every shadow model, plus a mirror of the champion, in simulation on the live bars.
    Same simulator for both, so the comparison is fair. Trades go to the journal with
    mode='shadow' and the model version."""

    def __init__(self, cfg: Config, registry: Registry, journal_path: str, providers=None):
        self.cfg = cfg
        self.registry = registry
        self.journal_path = journal_path
        self.providers = list(providers or [])
        self.engines: dict[str, Any] = {}

    def _engine(self, model: Model):
        from cryptobot.brokers.paper import PaperBroker
        from cryptobot.engine import Engine
        from cryptobot.journal import Journal
        from cryptobot.risk import RiskManager

        cm = backtest.cost_model(self.cfg)
        risk = RiskManager(
            self.cfg.risk,
            self.cfg.allow_short,
            self.cfg.max_entries_per_day,
            cost_model=cm,
        )
        engine = Engine(
            self.cfg,
            load_strategy(model.strategy),
            model.params,
            PaperBroker(self.cfg.starting_cash, cm),
            risk,
            Journal(self.journal_path, mode="shadow"),
            mode="shadow",
        )
        engine.version = model.id
        engine.filters = dict(model.filters)
        engine.context_providers = list(self.providers)
        return engine

    def sync(self) -> list[str]:
        """Start engines for new shadows and the current champion's mirror; drop old ones."""
        want: dict[str, Model] = {m.id: m for m in self.registry.shadows()}
        champ = self.registry.champion()
        if want and champ is not None:
            want[champ.id] = champ
        for vid in list(self.engines):
            if vid not in want:
                del self.engines[vid]
        started = []
        for vid, model in want.items():
            if vid not in self.engines:
                self.engines[vid] = self._engine(model)
                started.append(vid)
        return started

    def on_bar(self, symbol: str, window: pd.DataFrame) -> None:
        for engine in self.engines.values():
            try:
                engine.on_bar(symbol, window)
            except Exception as e:  # a shadow must never disturb the real bot
                log.warning("shadow %s: %s", engine.version, e)

    def on_tick(self, symbol: str, price: float, ts: datetime) -> None:
        for engine in self.engines.values():
            try:
                engine.on_tick(symbol, price, ts)
            except Exception as e:
                log.warning("shadow %s: %s", engine.version, e)

    def evaluate(self, now: datetime) -> list[tuple[str, str, str]]:
        """Decide on every shadow that has run long enough. Paper promotion only."""
        from cryptobot.journal import Journal

        out = []
        champ = self.registry.champion()
        journal = Journal(self.journal_path, mode="shadow")
        trades = journal.trades(mode="shadow")
        for model in self.registry.shadows():
            since = model.shadow_start or now.isoformat()
            mine = trades[(trades["version"] == model.id) & (trades["entry_ts"] >= since)]
            mirror = (
                trades[(trades["version"] == champ.id) & (trades["entry_ts"] >= since)]
                if champ is not None
                else trades.iloc[0:0]
            )
            verdict, reason = shadow_verdict(
                model, mine, mirror, self.cfg.learning, self.cfg.starting_cash, now
            )
            if verdict == "promote":
                self.registry.promote_to_paper(model.id, f"shadow run passed: {reason}")
            elif verdict == "reject":
                self.registry.set_status(model.id, "rejected", f"shadow run failed: {reason}")
            out.append((model.id, verdict, reason))
        self.sync()
        return out
