"""Layer 1 of the learning loop: deterministic walk-forward re-fit plus the gate.

The gate is the only way any parameter change reaches config.yaml, whether it came from the
grid search here or from the AI coach. Risk settings are never touched by code (RISK_KEYS).
"""

from __future__ import annotations

import itertools
import logging
from datetime import date
from types import ModuleType
from typing import Any

import pandas as pd

from cryptobot import backtest, config
from cryptobot.config import RISK_KEYS, Config
from cryptobot.journal import Journal
from cryptobot.strategies import default_params, load_strategy, param_grid, strategy_name

log = logging.getLogger(__name__)


class ProposalError(ValueError):
    pass


def param_combinations(grid: dict[str, list]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = sorted(grid)
    return [
        dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(grid[k] for k in keys))
    ]


def evaluate(
    cfg: Config,
    strategy: ModuleType,
    params: dict[str, Any],
    bars_by_symbol: dict[str, pd.DataFrame],
    days: list[date],
) -> dict[str, Any]:
    _, trades, equity = backtest.run_period(cfg, strategy, params, bars_by_symbol, days)
    return backtest.compute_metrics(trades, equity, cfg.starting_cash)


def grid_search(
    cfg: Config,
    strategy: ModuleType,
    bars_by_symbol: dict[str, pd.DataFrame],
    train_days: list[date],
    base_params: dict[str, Any],
    min_trades: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pick the grid point with the best in-sample Sharpe (ties: fewest changes from base).
    Returns (best_params, table of every point tried)."""
    table = []
    best, best_key = dict(base_params), None
    for combo in param_combinations(param_grid(strategy)):
        params = {**base_params, **combo}
        m = evaluate(cfg, strategy, params, bars_by_symbol, train_days)
        qualifies = m["n_trades"] >= min_trades
        changes = sum(1 for k, v in combo.items() if base_params.get(k) != v)
        table.append({"params": params, **m, "qualifies": qualifies})
        key = (m["sharpe"], -changes)
        if qualifies and (best_key is None or key > best_key):
            best, best_key = params, key
    return best, table


def gate(
    champion: dict[str, Any],
    challenger: dict[str, Any],
    min_oos_trades: int,
    min_sharpe_gain: float,
) -> tuple[bool, str]:
    """The champion/challenger rule. Both metric dicts are out-of-sample, after costs."""
    if challenger.get("n_trades", 0) < min_oos_trades:
        return False, (
            f"challenger has {challenger.get('n_trades', 0)} out-of-sample trades,"
            f" needs {min_oos_trades}"
        )
    if challenger.get("net_return_pct", 0.0) <= 0:
        return False, "challenger loses money out of sample after costs"
    needed = float(champion.get("sharpe", 0.0)) + float(min_sharpe_gain)
    if float(challenger.get("sharpe", 0.0)) < needed:
        return False, (
            f"challenger Sharpe {challenger.get('sharpe', 0.0):.2f} < champion"
            f" {champion.get('sharpe', 0.0):.2f} + {min_sharpe_gain}"
        )
    return True, "challenger beats champion out of sample after costs"


def check_proposal_params(params: dict[str, Any]) -> None:
    """Rule 4 guard: nothing automated may touch risk settings, and params must be plain."""
    touched = set(params) & RISK_KEYS
    if touched:
        raise ProposalError(f"proposal tries to change risk settings {sorted(touched)}")
    for k, v in params.items():
        if not isinstance(k, str) or not isinstance(v, int | float | str | bool | type(None)):
            raise ProposalError(f"param {k!r} must be a number, string or bool")


def evaluate_candidate(
    cfg: Config,
    bars_by_symbol: dict[str, pd.DataFrame],
    new_params: dict[str, Any],
    strategy: str | ModuleType | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Out-of-sample champion (current params) vs challenger (new params) through the gate."""
    module = (
        strategy if isinstance(strategy, ModuleType) else load_strategy(strategy or cfg.strategy)
    )
    check_proposal_params(new_params)
    base = {**default_params(module), **cfg.params}
    candidate = {**base, **new_params}
    days = backtest.all_session_days(bars_by_symbol)
    _, test_days = backtest.split_days(days, cfg.learning.train_fraction)
    champ = evaluate(cfg, module, base, bars_by_symbol, test_days)
    chall = evaluate(cfg, module, candidate, bars_by_symbol, test_days)
    ok, reason = gate(champ, chall, cfg.learning.min_oos_trades, cfg.learning.min_sharpe_gain)
    stats = {"champion": champ, "challenger": chall, "params": candidate, "reason": reason}
    return ok, reason, stats


def apply_params(
    cfg: Config, journal: Journal, old: dict, new: dict, stats: dict, source: str, applied: bool
) -> None:
    journal.add_params_change(cfg.strategy, old, new, stats, source, applied)
    if applied:
        config.write_params(cfg.path, new)
        cfg.params = dict(new)
        log.info("applied new params from %s: %s", source, new)


def refit(
    cfg: Config,
    bars_by_symbol: dict[str, pd.DataFrame],
    journal: Journal,
    apply: bool = True,
) -> dict[str, Any]:
    """Grid-search on the training window, then challenge the current params out of sample."""
    module = load_strategy(cfg.strategy)
    base = {**default_params(module), **cfg.params}
    days = backtest.all_session_days(bars_by_symbol)
    train_days, test_days = backtest.split_days(days, cfg.learning.train_fraction)
    best, table = grid_search(cfg, module, bars_by_symbol, train_days, base)
    champ = evaluate(cfg, module, base, bars_by_symbol, test_days)
    if best == base:
        return {
            "changed": False,
            "reason": "grid search kept the current params",
            "params": base,
            "champion": champ,
            "table": table,
        }
    chall = evaluate(cfg, module, best, bars_by_symbol, test_days)
    ok, reason = gate(champ, chall, cfg.learning.min_oos_trades, cfg.learning.min_sharpe_gain)
    stats = {"champion": champ, "challenger": chall, "reason": reason}
    apply_params(cfg, journal, base, best, stats, "refit", applied=ok and apply)
    return {
        "changed": ok and apply,
        "accepted": ok,
        "reason": reason,
        "params": best,
        "old_params": base,
        "champion": champ,
        "challenger": chall,
        "table": table,
        "strategy": strategy_name(module),
    }


def decider_description(cfg: Config) -> str:
    d = cfg.decider
    if d.provider == "none":
        return "none"
    model = d.model if d.provider == "claude" else d.ollama_model
    return f"{d.provider}:{model}:{d.role}"


def paper_days_with_trades(journal: Journal) -> int:
    daily = journal.daily()
    return int((daily["trades"] > 0).sum()) if len(daily) else 0


def live_gate(
    cfg: Config, journal: Journal, result: backtest.BacktestResult | None, min_paper_days: int = 20
) -> tuple[bool, list[str]]:
    """Rule 6: live needs (a) the out-of-sample backtest, after costs, to beat buy-and-hold on
    Sharpe and (b) at least `min_paper_days` paper days with results within the backtest's
    expected range (average R not more than 0.5R below the backtest, and no more than two
    daily-loss halts)."""
    problems = []
    if result is None:
        problems.append("no saved backtest for this strategy: run `python -m cryptobot backtest`")
    else:
        if not result.beats_buy_and_hold:
            problems.append("out-of-sample backtest does not beat buy-and-hold on Sharpe")
        if result.params != {**default_params(load_strategy(cfg.strategy)), **cfg.params}:
            problems.append("config params differ from the backtested params; re-run the backtest")
        if result.decider != decider_description(cfg):
            problems.append(
                f"backtest used decider '{result.decider}' but config says"
                f" '{decider_description(cfg)}'; re-run the backtest (--llm)"
            )
    paper = Journal(journal.path, mode="paper") if journal.mode != "paper" else journal
    daily = paper.daily()
    days = int((daily["trades"] > 0).sum()) if len(daily) else 0
    if days < min_paper_days:
        problems.append(f"only {days} paper trading days with trades, need {min_paper_days}")
    trades = paper.trades(mode="paper")
    if result is not None and len(trades):
        paper_r = float(trades["r_multiple"].mean())
        expected = float(result.metrics.get("avg_r", 0.0))
        if paper_r < expected - 0.5:
            problems.append(f"paper avg R {paper_r:.2f} is below backtest {expected:.2f} - 0.5")
    if len(daily) and int(daily["halted"].sum()) > 2:
        problems.append("more than two daily-loss halts in paper trading")
    return not problems, problems
