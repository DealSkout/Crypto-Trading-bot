"""Event-driven backtester: the same Engine as paper/live, fed historical bars through the
paper broker. Walk-forward: the first `train_fraction` of days is in-sample (used only to pick
parameters), the rest is out-of-sample and is what every report headlines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from cryptobot import sessions
from cryptobot.brokers.paper import PaperBroker
from cryptobot.config import Config
from cryptobot.costs import CostModel
from cryptobot.engine import Engine
from cryptobot.journal import Journal
from cryptobot.risk import RiskManager
from cryptobot.strategies import default_params, load_strategy, strategy_name

HISTORY_SESSIONS = 25  # how many sessions of history a strategy sees
PERIODS_PER_YEAR = 365  # crypto trades every day of the year: annualise daily returns by 365
TRADING_DAYS_PER_YEAR = PERIODS_PER_YEAR  # older name, kept for compatibility


def cost_model(cfg: Config) -> CostModel:
    return CostModel(**asdict(cfg.costs))


def fx_applies(cfg: Config, symbol: str) -> bool:
    """Whether the percentage fee applies. Crypto trades always pay the exchange fee."""
    return True


def result_currency(cfg: Config) -> str:
    currencies = {sessions.currency_for(s) for s in cfg.symbols}
    return currencies.pop() if len(currencies) == 1 else "mixed"


@dataclass
class BacktestResult:
    strategy: str
    params: dict[str, Any]
    symbols: list[str]
    timeframe: str
    train_start: str
    train_end: str
    start: str
    end: str
    starting_cash: float
    currency: str
    metrics: dict[str, Any]
    train_metrics: dict[str, Any]
    benchmarks: dict[str, dict[str, Any]]
    costs: dict[str, Any]
    trades: pd.DataFrame
    equity: pd.Series
    bench_equity: dict[str, pd.Series]
    params_history: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    decider: str = "none"
    regime_table: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def beats_buy_and_hold(self) -> bool:
        bh = self.benchmarks.get("buy_and_hold", {})
        return float(self.metrics.get("sharpe", 0)) > float(bh.get("sharpe", 0))

    @property
    def beats_random(self) -> bool:
        rnd = self.benchmarks.get("random_entry", {})
        return float(self.metrics.get("sharpe", 0)) > float(rnd.get("sharpe", 0))

    def to_json(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k not in ("trades", "equity", "bench_equity")}
        d["beats_buy_and_hold"] = self.beats_buy_and_hold
        d["beats_random"] = self.beats_random
        return d

    def save(self, reports_dir: str | Path, stem: str | None = None) -> Path:
        out = Path(reports_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = stem or f"latest_{self.strategy}"
        (out / f"{stem}.json").write_text(json.dumps(self.to_json(), indent=2, default=str))
        self.trades.to_parquet(out / f"{stem}_trades.parquet")
        curves = pd.DataFrame({"strategy": self.equity, **self.bench_equity})
        curves.to_parquet(out / f"{stem}_equity.parquet")
        return out / f"{stem}.json"

    @classmethod
    def load(cls, path: str | Path) -> BacktestResult:
        path = Path(path)
        d = json.loads(path.read_text())
        d.pop("beats_buy_and_hold", None)
        d.pop("beats_random", None)
        stem = path.stem
        trades = pd.read_parquet(path.with_name(f"{stem}_trades.parquet"))
        curves = pd.read_parquet(path.with_name(f"{stem}_equity.parquet"))
        equity = curves["strategy"].dropna()
        bench = {c: curves[c].dropna() for c in curves.columns if c != "strategy"}
        return cls(trades=trades, equity=equity, bench_equity=bench, **d)


# ----------------------------------------------------------------------------- helpers


def all_session_days(bars_by_symbol: dict[str, pd.DataFrame]) -> list[date]:
    days: set[date] = set()
    for bars in bars_by_symbol.values():
        days.update(bars.index.date)
    return sorted(days)


def split_days(days: list[date], train_fraction: float) -> tuple[list[date], list[date]]:
    if len(days) < 2:
        return list(days), list(days)
    k = int(round(len(days) * train_fraction))
    k = min(max(k, 1), len(days) - 1)
    return list(days[:k]), list(days[k:])


def _utc_index(timestamps) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(t).tz_convert("UTC") for t in timestamps])


def run_period(
    cfg: Config,
    strategy: ModuleType,
    params: dict[str, Any],
    bars_by_symbol: dict[str, pd.DataFrame],
    days: list[date],
    journal: Journal | None = None,
    decider=None,
    context_providers: list | None = None,
    filters: dict[str, Any] | None = None,
) -> tuple[Engine, pd.DataFrame, pd.Series]:
    """Drive the engine over `days`, giving each strategy call up to HISTORY_SESSIONS of
    history before the first active day. Returns (engine, trades, equity curve in UTC)."""
    cm = cost_model(cfg)
    broker = PaperBroker(cfg.starting_cash, cm, fx_for=lambda s: fx_applies(cfg, s))
    risk = RiskManager(
        cfg.risk,
        cfg.allow_short,
        cfg.max_entries_per_day,
        cost_model=cm,
        fx_for=lambda s: fx_applies(cfg, s),
    )
    engine = Engine(
        cfg, strategy, params, broker, risk, journal or Journal(":memory:"), decider=decider
    )
    engine.context_providers = list(context_providers or [])
    engine.filters = dict(filters or {})
    active = set(days)
    events: list[tuple[pd.Timestamp, str, int]] = []
    arrays: dict[str, tuple] = {}
    for symbol, bars in bars_by_symbol.items():
        dates = bars.index.date
        idx_utc = bars.index.tz_convert("UTC")
        arrays[symbol] = (
            bars["open"].to_numpy(dtype=float),
            bars["high"].to_numpy(dtype=float),
            bars["low"].to_numpy(dtype=float),
            bars["close"].to_numpy(dtype=float),
            idx_utc.to_pydatetime(),
        )
        for i in np.flatnonzero(np.isin(dates, list(active))):
            events.append((idx_utc[i], symbol, int(i)))
    events.sort(key=lambda e: (e[0], e[1]))
    hist = {
        s: HISTORY_SESSIONS * (sessions.market_for(s).session_minutes // cfg.timeframe_minutes)
        for s in bars_by_symbol
    }
    tf = int(cfg.timeframe_minutes)

    def window_for(symbol: str, i: int):
        def build() -> pd.DataFrame:
            bars = bars_by_symbol[symbol]
            w = bars.iloc[max(0, i + 1 - hist[symbol]) : i + 1]
            w.attrs = {**bars.attrs, "symbol": symbol, "timeframe": tf}
            return w

        return build

    last_ts = None
    for _ts, symbol, i in events:
        o, h, lo, c, stamps = arrays[symbol]
        last_ts = stamps[i]
        engine.process_bar(symbol, last_ts, tf, o[i], h[i], lo[i], c[i], window_for(symbol, i))
    engine.end_of_data(last_ts)
    trades = pd.DataFrame([t.to_dict() for t in engine.trades])
    if trades.empty:
        trades = pd.DataFrame(
            columns=[
                "symbol",
                "side",
                "qty",
                "entry_ts",
                "entry_price",
                "exit_ts",
                "exit_price",
                "stop",
                "target",
                "pnl",
                "r_multiple",
                "costs",
                "exit_reason",
                "signal_reason",
            ]
        )
    if engine.equity_curve:
        equity = pd.Series(
            [e for _, e in engine.equity_curve],
            index=_utc_index([t for t, _ in engine.equity_curve]),
            name="equity",
        )
        equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    else:
        equity = pd.Series(
            [cfg.starting_cash], index=pd.DatetimeIndex([pd.Timestamp.now(tz="UTC")]), name="equity"
        )
    return engine, trades, equity


def daily_returns(equity: pd.Series, starting_cash: float) -> pd.Series:
    """Daily returns from an equity curve, counting the first day against the starting cash
    (so the first day's result is not silently dropped)."""
    if not len(equity):
        return pd.Series(dtype=float)
    daily = equity.groupby(equity.index.normalize()).last()
    first = daily.index[0] - pd.Timedelta(days=1)
    daily = pd.concat([pd.Series([float(starting_cash)], index=[first]), daily])
    return daily.pct_change().dropna()


def sharpe_sortino(rets: pd.Series, periods: int = PERIODS_PER_YEAR) -> tuple[float, float]:
    """Annualised Sharpe and Sortino of per-period returns (0 when undefined)."""
    if len(rets) < 2:
        return 0.0, 0.0
    mean = float(rets.mean())
    std = float(rets.std(ddof=1))
    downside = float(np.sqrt(np.mean(np.minimum(rets.to_numpy(dtype=float), 0.0) ** 2)))
    sharpe = mean / std * np.sqrt(periods) if std > 0 else 0.0
    sortino = mean / downside * np.sqrt(periods) if downside > 0 else 0.0
    return float(sharpe), float(sortino)


def max_drawdown_pct(equity: pd.Series, starting_cash: float | None = None) -> float:
    if not len(equity):
        return 0.0
    curve = equity.astype(float)
    if starting_cash is not None:
        curve = pd.concat([pd.Series([float(starting_cash)]), curve.reset_index(drop=True)])
    running_max = curve.cummax()
    return float(((curve / running_max) - 1.0).min() * 100.0)


def compute_metrics(
    trades: pd.DataFrame, equity: pd.Series, starting_cash: float
) -> dict[str, Any]:
    final = float(equity.iloc[-1]) if len(equity) else float(starting_cash)
    rets = daily_returns(equity, starting_cash)
    sharpe, sortino = sharpe_sortino(rets)
    drawdown = max_drawdown_pct(equity, starting_cash)
    n = int(len(trades))
    pnl = trades["pnl"].astype(float) if n else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    profit_factor = (
        float(wins.sum() / -losses.sum())
        if len(losses) and losses.sum() < 0
        else (float("inf") if len(wins) else 0.0)
    )
    return {
        "net_return_pct": (final / starting_cash - 1.0) * 100.0,
        "final_equity": final,
        "max_drawdown_pct": drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "n_trades": n,
        "win_rate_pct": float((pnl > 0).mean() * 100.0) if n else 0.0,
        "avg_r": float(trades["r_multiple"].astype(float).mean()) if n else 0.0,
        "avg_pnl": float(pnl.mean()) if n else 0.0,
        "profit_factor": profit_factor,
        "total_costs": float(trades["costs"].astype(float).sum()) if n else 0.0,
        "trading_days": int(len(rets)),
    }


def buy_and_hold(
    cfg: Config, bars_by_symbol: dict[str, pd.DataFrame], days: list[date]
) -> tuple[dict[str, Any], pd.Series]:
    """Equal-weight buy at the first open of the window, sell at the last close, with costs."""
    cm = cost_model(cfg)
    active = set(days)
    per_symbol = cfg.starting_cash / max(1, len(bars_by_symbol))
    curves, rows = [], []
    for symbol, bars in bars_by_symbol.items():
        window = bars[np.isin(bars.index.date, list(active))]
        if window.empty:
            continue
        fx = fx_applies(cfg, symbol)
        buy = cm.fill_price(float(window["open"].iloc[0]), True)
        qty = per_symbol / buy
        fee_in = cm.fees(buy * qty, fx=fx)
        qty = (per_symbol - fee_in) / buy
        sell = cm.fill_price(float(window["close"].iloc[-1]), False)
        fee_out = cm.fees(sell * qty, fx=fx)
        value = qty * window["close"].astype(float)
        value.index = _utc_index(window.index)
        value.iloc[-1] = sell * qty - fee_out
        curves.append(value)
        implicit = (
            abs(buy - float(window["open"].iloc[0])) * qty
            + abs(sell - float(window["close"].iloc[-1])) * qty
        )
        rows.append(
            {
                "symbol": symbol,
                "side": "long",
                "qty": qty,
                "entry_ts": window.index[0],
                "entry_price": buy,
                "exit_ts": window.index[-1],
                "exit_price": sell,
                "stop": np.nan,
                "target": np.nan,
                "pnl": sell * qty - fee_out - per_symbol,
                "r_multiple": np.nan,
                "costs": fee_in + fee_out + implicit,
                "exit_reason": "end",
                "signal_reason": "buy and hold",
            }
        )
    if not curves:
        idx = pd.DatetimeIndex([pd.Timestamp.now(tz="UTC")])
        return compute_metrics(
            pd.DataFrame(), pd.Series([cfg.starting_cash], index=idx), cfg.starting_cash
        ), pd.Series(dtype=float)
    total = pd.concat(curves, axis=1).sort_index().ffill().bfill().sum(axis=1)
    unused = cfg.starting_cash - per_symbol * len(curves)
    equity = (total + unused).rename("buy_and_hold")
    trades = pd.DataFrame(rows)
    metrics = compute_metrics(trades, equity, cfg.starting_cash)
    metrics["avg_r"] = 0.0
    return metrics, equity


def random_params_from(trades: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """Mirror the strategy's typical stop distance and target multiple so only the entry timing
    is random."""
    params: dict[str, Any] = {"seed": seed, "side": "long"}
    if len(trades):
        entry = trades["entry_price"].astype(float)
        stop_pct = ((entry - trades["stop"].astype(float)).abs() / entry * 100.0).median()
        params["stop_pct"] = float(max(0.05, stop_pct))
        targets = trades["target"]
        if targets.notna().any():
            dist = (entry - trades["stop"].astype(float)).abs()
            r = ((targets.astype(float) - entry).abs() / dist).replace([np.inf], np.nan).dropna()
            if len(r):
                params["target_r"] = float(r.median())
    return params


def run_backtest(
    cfg: Config,
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy: str | ModuleType | None = None,
    params: dict[str, Any] | None = None,
    seed: int = 0,
    journal: Journal | None = None,
    notes: list[str] | None = None,
    decider=None,
    context_providers: list | None = None,
) -> BacktestResult:
    module = (
        strategy if isinstance(strategy, ModuleType) else load_strategy(strategy or cfg.strategy)
    )
    base = {**default_params(module), **(cfg.params if params is None else params)}
    days = all_session_days(bars_by_symbol)
    if not days:
        raise ValueError("no bars to backtest")
    train_days, test_days = split_days(days, cfg.learning.train_fraction)
    from cryptobot.regime import RegimeDetector, regime_performance

    regimes = RegimeDetector.from_bars(cfg.regime, bars_by_symbol)
    providers = [regimes.context, *(context_providers or [])]

    _, is_trades, is_equity = run_period(
        cfg, module, base, bars_by_symbol, train_days, context_providers=providers
    )
    oos_engine, oos_trades, oos_equity = run_period(
        cfg,
        module,
        base,
        bars_by_symbol,
        test_days,
        journal,
        decider=decider,
        context_providers=providers,
    )
    metrics = compute_metrics(oos_trades, oos_equity, cfg.starting_cash)
    metrics["cash_capped_entries"] = oos_engine.capped_entries
    metrics["skipped_signals"] = oos_engine.skipped
    notes = list(notes or [])
    if decider is not None:
        metrics["llm_calls"] = decider.calls
        metrics["llm_cost_usd"] = round(decider.spend, 4)
        notes.append(
            f"out-of-sample period run with the LLM decider {decider.describe()}:"
            f" {decider.calls} calls, {decider.spend:.2f} USD; the in-sample column is rules"
            " only. LLM answers are not deterministic: re-runs will differ."
        )
    train_metrics = compute_metrics(is_trades, is_equity, cfg.starting_cash)

    bh_metrics, bh_equity = buy_and_hold(cfg, bars_by_symbol, test_days)
    rnd_module = load_strategy("random_entry")
    rnd_params = {**default_params(rnd_module), **random_params_from(oos_trades, seed)}
    _, rnd_trades, rnd_equity = run_period(cfg, rnd_module, rnd_params, bars_by_symbol, test_days)
    rnd_metrics = compute_metrics(rnd_trades, rnd_equity, cfg.starting_cash)

    return BacktestResult(
        strategy=strategy_name(module),
        params=base,
        symbols=list(bars_by_symbol),
        timeframe=cfg.timeframe,
        train_start=train_days[0].isoformat(),
        train_end=train_days[-1].isoformat(),
        start=test_days[0].isoformat(),
        end=test_days[-1].isoformat(),
        starting_cash=cfg.starting_cash,
        currency=result_currency(cfg),
        metrics=metrics,
        train_metrics=train_metrics,
        benchmarks={"buy_and_hold": bh_metrics, "random_entry": rnd_metrics},
        costs=asdict(cfg.costs),
        trades=oos_trades,
        equity=oos_equity,
        bench_equity={"buy_and_hold": bh_equity, "random_entry": rnd_equity},
        notes=notes,
        decider=decider.describe() if decider is not None else "none",
        regime_table=regime_performance(oos_trades).to_dict("records"),
    )
