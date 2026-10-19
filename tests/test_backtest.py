import numpy as np
import pandas as pd
import pytest

from cryptobot.backtest import (
    BacktestResult,
    buy_and_hold,
    compute_metrics,
    run_backtest,
    split_days,
)


def test_split_days():
    days = list(range(10))
    train, test = split_days(days, 0.7)
    assert train == list(range(7)) and test == [7, 8, 9]
    assert split_days([1], 0.7) == ([1], [1])


def test_compute_metrics_known_series():
    idx = pd.date_range("2026-01-05", periods=3, freq="D", tz="UTC")
    equity = pd.Series([100.0, 110.0, 99.0], index=idx)
    trades = pd.DataFrame({"pnl": [10.0, -11.0], "r_multiple": [1.0, -1.1], "costs": [0.5, 0.5]})
    m = compute_metrics(trades, equity, 100.0)
    assert m["net_return_pct"] == pytest.approx(-1.0)
    assert m["max_drawdown_pct"] == pytest.approx(-10.0)
    assert m["n_trades"] == 2 and m["win_rate_pct"] == 50.0 and m["total_costs"] == 1.0
    assert np.isfinite(m["sharpe"]) and m["trading_days"] == 3


def test_run_backtest_end_to_end(cfg, bars, tmp_path):
    r = run_backtest(cfg, bars)
    assert r.start > r.train_end and r.strategy == "orb"
    assert set(r.benchmarks) == {"buy_and_hold", "random_entry"}
    for key in ("net_return_pct", "max_drawdown_pct", "sharpe", "n_trades", "total_costs"):
        assert key in r.metrics and key in r.benchmarks["buy_and_hold"]
    assert r.metrics["n_trades"] > 0 and r.metrics["total_costs"] > 0  # costs always on
    assert len(r.equity) > 0 and r.equity.index.tz is not None
    path = r.save(tmp_path / "reports")
    back = BacktestResult.load(path)
    assert back.metrics == r.metrics and len(back.trades) == len(r.trades)
    assert back.beats_buy_and_hold == r.beats_buy_and_hold


def test_buy_and_hold_pays_costs(cfg, bars):
    days = sorted({d for b in bars.values() for d in b.index.date})[-5:]
    m, eq = buy_and_hold(cfg, bars, days)
    assert m["n_trades"] == 2 and m["total_costs"] > 0 and len(eq) > 0
