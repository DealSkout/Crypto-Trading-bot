"""Section 3: regime detection, from completed days only, and results tracked per regime."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from cryptobot.backtest import run_backtest
from cryptobot.config import RegimeConfig
from cryptobot.regime import RegimeDetector, daily_bars, profitable_regimes, regime_performance


def intraday_from_daily(closes, start="2026-08-01", symbol="BTC/USD", bars_per_day=4):
    """A few bars per day that end each day at the given close."""
    rows, idx = [], []
    t0 = pd.Timestamp(start, tz="UTC")
    prev = closes[0]
    for d, c in enumerate(closes):
        path = np.linspace(prev, c, bars_per_day + 1)[1:]
        o = prev
        for i, p in enumerate(path):
            rows.append(
                {
                    "open": o,
                    "high": max(o, p) * 1.001,
                    "low": min(o, p) * 0.999,
                    "close": p,
                    "volume": 1.0,
                }
            )
            idx.append(t0 + timedelta(days=d, hours=6 * i))
            o = p
        prev = c
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="ts"))
    df.attrs.update({"symbol": symbol, "timeframe": 360})
    return df


def day_n(n, start="2026-08-01"):
    return (pd.Timestamp(start) + timedelta(days=n)).date()


def test_trending_and_ranging():
    up = [100 * 1.01**i for i in range(40)]  # steady climb
    chop = [100 + (1 if i % 2 else -1) for i in range(40)]  # back and forth
    det = RegimeDetector.from_bars(
        RegimeConfig(), {"BTC/USD": intraday_from_daily(up), "ETH/USD": intraday_from_daily(chop)}
    )
    assert det.classify("BTC/USD", day_n(35))["regime"] == "trending"
    assert det.classify("ETH/USD", day_n(35))["regime"] == "ranging"


def test_high_volatility():
    rng = np.random.default_rng(1)
    calm = list(100 * np.exp(np.cumsum(rng.normal(0, 0.005, 33))))
    wild = calm + [calm[-1] * f for f in (1.08, 0.95, 1.07, 0.94, 1.06)]
    det = RegimeDetector.from_bars(RegimeConfig(), {"BTC/USD": intraday_from_daily(wild)})
    assert det.classify("BTC/USD", day_n(30))["regime"] != "high_vol"
    assert det.classify("BTC/USD", day_n(38))["regime"] == "high_vol"


def test_risk_off_needs_weak_breadth_and_a_falling_market():
    down = [100 * 0.99**i for i in range(40)]
    det = RegimeDetector.from_bars(
        RegimeConfig(),
        {s: intraday_from_daily(down, symbol=s) for s in ("BTC/USD", "ETH/USD", "SOL/USD")},
    )
    info = det.classify("SOL/USD", day_n(35))
    assert info["regime"] == "risk_off" and info["regime_breadth"] == 0.0
    # extreme fear plus half the markets below average also means risk off
    flat = [100 + (0.2 if i % 2 else -0.2) for i in range(40)]
    det2 = RegimeDetector.from_bars(
        RegimeConfig(),
        {"BTC/USD": intraday_from_daily(down), "ETH/USD": intraday_from_daily(flat)},
        fear_greed=lambda when: 12.0,
    )
    assert det2.classify("ETH/USD", day_n(35))["regime"] == "risk_off"


def test_not_enough_history_is_unknown():
    det = RegimeDetector.from_bars(RegimeConfig(), {"BTC/USD": intraday_from_daily([100] * 10)})
    assert det.classify("BTC/USD", day_n(9))["regime"] == "unknown"


def test_no_look_ahead_a_days_label_ignores_that_day_and_later():
    closes = [100 * 1.01**i for i in range(40)]
    a = intraday_from_daily(closes)
    b = intraday_from_daily(closes[:35] + [c * 0.5 for c in closes[35:]])  # crash from day 35
    da = RegimeDetector.from_bars(RegimeConfig(), {"BTC/USD": a})
    db = RegimeDetector.from_bars(RegimeConfig(), {"BTC/USD": b})
    assert da.classify("BTC/USD", day_n(35)) == db.classify("BTC/USD", day_n(35))


def test_live_updates_add_completed_days_only():
    closes = [100 * 1.01**i for i in range(30)]
    full = intraday_from_daily(closes)
    det = RegimeDetector(RegimeConfig(), ["BTC/USD"])
    window = full.iloc[: 25 * 4 + 2]  # 25 full days and 2 bars of day 26
    det.update_from_bars("BTC/USD", window)
    assert det.daily["BTC/USD"].index[-1].date() == day_n(24)
    assert len(daily_bars(window)) == 26


def test_performance_by_regime_and_profitable_regimes():
    trades = pd.DataFrame(
        {
            "regime": ["trending"] * 6 + ["ranging"] * 6 + ["high_vol"] * 2,
            "pnl": [5, 3, -1, 4, 2, 1] + [-2, -1, 1, -3, -1, -2] + [10, 12],
            "r_multiple": [1.0] * 14,
            "strategy": ["orb"] * 14,
        }
    )
    perf = regime_performance(trades, by=["strategy"])
    row = perf[perf["regime"] == "trending"].iloc[0]
    assert row["trades"] == 6 and row["net_pnl"] == 14
    # high_vol is profitable but has too few trades to count
    assert profitable_regimes(trades, min_trades=5) == ["trending"]


def test_backtests_label_trades_with_their_regime(cfg, bars):
    r = run_backtest(cfg, bars)
    assert "regime" in r.trades.columns
    assert set(r.trades["regime"]) <= {"trending", "ranging", "high_vol", "risk_off", "unknown"}
    assert sum(row["trades"] for row in r.regime_table) == r.metrics["n_trades"]


@pytest.mark.parametrize("bad", [{"vol_short_days": 20, "vol_long_days": 10}, {"er_days": 1}])
def test_config_validation(cfg, bad):
    from cryptobot.config import ConfigError, validate

    for k, v in bad.items():
        setattr(cfg.regime, k, v)
    with pytest.raises(ConfigError):
        validate(cfg)


def test_classify_uses_dates_not_bar_positions():
    det = RegimeDetector.from_bars(RegimeConfig(), {"BTC/USD": intraday_from_daily([100] * 30)})
    assert det.classify("BTC/USD", date(2026, 8, 1))["regime_days"] == 0
