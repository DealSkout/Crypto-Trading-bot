from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
import yaml

from cryptobot import sessions
from cryptobot.config import from_dict
from cryptobot.data import get_bars

NOW = datetime(2026, 10, 16, 21, 30, tzinfo=UTC)  # a Friday evening, after the US close


def base_raw(tmp_path) -> dict:
    return {
        "mode": "paper",
        "broker": "paper",
        "symbols": ["BTC/USD", "ETH/USD"],
        "timeframe": "5m",
        "strategy": "orb",
        "params": {"range_minutes": 5, "target_r": 10},
        "starting_cash": 1000,
        "data": {
            "provider": "synthetic",
            "cache_dir": str(tmp_path / "cache"),
            "csv_dir": str(tmp_path / "data"),
            "lookback_days": 40,
        },
        "learning": {"min_oos_trades": 5, "min_sharpe_gain": 0.1},
        "reports_dir": str(tmp_path / "reports"),
        "journal_path": str(tmp_path / "journal.sqlite"),
        "coach": {"dir": str(tmp_path / "coach")},
        "telegram": False,
    }


@pytest.fixture
def cfg(tmp_path):
    raw = base_raw(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text("# test config\n" + yaml.safe_dump(raw, sort_keys=False))
    return from_dict(raw, path=str(path))


@pytest.fixture
def bars(cfg):
    return {
        s: get_bars(cfg, s, provider="synthetic", lookback_days=40, now=NOW) for s in cfg.symbols
    }


def bars_from_closes(
    closes, day="2026-10-05", symbol="BTC/USD", tf=5, wick=0.0, start_minute=0, first_open=None
):
    """Hand-built session bars: each bar opens at the previous close."""
    market = sessions.market_for(symbol)
    t0 = sessions.session_open(market, pd.Timestamp(day).date()) + timedelta(minutes=start_minute)
    rows, prev = [], (closes[0] if first_open is None else first_open)
    for c in closes:
        o = prev
        rows.append(
            {
                "open": o,
                "high": max(o, c) + wick,
                "low": min(o, c) - wick,
                "close": c,
                "volume": 1000.0,
            }
        )
        prev = c
    idx = pd.DatetimeIndex([t0 + timedelta(minutes=i * tf) for i in range(len(closes))], name="ts")
    df = pd.DataFrame(rows, index=idx)
    df.attrs.update({"symbol": symbol, "market": market.name, "timeframe": tf})
    return df


class StubStrategy:
    """Fires once per session at bar number `fire_at` (1-based) with a % stop/target."""

    NAME = "stub"
    DEFAULT_PARAMS: dict = {}

    def __init__(self, fire_at=2, side="long", stop_pct=1.0, target_pct=None):
        self.fire_at, self.side, self.stop_pct, self.target_pct = (
            fire_at,
            side,
            stop_pct,
            target_pct,
        )

    def signal(self, bars, params):
        from cryptobot.strategies import Signal
        from cryptobot.strategies.common import today_bars

        today = today_bars(bars)
        if len(today) != self.fire_at:
            return None
        entry = float(today["close"].iloc[-1])
        sgn = 1 if self.side == "long" else -1
        stop = entry * (1 - sgn * self.stop_pct / 100)
        target = entry * (1 + sgn * self.target_pct / 100) if self.target_pct else None
        return Signal(self.side, entry, stop, target, reason="stub")
