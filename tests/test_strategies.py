import pytest

from cryptobot.strategies import Signal, default_params, load_strategy, strategy_name
from cryptobot.strategies.common import daily_atr, today_bars
from tests.conftest import bars_from_closes


def test_signal_validation():
    with pytest.raises(ValueError):
        Signal("long", 100, 101)
    with pytest.raises(ValueError):
        Signal("short", 100, 99)
    with pytest.raises(ValueError):
        Signal("long", 100, 99, target=99.5)
    s = Signal("long", 100, 98, 110)
    assert s.risk_per_unit == 2


def test_orb_fires_once_at_range_end(bars):
    orb = load_strategy("orb")
    df = bars["BTC/USD"]
    today = today_bars(df)
    start = len(df) - len(today)
    fired = []
    for i in range(start + 1, len(df) + 1):
        s = orb.signal(df.iloc[:i], {"range_minutes": 15, "target_r": 10})
        if s:
            fired.append((i - start, s))
    assert len(fired) == 1 and fired[0][0] == 3  # 15 minutes of 5m bars
    k, s = fired[0]
    rng = today.iloc[:3]
    expected_side = "long" if rng["close"].iloc[-1] > rng["open"].iloc[0] else "short"
    assert s.side == expected_side
    assert abs(abs(s.target - s.entry) - 10 * s.risk_per_unit) < 1e-9
    atr = daily_atr(df, 14)
    assert s.risk_per_unit >= 0.1 * atr - 1e-9


def test_orb_stop_at_range_extreme_when_wide():
    import pandas as pd

    day1 = bars_from_closes([100] * 78, day="2026-10-05", wick=0.5)  # quiet history
    day2 = bars_from_closes([102, 103], day="2026-10-06", wick=0.5, first_open=100)  # big bar
    df = pd.concat([day1, day2.iloc[:1]])
    df.attrs.update(day1.attrs)
    s = load_strategy("orb").signal(df, {"range_minutes": 5, "target_r": 2})
    assert s is not None and s.side == "long" and s.entry == 102
    assert s.stop == pytest.approx(day2["low"].iloc[0])
    assert s.target == pytest.approx(102 + 2 * (102 - day2["low"].iloc[0]))


def test_intraday_momentum_fires_30min_before_close():
    closes = [100 + 0.02 * i for i in range(288)]  # up all (UTC) day
    df = bars_from_closes(closes)
    mom = load_strategy("intraday_momentum")
    fired = [(i, mom.signal(df.iloc[:i], {})) for i in range(1, 289)]
    fired = [(i, s) for i, s in fired if s]
    assert len(fired) == 1
    i, s = fired[0]
    assert df.index[i - 1].strftime("%H:%M") == "23:25"  # bar ending 23:30, 30 min before 00:00 UTC
    assert s.side == "long" and s.target is None and s.stop < s.entry


def test_vwap_revert_long_when_stretched_below():
    closes = [100.0] * 40 + [99.0, 98.0, 97.0]
    df = bars_from_closes(closes, wick=0.05)
    s = load_strategy("vwap_revert").signal(df, {"z_entry": 1.5, "lookback_bars": 20})
    assert s is not None and s.side == "long" and s.target > s.entry > s.stop


def test_random_entry_is_deterministic_and_once_per_day(bars):
    rnd = load_strategy("random_entry")
    df = bars["ETH/USD"]
    today = today_bars(df)
    start = len(df) - len(today)
    hits = [
        i
        for i in range(start + 1, len(df) + 1)
        if rnd.signal(df.iloc[:i], {"seed": 3, "stop_pct": 0.5}) is not None
    ]
    hits2 = [
        i
        for i in range(start + 1, len(df) + 1)
        if rnd.signal(df.iloc[:i], {"seed": 3, "stop_pct": 0.5}) is not None
    ]
    assert len(hits) == 1 and hits == hits2


def test_load_strategy_from_file(tmp_path):
    p = tmp_path / "mine.py"
    p.write_text(
        "from cryptobot.strategies import Signal\nDEFAULT_PARAMS={'x':1}\n"
        "def signal(bars, params):\n    return None\n"
    )
    mod = load_strategy(str(p))
    assert default_params(mod) == {"x": 1} and mod.signal(None, {}) is None
    with pytest.raises(ValueError):
        load_strategy("nope")
    assert strategy_name(load_strategy("orb")) == "orb"
