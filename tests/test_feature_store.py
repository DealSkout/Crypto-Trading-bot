"""Section 1: the trade journal and feature store. Every decision (taken, skipped, closed) is
journalled with the features known at that moment; nothing from later bars leaks in; a past
day replays exactly from the stored bars."""

import json
import sqlite3
from datetime import timedelta

import pandas as pd
import pytest

from cryptobot.backtest import cost_model
from cryptobot.brokers.paper import PaperBroker
from cryptobot.engine import Engine
from cryptobot.features import bar_features, decision_features, inputs_hash
from cryptobot.journal import Journal
from cryptobot.risk import RiskManager
from cryptobot.store import export_store, replay_day, window_at
from cryptobot.strategies import load_strategy
from tests.conftest import StubStrategy, bars_from_closes
from tests.test_engine import feed


def paper_engine(cfg, strategy, journal, params=None):
    cm = cost_model(cfg)
    broker = PaperBroker(10_000, cm)
    risk = RiskManager(cfg.risk, cfg.allow_short, cfg.max_entries_per_day, cost_model=cm)
    return Engine(cfg, strategy, params or {}, broker, risk, journal, mode="paper")


def history(days=20, symbol="BTC/USD"):
    frames = []
    for d in range(days):
        day = (pd.Timestamp("2026-09-01") + timedelta(days=d)).strftime("%Y-%m-%d")
        closes = [100 + d * 0.3 + (i % 7) * 0.05 for i in range(288)]
        frames.append(bars_from_closes(closes, day=day, symbol=symbol, wick=0.05))
    df = pd.concat(frames)
    df.attrs.update(frames[0].attrs)
    return df


def test_enter_skip_and_exit_decisions_are_journalled_with_features(cfg):
    journal = Journal(":memory:", mode="paper")
    df = bars_from_closes([100, 100, 100, 103, 103], wick=0.0)
    engine = paper_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0, target_pct=2.0), journal)
    engine.version = "v7"
    feed(engine, df)
    d = journal.decisions()
    assert list(d["action"]) == ["enter", "exit"]
    enter = d.iloc[0]
    assert enter["version"] == "v7" and enter["signal_price"] == pytest.approx(100.0)
    feats = json.loads(enter["features"])
    assert feats["last_close"] == pytest.approx(100.0) and "minute_of_day" in feats
    assert len(enter["inputs_hash"]) == 24
    exit_ = d.iloc[1]
    assert exit_["reason"] == "target" and exit_["pnl"] > 0 and exit_["trade_id"] >= 1
    t = journal.trades().iloc[0]
    assert t["version"] == "v7" and t["signal_price"] == pytest.approx(100.0)
    assert t["fees"] > 0 and json.loads(t["features"])["last_close"] == pytest.approx(100.0)
    assert t["entry_slippage_bps"] > 0  # paid the half-spread and slippage


def test_skipped_signals_are_decisions_too(cfg):
    journal = Journal(":memory:", mode="paper")
    df = bars_from_closes([100.0] * 288)
    engine = paper_engine(cfg, StubStrategy(fire_at=287, stop_pct=1.0), journal)
    feed(engine, df)
    skips = journal.decisions(actions=["skip"])
    assert len(skips) == 1 and "close" in skips.iloc[0]["reason"]
    assert json.loads(skips.iloc[0]["features"])["last_close"] == pytest.approx(100.0)


def test_no_look_ahead_future_bars_never_change_a_recorded_decision(cfg):
    """Run the same day twice; the second time everything AFTER the signal bar is wildly
    different. The features stored for the entry decision must be identical."""
    closes = [100.0] * 10 + [101.0] * 10
    a = bars_from_closes(closes, wick=0.1)
    b = a.copy()
    b.iloc[3:, :4] = b.iloc[3:, :4] * 1.5  # the future: +50% after the signal bar
    b.attrs.update(a.attrs)
    stored = []
    for df in (a, b):
        journal = Journal(":memory:", mode="paper")
        engine = paper_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0), journal)
        feed(engine, df)
        stored.append(journal.decisions(actions=["enter"]).iloc[0])
    assert stored[0]["features"] == stored[1]["features"]
    assert stored[0]["inputs_hash"] == stored[1]["inputs_hash"]


def test_daily_features_use_completed_days_only():
    df = history(20)
    base = bar_features(df.iloc[:-100])
    changed = df.iloc[:-100].copy()
    today = changed.index.normalize() == changed.index[-1].normalize()
    changed.loc[today & (changed.index < changed.index[-1]), ["high", "low"]] *= [1.2, 0.8]
    changed.attrs.update(df.attrs)
    again = bar_features(changed)
    for k in ("vol_7d_pct", "trend_er_14d", "atr_14d_pct", "dist_ma20d_pct"):
        assert again[k] == pytest.approx(base[k]), k  # today's bars never feed daily features


def test_context_providers_add_keys_and_failures_never_block(cfg):
    df = history(3)

    def regime(symbol, ts, bars):
        return {"regime": "trending"}

    def broken(symbol, ts, bars):
        raise RuntimeError("feed down")

    feats = decision_features("BTC/USD", df, [regime, broken])
    assert feats["regime"] == "trending" and "broken" in feats["context_errors"][0]


def test_replay_reproduces_a_paper_day_exactly(cfg, tmp_path):
    cfg.max_entries_per_day = 1
    df = history(20)
    orb = load_strategy("orb")
    params = {"range_minutes": 15, "target_r": 2}
    journal = Journal(str(tmp_path / "j.sqlite"), mode="paper")
    engine = paper_engine(cfg, orb, journal, params)
    n = 25 * 288
    last_day = df.index[-1].normalize()
    start = int((df.index < last_day).sum())
    for i in range(start, len(df)):
        w = df.iloc[max(0, i + 1 - n) : i + 1]
        w.attrs = dict(df.attrs)
        engine.on_bar("BTC/USD", w)
    day = last_day.date()
    rows = replay_day(cfg, journal, day, {"BTC/USD": df}, mode="paper")
    assert rows and all(r["status"] == "exact" for r in rows), rows
    # a stored bar that changed afterwards is caught by the fingerprint
    revised = df.copy()
    revised.iloc[start - 5, revised.columns.get_loc("close")] += 1.0
    revised.attrs.update(df.attrs)
    rows2 = replay_day(cfg, journal, day, {"BTC/USD": revised}, mode="paper")
    assert all(r["status"] == "inputs changed" for r in rows2)


def test_window_at_matches_what_the_engine_saw():
    df = history(30)
    ts = df.index[-50]
    w = window_at(df, ts, 5)
    assert w.index[-1] == ts and len(w) == 25 * 288
    assert inputs_hash(w, "orb", {}) == inputs_hash(window_at(df, ts, 5), "orb", {})


def test_export_writes_parquet_with_feature_columns(cfg, tmp_path):
    journal = Journal(":memory:", mode="paper")
    df = bars_from_closes([100, 100, 100, 103, 103], wick=0.0)
    feed(paper_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0, target_pct=2.0), journal), df)
    paths = export_store(journal, tmp_path / "store")
    dec = pd.read_parquet(paths["decisions"])
    assert "f_last_close" in dec.columns and len(dec) == 2
    assert len(pd.read_parquet(paths["trades"])) == 1


def test_old_journals_gain_the_new_trade_columns(tmp_path):
    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE trades (id INTEGER PRIMARY KEY, mode TEXT, strategy TEXT, symbol TEXT,"
        " side TEXT, qty REAL, entry_ts TEXT, entry_price REAL, exit_ts TEXT, exit_price REAL,"
        " stop REAL, target REAL, pnl REAL, r_multiple REAL, costs REAL, exit_reason TEXT,"
        " signal_reason TEXT, asset_class TEXT DEFAULT 'crypto')"
    )
    con.commit()
    con.close()
    j = Journal(str(path))
    cols = {r[1] for r in j.conn.execute("PRAGMA table_info(trades)")}
    assert {"version", "regime", "features", "entry_slippage_bps"} <= cols
