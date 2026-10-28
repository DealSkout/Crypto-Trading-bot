from datetime import UTC, datetime

import pytest

from cryptobot.team.store import TeamStore

T0 = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
T1 = datetime(2026, 10, 16, 13, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path):
    s = TeamStore(str(tmp_path / "team.sqlite"))
    yield s
    s.close()


def test_wal_mode_is_on(store):
    mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"


def test_bots_upsert(store):
    store.upsert_bot("mom-1", name="Momentum", repo="bots/mom", version="0.1", ts=T0)
    store.upsert_bot(
        "mom-1", name="Momentum", repo="bots/mom", version="0.2", status="paused", ts=T1
    )
    df = store.bots()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["version"] == "0.2" and row["status"] == "paused"
    assert row["updated_ts"] == T1.isoformat()


def test_signals_round_trip(store):
    store.log_signal(
        "mom-1",
        "BTC/USD",
        "buy",
        entry=100.0,
        stop=98.0,
        target=105.0,
        reason="breakout",
        taken=True,
        ts=T0,
    )
    row = store.conn.execute("SELECT * FROM signals").fetchone()
    assert row[1] == T0.isoformat() and row[2] == "mom-1" and row[3] == "BTC/USD"
    assert row[5] == 100.0 and row[6] == 98.0 and row[7] == 105.0
    assert row[8] == "breakout" and row[9] == 1


def test_fills_round_trip_and_filter(store):
    store.log_fill(
        "mom-1",
        "BTC/USD",
        "buy",
        qty=2.0,
        price=100.5,
        fees=0.5,
        internal=False,
        order_id="paper-1",
        ts=T0,
    )
    store.log_fill(
        "rev-1", "BTC/USD", "sell", qty=1.0, price=101.0, fees=0.25, internal=True, ts=T0
    )
    df = store.fills()
    assert len(df) == 2 and list(df["bot_id"]) == ["mom-1", "rev-1"]
    assert df.iloc[0]["order_id"] == "paper-1" and df.iloc[0]["internal"] == 0
    assert df.iloc[1]["internal"] == 1
    assert len(store.fills(bot_id="rev-1")) == 1


def test_daily_metrics_round_trip(store):
    store.record_daily_metrics(
        "2026-10-16",
        "mom-1",
        equity=5100.0,
        pnl=100.0,
        trades=4,
        sharpe=1.2,
        drawdown_pct=2.5,
        regime="trend",
    )
    store.record_daily_metrics(
        "2026-10-16",
        "mom-1",
        equity=5050.0,
        pnl=50.0,
        trades=5,
        sharpe=1.1,
        drawdown_pct=3.0,
        regime="trend",
    )
    df = store.daily_metrics()
    assert len(df) == 1  # same (day, bot_id) replaces, not duplicates
    assert df.iloc[0]["equity"] == 5050.0 and df.iloc[0]["trades"] == 5
    assert len(store.daily_metrics(bot_id="other")) == 0


def test_regimes_latest(store):
    assert store.latest_regime() is None
    store.set_regime("chop", vol_pctile=40.0, trend=0.1, breadth=0.5, ts=T0)
    store.set_regime("trend", vol_pctile=60.0, trend=0.8, breadth=0.7, ts=T1)
    latest = store.latest_regime()
    assert latest["label"] == "trend" and latest["ts"] == T1.isoformat()
    assert latest["vol_pctile"] == 60.0 and latest["trend"] == 0.8 and latest["breadth"] == 0.7


def test_current_allocations_latest_per_bot(store):
    store.record_allocation("mom-1", 0.5, reason="start", ts=T0)
    store.record_allocation("rev-1", 0.3, reason="start", ts=T0)
    store.record_allocation("mom-1", 0.4, reason="drawdown trim", ts=T1)
    assert store.current_allocations() == {"mom-1": 0.4, "rev-1": 0.3}


def test_consensus_round_trip(store):
    store.log_consensus(
        "BTC/USD", "buy", n_bots=3, bot_ids=["mom-1", "rev-1", "orb-1"], outcome_pnl=12.5, ts=T0
    )
    row = store.conn.execute(
        "SELECT ts, symbol, side, n_bots, bot_ids, outcome_pnl FROM consensus"
    ).fetchone()
    assert row == (T0.isoformat(), "BTC/USD", "buy", 3, "mom-1,rev-1,orb-1", 12.5)


def test_consensus_entry_mark_and_scoring(store):
    cid = store.log_consensus("ETH/USD", "sell", 3, ["a", "b", "c"], entry_mark=50.0, ts=T0)
    pending = store.unscored_consensus()
    assert len(pending) == 1
    assert pending[0]["id"] == cid and pending[0]["entry_mark"] == 50.0
    store.score_consensus(cid, -0.02)
    assert store.unscored_consensus() == []


def test_schema_migration_adds_entry_mark_to_an_old_store(tmp_path):
    """A store created before entry_mark existed is upgraded in place on open."""
    import sqlite3

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE consensus (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
        " symbol TEXT, side TEXT, n_bots INTEGER, bot_ids TEXT, outcome_pnl REAL)"
    )
    conn.execute(
        "INSERT INTO consensus (ts, symbol, side, n_bots, bot_ids) VALUES (?,?,?,?,?)",
        (T0.isoformat(), "BTC/USD", "buy", 3, "a,b,c"),
    )
    conn.commit()
    conn.close()
    upgraded = TeamStore(str(path))
    try:
        pending = upgraded.unscored_consensus()
        assert len(pending) == 1 and pending[0]["entry_mark"] is None
        upgraded.log_consensus("BTC/USD", "buy", 3, ["a", "b", "c"], entry_mark=1.0, ts=T1)
        assert len(upgraded.unscored_consensus()) == 2
    finally:
        upgraded.close()


def test_bot_and_team_state_round_trip(store):
    assert store.bot_state("mom-1") is None
    store.save_bot_state("mom-1", {"cash": 1234.5, "positions": {"BTC/USD": [1, 100]}}, ts=T0)
    store.save_bot_state("mom-1", {"cash": 1200.0, "positions": {}}, ts=T1)
    assert store.bot_state("mom-1") == {"cash": 1200.0, "positions": {}}
    row = store.conn.execute("SELECT updated_ts FROM bot_state WHERE bot_id = 'mom-1'").fetchone()
    assert row[0] == T1.isoformat()
    assert store.team_state("internal_fees", 0.0) == 0.0
    store.set_team_state("internal_fees", 3.25, ts=T0)
    store.set_team_state("internal_fees", 4.5, ts=T1)
    assert store.team_state("internal_fees") == 4.5


def test_challengers_add_and_update(store):
    cid = store.add_challenger("mom-2", "wider stops variant", ts=T0)
    assert cid is not None
    store.update_challenger(cid, status="testing", metrics={"sharpe": 0.9})
    df = store.challengers()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["status"] == "testing" and '"sharpe": 0.9' in row["metrics_json"]
    assert row["description"] == "wider stops variant"


def test_bot_versions_round_trip(store):
    store.add_bot_version("mom-1", "0.1", reason="initial", ts=T0)
    store.add_bot_version("mom-1", "0.2", reason="tuned params", metrics={"sharpe": 1.3}, ts=T1)
    store.add_bot_version("rev-1", "0.1", reason="initial", ts=T0)
    df = store.versions("mom-1")
    assert list(df["version"]) == ["0.1", "0.2"]
    assert '"sharpe": 1.3' in df.iloc[1]["metrics_json"]


def test_events_round_trip(store):
    store.log_event("halt", message="kill switch", bot_id="mom-1", data={"source": "router"}, ts=T0)
    store.log_event("resume", message="all clear", ts=T1)
    df = store.events()
    assert len(df) == 2
    assert df.iloc[0]["kind"] == "resume"  # newest first
    assert df.iloc[1]["bot_id"] == "mom-1" and '"source": "router"' in df.iloc[1]["data_json"]


def test_daily_metrics_carry_start_equity_and_transfers(store):
    store.record_daily_metrics("2026-10-20", "b", equity=1010.0, pnl=-15.0, trades=2,
                               start_equity=1000.0, transfers=25.0, realized_pnl=-12.0)
    row = store.daily_metrics().iloc[0]
    assert row["start_equity"] == 1000.0 and row["transfers"] == 25.0
    assert row["realized_pnl"] == -12.0 and row["pnl"] == -15.0
