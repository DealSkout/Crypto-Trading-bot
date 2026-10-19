"""Section 4: the self-learning loop. The promotion gate must reject an overfit challenger (the
luckiest of many noise strategies) and accept a genuinely better one; winners shadow before
they trade; nothing automated ever reaches live trading."""

import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryptobot.config import LearningConfig
from cryptobot.journal import Journal
from cryptobot.learning import (
    Candidate,
    ShadowRunner,
    deflated_sharpe,
    evaluate,
    expected_max_sharpe,
    generate_challengers,
    max_drawdown,
    pbo_cscv,
    run_learning,
    shadow_verdict,
    walk_forward,
)
from cryptobot.registry import Registry, RegistryError

REGIMES = ["trending", "ranging", "high_vol", "risk_off"]
DAYS = [date(2025, 9, 1) + timedelta(days=i) for i in range(400)]


def trades_for(returns: np.ndarray, equity: float = 1000.0) -> pd.DataFrame:
    """One trade a day carrying that day's return; regimes rotate."""
    return pd.DataFrame(
        {
            "entry_ts": [datetime(d.year, d.month, d.day, 1, tzinfo=UTC) for d in DAYS],
            "pnl": returns * equity,
            "r_multiple": returns / 0.005,
            "regime": [REGIMES[(i // 7) % 4] for i in range(len(DAYS))],
            "symbol": "BTC/USD",
        }
    )


def noise_matrix(k: int, seed: int, sd: float = 0.01) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sd, size=(len(DAYS), k))


def lc(**kw) -> LearningConfig:
    return LearningConfig(**kw)


# ----------------------------------------------------------------------------- statistics


def test_deflated_sharpe_penalises_many_trials():
    x = np.random.default_rng(0).normal(0.002, 0.01, 200)
    one = deflated_sharpe(x, 1, 0.01)["dsr"]
    many = deflated_sharpe(x, 100, 0.01)["dsr"]
    assert many < one
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01) > 0
    assert expected_max_sharpe(1, 0.01) == 0.0


def test_pbo_is_high_for_pure_noise_and_low_for_a_real_edge():
    noise = noise_matrix(20, 1)[:280]
    assert pbo_cscv(noise) > 0.3
    edge = noise.copy()
    edge[:, 3] += 0.006
    assert pbo_cscv(edge) < 0.1


def test_walk_forward_only_uses_the_past():
    r = noise_matrix(5, 2)[:250]
    picks = walk_forward(r, folds=4)["picks"]
    future = r.copy()
    future[150:, 4] += 0.05  # change everything from fold 3's start onwards
    changed = walk_forward(future, folds=4)["picks"]
    assert changed[:3] == picks[:3]  # folds 1-3 were picked before day 150: unaffected
    assert changed[3] == 4  # fold 4 (starting day 200) may use the new past


def test_max_drawdown():
    assert max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(-50.0)


# ----------------------------------------------------------------------------- the gate


@pytest.mark.parametrize("seed", range(8))
def test_gate_rejects_the_luckiest_of_many_noise_challengers(seed):
    """Overfitting: 30 skill-less challengers. The best-looking one must not be promoted."""
    r = noise_matrix(31, seed)
    trades = [trades_for(r[:, j]) for j in range(r.shape[1])]
    res = evaluate(r, trades, DAYS, lc(), n_trials=30)
    assert res.best is not None
    assert not res.accepted, res.stats["holdout"]


def variants_of(champion: np.ndarray, k: int, seed: int, sd: float = 0.003) -> np.ndarray:
    """Challengers are variants of the champion (other params, filters): their returns are
    the champion's plus a little noise of their own, not independent series."""
    noise = np.random.default_rng(seed).normal(0.0, sd, size=(len(champion), k))
    return champion[:, None] + noise


@pytest.mark.parametrize("seed", range(8))
def test_gate_rejects_the_luckiest_of_many_correlated_variants(seed):
    champ = np.random.default_rng(100 + seed).normal(0.0005, 0.01, len(DAYS))
    r = np.column_stack([champ, variants_of(champ, 30, seed)])
    trades = [trades_for(r[:, j]) for j in range(r.shape[1])]
    res = evaluate(r, trades, DAYS, lc(), n_trials=1)
    assert not res.accepted, res.stats["holdout"]


def test_gate_accepts_a_genuinely_better_challenger():
    champ = np.random.default_rng(5).normal(0.0005, 0.01, len(DAYS))
    r = np.column_stack([champ, variants_of(champ, 20, 6)])
    # challenger 7 avoids part of the champion's losses: a real, persistent improvement
    better = np.where(champ < -0.012, champ * 0.4, champ)
    r[:, 7] = better + np.random.default_rng(8).normal(0.0, 0.001, len(DAYS))
    trades = [trades_for(r[:, j]) for j in range(r.shape[1])]
    res = evaluate(r, trades, DAYS, lc(), n_trials=26)  # half a year of weekly looks
    assert res.best == 7
    assert res.accepted, res.reasons
    h = res.stats["holdout"]
    assert h["deflated_sharpe"]["dsr"] >= 0.95 and len(h["challenger"]["profitable_regimes"]) >= 2


def test_gate_each_rule_can_reject():
    r = noise_matrix(3, 5)
    r[:, 1] = np.random.default_rng(1).normal(0.004, 0.01, len(DAYS))
    good = [trades_for(r[:, j]) for j in range(3)]
    # too few holdout trades
    few = [t.iloc[::10] for t in good]
    res = evaluate(r, few, DAYS, lc(), n_trials=2)
    assert any("holdout trades" in x or "selection window" in x for x in res.reasons)
    # profitable in one regime only
    one = [t.assign(regime="trending") for t in good]
    res = evaluate(r, one, DAYS, lc(), n_trials=2)
    assert any("regime" in x for x in res.reasons)
    # a bigger drawdown than the champion's
    r2 = r.copy()
    r2[-60, 1] = -0.4
    res = evaluate(r2, good, DAYS, lc(), n_trials=2)
    assert any("drawdown" in x for x in res.reasons)


# ----------------------------------------------------------------------------- the registry


def test_registry_lifecycle_and_rollback():
    reg = Registry(":memory:")
    v1 = reg.sync_with_config("orb", {"range_minutes": 30})
    assert v1.status == "champion" and v1.reason == "initial config.yaml"
    v2 = reg.add("orb", {"range_minutes": 15}, {}, "candidate", "test")
    reg.set_status(v2.id, "shadow", "passed")
    reg.promote_to_paper(v2.id, "shadow passed")
    assert reg.champion().id == v2.id and reg.get(v1.id).status == "retired"
    # unchanged config keeps the learned champion
    assert reg.sync_with_config("orb", {"range_minutes": 30}).id == v2.id
    # one-command rollback
    reg.rollback(v1.id)
    assert reg.champion().id == v1.id
    # a human edit of config.yaml becomes the champion
    v3 = reg.sync_with_config("orb", {"range_minutes": 5})
    assert reg.champion().id == v3.id and "human" in v3.reason
    events = [e["event"] for e in reg.events()]
    assert "retired" in events and "champion" in events


def test_live_approval_is_human_only_and_needs_a_paper_record():
    reg = Registry(":memory:")
    reg.sync_with_config("orb", {"range_minutes": 30})
    cand = reg.add("orb", {"range_minutes": 5}, {}, "candidate", "x")
    with pytest.raises(RegistryError):
        reg.approve_live(cand.id, who="me")  # never paper champion
    champ = reg.champion()
    reg.approve_live(champ.id, who="me")
    assert reg.live_version().id == champ.id
    # no automated module calls approve_live or writes live_approved
    root = Path(__file__).resolve().parent.parent / "cryptobot"
    for path in root.rglob("*.py"):
        if path.name in ("registry.py", "main.py"):
            continue
        assert "approve_live" not in path.read_text(), path


def test_challengers_only_shrink_positions_and_respect_data_coverage(cfg):
    reg = Registry(":memory:")
    champ = reg.sync_with_config("orb", {"range_minutes": 30, "target_r": 3})
    cands = generate_challengers(cfg, champ, available=set())
    assert cands and all(c.key() != Candidate.from_model(champ).key() for c in cands)
    assert not any("skip_if" in c.filters for c in cands)  # no research history -> no filters
    for c in cands:
        for w in c.filters.get("regime_weights", {}).values():
            assert 0 <= w <= 1
        assert not set(c.params) & {"risk_per_trade_pct", "max_daily_loss_pct"}
    with_fg = generate_challengers(cfg, champ, available={"fear_greed"})
    assert any(r["feature"] == "fear_greed" for c in with_fg for r in c.filters.get("skip_if", []))
    assert len(with_fg) <= cfg.learning.max_challengers


# ----------------------------------------------------------------------------- the loop


def fake_sims(edge_col: int | None, seed: int):
    """A simulate_fn that serves canned daily returns per candidate (in call order):
    the champion first, then variants of it; optionally one variant is genuinely better."""
    champ = np.random.default_rng(seed).normal(0.0005, 0.01, len(DAYS))
    r = np.column_stack([champ, variants_of(champ, 40, seed + 1)])
    if edge_col is not None:
        better = np.where(champ < -0.012, champ * 0.4, champ)
        r[:, edge_col] = better + np.random.default_rng(99).normal(0.0, 0.001, len(DAYS))
    state = {"i": 0}

    def sim(cand):
        j = state["i"]
        state["i"] += 1
        return {"daily": pd.Series(r[:, j], index=DAYS), "trades": trades_for(r[:, j])}

    return sim


def bars_spanning_days(cfg):
    idx = pd.DatetimeIndex([datetime(d.year, d.month, d.day, tzinfo=UTC) for d in DAYS], name="ts")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, idx)
    return {s: df for s in cfg.symbols}


def test_loop_rejects_noise_and_records_why(cfg, tmp_path):
    reg = Registry(":memory:")
    report = run_learning(
        cfg,
        bars_spanning_days(cfg),
        reg,
        simulate_fn=fake_sims(None, 3),
        now=datetime(2026, 10, 19, tzinfo=UTC),
        out_dir=tmp_path,
    )
    assert not report["accepted"] and report["reasons"]
    assert not reg.shadows()
    assert any(m.status == "rejected" for m in reg.all())
    assert list(tmp_path.glob("*.md")) and list(tmp_path.glob("*.json"))
    # the run is remembered: it counts as a holdout look for the next runs' deflation
    assert reg.runs_since(datetime(2026, 1, 1, tzinfo=UTC)) == 1
    assert reg.trials_since(datetime(2026, 1, 1, tzinfo=UTC)) == len(report["challengers"])


def test_loop_starts_a_shadow_for_a_real_edge_and_never_touches_live(cfg, tmp_path):
    reg = Registry(":memory:")
    report = run_learning(
        cfg,
        bars_spanning_days(cfg),
        reg,
        simulate_fn=fake_sims(5, 4),
        now=datetime(2026, 10, 19, tzinfo=UTC),
        out_dir=tmp_path,
    )
    assert report["accepted"], report["reasons"]
    shadows = reg.shadows()
    assert len(shadows) == 1 and shadows[0].shadow_start
    assert reg.champion().reason == "initial config.yaml"  # not promoted yet: shadow first
    assert reg.live_version() is None


def test_dry_run_changes_nothing(cfg, tmp_path):
    reg = Registry(":memory:")
    run_learning(
        cfg,
        bars_spanning_days(cfg),
        reg,
        simulate_fn=fake_sims(5, 4),
        dry_run=True,
        now=datetime(2026, 10, 19, tzinfo=UTC),
        out_dir=tmp_path,
    )
    assert [m.status for m in reg.all()] == ["champion"]


def test_loop_runs_end_to_end_on_the_real_engine(cfg, bars, tmp_path):
    """Synthetic bars through the real simulator (small and fast, not a statement about
    markets): the pipeline runs and writes its report."""
    cfg.learning.max_challengers = 3
    cfg.learning.workers = 1
    cfg.learning.min_oos_trades = 3
    reg = Registry(":memory:")
    report = run_learning(cfg, bars, reg, out_dir=tmp_path)
    assert len(report["challengers"]) == 3 and "holdout" in report["stats"]
    assert report["action"] != "none"


# ----------------------------------------------------------------------------- shadow


def shadow_trades(pnls, version, start):
    return pd.DataFrame(
        {
            "version": version,
            "entry_ts": [(start + timedelta(days=i + 1)).isoformat() for i in range(len(pnls))],
            "pnl": pnls,
            "r_multiple": [p / 5 for p in pnls],
        }
    )


def test_shadow_verdicts():
    lcfg = lc(shadow_days=14, min_shadow_trades=5)
    start = datetime(2026, 10, 1, tzinfo=UTC)
    reg = Registry(":memory:")
    reg.sync_with_config("orb", {})
    m = reg.add("orb", {"x": 1}, {}, "candidate", "", metrics={"holdout": {"avg_r": 0.2}})
    m = reg.set_status(m.id, "shadow", "go")
    m.shadow_start = start.isoformat()
    good = shadow_trades([3, -1, 2, 4, -1, 2], m.id, start)
    worse = shadow_trades([1, -2, 1, -1, 0, 1], "v1", start)
    assert shadow_verdict(m, good, worse, lcfg, 1000, start + timedelta(days=5))[0] == "wait"
    assert shadow_verdict(m, good, worse, lcfg, 1000, start + timedelta(days=15))[0] == "promote"
    assert shadow_verdict(m, worse, good, lcfg, 1000, start + timedelta(days=15))[0] == "reject"
    few = good.iloc[:2]
    assert shadow_verdict(m, few, worse, lcfg, 1000, start + timedelta(days=15))[0] == "wait"
    assert shadow_verdict(m, few, worse, lcfg, 1000, start + timedelta(days=30))[0] == "reject"


def test_shadow_runner_promotes_on_paper_and_the_runner_switches_models(cfg, tmp_path):
    from cryptobot.backtest import cost_model
    from cryptobot.brokers.paper import PaperBroker
    from cryptobot.engine import Engine
    from cryptobot.risk import RiskManager
    from cryptobot.runner import Runner
    from cryptobot.strategies import load_strategy

    path = str(tmp_path / "j.sqlite")
    reg = Registry(path)
    champ = reg.sync_with_config("orb", {"range_minutes": 30})
    m = reg.add(
        "orb", {"range_minutes": 15}, {}, "candidate", "", metrics={"holdout": {"avg_r": 0}}
    )
    reg.set_status(m.id, "shadow", "go")
    start = pd.Timestamp(reg.get(m.id).shadow_start).to_pydatetime()
    j = Journal(path, mode="shadow")
    for i, (a, b) in enumerate([(5, 1), (4, -1), (3, 2), (6, 0), (2, -2), (4, 1)]):
        for ver, pnl in ((m.id, a), (champ.id, b)):
            ts = (start + timedelta(days=i + 1)).isoformat()
            j.record_trade(
                {
                    "symbol": "BTC/USD",
                    "side": "long",
                    "qty": 1,
                    "entry_ts": ts,
                    "entry_price": 100,
                    "exit_ts": ts,
                    "exit_price": 100 + pnl,
                    "stop": 99,
                    "target": None,
                    "pnl": pnl,
                    "r_multiple": pnl,
                    "version": ver,
                },
                strategy="orb",
            )
    shadows = ShadowRunner(cfg, reg, path)
    assert set(shadows.sync()) == {m.id, champ.id}  # the shadow and a champion mirror
    out = shadows.evaluate(start + timedelta(days=15))
    assert out[0][1] == "promote" and reg.champion().id == m.id
    assert reg.live_version() is None  # paper only

    cm = cost_model(cfg)
    engine = Engine(
        cfg,
        load_strategy("orb"),
        champ.params,
        PaperBroker(1000, cm),
        RiskManager(cfg.risk, cost_model=cm),
        Journal(path, mode="paper"),
        mode="paper",
    )
    engine.version = champ.id
    runner = Runner(cfg, "paper", engine, lambda s: pd.DataFrame(), engine.journal, registry=reg)
    runner.apply_champion()
    assert engine.version == m.id and engine.params == {"range_minutes": 15}


# ----------------------------------------------------------------------------- filters


def test_filters_skip_and_shrink_but_never_grow(cfg):
    from cryptobot.backtest import cost_model
    from cryptobot.brokers.paper import PaperBroker
    from cryptobot.engine import Engine
    from cryptobot.risk import RiskManager
    from tests.conftest import StubStrategy, bars_from_closes
    from tests.test_engine import feed

    def run(filters, regime="trending"):
        cm = cost_model(cfg)
        j = Journal(":memory:", mode="paper")
        e = Engine(
            cfg,
            StubStrategy(fire_at=2, stop_pct=1.0),
            {},
            PaperBroker(10_000, cm),
            RiskManager(cfg.risk, cost_model=cm),
            j,
            mode="paper",
        )
        e.context_providers = [lambda s, ts, b: {"regime": regime, "fear_greed": 15.0}]
        e.filters = filters
        feed(e, bars_from_closes([100.0, 100.0, 100.0], wick=0.0))
        return e, j

    base, _ = run({})
    q = base.open["BTC/USD"][0].qty
    half, _ = run({"regime_weights": {"trending": 0.5}})
    assert half.open["BTC/USD"][0].qty == pytest.approx(q * 0.5, rel=1e-3)
    capped, _ = run({"regime_weights": {"trending": 3.0}})
    assert capped.open["BTC/USD"][0].qty == pytest.approx(q)  # never bigger than the risk limit
    skipped, j = run({"skip_if": [{"feature": "fear_greed", "op": "<=", "value": 20}]})
    assert not skipped.open
    assert "filter: fear_greed" in j.decisions(actions=["skip"]).iloc[0]["reason"]
    zero, _ = run({"regime_weights": {"trending": 0.0}})
    assert not zero.open
    assert math.isfinite(q)


def test_extra_candidates_enter_the_same_gate(cfg, monkeypatch):
    """Cross-learning and evolution candidates are appended to the pool and
    simulated like everything else, deduplicated by their key."""
    from cryptobot.learning import Candidate, generate_challengers, run_learning
    from cryptobot.registry import Registry

    registry = Registry(":memory:")
    champ = registry.sync_with_config("orb", {"range_minutes": 30, "target_r": 3})
    labels = []

    def fake_sim(candidate):
        labels.append(candidate.label)
        import numpy as np
        import pandas as pd

        return {
            "candidate": candidate,
            "label": candidate.label,
            "daily": pd.Series(dtype=float),
            "trades": pd.DataFrame(columns=["pnl", "r_multiple", "entry_ts"]),
            "returns": np.zeros(0),
        }

    extra = Candidate(strategy="vwap_revert", params={"band_pct": 9.9}, filters={})
    bars = {"BTC/USD": __import__("tests.conftest", fromlist=["x"]).bars_from_closes([100.0] * 60)}
    try:
        run_learning(cfg, bars, registry, simulate_fn=fake_sim, extra_candidates=[extra, extra])
    except Exception:
        pass  # the fake simulate output may not satisfy the full evaluate path
    assert sum(1 for label in labels if "vwap_revert" in label and "9.9" in label) == 1
