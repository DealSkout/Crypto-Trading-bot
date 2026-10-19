import pytest

from cryptobot import learn
from cryptobot.journal import Journal
from cryptobot.strategies import load_strategy


def test_gate_rules():
    champ = {"sharpe": 1.0, "n_trades": 50, "net_return_pct": 5.0}
    assert learn.gate(champ, {"sharpe": 1.5, "n_trades": 40, "net_return_pct": 6}, 30, 0.1)[0]
    assert not learn.gate(champ, {"sharpe": 1.5, "n_trades": 10, "net_return_pct": 6}, 30, 0.1)[0]
    assert not learn.gate(champ, {"sharpe": 1.05, "n_trades": 40, "net_return_pct": 6}, 30, 0.1)[0]
    assert not learn.gate(champ, {"sharpe": 2.0, "n_trades": 40, "net_return_pct": -1}, 30, 0.1)[0]


def test_param_combinations_and_risk_guard():
    combos = learn.param_combinations({"a": [1, 2], "b": ["x"]})
    assert combos == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]
    with pytest.raises(learn.ProposalError):
        learn.check_proposal_params({"max_daily_loss_pct": 50})
    with pytest.raises(learn.ProposalError):
        learn.check_proposal_params({"x": [1, 2]})
    learn.check_proposal_params({"range_minutes": 15, "name": "a", "flag": True})


def test_grid_search_and_refit(cfg, bars, monkeypatch):
    orb = load_strategy("orb")
    monkeypatch.setattr(orb, "PARAM_GRID", {"range_minutes": [5, 15], "target_r": [3]})
    days = sorted({d for b in bars.values() for d in b.index.date})
    best, table = learn.grid_search(
        cfg, orb, bars, days[:20], {"range_minutes": 5, "target_r": 10}, min_trades=1
    )
    assert len(table) == 2 and best["target_r"] == 3 and best["range_minutes"] in (5, 15)
    journal = Journal(cfg.journal_path, mode="paper")
    out = learn.refit(cfg, bars, journal, apply=True)
    assert "params" in out and out["params"]["target_r"] == 3
    hist = journal.params_history()
    assert len(hist) == 1 and hist[0]["applied"] == out["changed"]
    if out["changed"]:
        from cryptobot.config import load_config

        assert load_config(cfg.path).params["target_r"] == 3


def test_evaluate_candidate_and_live_gate(cfg, bars):
    ok, reason, stats = learn.evaluate_candidate(cfg, bars, {"range_minutes": 30})
    assert isinstance(ok, bool) and "champion" in stats and "challenger" in stats
    journal = Journal(cfg.journal_path, mode="paper")
    passed, problems = learn.live_gate(cfg, journal, None)
    assert not passed and any("no saved backtest" in p for p in problems)
    assert any("paper trading days" in p for p in problems)
