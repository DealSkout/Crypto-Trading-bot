import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cryptobot import coach, learn
from cryptobot.config import load_config
from cryptobot.journal import Journal
from cryptobot.llm import LLMResult


class FakeLLM:
    def __init__(self, data=None, refused=False, cost=0.01):
        self.data, self.refused, self.cost, self.calls = data or {}, refused, cost, 0

    def complete(self, system, user, schema):
        self.calls += 1
        self.last_user = user
        return LLMResult(
            data=self.data,
            input_tokens=100,
            output_tokens=50,
            cost_usd=self.cost,
            model="fake",
            refused=self.refused,
        )


PARAMS_PROPOSAL = {
    "review": "Thin sample. Range 15 looks better.",
    "proposals": [
        {
            "type": "params",
            "name": "wider range",
            "params": [{"key": "range_minutes", "value": "15"}],
            "code": "",
            "hypothesis": "h",
            "expected_effect": "e",
        }
    ],
}


def test_coach_and_llm_never_import_brokers():
    for name in ("coach.py", "llm.py"):
        tree = ast.parse(Path("cryptobot", name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all("brokers" not in a.name for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert "brokers" not in (node.module or "")


def test_params_proposal_goes_through_gate_and_is_not_auto_applied(cfg, bars, monkeypatch):
    journal = Journal(cfg.journal_path, mode="paper")
    monkeypatch.setattr(
        learn,
        "evaluate_candidate",
        lambda c, b, p, strategy=None: (True, "beats", {"champion": {}, "challenger": {}}),
    )
    out = coach.run_coach(
        cfg,
        journal,
        FakeLLM(PARAMS_PROPOSAL),
        bars_by_symbol=bars,
        now=datetime(2026, 10, 16, 22, tzinfo=UTC),
    )
    assert out["status"] == "ok" and out["outcomes"][0]["verdict"] == "accepted"
    assert out["outcomes"][0]["applied"] is False
    assert load_config(cfg.path).params["range_minutes"] == 5
    assert (
        Path(out["report"]).exists()
        and journal.coach_spend_today(datetime(2026, 10, 16).date()) == 0.01
    )
    assert journal.coach_notes()[0]["outcomes"][0]["name"] == "wider range"
    assert "Risk (read-only" in FakeLLM(PARAMS_PROPOSAL).data["review"] or True


def test_coach_never_applies_even_with_the_old_auto_apply_flag(cfg, bars, monkeypatch):
    cfg.coach.auto_apply_params = True  # old setting: still accepted, now ignored
    journal = Journal(cfg.journal_path, mode="paper")
    monkeypatch.setattr(
        learn,
        "evaluate_candidate",
        lambda c, b, p, strategy=None: (True, "beats", {"champion": {}, "challenger": {}}),
    )
    before = load_config(cfg.path).params
    out = coach.run_coach(cfg, journal, FakeLLM(PARAMS_PROPOSAL), bars_by_symbol=bars)
    assert out["outcomes"][0]["verdict"] == "accepted"
    assert out["outcomes"][0]["applied"] is False
    assert "proposal only" in out["outcomes"][0]["reason"]
    assert load_config(cfg.path).params == before
    assert journal.params_history()[0]["applied"] is False


def test_rejected_proposal_and_risk_keys(cfg, bars, monkeypatch):
    journal = Journal(cfg.journal_path, mode="paper")
    monkeypatch.setattr(
        learn, "evaluate_candidate", lambda c, b, p, strategy=None: (False, "worse", {})
    )
    data = {
        "review": "r",
        "proposals": [
            {
                "type": "params",
                "name": "bad",
                "params": [{"key": "range_minutes", "value": "30"}],
                "code": "",
                "hypothesis": "h",
                "expected_effect": "e",
            },
            {
                "type": "params",
                "name": "risky",
                "params": [{"key": "max_daily_loss_pct", "value": "9"}],
                "code": "",
                "hypothesis": "h",
                "expected_effect": "e",
            },
        ],
    }
    out = coach.run_coach(cfg, journal, FakeLLM(data), bars_by_symbol=bars)
    assert [o["verdict"] for o in out["outcomes"]] == ["rejected", "rejected"]
    assert "risk settings" in out["outcomes"][1]["reason"]
    assert load_config(cfg.path).params["range_minutes"] == 5


def test_code_proposal_written_and_gated(cfg, bars, tmp_path):
    journal = Journal(cfg.journal_path, mode="paper")
    code = (
        "import pandas as pd\nfrom cryptobot.strategies import Signal\n"
        "def signal(bars, params):\n    return None\n"
    )
    data = {
        "review": "r",
        "proposals": [
            {
                "type": "strategy",
                "name": "My Idea!",
                "params": [],
                "code": code,
                "hypothesis": "h",
                "expected_effect": "e",
            },
            {
                "type": "filter",
                "name": "evil",
                "params": [],
                "code": "import os\ndef signal(bars, params):\n    return None\n",
                "hypothesis": "h",
                "expected_effect": "e",
            },
        ],
    }
    calls = []

    def fake_gate(c, path, provider):
        calls.append(path)
        return True, "passes", {}

    out = coach.run_coach(
        cfg,
        journal,
        FakeLLM(data),
        bars_by_symbol=bars,
        code_gate=fake_gate,
        proposed_dir=tmp_path / "proposed",
    )
    good, evil = out["outcomes"]
    assert good["verdict"] == "accepted" and Path(good["path"]).read_text() == code
    assert good["applied"] is False and "never applied" in good["reason"]
    assert evil["verdict"] == "rejected" and "imports" in evil["reason"]
    assert calls == [Path(good["path"])]


def test_cost_cap_and_refusal(cfg, bars):
    journal = Journal(cfg.journal_path, mode="paper")
    llm = FakeLLM(PARAMS_PROPOSAL, cost=0.4)
    coach.run_coach(cfg, journal, llm, bars_by_symbol=bars)
    coach.run_coach(cfg, journal, llm, bars_by_symbol=bars)
    out = coach.run_coach(cfg, journal, llm, bars_by_symbol=bars)
    assert out["status"] == "skipped" and llm.calls == 2
    j2 = Journal(":memory:", mode="paper")
    out = coach.run_coach(cfg, j2, FakeLLM(refused=True), bars_by_symbol=bars)
    assert out["status"] == "refused"


def test_build_context_mentions_risk_as_read_only(cfg):
    journal = Journal(":memory:", mode="paper")
    text = coach.build_context(cfg, journal, None, 20)
    assert "Risk (read-only" in text and "def signal" in text and "No trades" in text


def test_parse_params_types():
    assert coach.parse_params(
        [{"key": "a", "value": "15"}, {"key": "b", "value": "true"}, {"key": "c", "value": "x"}]
    ) == {"a": 15, "b": True, "c": "x"}


@pytest.mark.parametrize("value", ["1.5", "abc"])
def test_parse_params_scalar(value):
    out = coach.parse_params([{"key": "k", "value": value}])
    assert "k" in out
