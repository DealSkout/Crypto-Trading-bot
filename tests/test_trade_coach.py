import ast
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from cryptobot import trade_coach as tc
from cryptobot.config import load_config
from cryptobot.journal import Journal
from cryptobot.llm import LLMError, LLMResult

DAY = date(2026, 10, 16)
OPEN = datetime(2026, 10, 16, 13, 35, tzinfo=UTC)


class FakeLLM:
    def __init__(self, data=None, refused=False, raise_exc=None, model="claude-haiku-5-5"):
        self.data, self.refused, self.raise_exc, self.model = data or {}, refused, raise_exc, model
        self.calls = []

    def complete(self, system, user, schema, effort=None):
        self.calls.append({"system": system, "user": user, "schema": schema, "effort": effort})
        if self.raise_exc:
            raise self.raise_exc
        return LLMResult(
            data=self.data,
            input_tokens=900,
            output_tokens=300,
            cost_usd=0.00024,
            model=self.model,
            refused=self.refused,
        )


def add_trade(j, pnl, r, day_offset=0, minutes=60, reason="target", costs=0.4):
    entry = OPEN - timedelta(days=day_offset)
    j.record_trade(
        {
            "symbol": "ETH/USD",
            "side": "long",
            "qty": 2,
            "entry_ts": entry,
            "entry_price": 500.0,
            "exit_ts": entry + timedelta(minutes=minutes),
            "exit_price": 500.0 + pnl / 2,
            "stop": 498.0,
            "target": 520.0,
            "pnl": pnl,
            "r_multiple": r,
            "costs": costs,
            "exit_reason": reason,
            "signal_reason": "ORB long",
        },
        strategy="orb",
    )


def setup(cfg, tmp_path, n_old=0):
    cfg.coach.dir = str(tmp_path / "coach")
    j = Journal(cfg.journal_path, mode="paper")
    for i in range(n_old):
        add_trade(j, 1.0, 0.25, day_offset=2 + i)
    add_trade(j, 8.0, 2.0, minutes=90)
    add_trade(j, -4.0, -1.0, minutes=5, reason="stop", costs=1.5)
    j.log("skip", "spread 0.12 too wide", ts=OPEN + timedelta(hours=1))
    j.log("order_rejected", "insufficient buying power", ts=OPEN + timedelta(hours=2))
    return j


def ids(j):
    return [int(x) for x in j.trades(mode="paper")["id"].tail(2)]


def good_reply(trade_ids, proposals=None):
    return {
        "trades": [
            {
                "trade_id": trade_ids[0],
                "verdict": "good",
                "mistake": "",
                "lesson": "Let winners run.",
            },
            {
                "trade_id": trade_ids[1],
                "verdict": "mistake",
                "mistake": "Stop too tight.",
                "lesson": "Use the range low for the stop.",
            },
        ],
        "daily_summary": "One clean breakout, one rushed entry.",
        "proposed_changes": proposals or [],
    }


def test_stats_are_computed_in_python(cfg, tmp_path):
    j = setup(cfg, tmp_path)
    s = tc.build_summary(cfg, j, DAY)
    assert s["today"]["trades"] == 2 and s["today"]["wins"] == 1
    assert s["today"]["net_pnl"] == 4.0 and s["today"]["avg_r"] == 0.5
    assert s["today"]["profit_factor"] == 2.0
    assert s["today"]["rejected_orders"] == 1 and s["today"]["skips"]
    quick = s["trades"][1]
    assert quick["hold_minutes"] == 5.0 and quick["planned_r"] == 10.0
    assert any("held only" in f for f in quick["flags"])
    assert any("costs were" in f for f in quick["flags"])  # 1.5 of 4.0 at risk
    assert s["param_proposals_allowed"] is False and s["rolling"]["closed_trades"] == 2


def test_one_compact_capped_call_per_day_with_cost_logged(cfg, tmp_path):
    j = setup(cfg, tmp_path)
    llm = FakeLLM(good_reply(ids(j)))
    out = tc.run_daily_review(cfg, j, DAY, llm=llm)
    assert out["status"] == "done" and out["ai"] == "reviewed" and len(llm.calls) == 1
    call = llm.calls[0]
    assert call["effort"] == "low" and call["schema"] is tc.REVIEW_SCHEMA
    assert len(call["user"]) < 4000 and json.loads(call["user"])["date"] == "2026-10-16"
    today = datetime.now(UTC).date()
    assert out["cost_usd"] == 0.00024 and j.llm_spend_today(tc.COMPONENT, today) == 0.00024
    again = tc.run_daily_review(cfg, j, DAY, llm=llm)
    assert again["status"] == "skipped" and len(llm.calls) == 1
    tc.run_daily_review(cfg, j, DAY, llm=llm, force=True)
    assert len(llm.calls) == 2


def test_per_trade_verdicts_summary_and_files(cfg, tmp_path):
    j = setup(cfg, tmp_path)
    t = ids(j)
    out = tc.run_daily_review(cfg, j, DAY, llm=FakeLLM(good_reply(t)))
    assert [x["verdict"] for x in out["trades"]] == ["good", "mistake"]
    assert out["trades"][1]["mistake"] == "Stop too tight."
    assert out["daily_summary"].startswith("One clean breakout")
    base = Path(cfg.coach.dir)
    review = json.loads((base / "reviews" / "2026-10-16.json").read_text())
    assert review["trades"][0]["lesson"] == "Let winners run."
    md = (base / "journal.md").read_text()
    assert md.startswith("# Coach journal") and "## 2026-10-16" in md
    assert "lesson (ETH/USD" in md and "cost $0.0002" in md
    assert not (base / "proposals").exists()  # nothing proposed, no file
    tc.run_daily_review(cfg, j, DAY + timedelta(days=1), llm=None)
    md = (base / "journal.md").read_text()
    assert md.count("# Coach journal") == 1 and "## 2026-10-17" in md


def test_missing_verdict_is_marked_not_reviewed(cfg, tmp_path):
    j = setup(cfg, tmp_path)
    reply = good_reply(ids(j))
    reply["trades"] = reply["trades"][:1]
    out = tc.run_daily_review(cfg, j, DAY, llm=FakeLLM(reply))
    assert out["trades"][1]["verdict"] == "not reviewed"


def test_proposals_are_saved_never_applied(cfg, tmp_path):
    j = setup(cfg, tmp_path)
    proposals = [
        {
            "type": "filter",
            "description": "Skip entries in the first 5 minutes",
            "params": [],
            "rationale": "Two fast stop-outs.",
        },
    ]
    before = load_config(cfg.path).params
    out = tc.run_daily_review(cfg, j, DAY, llm=FakeLLM(good_reply(ids(j), proposals)))
    assert len(out["proposed_changes"]) == 1
    saved = json.loads((Path(cfg.coach.dir) / "proposals" / "2026-10-16.json").read_text())
    assert saved["proposed_changes"][0]["status"].startswith("proposed")
    assert (
        "never applied" in saved["proposed_changes"][0]["status"]
        or "never auto" in saved["proposed_changes"][0]["status"]
    )
    assert load_config(cfg.path).params == before
    assert j.params_history() == []


def test_no_param_proposals_before_30_closed_trades(cfg, tmp_path):
    j = setup(cfg, tmp_path)  # 2 closed trades
    p = [
        {
            "type": "params",
            "description": "Wider range",
            "rationale": "x",
            "params": [{"key": "range_minutes", "value": "15"}],
        }
    ]
    out = tc.run_daily_review(cfg, j, DAY, llm=FakeLLM(good_reply(ids(j), p)))
    assert out["proposed_changes"] == []
    assert "need 30+ closed trades; there are 2" in out["dropped_proposals"][0]["dropped"]


def test_param_proposals_allowed_from_30_trades_but_only_strategy_params(cfg, tmp_path):
    j = setup(cfg, tmp_path, n_old=28)  # 30 closed trades
    p = [
        {
            "type": "params",
            "description": "Wider range",
            "rationale": "x",
            "params": [{"key": "range_minutes", "value": "15"}],
        },
        {
            "type": "params",
            "description": "Tune something",
            "rationale": "x",
            "params": [{"key": "made_up", "value": "1"}],
        },
    ]
    out = tc.run_daily_review(cfg, j, DAY, llm=FakeLLM(good_reply(ids(j), p)))
    assert [x["params"][0]["key"] for x in out["proposed_changes"]] == ["range_minutes"]
    assert "not a strategy parameter" in out["dropped_proposals"][0]["dropped"]
    assert load_config(cfg.path).params["range_minutes"] == 5  # still never applied


def test_risk_limits_and_paper_mode_are_off_limits(cfg, tmp_path):
    j = setup(cfg, tmp_path, n_old=28)
    p = [
        {
            "type": "params",
            "description": "Risk more",
            "rationale": "x",
            "params": [{"key": "risk_per_trade_pct", "value": "1.0"}],
        },
        {
            "type": "process",
            "description": "Raise the daily loss cap to 4%",
            "params": [],
            "rationale": "x",
        },
        {
            "type": "process",
            "description": "Switch to live trading now",
            "params": [],
            "rationale": "Results look good.",
        },
    ]
    out = tc.run_daily_review(cfg, j, DAY, llm=FakeLLM(good_reply(ids(j), p)))
    assert out["proposed_changes"] == []
    reasons = [d["dropped"] for d in out["dropped_proposals"]]
    assert reasons[0] == "touches a risk limit"
    assert all("protected setting" in r for r in reasons[1:])
    prompt = tc.SYSTEM_PROMPT.lower()
    assert "never propose changing risk limits" in prompt and "going live" in prompt


def test_one_switch_turns_the_coach_off(cfg, tmp_path):
    j = setup(cfg, tmp_path)
    cfg.coach.enabled = False
    llm = FakeLLM(good_reply(ids(j)))
    out = tc.run_daily_review(cfg, j, DAY, llm=llm)
    assert out["status"] == "off" and llm.calls == [] and not Path(cfg.coach.dir).exists()


def test_without_a_key_errors_or_refusals_it_falls_back_to_stats(cfg, tmp_path):
    j = setup(cfg, tmp_path)
    out = tc.run_daily_review(cfg, j, DAY, llm=None)
    assert "stats only" in out["ai"] and out["daily_summary"].startswith("2 trades, 1 won")
    out = tc.run_daily_review(cfg, j, DAY, llm=FakeLLM(raise_exc=LLMError("boom")), force=True)
    assert "model error" in out["ai"] and out["trades"][0]["verdict"] == "not reviewed"
    out = tc.run_daily_review(cfg, j, DAY, llm=FakeLLM(refused=True), force=True)
    assert "declined" in out["ai"] and out["cost_usd"] > 0  # a refusal still costs, and is logged


def test_daily_budget_and_quiet_days_make_no_call(cfg, tmp_path):
    j = setup(cfg, tmp_path)
    j.add_llm_spend(tc.COMPONENT, 1.0)
    llm = FakeLLM(good_reply(ids(j)))
    out = tc.run_daily_review(cfg, j, DAY, llm=llm)
    assert "budget" in out["ai"] and llm.calls == []
    out = tc.run_daily_review(cfg, j, DAY - timedelta(days=5), llm=llm)
    assert "no closed trades" in out["ai"] and llm.calls == []


def test_review_client_uses_haiku_and_the_token_cap(cfg, monkeypatch):
    seen = {}

    class Spy:
        def __init__(self, model, max_tokens, timeout):
            seen.update(model=model, max_tokens=max_tokens)

    monkeypatch.setattr("cryptobot.llm.ClaudeLLM", Spy)
    cfg.coach.review_max_tokens = 1500
    tc.make_review_llm(cfg)
    assert seen == {"model": "claude-haiku-5-5", "max_tokens": 1500}


def test_cli_review_and_runner_hook(cfg, tmp_path, capsys, monkeypatch):
    import yaml

    from cryptobot.main import main

    j = setup(cfg, tmp_path)
    j.close()
    raw = yaml.safe_load(Path(cfg.path).read_text())
    raw["coach"] = {"dir": str(tmp_path / "coach")}  # never write into the working tree
    Path(cfg.path).write_text(yaml.safe_dump(raw))
    assert (
        main(["--config", cfg.path, "review", "--date", "2026-10-16", "--no-ai", "--mode", "paper"])
        == 0
    )
    out = capsys.readouterr().out
    assert "coach review for 2026-10-16" in out and "not reviewed" in out
    assert (tmp_path / "coach" / "journal.md").exists()


def test_trade_coach_never_imports_brokers():
    tree = ast.parse(Path("cryptobot/trade_coach.py").read_text())
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any("brokers" in n for n in names)
