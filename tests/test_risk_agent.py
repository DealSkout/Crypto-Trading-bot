import io
import json
from datetime import timedelta
from pathlib import Path

from cryptobot.brokers.base import Order, Position
from cryptobot.journal import Journal
from cryptobot.notify import Notifier
from cryptobot.risk_agent import RiskAgentServer
from tests.conftest import NOW


class FakeT212:
    """Stands in for Trading212Broker; records every call so we can prove it only reads."""

    live = False

    def __init__(self, positions=None, orders=None, fail=False):
        self.positions = positions or []
        self.orders = orders or []
        self.fail = fail
        self.calls: list[str] = []

    def get_positions(self):
        self.calls.append("get_positions")
        if self.fail:
            raise RuntimeError("401 bad key")
        return self.positions

    def open_orders(self):
        self.calls.append("open_orders")
        return self.orders

    def get_cash(self):
        self.calls.append("get_cash")
        return 400.0

    def get_equity(self):
        self.calls.append("get_equity")
        return 1005.0

    def place_order(self, *a, **k):  # pragma: no cover - must never be called
        raise AssertionError("the risk agent must never place orders")

    def cancel(self, *a, **k):  # pragma: no cover - must never be called
        raise AssertionError("the risk agent must never cancel orders")


def make(cfg, tmp_path, broker=None, read_only=False, t212=True):
    cfg.guardian.stop_file = str(tmp_path / "cryptobot.STOPPED")
    cfg.broker = "alpaca" if t212 else "paper"
    srv = RiskAgentServer(
        cfg,
        "paper",
        read_only=read_only,
        broker_factory=(lambda: broker) if broker is not None else None,
        notifier=Notifier(enabled=False),
        clock=lambda: NOW,
    )
    return srv


def seed_engine(cfg, open_trades, equity=1000.0, hb_age=30, peak=None):
    j = Journal(cfg.journal_path, mode="paper")
    j.set_state(
        "engine",
        {
            "ts": (NOW - timedelta(seconds=hb_age)).isoformat(),
            "equity": equity,
            "cash": 500.0,
            "day_pnl": -3.0,
            "open_trades": open_trades,
            "halted": False,
        },
    )
    j.set_state("heartbeat", {"ts": (NOW - timedelta(seconds=hb_age)).isoformat(), "active": True})
    if peak is not None:
        j.set_state("peak_equity", peak)
    j.close()


def call(srv, name, args=None):
    out = srv.call_tool(name, args or {})
    return out["isError"], json.loads(out["content"][0]["text"])


def rpc(srv, lines):
    stdout = io.StringIO()
    srv.serve(
        io.StringIO(
            "\n".join(json.dumps(x) if not isinstance(x, str) else x for x in lines) + "\n"
        ),
        stdout,
    )
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


TRADE = {"symbol": "AAPL", "side": "long", "qty": 2.0, "entry_price": 200.0, "stop": 196.0}


def test_protocol_handshake_lists_tools_and_ignores_notifications(cfg, tmp_path):
    srv = make(cfg, tmp_path)
    out = rpc(
        srv,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "nope"},
            "not json",
            {"jsonrpc": "2.0", "id": 4, "method": "prompts/get", "params": {"name": "risk_review"}},
        ],
    )
    assert [m.get("id") for m in out] == [1, 2, 3, None, 4]  # no reply to the notification
    assert out[0]["result"]["protocolVersion"] == "2025-06-18"
    assert "risk manager" in out[0]["result"]["instructions"]
    names = [t["name"] for t in out[1]["result"]["tools"]]
    assert names == [
        "bot_status",
        "account_snapshot",
        "recent_trades",
        "daily_results",
        "alerts",
        "risk_check",
        "stop_bot",
    ]
    assert all("handler" not in t for t in out[1]["result"]["tools"])
    assert out[2]["error"]["code"] == -32601
    assert out[3]["error"]["code"] == -32700
    assert "risk review" in out[4]["result"]["messages"][0]["content"]["text"]


def test_unknown_protocol_version_gets_the_latest(cfg, tmp_path):
    out = make(cfg, tmp_path).handle(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1999"}}
    )
    assert out["result"]["protocolVersion"] == "2025-06-18"


def test_agent_can_never_trade_resume_or_change_settings(cfg, tmp_path):
    srv = make(cfg, tmp_path)
    names = set(srv.tools)
    assert names == {
        "bot_status",
        "account_snapshot",
        "recent_trades",
        "daily_results",
        "alerts",
        "risk_check",
        "stop_bot",
    }
    src = Path("cryptobot/risk_agent.py").read_text()
    for forbidden in ("place_order", ".resume(", ".cancel(", "flatten_all", "write_params"):
        assert forbidden not in src, forbidden
    assert set(make(cfg, tmp_path, read_only=True).tools) == names - {"stop_bot"}


def test_bot_status_reports_running_drawdown_and_limits(cfg, tmp_path):
    seed_engine(cfg, [TRADE], equity=950.0, hb_age=30, peak=1000.0)
    err, st = call(make(cfg, tmp_path), "bot_status")
    assert not err
    assert st["running"] is True and st["guardian_stop"] is None
    assert st["open_trades"][0]["symbol"] == "AAPL"
    assert st["drawdown"]["drawdown_pct"] == 5.0
    assert st["limits"]["max_stop_pct"] == cfg.risk.max_stop_pct


def test_bot_status_on_an_empty_journal(cfg, tmp_path):
    err, st = call(make(cfg, tmp_path), "bot_status")
    assert not err and st["running"] is False and st["drawdown"]["drawdown_pct"] is None


def test_account_snapshot_reads_t212_and_flags_mismatches(cfg, tmp_path):
    seed_engine(cfg, [TRADE, {**TRADE, "symbol": "MSFT", "qty": 1.0}])
    broker = FakeT212(
        positions=[Position("AAPL", 2.0, "long", 200.0), Position("TSLA", 3.0, "long", 250.0)],
        orders=[Order("9", "AAPL", "sell", 2.0, status="new")],
    )
    err, snap = call(make(cfg, tmp_path, broker), "account_snapshot")
    assert not err
    assert snap["account"] == "practice (demo)" and snap["equity"] == 1005.0
    assert snap["pending_orders"][0]["id"] == "9"
    assert snap["mismatches"] == [
        "MSFT: the bot thinks it holds 1 but the broker shows none",
        "TSLA: 3 held at the broker but the bot has no open trade",
    ]
    assert set(broker.calls) <= {"get_positions", "open_orders", "get_cash", "get_equity"}


def test_account_snapshot_is_cached_to_respect_t212_rate_limits(cfg, tmp_path):
    broker = FakeT212()
    srv = make(cfg, tmp_path, broker)
    call(srv, "account_snapshot")
    call(srv, "account_snapshot")
    assert broker.calls.count("get_equity") == 1


def test_account_snapshot_errors_are_reported_not_raised(cfg, tmp_path):
    err, out = call(make(cfg, tmp_path, FakeT212(fail=True)), "account_snapshot")
    assert err and "401 bad key" in out["error"]
    err, out = call(make(cfg, tmp_path, t212=False), "account_snapshot")
    assert err and "broker: alpaca" in out["error"]


def test_reconcile_tolerates_rounding():
    assert (
        RiskAgentServer.reconcile([{"symbol": "aapl", "side": "long", "qty": 2.0001}], [{**TRADE}])
        == []
    )


def test_trades_daily_and_alerts(cfg, tmp_path):
    j = Journal(cfg.journal_path, mode="paper")
    for i, pnl in enumerate([5.0, -2.0, -1.0]):
        j.record_trade(
            {
                "symbol": "AAPL",
                "side": "long",
                "qty": 1,
                "entry_ts": NOW - timedelta(days=i + 1),
                "entry_price": 100,
                "exit_ts": NOW - timedelta(days=i + 1),
                "exit_price": 100 + pnl,
                "stop": 99,
                "target": 110,
                "pnl": pnl,
                "r_multiple": pnl,
                "costs": 0.1,
                "exit_reason": "stop",
                "signal_reason": "test",
            },
            strategy="orb",
        )
        j.record_daily((NOW - timedelta(days=i + 1)).date(), 1000, 1000 + pnl, pnl, 1, 0.1, i == 2)
    j.log("order_rejected", "insufficient funds", ts=NOW - timedelta(hours=2))
    j.log("signal", "not an alert", ts=NOW - timedelta(hours=1))
    j.log("error", "too old", ts=NOW - timedelta(days=3))
    j.close()
    srv = make(cfg, tmp_path)
    _, tr = call(srv, "recent_trades", {"days": 7})
    assert tr["count"] == 3 and tr["net_pnl"] == 2.0 and tr["win_rate"] == round(1 / 3, 3)
    _, dr = call(srv, "daily_results", {"days": 30})
    assert dr["count"] == 3 and dr["halts"] == 1
    _, al = call(srv, "alerts", {"hours": 24})
    assert al["counts"] == {"order_rejected": 1}
    err, bad = call(srv, "recent_trades", {"days": "lots"})
    assert err and "number" in bad["error"]


def test_risk_check_finds_problems_without_stopping(cfg, tmp_path):
    seed_engine(cfg, [TRADE], equity=880.0, hb_age=3600, peak=1000.0)
    broker = FakeT212(positions=[])
    srv = make(cfg, tmp_path, broker)
    err, rc = call(srv, "risk_check")
    assert not err
    assert any("drawdown 12.00%" in f for f in rc["rule_findings"])
    assert rc["guardian_would_stop"] is True
    assert any("heartbeat is stale" in w for w in rc["warnings"])
    assert any("the broker shows none" in w for w in rc["warnings"])
    assert not Path(cfg.guardian.stop_file).exists()


def test_risk_check_quiet_when_healthy(cfg, tmp_path):
    seed_engine(cfg, [TRADE], equity=1000.0, hb_age=10, peak=1000.0)
    broker = FakeT212(positions=[Position("AAPL", 2.0, "long", 200.0)])
    _, rc = call(make(cfg, tmp_path, broker), "risk_check")
    assert rc["rule_findings"] == [] and rc["warnings"] == []


def test_stop_bot_writes_the_flag_and_the_bot_refuses_to_start(cfg, tmp_path):
    from cryptobot.guardian import Guardian

    srv = make(cfg, tmp_path)
    err, out = call(srv, "stop_bot", {"reason": "short"})
    assert err and "specific reason" in out["error"]
    err, out = call(srv, "stop_bot", {"reason": "drawdown 12% beyond the 10% limit"})
    assert not err and out["stopped"] and not out["already"]
    info = json.loads(Path(cfg.guardian.stop_file).read_text())
    assert info["source"] == "risk-agent (MCP)"
    _, again = call(srv, "stop_bot", {"reason": "a second, different reason"})
    assert again["already"] and again["info"]["reason"].startswith("drawdown 12%")
    j = Journal(cfg.journal_path, mode="paper")
    assert Guardian(cfg, j).stopped  # the runner checks exactly this before and during trading
    assert len(j.events(kinds=["guardian_stop"])) == 1
    j.close()


def test_read_only_server_cannot_stop(cfg, tmp_path):
    srv = make(cfg, tmp_path, read_only=True)
    err, out = call(srv, "stop_bot", {"reason": "drawdown 12% beyond the 10% limit"})
    assert err and "unknown tool" in out["error"]
    assert not Path(cfg.guardian.stop_file).exists()


def test_cli_mcp_resolves_paths_from_the_config_folder(cfg, tmp_path, monkeypatch):
    import yaml

    from cryptobot.main import main

    raw = yaml.safe_load(Path(cfg.path).read_text())
    raw["journal_path"] = "journal.sqlite"  # relative: must resolve next to config.yaml
    raw["guardian"] = {"provider": "none", "stop_file": "cryptobot.STOPPED"}
    Path(cfg.path).write_text(yaml.safe_dump(raw))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    lines = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "stop_bot", "arguments": {"reason": "testing the CLI stop path"}},
        },
    ]
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(lines[0]) + "\n"))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    assert main(["--config", str(Path(cfg.path).resolve()), "mcp"]) == 0
    reply = json.loads(out.getvalue().splitlines()[-1])
    assert reply["result"]["isError"] is False
    assert (tmp_path / "cryptobot.STOPPED").exists()
    assert not (elsewhere / "cryptobot.STOPPED").exists()
