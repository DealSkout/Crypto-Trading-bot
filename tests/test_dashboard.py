import http.client
import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from cryptobot import dashboard
from cryptobot.journal import Journal
from cryptobot.main import main
from tests.conftest import NOW


def test_collect_empty_journal(cfg):
    data = dashboard.collect(cfg, now=NOW)
    assert data["journal_exists"] is False and data["status"]["alive"] is False
    assert data["daily"] == [] and data["trades"] == [] and data["backtest"] is None
    assert data["live_gate"]["ok"] is False and data["status"]["markets"][0]["name"] == "CRYPTO"
    html = dashboard.render_html(data, static=True)
    assert "__DATA__" not in html and 'id="data"' in html and "const REFRESH = 0;" in html


def test_collect_demo_has_everything(cfg, tmp_path):
    demo = dashboard.build_demo(cfg, tmp_path / "demo", days=30, now=NOW)
    data = dashboard.collect(demo, now=NOW)
    assert data["journal_exists"] and data["status"]["alive"] and data["engine"]["open_trades"]
    assert len(data["daily"]) > 0 and len(data["trades"]) > 0 and len(data["events"]) > 0
    assert data["backtest"] is not None and set(data["backtest"]["curves"]) >= {
        "strategy",
        "buy_and_hold",
    }
    assert all(len(c) <= dashboard.MAX_POINTS for c in data["backtest"]["curves"].values())
    assert (
        data["params_history"][0]["applied"] is True and data["coach_notes"][0]["cost_usd"] == 0.03
    )
    assert data["llm_spend"]["today"]["coach"] == 0.03
    json.dumps(data)  # must be JSON-serialisable
    html = dashboard.render_html(data)
    assert (
        "const REFRESH = 30;" in html
        and "</script>" in html
        and "<\\/" in html
        or "</" not in json.dumps(data)
    )


def test_engine_snapshot_and_heartbeat(cfg, tmp_path):
    from cryptobot.backtest import cost_model
    from cryptobot.brokers.paper import PaperBroker
    from cryptobot.engine import Engine
    from cryptobot.risk import RiskManager
    from tests.conftest import StubStrategy, bars_from_closes

    journal = Journal(str(tmp_path / "j.sqlite"), mode="paper")
    broker = PaperBroker(1000, cost_model(cfg))
    engine = Engine(
        cfg,
        StubStrategy(fire_at=2, stop_pct=1.0),
        {},
        broker,
        RiskManager(cfg.risk),
        journal,
        mode="paper",
    )
    df = bars_from_closes([100.0, 100.0, 100.5])
    for i in range(1, 4):
        w = df.iloc[:i]
        w.attrs = dict(df.attrs)
        engine.on_bar("BTC/USD", w)
    st = journal.get_state("engine")
    assert st["open_trades"][0]["symbol"] == "BTC/USD" and st["equity"] > 0 and st["day"]
    assert (
        abs(
            st["open_trades"][0]["unrealized"]
            - (100.5 - st["open_trades"][0]["entry_price"]) * st["open_trades"][0]["qty"]
        )
        < 1e-9
    )
    journal.set_state("heartbeat", {"ts": datetime.now(UTC).isoformat(), "active": True})
    data = dashboard.collect(cfg, journal_path=journal.path)
    assert data["status"]["alive"] and data["engine"]["open_trades"]


def test_server_routes(cfg, tmp_path):
    demo = dashboard.build_demo(cfg, tmp_path / "demo", days=12, now=NOW)
    server = dashboard.make_server(demo, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/")
        r = conn.getresponse()
        body = r.read().decode()
        assert r.status == 200 and "Cryptobot dashboard" in body and "const REFRESH = 30;" in body
        conn.request("GET", "/api/data")
        r = conn.getresponse()
        assert r.status == 200
        js = json.loads(r.read())
        assert js["mode"] == "paper" and js["trades"]
        conn.request("GET", "/healthz")
        assert conn.getresponse().read() == b"ok"
        conn.request("GET", "/nope")
        assert conn.getresponse().status == 404
    finally:
        server.shutdown()
        server.server_close()


def test_cli_static_demo(cfg, tmp_path):
    out = tmp_path / "snap.html"
    rc = main(
        [
            "--config",
            cfg.path,
            "dashboard",
            "--demo",
            "--demo-dir",
            str(tmp_path / "d"),
            "--static",
            str(out),
        ]
    )
    assert rc == 0 and out.exists() and "Cryptobot dashboard" in out.read_text()
    assert Path(tmp_path / "d" / "demo.sqlite").exists()


def test_demo_seed_is_idempotent(cfg, tmp_path):
    dashboard.build_demo(cfg, tmp_path / "demo", days=12, now=NOW)
    first = len(
        dashboard.collect(dashboard.build_demo(cfg, tmp_path / "demo", days=12, now=NOW), now=NOW)[
            "trades"
        ]
    )
    again = len(
        dashboard.collect(dashboard.build_demo(cfg, tmp_path / "demo", days=12, now=NOW), now=NOW)[
            "trades"
        ]
    )
    assert first == again > 0
