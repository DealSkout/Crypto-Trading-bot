"""A rich local dashboard for watching the bot: equity and P&L, open positions, trade
analytics, the latest backtest against its benchmarks, the learning loop and the decision log.

    python -m cryptobot dashboard            # http://127.0.0.1:8787
    python -m cryptobot dashboard --demo     # seeded with synthetic paper trades, for a look
    python -m cryptobot dashboard --static out.html   # a self-contained snapshot

Everything is read from the SQLite journal and the saved backtest reports; nothing here can
place an order. The server is plain stdlib and binds to localhost by default: on a VPS, reach
it through an SSH tunnel rather than opening the port.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd

from cryptobot import learn, sessions
from cryptobot.backtest import BacktestResult, result_currency
from cryptobot.config import Config
from cryptobot.guardian import stopped_info
from cryptobot.journal import Journal

log = logging.getLogger(__name__)

MAX_POINTS = 400  # curves are thinned to this many points before they reach the browser
STALE_AFTER = 180  # seconds without a heartbeat before the bot is reported as stale


def _f(x, default=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) or math.isinf(v) else v


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if isinstance(v, pd.Timestamp | datetime):
                clean[k] = v.isoformat()
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            elif hasattr(v, "item"):  # numpy scalars
                clean[k] = v.item()
            else:
                clean[k] = v
        out.append(clean)
    return out


def thin(series: pd.Series, max_points: int = MAX_POINTS) -> list[list[Any]]:
    s = series.dropna()
    if len(s) > max_points:
        step = int(math.ceil(len(s) / max_points))
        s = pd.concat([s.iloc[::step], s.iloc[[-1]]])
        s = s[~s.index.duplicated(keep="last")]
    return [[pd.Timestamp(t).isoformat(), _f(v)] for t, v in s.items()]


def market_status(cfg: Config, now: datetime) -> list[dict[str, Any]]:
    out = []
    for name in sorted({sessions.market_for(s).name for s in cfg.symbols}):
        m = sessions.MARKETS[name]
        is_open = sessions.is_in_session(m, now)
        out.append(
            {
                "name": name,
                "open": is_open,
                "minutes_to_close": round(sessions.minutes_to_close(m, now)) if is_open else None,
                "next_open": None if is_open else sessions.next_session_open(m, now).isoformat(),
                "symbols": [s for s in cfg.symbols if sessions.market_for(s) is m],
            }
        )
    return out


def backtest_summary(cfg: Config) -> dict[str, Any] | None:
    path = Path(cfg.reports_dir) / f"latest_{cfg.strategy}.json"
    if not path.exists():
        return None
    try:
        r = BacktestResult.load(path)
    except Exception as e:  # a corrupt report should not take the dashboard down
        log.warning("could not load %s: %s", path, e)
        return None
    base = r.starting_cash or 1.0

    def pct_curve(eq: pd.Series) -> list[list[Any]]:
        return thin((eq / base - 1.0) * 100.0) if eq is not None and len(eq) else []

    return {
        "strategy": r.strategy,
        "params": r.params,
        "symbols": r.symbols,
        "start": r.start,
        "end": r.end,
        "train_start": r.train_start,
        "train_end": r.train_end,
        "currency": r.currency,
        "decider": r.decider,
        "created_at": r.created_at,
        "metrics": {k: _f(v, v) for k, v in r.metrics.items()},
        "benchmarks": {n: {k: _f(v, v) for k, v in m.items()} for n, m in r.benchmarks.items()},
        "beats_buy_and_hold": r.beats_buy_and_hold,
        "beats_random": r.beats_random,
        "curves": {
            "strategy": pct_curve(r.equity),
            **{k: pct_curve(v) for k, v in r.bench_equity.items()},
        },
        "notes": r.notes,
        "n_trades_table": len(r.trades),
    }


def collect(
    cfg: Config,
    mode: str | None = None,
    now: datetime | None = None,
    journal_path: str | None = None,
) -> dict[str, Any]:
    """Everything the page needs, as plain JSON-able data."""
    now = now or datetime.now(UTC)
    mode = mode or cfg.mode
    path = journal_path or cfg.journal_path
    exists = Path(path).exists()
    journal = Journal(path if exists else ":memory:", mode=mode)
    try:
        state = journal.all_state()
        daily = _records(journal.daily())
        trades = _records(journal.trades(mode=mode).tail(500))
        events = _records(journal.events(limit=250))
        decisions = _records(journal.events(kinds=["decider", "decider_error"], limit=300))
        params_history = journal.params_history(limit=20)
        coach_notes = journal.coach_notes(limit=5)
        spend_today = {c: journal.llm_spend_today(c, now.date()) for c in ("decider", "coach")}
        spend_today["coach"] += journal.llm_spend_today("coach_review", now.date())
        spend_30d = journal.llm_spend_since(30, now.date())
    finally:
        journal.close()
    engine = (state.get("engine") or {}).get("value") or {}
    hb = (state.get("heartbeat") or {}).get("value") or {}
    hb_ts = hb.get("ts") or engine.get("ts")
    age = None
    if hb_ts:
        try:
            age = max(0.0, (now - datetime.fromisoformat(hb_ts)).total_seconds())
        except ValueError:
            age = None
    bt = backtest_summary(cfg)
    gate_ok, problems = (False, ["no journal yet: run `python -m cryptobot paper`"])
    if exists:
        paper = Journal(path, mode="paper")
        try:
            result = None
            bt_path = Path(cfg.reports_dir) / f"latest_{cfg.strategy}.json"
            if bt_path.exists():
                try:
                    result = BacktestResult.load(bt_path)
                except Exception:
                    result = None
            gate_ok, problems = learn.live_gate(cfg, paper, result)
        finally:
            paper.close()
    return {
        "generated_at": now.isoformat(),
        "mode": mode,
        "broker": cfg.broker,
        "strategy": cfg.strategy,
        "params": cfg.params,
        "symbols": cfg.symbols,
        "timeframe": cfg.timeframe,
        "currency": result_currency(cfg),
        "starting_cash": cfg.starting_cash,
        "risk": cfg.risk.__dict__,
        "costs": cfg.costs.__dict__,
        "decider": {
            "provider": cfg.decider.provider,
            "role": cfg.decider.role,
            "description": learn.decider_description(cfg),
            "max_daily_cost_usd": cfg.decider.max_daily_cost_usd,
        },
        "coach": {
            "provider": cfg.coach.provider,
            "enabled": cfg.coach.enabled,
            "review_model": cfg.coach.review_model,
            "max_daily_cost_usd": cfg.coach.max_daily_cost_usd,
        },
        "journal_path": path,
        "journal_exists": exists,
        "status": {
            "heartbeat": hb_ts,
            "age_seconds": age,
            "alive": age is not None and age <= STALE_AFTER,
            "loop_active": bool(hb.get("active")),
            "markets": market_status(cfg, now),
            "stopped": stopped_info(cfg),
            "halted": bool(engine.get("halted")),
            "halt_reason": engine.get("halt_reason", ""),
        },
        "engine": engine,
        "daily": daily,
        "trades": trades,
        "events": events,
        "decisions": decisions,
        "params_history": params_history,
        "coach_notes": coach_notes,
        "llm_spend": {"today": spend_today, "last_30d": spend_30d},
        "backtest": bt,
        "live_gate": {"ok": gate_ok, "problems": problems},
    }


# ------------------------------------------------------------------------------ rendering


def template() -> str:
    return resources.files("cryptobot").joinpath("dashboard.html").read_text(encoding="utf-8")


def render_html(data: dict[str, Any], refresh_seconds: int = 30, static: bool = False) -> str:
    payload = json.dumps(data, default=str).replace("</", "<\\/")
    html = template()
    html = html.replace("__DATA__", payload)
    html = html.replace("__REFRESH__", "0" if static else str(int(refresh_seconds)))
    return html


# ------------------------------------------------------------------------------ server


class _Handler(BaseHTTPRequestHandler):
    cfg: Config
    mode: str
    refresh: int

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        route = self.path.split("?", 1)[0]
        try:
            if route in ("/", "/index.html"):
                data = collect(self.cfg, mode=self.mode)
                body = render_html(data, refresh_seconds=self.refresh).encode("utf-8")
                self._send(200, body, "text/html; charset=utf-8")
            elif route == "/api/data":
                data = collect(self.cfg, mode=self.mode)
                self._send(
                    200,
                    json.dumps(data, default=str).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            elif route == "/healthz":
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:  # keep serving; show the error
            log.exception("dashboard request failed")
            self._send(500, f"dashboard error: {e!r}".encode(), "text/plain; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:  # quieter than the stdlib default
        log.debug("dashboard: " + fmt, *args)


def make_server(
    cfg: Config,
    host: str = "127.0.0.1",
    port: int = 8787,
    mode: str | None = None,
    refresh_seconds: int = 30,
) -> ThreadingHTTPServer:
    handler = type(
        "DashboardHandler",
        (_Handler,),
        {"cfg": cfg, "mode": mode or cfg.mode, "refresh": refresh_seconds},
    )
    return ThreadingHTTPServer((host, port), handler)


def serve(
    cfg: Config,
    host: str = "127.0.0.1",
    port: int = 8787,
    mode: str | None = None,
    refresh_seconds: int = 30,
    open_browser: bool = False,
) -> None:
    server = make_server(cfg, host, port, mode, refresh_seconds)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"Cryptobot dashboard at {url}  (Ctrl-C to stop; mode={mode or cfg.mode})")
    if open_browser:
        import webbrowser

        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ------------------------------------------------------------------------------ demo data


def build_demo(cfg: Config, workdir: Path, days: int = 45, now: datetime | None = None) -> Config:
    """Seed a temporary journal and backtest with synthetic paper trading so the dashboard has
    something to show. Returns a config pointing at the seeded files."""
    from dataclasses import replace

    from cryptobot.backtest import run_backtest
    from cryptobot.data import get_bars
    from cryptobot.report import write_report

    now = now or datetime.now(UTC)
    workdir.mkdir(parents=True, exist_ok=True)
    for stale in ("demo.sqlite", "demo.sqlite-journal"):  # seed from scratch every time
        (workdir / stale).unlink(missing_ok=True)
    demo = replace(
        cfg, journal_path=str(workdir / "demo.sqlite"), reports_dir=str(workdir / "reports")
    )
    demo.data = replace(cfg.data, provider="synthetic", cache_dir=str(workdir / "cache"))
    bars = {
        s: get_bars(demo, s, provider="synthetic", lookback_days=days, now=now)
        for s in demo.symbols
    }
    journal = Journal(demo.journal_path, mode="paper")
    try:
        r = run_backtest(demo, bars, journal=journal)
        write_report(r, demo.reports_dir)
        last_day = max(d for b in bars.values() for d in b.index.date)
        equity = float(r.equity.iloc[-1]) if len(r.equity) else demo.starting_cash
        last_prices = {s: float(b["close"].iloc[-1]) for s, b in bars.items()}
        sym = demo.symbols[0]
        entry = last_prices[sym] * 0.998
        journal.set_state(
            "engine",
            {
                "ts": now.isoformat(),
                "day": str(last_day),
                "equity": equity,
                "cash": equity - entry * 2,
                "day_pnl": float(r.trades["pnl"].tail(1).sum()) if len(r.trades) else 0.0,
                "day_costs": 0.0,
                "day_trades": 1,
                "halted": False,
                "halt_reason": "",
                "open_trades": [
                    {
                        "symbol": sym,
                        "side": "long",
                        "qty": 2.0,
                        "entry_price": entry,
                        "entry_ts": (now - timedelta(minutes=35)).isoformat(),
                        "stop": entry * 0.99,
                        "target": entry * 1.02,
                        "reason": "ORB 5m long (demo)",
                        "last_price": last_prices[sym],
                        "unrealized": (last_prices[sym] - entry) * 2.0,
                        "risk": entry * 0.01 * 2.0,
                    }
                ],
                "last_prices": last_prices,
                "strategy": demo.strategy,
                "params": r.params,
                "mode": "paper",
                "broker": "paper",
                "decider": None,
            },
            ts=now,
        )
        journal.set_state(
            "heartbeat",
            {"ts": now.isoformat(), "active": True, "steps": 1234, "mode": "paper"},
            ts=now,
        )
        journal.add_params_change(
            demo.strategy,
            {"range_minutes": 5, "target_r": 10},
            r.params,
            {"champion": {"sharpe": 0.4}, "challenger": {"sharpe": 0.9}, "reason": "demo"},
            "refit",
            True,
        )
        journal.add_coach_note(
            {
                "review": "Demo review: sample is thin; the stop is hit on"
                " most losers within three bars. Nothing to change yet.",
                "outcomes": [
                    {
                        "name": "wider range",
                        "type": "params",
                        "verdict": "rejected",
                        "reason": "worse out of sample",
                    }
                ],
                "cost_usd": 0.03,
                "model": "demo",
            }
        )
        journal.add_llm_spend("coach", 0.03, now.date())
    finally:
        journal.close()
    return demo
