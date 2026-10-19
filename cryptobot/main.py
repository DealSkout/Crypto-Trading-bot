"""Command line: backtest | paper | live | report | coach | refit | check | review | account |
dashboard | guardian | resume | mcp | export | replay | research |
regimes | learn | models | rollback | approve-live | revoke-live."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cryptobot import __version__, learn, report
from cryptobot.backtest import BacktestResult, cost_model, fx_applies, run_backtest
from cryptobot.config import Config, ConfigError, live_confirmed, load_config
from cryptobot.data import DataError, get_bars
from cryptobot.journal import Journal
from cryptobot.strategies import BUILTIN, default_params, load_strategy, strategy_name

log = logging.getLogger("cryptobot")

LIVE_BANNER = """
############################################################
#  LIVE MODE: REAL MONEY. Orders go to {broker}.
#  Stops are SOFTWARE stops polled every {poll}s: a gap can
#  exceed your planned loss. Ctrl-C flattens everything.
#  LLM decider: {decider}
############################################################
"""


def make_decider(cfg: Config, journal: Journal, max_cost: float | None = None):
    """The in-loop LLM decider, or None when decider.provider is none."""
    if cfg.decider.provider == "none":
        return None
    from cryptobot.decider import Decider
    from cryptobot.llm import make_llm

    llm = make_llm(cfg, component="decider")
    section = cfg.decider
    if max_cost is not None:
        from dataclasses import replace

        section = replace(section, max_daily_cost_usd=max_cost)
    return Decider(section, llm, journal, name=cfg.decider.provider)


def estimate_decider_cost(cfg: Config, n_days: int) -> tuple[int, float]:
    from cryptobot import sessions
    from cryptobot.llm import estimate_cost

    calls = 0
    for symbol in cfg.symbols:
        per_day = max(1, sessions.market_for(symbol).session_minutes // cfg.decider.every_minutes)
        calls += per_day * n_days
    per_call = (
        estimate_cost(cfg.decider.model, 3500, 150) if cfg.decider.provider == "claude" else 0
    )
    return calls, calls * per_call


def setup_logging(verbose: bool, logfile: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if logfile:
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def load_bars(cfg: Config, provider: str | None, days: int | None) -> dict:
    bars = {}
    for symbol in cfg.symbols:
        bars[symbol] = get_bars(cfg, symbol, provider=provider, lookback_days=days)
        if bars[symbol].empty:
            raise DataError(f"no bars for {symbol}")
        log.info(
            "%s: %d bars from %s to %s",
            symbol,
            len(bars[symbol]),
            bars[symbol].index[0],
            bars[symbol].index[-1],
        )
    return bars


def latest_result(cfg: Config, strategy: str | None = None) -> BacktestResult | None:
    path = Path(cfg.reports_dir) / f"latest_{strategy or cfg.strategy}.json"
    return BacktestResult.load(path) if path.exists() else None


# ------------------------------------------------------------------------------ commands


def cmd_backtest(cfg: Config, args: argparse.Namespace) -> int:
    bars = load_bars(cfg, args.data, args.days)
    if args.strategy_file:
        strategies = [args.strategy_file]
    elif args.strategy == "all":
        strategies = [s for s in BUILTIN if s != "random_entry"]
    else:
        strategies = [args.strategy or cfg.strategy]
    journal = Journal(cfg.journal_path) if Path(cfg.journal_path).exists() else None
    decider = None
    if args.llm:
        if cfg.decider.provider == "none":
            print("--llm needs decider.provider: claude or ollama in config.yaml")
            return 2
        days = sorted({d for b in bars.values() for d in b.index.date})
        _, test_days = __import__("cryptobot.backtest", fromlist=["split_days"]).split_days(
            days, cfg.learning.train_fraction
        )
        calls, usd = estimate_decider_cost(cfg, len(test_days))
        print(
            f"LLM decider {learn.decider_description(cfg)} over {len(test_days)} out-of-sample"
            f" days: up to ~{calls} calls, roughly {usd:.2f} USD; hard cap {args.max_llm_cost:.2f}"
            " USD for this run. Answers are not deterministic."
        )
        decider = make_decider(cfg, Journal(":memory:"), max_cost=args.max_llm_cost)
    results = []
    for name in strategies:
        module = load_strategy(name)
        params = cfg.params if strategy_name(module) == cfg.strategy else None
        notes = []
        if args.grid:
            base = {**default_params(module), **(params or {})}
            days = sorted({d for b in bars.values() for d in b.index.date})
            from cryptobot.backtest import split_days

            train_days, _ = split_days(days, cfg.learning.train_fraction)
            params, table = learn.grid_search(cfg, module, bars, train_days, base)
            notes.append(
                f"grid search over {len(table)} parameter sets on the training window"
                f" picked {params}"
            )
        r = run_backtest(
            cfg, bars, strategy=module, params=params, seed=args.seed, notes=notes, decider=decider
        )
        if journal is not None:
            r.params_history = [h for h in journal.params_history() if h["applied"]]
        results.append(r)
        if args.json:
            continue
        paths = report.write_report(r, cfg.reports_dir)
        print(report.summary_line(r))
        print(f"  report: {paths['markdown']}  |  {paths['html']}")
    if args.json:
        print(json.dumps(results[-1].to_json(), default=str))
        return 0
    if len(results) > 1:
        table = report.comparison_markdown(results)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = Path(cfg.reports_dir) / f"comparison_{stamp}.md"
        path.write_text("# Strategy comparison (out of sample, after costs)\n\n" + table)
        print("\n" + table)
        print(f"  comparison: {path}")
    print(
        "\nAll numbers are out of sample and after costs. "
        + (
            "Synthetic data: this says nothing about markets."
            if args.data == "synthetic"
            else "Small samples are noise; read the trade counts."
        )
    )
    return 0


def cmd_report(cfg: Config, args: argparse.Namespace) -> int:
    r = latest_result(cfg, args.strategy)
    if r is None:
        print("no saved backtest; run `python -m cryptobot backtest` first")
        return 1
    paths = report.write_report(r, cfg.reports_dir)
    print(report.render_markdown(r))
    print(f"written: {paths['markdown']}  |  {paths['html']}")
    return 0


def make_broker(cfg: Config, mode: str):
    if mode == "paper" and cfg.broker == "paper":
        from cryptobot.brokers.paper import PaperBroker

        return PaperBroker(cfg.starting_cash, cost_model(cfg), fx_for=lambda s: fx_applies(cfg, s))
    if cfg.broker == "alpaca":
        from cryptobot.brokers.alpaca import AlpacaCryptoBroker

        return AlpacaCryptoBroker(
            os.environ.get("ALPACA_API_KEY", ""),
            os.environ.get("ALPACA_API_SECRET", ""),
            live=(mode == "live"),
            time_in_force=cfg.orders.time_in_force,
        )
    raise ConfigError(f"no broker for mode={mode} broker={cfg.broker}")


def run_loop(cfg: Config, mode: str, args: argparse.Namespace) -> int:
    from cryptobot.engine import Engine
    from cryptobot.notify import make_notifier
    from cryptobot.risk import RiskManager
    from cryptobot.runner import Runner

    setup_logging(args.verbose, logfile=f"cryptobot_{mode}.log")
    from cryptobot.registry import Registry

    registry = Registry(cfg.journal_path)
    model = trading_model(cfg, mode, registry)
    module = load_strategy(model.strategy)
    params = dict(model.params)
    journal = Journal(cfg.journal_path, mode=mode)
    notifier = make_notifier(cfg)
    broker = make_broker(cfg, mode)
    cm = cost_model(cfg)
    risk = RiskManager(
        cfg.risk,
        cfg.allow_short,
        cfg.max_entries_per_day,
        cost_model=cm,
        fx_for=lambda s: fx_applies(cfg, s),
    )
    decider = make_decider(cfg, journal)
    engine = Engine(
        cfg, module, params, broker, risk, journal, mode=mode, notifier=notifier, decider=decider
    )
    engine.version = model.id
    engine.filters = dict(model.filters)
    provider = args.data or cfg.data.provider

    from cryptobot.backtest import HISTORY_SESSIONS

    def fetch(symbol: str):
        # the same history a backtest strategy sees (ATR etc. need it); incremental after the
        # first call, so polling stays cheap
        return get_bars(cfg, symbol, provider=provider, lookback_days=HISTORY_SESSIONS + 2)

    from cryptobot.guardian import make_guardian

    guardian = make_guardian(cfg, journal, notifier=notifier)
    if guardian.stopped:
        info = guardian.stop_info() or {}
        print(
            f"refusing to trade: the guardian stopped the bot ({info.get('source')}) on"
            f" {str(info.get('ts', ''))[:16]}: {info.get('reason')}\n"
            "Review the dashboard and journal, then run `python -m cryptobot resume`."
        )
        return 4
    from cryptobot.regime import RegimeDetector

    research = None
    regimes = RegimeDetector(cfg.regime, cfg.symbols)
    engine.context_providers.append(regimes.context)
    if cfg.research.enabled:
        from cryptobot.research.feed import ResearchFeed
        from cryptobot.research.store import ResearchStore

        research = ResearchFeed(cfg, ResearchStore(cfg.journal_path))
        research.start()  # background thread; never blocks or breaks the trading loop
        engine.context_providers.append(research.context)
        store = research.store

        def fear_greed(when):
            row = store.asof("fear_greed", "index", when)
            return row["value"] if row else None

        regimes.fear_greed = fear_greed
    shadows = None
    if mode == "paper":  # shadow runs never happen next to live money
        from cryptobot.learning import ShadowRunner

        shadows = ShadowRunner(cfg, registry, cfg.journal_path, engine.context_providers)
        shadows.sync()
    runner = Runner(
        cfg,
        mode,
        engine,
        fetch,
        journal,
        notifier,
        guardian=guardian,
        research=research,
        regimes=regimes,
        registry=registry,
        shadows=shadows,
    )
    runner.install_signal_handlers()
    print(
        f"{mode} mode: broker={broker.name} model={model.id} strategy={model.strategy}"
        f" params={params} filters={model.filters or 'none'}"
        f" symbols={cfg.symbols} data={provider}. Ctrl-C to stop (flattens first)."
    )
    runner.run_forever()
    return 0


def trading_model(cfg: Config, mode: str, registry):
    """Which model trades. Paper: the registry's champion (config.yaml is registered first, and
    a human edit of it becomes the champion). Live: ONLY a version a human approved with
    `approve-live`; without one, the human-set config.yaml strategy and params."""
    from cryptobot.registry import Model

    module = load_strategy(cfg.strategy)
    config_params = {**default_params(module), **cfg.params}
    if mode == "paper":
        registry.sync_with_config(cfg.strategy, config_params)
        return registry.champion()
    approved = registry.live_version()
    if approved is not None:
        return approved
    return Model(id="config", strategy=cfg.strategy, params=config_params)


def cmd_paper(cfg: Config, args: argparse.Namespace) -> int:
    if cfg.mode != "paper":
        print("config says mode: live; the `paper` command still runs in paper mode.")
    return run_loop(cfg, "paper", args)


def cmd_live(cfg: Config, args: argparse.Namespace) -> int:
    if cfg.mode != "live":
        print("refusing: set `mode: live` in config.yaml first (Rule 1).")
        return 2
    if not live_confirmed():
        print("refusing: set CRYPTOBOT_LIVE_CONFIRM=yes in .env as well (Rule 1).")
        return 2
    ok, problems = learn.live_gate(cfg, Journal(cfg.journal_path, mode="paper"), latest_result(cfg))
    if not ok:
        print("refusing to go live (Rule 6):")
        for p in problems:
            print(f"  - {p}")
        return 3
    print(
        LIVE_BANNER.format(
            broker=cfg.broker, poll=cfg.poll_seconds, decider=learn.decider_description(cfg)
        )
    )
    return run_loop(cfg, "live", args)


def cmd_check(cfg: Config, args: argparse.Namespace) -> int:
    ok, problems = learn.live_gate(cfg, Journal(cfg.journal_path, mode="paper"), latest_result(cfg))
    print("Rule 6 live gate: " + ("PASSED" if ok else "NOT MET"))
    for p in problems:
        print(f"  - {p}")
    from cryptobot.guardian import stopped_info

    info = stopped_info(cfg)
    if info:
        print(
            f"GUARDIAN STOP in force ({info.get('source')}): {info.get('reason')}"
            " -> `python -m cryptobot resume` after reviewing"
        )
        return 1
    return 0 if ok else 1


def cmd_refit(cfg: Config, args: argparse.Namespace) -> int:
    bars = load_bars(cfg, args.data, args.days)
    journal = Journal(cfg.journal_path, mode="paper")
    # report only unless a human asks with --apply (the weekly `learn` loop is the automated
    # path, and it goes through the full gate and a shadow run)
    out = learn.refit(cfg, bars, journal, apply=args.apply)
    print(f"refit: changed={out['changed']} reason={out.get('reason')} params={out['params']}")
    if out.get("challenger"):
        c, ch = out["champion"], out["challenger"]
        print(
            f"  champion OOS Sharpe {c['sharpe']:.2f} ({c['n_trades']} trades) vs challenger"
            f" {ch['sharpe']:.2f} ({ch['n_trades']} trades)"
        )
    return 0


def cmd_coach(cfg: Config, args: argparse.Namespace) -> int:
    from cryptobot.coach import run_coach
    from cryptobot.llm import LLMError, make_llm

    if not cfg.coach.enabled:
        print("the coach is off (coach.enabled: false in config.yaml)")
        return 0
    provider = args.provider or cfg.coach.provider
    if provider == "none":
        print("coach.provider is none; set claude or ollama in config.yaml (or --provider)")
        return 2
    try:
        llm = make_llm(cfg, provider)
    except LLMError as e:
        print(f"cannot start the coach: {e}")
        return 2
    journal = Journal(cfg.journal_path, mode="paper")
    try:
        bars = load_bars(cfg, args.data, args.days)
    except DataError as e:
        log.warning("no bars for the gate: %s", e)
        bars = None
    out = run_coach(
        cfg,
        journal,
        llm,
        bars_by_symbol=bars,
        latest=latest_result(cfg),
        dry_run=args.dry_run,
        data_provider=args.data,
    )
    print(f"coach: {out['status']}" + (f" ({out['reason']})" if out.get("reason") else ""))
    if out.get("review"):
        print("\n" + out["review"] + "\n")
    for o in out.get("outcomes", []):
        print(f"- {o['name']} [{o['type']}] -> {o['verdict']}: {o['reason']}")
    if out.get("report"):
        print(f"\nreview written to {out['report']} (cost {out.get('cost_usd', 0):.4f} USD)")
    return 0 if out["status"] in ("ok", "skipped") else 1


def cmd_dashboard(cfg: Config, args: argparse.Namespace) -> int:
    from cryptobot import dashboard

    if args.demo:
        workdir = Path(args.demo_dir or (Path(cfg.reports_dir) / "demo"))
        print(f"seeding demo journal and backtest under {workdir} (synthetic data)")
        cfg = dashboard.build_demo(cfg, workdir)
    mode = args.mode or cfg.mode
    if args.static:
        out = Path(args.static)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(dashboard.render_html(dashboard.collect(cfg, mode=mode), static=True))
        print(f"dashboard snapshot written to {out}")
        return 0
    dashboard.serve(
        cfg,
        host=args.host,
        port=args.port,
        mode=mode,
        refresh_seconds=args.refresh,
        open_browser=args.open,
    )
    return 0


def cmd_guardian(cfg: Config, args: argparse.Namespace) -> int:
    """Run the guardian's review now (rules + AI reviewer) against the paper or live journal."""
    from cryptobot.guardian import make_guardian
    from cryptobot.notify import make_notifier

    mode = args.mode or cfg.mode
    journal = Journal(cfg.journal_path, mode=mode)
    guardian = make_guardian(cfg, journal, notifier=make_notifier(cfg))
    if guardian.stopped and not args.dry_run:
        info = guardian.stop_info() or {}
        print(f"already stopped ({info.get('source')}): {info.get('reason')}")
        return 1
    if args.dry_run:
        reasons = guardian.rule_reasons()
        print("rules: " + ("; ".join(reasons) if reasons else "nothing fired"))
        v = guardian.ai_review()
        if v is not None:
            print(
                f"AI reviewer ({v.model}): {v.verdict} (confidence {v.confidence:.2f}) {v.reason}"
            )
        else:
            print("AI reviewer: no opinion (not configured, no key, or an error)")
        return 0
    out = guardian.daily_review()
    if out.get("stopped"):
        print(
            "STOPPED: "
            + "; ".join(out.get("reasons") or [])
            + (f" AI: {out['verdict'].reason}" if out.get("verdict") else "")
        )
        return 1
    v = out.get("verdict")
    print("review passed: nothing fired" + (f"; AI said continue: {v.reason}" if v else ""))
    return 0


def cmd_review(cfg: Config, args: argparse.Namespace) -> int:
    """Run the daily trade coach now (it also runs by itself after each close)."""
    from datetime import date as date_cls

    from cryptobot.guardian import backtest_metrics_for
    from cryptobot.trade_coach import make_review_llm, run_daily_review

    if not cfg.coach.enabled:
        print("the coach is off (coach.enabled: false in config.yaml)")
        return 0
    day = date_cls.fromisoformat(args.date) if args.date else None
    journal = Journal(cfg.journal_path, mode=args.mode or cfg.mode)
    out = run_daily_review(
        cfg,
        journal,
        day,
        llm=None if args.no_ai else make_review_llm(cfg),
        backtest=backtest_metrics_for(cfg),
        force=args.force,
    )
    if out.get("status") != "done":
        print(f"coach: {out.get('status')} ({out.get('reason')})")
        return 0
    print(f"coach review for {out['date']}: {out['ai']} (cost ${out['cost_usd']:.4f})")
    for t in out["trades"]:
        print(f"  #{t['trade_id']} {t['symbol']} {t['r_multiple']:+.2f}R  {t['verdict']}")
        if t["mistake"]:
            print(f"      mistake: {t['mistake']}")
        if t["lesson"]:
            print(f"      lesson:  {t['lesson']}")
    print(f"summary: {out['daily_summary']}")
    if out["proposed_changes"]:
        print(f"{len(out['proposed_changes'])} proposal(s) saved to {cfg.coach.dir}/proposals/")
    for d in out["dropped_proposals"]:
        print(f"dropped a proposal: {d['dropped']}")
    return 0


def cmd_resume(cfg: Config, args: argparse.Namespace) -> int:
    from cryptobot.guardian import Guardian

    journal = Journal(cfg.journal_path, mode=args.mode or cfg.mode)
    g = Guardian(cfg, journal)
    info = g.stop_info()
    if g.resume("human"):
        print(f"resumed. The guardian had stopped the bot for: {(info or {}).get('reason')}")
        return 0
    print("nothing to resume: the bot is not stopped")
    return 0


ACCOUNT_BROKERS = ("alpaca",)


def cmd_account(cfg: Config, args: argparse.Namespace) -> int:
    """Connect to the broker and print the account: the quickest way to test your keys."""
    from cryptobot.brokers.base import BrokerError

    mode = args.mode or cfg.mode
    if cfg.broker not in ACCOUNT_BROKERS:
        print(f"broker: {cfg.broker} has no account to read; set broker: alpaca")
        return 1
    try:
        broker = make_broker(cfg, mode)
        equity, cash = broker.get_equity(), broker.get_cash()
        positions, pending = broker.get_positions(), broker.open_orders()
    except BrokerError as e:
        print(f"could not read the {cfg.broker} account: {e}")
        return 1
    kind = "LIVE (real money)" if getattr(broker, "live", False) else "paper / practice"
    print(f"connected to {cfg.broker} ({kind})")
    print(f"equity {equity:,.2f}   cash available to the bot {cash:,.2f}")
    print(f"{len(positions)} open position(s)" + (":" if positions else ""))
    for p in positions:
        print(f"  {p.symbol:<10} {p.side:<5} {p.qty:g} @ {p.avg_price:,.2f}")
    print(f"{len(pending)} pending order(s)")
    return 0


def cmd_mcp(cfg: Config, args: argparse.Namespace) -> int:
    """Serve the AI risk agent's tools over MCP (stdio) for Claude Desktop, Claude Code, etc."""
    from cryptobot.notify import make_notifier
    from cryptobot.risk_agent import RiskAgentServer

    # MCP clients start servers from any folder: resolve journal, stop flag and .env like the bot.
    os.chdir(Path(args.config).resolve().parent)
    load_dotenv(Path(".env"))
    mode = args.mode or cfg.mode
    factory = (lambda: make_broker(cfg, mode)) if cfg.broker in ACCOUNT_BROKERS else None
    server = RiskAgentServer(
        cfg,
        mode,
        read_only=args.read_only,
        broker_factory=factory,
        notifier=make_notifier(cfg),
    )
    log.info(
        "risk agent MCP server on stdio (mode=%s, broker=%s%s)",
        mode,
        cfg.broker,
        ", read-only" if args.read_only else "",
    )
    server.serve()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cryptobot", description="A small, paper-first day-trading bot."
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--version", action="version", version=f"cryptobot {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_data(sp):
        sp.add_argument(
            "--data",
            choices=["alpaca", "csv", "synthetic"],
            default=None,
            help="override data.provider",
        )
        sp.add_argument("--days", type=int, default=None, help="override data.lookback_days")

    bt = sub.add_parser("backtest", help="walk-forward backtest, out of sample, after costs")
    bt.add_argument("--strategy", default=None, help="orb | intraday_momentum | vwap_revert | all")
    bt.add_argument("--strategy-file", default=None, help="path to a .py file with signal()")
    bt.add_argument("--grid", action="store_true", help="grid-search params on the training window")
    bt.add_argument("--seed", type=int, default=0)
    bt.add_argument("--json", action="store_true", help="print the result as JSON only")
    bt.add_argument(
        "--llm",
        action="store_true",
        help="consult the configured LLM decider out of sample (costs money, slow)",
    )
    bt.add_argument(
        "--max-llm-cost",
        type=float,
        default=5.0,
        help="hard USD cap for --llm in this run (default 5)",
    )
    add_data(bt)
    pp = sub.add_parser("paper", help="paper trade (simulator or broker demo account)")
    add_data(pp)
    lv = sub.add_parser("live", help="trade real money (needs mode: live + CRYPTOBOT_LIVE_CONFIRM)")
    add_data(lv)
    rp = sub.add_parser("report", help="re-render the latest saved backtest report")
    rp.add_argument("--strategy", default=None)
    ch = sub.add_parser("coach", help="run the offline AI coach once")
    ch.add_argument("--dry-run", action="store_true", help="never apply anything")
    ch.add_argument("--provider", choices=["claude", "ollama"], default=None)
    add_data(ch)
    rf = sub.add_parser("refit", help="legacy quick re-fit of params (reports only by default)")
    rf.add_argument("--apply", action="store_true", help="write passing params to config.yaml")
    add_data(rf)
    sub.add_parser("check", help="show whether Rule 6 would allow going live")
    gd = sub.add_parser("guardian", help="run the watchdog review now; it may stop the bot")
    gd.add_argument("--mode", choices=["paper", "live"], default=None)
    gd.add_argument("--dry-run", action="store_true", help="report without stopping anything")
    rv = sub.add_parser("review", help="run the daily trade coach now (also runs after each close)")
    rv.add_argument("--date", default=None, help="YYYY-MM-DD (default: today, UTC)")
    rv.add_argument("--mode", choices=["paper", "live"], default=None)
    rv.add_argument("--force", action="store_true", help="review again even if already done")
    rv.add_argument("--no-ai", action="store_true", help="stats only, no Claude call")
    rs = sub.add_parser("resume", help="clear a guardian stop (humans only)")
    rs.add_argument("--mode", choices=["paper", "live"], default=None)
    db = sub.add_parser(
        "dashboard", help="serve the local dashboard (reads the journal; never trades)"
    )
    db.add_argument("--host", default="127.0.0.1", help="bind address (keep localhost on a VPS)")
    db.add_argument("--port", type=int, default=8787)
    db.add_argument("--mode", choices=["paper", "live"], default=None, help="which journal to show")
    db.add_argument("--refresh", type=int, default=30, help="auto-refresh seconds")
    db.add_argument("--open", action="store_true", help="open a browser tab")
    db.add_argument("--static", default=None, help="write a self-contained HTML snapshot and exit")
    db.add_argument("--demo", action="store_true", help="seed synthetic paper trades first")
    db.add_argument("--demo-dir", default=None)
    ac = sub.add_parser("account", help="connect to the broker and show the account (tests keys)")
    ac.add_argument("--mode", choices=["paper", "live"], default=None)
    mc = sub.add_parser(
        "mcp", help="serve the AI risk agent's tools over MCP (for Claude Desktop / Claude Code)"
    )
    mc.add_argument("--mode", choices=["paper", "live"], default=None, help="which journal")
    mc.add_argument("--read-only", action="store_true", help="leave out the stop_bot tool")
    rs2 = sub.add_parser("research", help="the market research feed: status, refresh, digest")
    rs2.add_argument("--refresh", action="store_true", help="refresh every source now")
    rs2.add_argument("--backfill", action="store_true", help="pull history (for backtests)")
    rs2.add_argument("--days", type=int, default=None, help="backfill depth in days")
    rs2.add_argument("--digest", action="store_true", help="build today's digest now")
    rs2.add_argument("--force", action="store_true", help="rebuild the digest if it exists")
    rs2.add_argument("--no-ai", action="store_true", help="digest from the numbers only")
    rg = sub.add_parser("regimes", help="today's regime per market and results per regime")
    add_data(rg)
    le = sub.add_parser("learn", help="weekly self-learning run: challengers, gate, shadow start")
    le.add_argument("--dry-run", action="store_true", help="report only; registry unchanged")
    le.add_argument("--workers", type=int, default=None)
    add_data(le)
    sub.add_parser("models", help="list the model registry (versions, status, reasons)")
    rb = sub.add_parser("rollback", help="make an earlier version the paper champion again")
    rb.add_argument("version")
    rb.add_argument("--live", action="store_true", help="roll the LIVE version back instead")
    al = sub.add_parser("approve-live", help="HUMAN ONLY: approve a version for live trading")
    al.add_argument("version")
    al.add_argument("--i-approve", action="store_true", help="required: you approve real money")
    sub.add_parser("revoke-live", help="withdraw the live approval (live falls back to config)")
    ex = sub.add_parser("export", help="write the journal's decisions and trades to Parquet")
    ex.add_argument("--out", default="store", help="output folder (default: store)")
    ex.add_argument("--mode", choices=["paper", "live", "backtest"], default=None)
    rr = sub.add_parser("replay", help="re-run a past day's decisions from stored bars and compare")
    rr.add_argument("--date", required=True, help="YYYY-MM-DD (UTC)")
    rr.add_argument("--mode", choices=["paper", "live"], default=None)
    add_data(rr)
    return p


def cmd_research(cfg: Config, args: argparse.Namespace) -> int:
    from cryptobot.research.digest import make_digest_llm, run_daily_digest
    from cryptobot.research.feed import ResearchFeed
    from cryptobot.research.store import ResearchStore

    feed = ResearchFeed(cfg, ResearchStore(cfg.journal_path))
    if args.backfill:
        for source, result in feed.backfill(args.days).items():
            print(f"backfill {source}: {result}")
    elif args.refresh:
        for source, result in feed.refresh(force=True).items():
            print(f"refresh {source}: {result}")
    if args.digest:
        now = datetime.now(UTC)
        llm = None if args.no_ai else make_digest_llm(cfg)
        journal = Journal(cfg.journal_path, mode=cfg.mode)
        out = run_daily_digest(cfg, journal, feed, now.date(), llm=llm, now=now, force=args.force)
        d = out.get("digest", out)
        print(f"digest ({out.get('source', out['status'])}): risk {d.get('risk_level')}")
        print(d.get("summary", ""))
    now = datetime.now(UTC)
    print("source          last ok (UTC)      age")
    ages = feed.staleness_minutes(now)
    status = feed.store.status()
    for source, age in ages.items():
        info = status.get(source) or {}
        last = (info.get("last_ok") or "never")[:16]
        err = f"  last error: {info['last_error'][:80]}" if info.get("last_error") else ""
        age_txt = "-" if age is None else f"{age:.0f} min"
        print(f"{source:<15} {last:<18} {age_txt}{err}")
    cover = feed.calendar.days_of_cover(now)
    print(f"macro calendar: {len(feed.calendar.events)} events, {cover:.0f} days of cover")
    if cover < cfg.research.calendar_warn_days:
        print(f"  WARNING: add upcoming events to {cfg.research.calendar_file}")
    return 0


def cmd_regimes(cfg: Config, args: argparse.Namespace) -> int:
    from cryptobot.regime import RegimeDetector, regime_performance

    bars = load_bars(cfg, args.data, args.days or 60)
    det = RegimeDetector.from_bars(cfg.regime, bars)
    today = datetime.now(UTC).date()
    print(f"Regimes for {today} (UTC), from completed days only:")
    for symbol in cfg.symbols:
        info = det.classify(symbol, today)
        detail = ", ".join(
            f"{k.removeprefix('regime_')} {v:.2f}"
            for k, v in info.items()
            if k != "regime" and isinstance(v, float)
        )
        print(f"  {symbol:<9} {info['regime']:<9} {detail}")
    journal = Journal(cfg.journal_path, mode=cfg.mode)
    perf = regime_performance(journal.trades(mode=cfg.mode), by=["strategy"])
    print(f"\n{cfg.mode} results by strategy and regime:")
    print("  (no trades yet)" if perf.empty else perf.to_string(index=False))
    result = latest_result(cfg)
    if result is not None and result.regime_table:
        print(f"\nlatest {result.strategy} backtest, out of sample, by regime:")
        print(pd.DataFrame(result.regime_table).to_string(index=False))
    return 0


def cmd_learn(cfg: Config, args: argparse.Namespace) -> int:
    from cryptobot.learning import run_learning
    from cryptobot.registry import Registry

    bars = load_bars(cfg, args.data, args.days)
    registry = Registry(cfg.journal_path)
    module = load_strategy(cfg.strategy)
    registry.sync_with_config(cfg.strategy, {**default_params(module), **cfg.params})
    research = cfg.journal_path if Path(cfg.journal_path).exists() else None
    report = run_learning(
        cfg, bars, registry, research_path=research, dry_run=args.dry_run, workers=args.workers
    )
    print(f"champion: {report['champion']}")
    print(f"challengers tried: {len(report['challengers'])}; best: {report.get('best', 'none')}")
    print(
        f"verdict: {'ACCEPTED' if report['accepted'] else 'rejected'}; action: {report['action']}"
    )
    for r in report["reasons"]:
        print(f"  - {r}")
    print(f"report: {Path(cfg.reports_dir) / 'learning'}")
    return 0


def cmd_models(cfg: Config, args: argparse.Namespace) -> int:
    from cryptobot.registry import Registry

    reg = Registry(cfg.journal_path)
    live = reg.live_version()
    print(f"{'ver':<5} {'status':<10} {'live':<5} model / reason")
    for m in reg.all():
        flag = "LIVE" if live is not None and live.id == m.id else ""
        print(f"{m.id:<5} {m.status:<10} {flag:<5} {m.describe().split(' ', 1)[1]}")
        if m.reason:
            print(f"{'':<22}{m.reason[:110]}")
    return 0


def cmd_rollback(cfg: Config, args: argparse.Namespace) -> int:
    from cryptobot.registry import Registry, RegistryError

    reg = Registry(cfg.journal_path)
    try:
        if args.live:
            m = reg.approve_live(args.version, who="human (rollback)")
            print(f"live now uses {m.describe()} (restart the live service to apply)")
        else:
            m = reg.rollback(args.version)
            print(f"paper champion is now {m.describe()} (applied at the next UTC day or restart)")
    except RegistryError as e:
        print(f"rollback refused: {e}")
        return 1
    return 0


def cmd_approve_live(cfg: Config, args: argparse.Namespace) -> int:
    from cryptobot.registry import Registry, RegistryError

    if not args.i_approve:
        print("refusing: add --i-approve to confirm you approve this version for real money")
        return 2
    reg = Registry(cfg.journal_path)
    try:
        m = reg.approve_live(args.version, who=os.environ.get("USER", "human"))
    except RegistryError as e:
        print(f"refused: {e}")
        return 1
    print(f"approved for live: {m.describe()}")
    print("Live trading still needs mode: live, CRYPTOBOT_LIVE_CONFIRM=yes and the Rule 6 gate.")
    return 0


def cmd_revoke_live(cfg: Config, args: argparse.Namespace) -> int:
    from cryptobot.registry import Registry

    Registry(cfg.journal_path).revoke_live(who=os.environ.get("USER", "human"))
    print("live approval withdrawn: live mode falls back to config.yaml's strategy and params")
    return 0


def cmd_export(cfg: Config, args: argparse.Namespace) -> int:
    from cryptobot.store import export_store

    journal = Journal(cfg.journal_path, mode=args.mode or cfg.mode)
    for name, path in export_store(journal, args.out).items():
        print(f"{name}: {path}")
    return 0


def cmd_replay(cfg: Config, args: argparse.Namespace) -> int:
    from datetime import date as date_cls

    from cryptobot.store import replay_day

    day = date_cls.fromisoformat(args.date)
    days_back = (datetime.now(UTC).date() - day).days + 30
    bars = load_bars(cfg, args.data, days_back)
    journal = Journal(cfg.journal_path, mode=args.mode or cfg.mode)
    rows = replay_day(cfg, journal, day, bars, mode=args.mode or cfg.mode)
    if not rows:
        print(f"no enter/skip decisions journalled on {day}")
        return 0
    exact = sum(1 for r in rows if r.get("status") == "exact")
    for r in rows:
        extra = f" diffs={r['feature_diffs']}" if r.get("feature_diffs") else ""
        print(f"{r['ts'][:16]} {r['symbol']:<9} {r['action']:<5} {r['status']}{extra}")
    print(f"{exact} of {len(rows)} decisions replayed exactly")
    return 0 if all(r.get("status") != "MISMATCH" for r in rows) else 1


COMMANDS = {
    "backtest": cmd_backtest,
    "paper": cmd_paper,
    "live": cmd_live,
    "report": cmd_report,
    "coach": cmd_coach,
    "refit": cmd_refit,
    "check": cmd_check,
    "dashboard": cmd_dashboard,
    "guardian": cmd_guardian,
    "resume": cmd_resume,
    "review": cmd_review,
    "mcp": cmd_mcp,
    "account": cmd_account,
    "export": cmd_export,
    "research": cmd_research,
    "regimes": cmd_regimes,
    "learn": cmd_learn,
    "models": cmd_models,
    "rollback": cmd_rollback,
    "approve-live": cmd_approve_live,
    "revoke-live": cmd_revoke_live,
    "replay": cmd_replay,
}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # Rule 5: secrets come from .env, never from config.yaml
    args = build_parser().parse_args(argv)
    if args.command not in ("paper", "live"):
        setup_logging(args.verbose)
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}")
        return 2
    try:
        return COMMANDS[args.command](cfg, args)
    except DataError as e:
        print(f"data error: {e}")
        return 1
    except ConfigError as e:
        print(f"config error: {e}")
        return 2
    except KeyboardInterrupt:
        print("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
