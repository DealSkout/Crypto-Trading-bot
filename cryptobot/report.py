"""Backtest reports: Markdown and a self-contained HTML page with an inline SVG equity curve.
Every report states net return, max drawdown, Sharpe, trades, total costs paid, and the
comparison with buy-and-hold and random entry, all out of sample and after costs."""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from cryptobot.backtest import BacktestResult

ROWS = [
    ("Net return", "net_return_pct", "{:+.2f}%"),
    ("Max drawdown", "max_drawdown_pct", "{:.2f}%"),
    ("Sharpe (annualised, daily)", "sharpe", "{:.2f}"),
    ("Trades", "n_trades", "{:d}"),
    ("Win rate", "win_rate_pct", "{:.1f}%"),
    ("Average R", "avg_r", "{:+.2f}"),
    ("Profit factor", "profit_factor", "{:.2f}"),
    ("Total costs paid", "total_costs", "{:.2f}"),
    ("Trading days", "trading_days", "{:d}"),
]
COLORS = {"strategy": "#1f6feb", "buy_and_hold": "#8b949e", "random_entry": "#d29922"}


def _fmt(fmt: str, value) -> str:
    try:
        if value is None:
            return "-"
        if fmt.endswith("d}"):
            return fmt.format(int(value))
        if value == float("inf"):
            return "inf"
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def metrics_columns(r: BacktestResult) -> list[tuple[str, dict]]:
    return [
        (f"{r.strategy} (out-of-sample)", r.metrics),
        ("Buy & hold", r.benchmarks.get("buy_and_hold", {})),
        ("Random entry, same exits", r.benchmarks.get("random_entry", {})),
        (f"{r.strategy} (in-sample)", r.train_metrics),
    ]


def verdict_lines(r: BacktestResult) -> list[str]:
    bh = r.benchmarks.get("buy_and_hold", {})
    rnd = r.benchmarks.get("random_entry", {})
    lines = [
        f"Beats buy-and-hold on Sharpe (after costs, out of sample): "
        f"{'YES' if r.beats_buy_and_hold else 'NO'} "
        f"({r.metrics['sharpe']:.2f} vs {bh.get('sharpe', 0):.2f})",
        f"Beats random entry with the same exits: {'YES' if r.beats_random else 'NO'} "
        f"({r.metrics['sharpe']:.2f} vs {rnd.get('sharpe', 0):.2f})",
        f"Costs paid out of sample: {r.metrics['total_costs']:.2f} {r.currency} on "
        f"{r.metrics['n_trades']} trades",
        "Rule 6 (backtest half): "
        + ("PASSED" if r.beats_buy_and_hold else "NOT MET")
        + ". Live also needs 20+ paper days within the expected range.",
    ]
    if r.metrics["n_trades"] < 30:
        lines.append(
            f"Only {r.metrics['n_trades']} out-of-sample trades: treat every number"
            " above as noise until there are more."
        )
    return lines


def header_lines(r: BacktestResult) -> list[str]:
    return [
        f"Strategy: {r.strategy}  params: {r.params}",
        f"Symbols: {', '.join(r.symbols)}  timeframe: {r.timeframe}  starting cash:"
        f" {r.starting_cash:.0f} {r.currency}",
        f"In-sample (train): {r.train_start} to {r.train_end}  |  Out-of-sample (test):"
        f" {r.start} to {r.end}",
        f"Costs: commission {r.costs['commission']}, FX fee {r.costs['fee_pct']}%/side,"
        f" spread {r.costs['spread_bps']} bps, slippage {r.costs['slippage_bps']} bps/side",
        f"Generated: {r.created_at[:19].replace('T', ' ')} UTC",
    ] + (
        [
            f"LLM decider in the loop: {r.decider} (calls {r.metrics.get('llm_calls', 0)},"
            f" cost {r.metrics.get('llm_cost_usd', 0)} USD)"
        ]
        if r.decider != "none"
        else []
    )


def render_markdown(r: BacktestResult) -> str:
    cols = metrics_columns(r)
    out = [f"# Cryptobot backtest: {r.strategy}", ""]
    out += [f"- {line}" for line in header_lines(r)]
    out += ["", "## Results (after costs)", ""]
    out.append("| Metric | " + " | ".join(name for name, _ in cols) + " |")
    out.append("|---|" + "---|" * len(cols))
    for label, key, fmt in ROWS:
        out.append(f"| {label} | " + " | ".join(_fmt(fmt, m.get(key)) for _, m in cols) + " |")
    out += ["", "## Verdict", ""]
    out += [f"- {line}" for line in verdict_lines(r)]
    if r.notes:
        out += ["", "## Notes", ""] + [f"- {n}" for n in r.notes]
    if r.params_history:
        out += [
            "",
            "## Parameter changes applied by the learning loop",
            "",
            "| When | Source | Old | New | OOS Sharpe before -> after |",
            "|---|---|---|---|---|",
        ]
        for h in r.params_history:
            st = h.get("stats", {})
            before = st.get("champion", {}).get("sharpe")
            after = st.get("challenger", {}).get("sharpe")
            out.append(
                f"| {h['ts'][:10]} | {h['source']} | {h['old_params']} | {h['new_params']} |"
                f" {_fmt('{:.2f}', before)} -> {_fmt('{:.2f}', after)} |"
            )
    out += ["", "## Out-of-sample trades (last 20)", ""]
    out.append(trades_markdown(r.trades.tail(20)))
    return "\n".join(out) + "\n"


def trades_markdown(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "_no trades_"
    lines = [
        "| Entry | Symbol | Side | Qty | Entry | Exit | Exit reason | PnL | R | Costs |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, t in trades.iterrows():
        lines.append(
            f"| {str(t['entry_ts'])[:16]} | {t['symbol']} | {t['side']} | {float(t['qty']):.4g} |"
            f" {float(t['entry_price']):.2f} | {float(t['exit_price']):.2f} | {t['exit_reason']} |"
            f" {float(t['pnl']):+.2f} | {float(t['r_multiple']):+.2f} | {float(t['costs']):.2f} |"
        )
    return "\n".join(lines)


def comparison_markdown(results: list[BacktestResult]) -> str:
    lines = [
        "| Strategy | Net return | Max DD | Sharpe | Trades | Win rate | Avg R | Costs |"
        " Buy & hold Sharpe | Beats B&H | Beats random |",
        "|---|" + "---|" * 10,
    ]
    for r in results:
        m, bh = r.metrics, r.benchmarks.get("buy_and_hold", {})
        lines.append(
            f"| {r.strategy} | {m['net_return_pct']:+.2f}% | {m['max_drawdown_pct']:.2f}% |"
            f" {m['sharpe']:.2f} | {m['n_trades']} | {m['win_rate_pct']:.1f}% | {m['avg_r']:+.2f} |"
            f" {m['total_costs']:.2f} | {bh.get('sharpe', 0):.2f} |"
            f" {'yes' if r.beats_buy_and_hold else 'no'} | {'yes' if r.beats_random else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def summary_line(r: BacktestResult) -> str:
    m = r.metrics
    return (
        f"{r.strategy}: net {m['net_return_pct']:+.2f}%  maxDD {m['max_drawdown_pct']:.2f}%"
        f"  Sharpe {m['sharpe']:.2f}  trades {m['n_trades']}  costs {m['total_costs']:.2f}"
        f"  | B&H Sharpe {r.benchmarks['buy_and_hold'].get('sharpe', 0):.2f}"
        f"  | beats B&H: {'yes' if r.beats_buy_and_hold else 'no'}"
    )


# ------------------------------------------------------------------------------- HTML


def svg_equity(
    curves: dict[str, pd.Series], starting_cash: float, width: int = 900, height: int = 320
) -> str:
    series = {k: v.dropna() for k, v in curves.items() if v is not None and len(v.dropna()) > 1}
    if not series:
        return "<p>no equity data</p>"
    pct = {k: (v / starting_cash - 1.0) * 100.0 for k, v in series.items()}
    lo = min(float(v.min()) for v in pct.values())
    hi = max(float(v.max()) for v in pct.values())
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    t0 = min(v.index[0] for v in pct.values())
    t1 = max(v.index[-1] for v in pct.values())
    span = max((t1 - t0).total_seconds(), 1.0)
    pad_l, pad_r, pad_t, pad_b = 60, 20, 20, 30

    def x(ts):
        return pad_l + (ts - t0).total_seconds() / span * (width - pad_l - pad_r)

    def y(val):
        return pad_t + (hi - val) / (hi - lo) * (height - pad_t - pad_b)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img"'
        ' aria-label="equity curves, percent of starting cash">'
    ]
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    for level in (lo, 0.0, hi):
        yy = y(level)
        parts.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width - pad_r}" y2="{yy:.1f}"'
            ' stroke="#e6e6e6"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6}" y="{yy + 4:.1f}" font-size="11" text-anchor="end"'
            f' fill="#555">{level:+.1f}%</text>'
        )
    for name, v in pct.items():
        pts = " ".join(f"{x(ts):.1f},{y(val):.1f}" for ts, val in v.items())
        color = COLORS.get(name, "#333")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{pts}"/>')
    legend_x = pad_l
    for name in pct:
        color = COLORS.get(name, "#333")
        parts.append(
            f'<rect x="{legend_x}" y="{height - 16}" width="12" height="12" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 16}" y="{height - 6}" font-size="12" fill="#333">'
            f"{html.escape(name)}</text>"
        )
        legend_x += 16 + 8 * len(name) + 24
    parts.append("</svg>")
    return "".join(parts)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    trs = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def render_html(r: BacktestResult) -> str:
    cols = metrics_columns(r)
    rows = [[label] + [_fmt(fmt, m.get(key)) for _, m in cols] for label, key, fmt in ROWS]
    trade_rows = [
        [
            str(t["entry_ts"])[:16],
            str(t["symbol"]),
            str(t["side"]),
            f"{float(t['qty']):.4g}",
            f"{float(t['entry_price']):.2f}",
            f"{float(t['exit_price']):.2f}",
            str(t["exit_reason"]),
            f"{float(t['pnl']):+.2f}",
            f"{float(t['r_multiple']):+.2f}",
            f"{float(t['costs']):.2f}",
        ]
        for _, t in r.trades.tail(50).iterrows()
    ]
    curves = {"strategy": r.equity, **r.bench_equity}
    css = (
        "body{font-family:system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;"
        "color:#222}table{border-collapse:collapse;margin:1rem 0;width:100%}th,td{border:1px solid"
        " #ddd;padding:6px 8px;text-align:right}th:first-child,td:first-child{text-align:left}"
        "li{margin:.25rem 0}.verdict{background:#f6f8fa;padding:1rem;border-radius:6px}"
    )
    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'><title>Cryptobot backtest: "
        f"{html.escape(r.strategy)}</title><style>{css}</style></head><body>",
        f"<h1>Cryptobot backtest: {html.escape(r.strategy)}</h1>",
        "<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in header_lines(r)) + "</ul>",
        "<h2>Equity, out of sample (% of starting cash)</h2>",
        svg_equity(curves, r.starting_cash),
        "<h2>Results (after costs)</h2>",
        _table(["Metric"] + [n for n, _ in cols], rows),
        "<div class='verdict'><h2>Verdict</h2><ul>"
        + "".join(f"<li>{html.escape(x)}</li>" for x in verdict_lines(r))
        + "</ul></div>",
    ]
    if r.notes:
        parts.append(
            "<h2>Notes</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in r.notes) + "</ul>"
        )
    if r.params_history:
        hist_rows = [
            [
                h["ts"][:10],
                h["source"],
                str(h["old_params"]),
                str(h["new_params"]),
                f"{_fmt('{:.2f}', h.get('stats', {}).get('champion', {}).get('sharpe'))} -> "
                f"{_fmt('{:.2f}', h.get('stats', {}).get('challenger', {}).get('sharpe'))}",
            ]
            for h in r.params_history
        ]
        parts.append(
            "<h2>Parameter changes applied by the learning loop</h2>"
            + _table(["When", "Source", "Old", "New", "OOS Sharpe"], hist_rows)
        )
    parts.append("<h2>Out-of-sample trades (last 50)</h2>")
    parts.append(
        _table(
            ["Entry", "Symbol", "Side", "Qty", "Entry px", "Exit px", "Exit", "PnL", "R", "Costs"],
            trade_rows,
        )
        if trade_rows
        else "<p>no trades</p>"
    )
    parts.append("</body></html>")
    return "".join(parts)


def write_report(
    r: BacktestResult, reports_dir: str | Path, stem: str | None = None
) -> dict[str, Path]:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    stem = stem or f"backtest_{r.strategy}_{stamp}"
    md, htm = out / f"{stem}.md", out / f"{stem}.html"
    md.write_text(render_markdown(r))
    htm.write_text(render_html(r))
    saved = r.save(out)
    return {"markdown": md, "html": htm, "result": saved}
