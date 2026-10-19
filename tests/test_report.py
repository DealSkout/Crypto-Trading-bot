from cryptobot.backtest import run_backtest
from cryptobot.report import comparison_markdown, render_html, render_markdown, write_report


def test_report_contents(cfg, bars):
    r = run_backtest(cfg, bars)
    md = render_markdown(r)
    for label in (
        "Net return",
        "Max drawdown",
        "Sharpe",
        "Trades",
        "Total costs paid",
        "Buy & hold",
        "Random entry",
        "Rule 6",
    ):
        assert label in md
    html = render_html(r)
    assert "<svg" in html and "buy_and_hold" in html
    assert "| orb |" in comparison_markdown([r])
    paths = write_report(r, cfg.reports_dir)
    assert paths["markdown"].exists() and paths["html"].exists() and paths["result"].exists()
