# Cryptobot

A small, readable, **paper-first** crypto day-trading bot for **Alpaca**, trading USD pairs such
as BTC/USD and ETH/USD around the clock. One config file, one command to backtest, one to paper
trade, one to go live. Deterministic Python makes the trading decisions; risk limits live in
code; a guardian can stop everything; a daily AI coach reviews every trade but never changes
anything by itself.

It is the crypto sibling of the stock bot in
[kelvora123/trade-bot](https://github.com/kelvora123/trade-bot) and shares its design.

> **Read this first.** Crypto is volatile and Alpaca's crypto fees (0.25% per side at the
> lowest volume tier, plus the spread) are large next to the edge most intraday strategies
> have. Every backtest here includes those costs. Expect most of them to say "no", and run on
> a paper account for at least a month before thinking about real money.

## What you get

- **24/7 trading of USD crypto pairs** on Alpaca through the official `alpaca-py` library:
  `TradingClient` for orders and the account, `CryptoHistoricalDataClient` for price bars.
- **Day trading on the UTC day.** Crypto never closes, so the bot's trading day is the UTC
  calendar day: it flattens everything 10 minutes before 00:00 UTC, the daily loss limit resets
  at midnight UTC, and the end-of-day jobs (summary, guardian review, coach) run then.
- **Long-only, no margin.** Alpaca crypto cannot be shorted or bought on margin. Short signals
  are refused, a sell never exceeds what is held, and sizing uses cash only.
- **Alpaca's crypto order rules.** Market, limit and stop-limit orders only, time in force
  `gtc` or `ioc`. Entries are market orders, or marketable limits if you prefer
  (`orders.entry_type: limit`); exits are always market orders. Quantities are rounded down to
  each coin's minimum increment, and orders below its minimum size are not sent.
- **Risk limits in code, not in the strategy:** at most 0.5% of equity at risk per trade, a 2%
  daily loss stop (flatten and stop for the day), at most 3 open positions, no stop further than
  3% from the entry, and **all open positions together capped at `max_total_exposure_pct` of
  equity** (100% by default, so never any leverage).
- **The guardian**, a watchdog that stops the bot **completely** on a 10% drawdown from peak, a
  6% loss over 30 days, 5 losing days in a row, repeated halts or errors, or results far below
  the backtest. An optional Claude reviewer can add a stop but never undo one. Only
  `python -m cryptobot resume` (you) restarts it.
- **The daily trade coach.** After each UTC day, Python computes every number and one cheap
  Claude Haiku call reviews each trade (verdict, mistake, lesson). Proposals are saved to
  `coach/proposals/` and never applied, with no parameter ideas before 30 closed trades and
  none ever touching risk limits or paper/live mode. Every trade is tagged with its
  `asset_class` (`crypto`) in the journal.
- **Honest backtests:** walk-forward (first 70% of days trains, last 30% is out of sample),
  always after fees, spread and slippage, compared with buy-and-hold and random entries.
- **Dashboard, journal, Telegram alerts, an AI risk agent over MCP**, systemd units and a
  Dockerfile, all as in the stock bot.
- **Live trading stays locked** until `mode: live` **and** `CRYPTOBOT_LIVE_CONFIRM=yes` **and**
  the strategy beat buy-and-hold out of sample after costs **and** has 20+ paper days within
  the expected range.
- **A self-learning loop that only adopts what is proven** (weekly, `learn`): challengers are
  validated with walk-forward and purged cross-validation, then must beat the champion on a
  holdout by a deflated-Sharpe margin, with 50+ trades, no worse drawdown and profits in 2+
  regimes; winners shadow the champion for 2 weeks before they trade on paper. Live trading
  only ever changes when you run `approve-live`. Details: [docs/SELF_LEARNING.md](docs/SELF_LEARNING.md).
- **A feature store**: every decision, taken or skipped, is journalled with the features known
  at that moment, exported to Parquet daily, and can be replayed exactly (`replay`).
- **A market research feed**: funding rates and open interest (OKX, Bybit, Binance), the Fear &
  Greed index, headlines scored locally with VADER, and a macro calendar (FOMC, CPI, token
  unlocks); at most one Haiku call a day writes a research digest. It never trades.
- **Market regimes** (trending, ranging, high-volatility, risk-off) on every trade, with results
  tracked per regime.
- **Sizing and pauses that only reduce risk**: volatility targeting and quarter Kelly, both
  capped by the per-trade risk limit; no new entries in macro windows or on stale data.
- **A daily report** (file, Telegram, email): P&L, metrics against the baseline, regimes, the
  digest, champion against challengers and every risk event.

## Setup

You need Python 3.11 to 3.13 and an Alpaca account.

1. **Give this bot its own Alpaca paper account.** In Alpaca's dashboard, open the account
   menu and add a new paper account, then generate API keys for it (*API Keys, Generate*).
   A separate account means the crypto bot and the stock bot never share a balance or see each
   other's positions. Paper and live keys are different.
2. Install:

   ```bash
   git clone https://github.com/DealSkout/Crypto-Trading-bot.git && cd Crypto-Trading-bot
   python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
   pip install -e ".[dev,claude]"
   cp .env.example .env                                 # Windows: copy .env.example .env
   ```

3. Put the paper keys in `.env` as `ALPACA_API_KEY` and `ALPACA_API_SECRET`.
4. Check the connection, backtest, then paper trade:

   ```bash
   python -m cryptobot account     # shows the paper balance: proves the keys work
   python -m cryptobot backtest    # BTC/USD + ETH/USD, 60 days of 5-minute Alpaca bars
   python -m cryptobot paper       # trades your Alpaca paper account, 24/7
   ```

No keys yet? `python -m cryptobot backtest --data synthetic` runs on fake data (clearly
labelled; it says nothing about markets).

## Commands

| Command | What it does |
| --- | --- |
| `backtest` | Walk-forward backtest after costs (`--strategy all`, `--grid`, `--data`, `--days`) |
| `paper` / `live` | The 24/7 trading loop (live needs the gate below) |
| `account` | Equity, cash, crypto positions and open orders on Alpaca |
| `dashboard` | Local web dashboard on http://127.0.0.1:8787 |
| `review` | Run the daily trade coach now (`--date`, `--force`, `--no-ai`) |
| `guardian` / `resume` | Run the watchdog review now / clear a guardian stop (humans only) |
| `refit`, `coach` | Re-tune parameters through the backtest gate; research coach |
| `check` | Whether the live-trading gate is met |
| `mcp` | Tools for an AI risk agent (Claude Desktop, Claude Code) |
| `baseline` | Baseline metrics on real data (every strategy, after fees) saved for the daily report |
| `learn` | The weekly learning run (`--dry-run` changes nothing) |
| `models` / `rollback <ver>` | The model registry / make an earlier version the paper champion |
| `approve-live <ver> --i-approve` / `revoke-live` | Human-only: choose the version live trading uses |
| `research` | Research feed status; `--refresh`, `--backfill` (history for backtests), `--digest` |
| `regimes` | Today's regime per market and results per regime |
| `daily-report` | Build and send a day's report (`--date`, `--no-send`) |
| `export` / `replay --date` | Parquet copy of the feature store / re-run a day's decisions exactly |

## config.yaml

Everything lives in `config.yaml`; secrets only in `.env`. The crypto-specific parts:

| Key | Meaning |
| --- | --- |
| `symbols` | USD pairs, e.g. `[BTC/USD, ETH/USD]` (USD, USDT or USDC quotes) |
| `risk.max_total_exposure_pct` | All open positions together, % of equity (max 100: no leverage) |
| `risk.flatten_minutes_before_close` | Minutes before 00:00 UTC when everything is closed |
| `costs.fee_pct` | Alpaca's fee per side; 0.25% is the lowest-volume taker fee. Lower it if your tier is cheaper |
| `orders.time_in_force` | `gtc` or `ioc` |
| `orders.entry_type` | `market`, or `limit` for a marketable limit `limit_offset_bps` through the price |
| `orders.fill_timeout_seconds` | An entry not filled by then is cancelled: no trade |

`allow_short` must stay `false`. The strategies (`orb`, `intraday_momentum`, `vwap_revert`)
treat the UTC day as their session: the opening range is the first minutes after 00:00 UTC.
The default parameters are a starting point, not a tested result; run `backtest --grid` and
`refit` before trusting them.

## Fees and fills

Alpaca takes its crypto fee out of what you receive: a buy delivers slightly fewer coins than
ordered, and the order itself reports no fee. The bot books the fee from `costs.fee_pct` and
tracks the quantity actually held, so its profit numbers and its exits match the account.

## Running it 24/7 on a server

Crypto never sleeps, so run the bot on a small always-on server, not a laptop. With the stock
bot already on a server, use the same machine:

```bash
git clone https://github.com/DealSkout/Crypto-Trading-bot.git /opt/cryptobot
cd /opt/cryptobot && python3 -m venv .venv && .venv/bin/pip install -e ".[claude]"
cp .env.example .env && nano .env          # the crypto paper account's keys
.venv/bin/python -m cryptobot account
adduser --disabled-password --gecos "" cryptobot && chown -R cryptobot:cryptobot /opt/cryptobot
cp deploy/cryptobot-paper.service deploy/cryptobot-dashboard.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now cryptobot-paper cryptobot-dashboard
```

The crypto dashboard also uses port 8787. If the stock bot's dashboard runs on the same server,
change one of them with `--port` in its service file.

Then, once: pull the research history (so backtests can use it), compute the baseline on real
data, and switch on the weekly learning run:

```bash
sudo -u cryptobot .venv/bin/python -m cryptobot research --backfill
sudo -u cryptobot .venv/bin/python -m cryptobot baseline
cp deploy/cryptobot-learn.service deploy/cryptobot-learn.timer /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now cryptobot-learn.timer
```

To upgrade an existing install: `git pull`, `.venv/bin/pip install -e ".[claude]"` (adds
`vaderSentiment`), `systemctl restart cryptobot-paper`, then the three steps above.

## Going live

1. `python -m cryptobot check` must say the gate is met. Choose the version live trades with
   `python -m cryptobot approve-live <version> --i-approve` (it must have been the paper
   champion); without one, live uses config.yaml's strategy and params.
2. Put the **live** keys in `.env`, set `mode: live` in `config.yaml` and
   `CRYPTOBOT_LIVE_CONFIRM=yes` in `.env`.
3. `python -m cryptobot live`. It starts at whatever size your limits allow; keep them small.

## Tests and CI

```bash
pytest -q          # no network: Alpaca is replaced by fakes
ruff check . && ruff format --check .
python scripts/secret_scan.py --all
```

CI runs all of that on Python 3.11 to 3.13, plus a synthetic backtest.

## The hard rules

1. Paper first; live only with `mode: live` **and** `CRYPTOBOT_LIVE_CONFIRM=yes`.
2. No LLM in the trading loop unless you switch on the optional decider; risk limits apply to it
   anyway.
3. Costs always on: fees, spread and slippage in every backtest and paper fill.
4. Risk limits enforced in code, plus the stop-distance cap, the exposure cap and the guardian.
5. Secrets only in `.env` (git-ignored); a secret scan runs before every commit and in CI.
6. No live trading until an out-of-sample backtest after costs beats buy-and-hold and 20+ paper
   days look as expected.
7. The learning loop never auto-promotes to live and never touches `risk:`; an LLM never places
   or sizes a trade.
