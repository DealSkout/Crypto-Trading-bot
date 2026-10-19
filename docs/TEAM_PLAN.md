# Step 0: audit and build plan for the 7-bot trading team

Repo audited: `DealSkout/Crypto-Trading-bot` at `4f81563` (main). Paper trading only.
No LLM calls anywhere in the trading loop. **Nothing beyond this document has been
built. This is the plan, for your approval.**

Owner instruction incorporated: **every bot gets its own repository** (see "Repo
layout"). The shared engine is still not copied seven times; it becomes a library
the bot repos depend on.

## 1. What exists today (Bot 1)

One bot, one strategy at a time, one process. Verified against the code at `4f81563`:

| Layer | Module | Today |
|---|---|---|
| Data | `data.py` | Alpaca crypto bars via alpaca-py REST (works keyless; keys only raise rate limits), parquet cache, CSV and synthetic providers. No websocket. |
| Strategy contract | `strategies/__init__.py` | Pure plugin: `signal(bars, params) -> Signal \| None`, loadable from any `.py` file (`--strategy-file`). `Signal` = side, entry, stop, target, reason. Long-only enforced. |
| Live config | `config.yaml` | ORB (opening-range breakout, 30 min range, 3R target) on BTC/USD + ETH/USD, 5-minute bars, $1,000 start. |
| Engine | `engine.py` | One bar-by-bar loop shared by backtest, paper and live. Software stops/targets checked each bar and each 30 s price poll. Daily-loss halt, flatten 10 min before 00:00 UTC (day-trading only). |
| Risk | `risk.py` + `guardian.py` | 0.5% risk per trade, 2% daily loss halt, 3 positions, 3% max stop distance, 100% exposure cap. Guardian kill switch: 10% drawdown, 6% 30-day loss, 5 losing days, 3 halts/30d, 10 errors/day; stop file + `resume` command. |
| Costs | `costs.py` | 0.25% taker fee per side + 5 bps half-spread + 5 bps slippage per side. About 0.6% per round trip, applied in every path including benchmarks. |
| Broker | `brokers/alpaca.py` | alpaca-py TradingClient, paper/live split, market/limit/stop_limit + gtc/ioc only, fractional qty rounded to the asset increment, long-only guards, `client_order_id` already tagged (`cryptobot-<uuid>`). |
| Runner | `runner.py` | 24/7 polling loop (30 s), completed-bars-only (no look-ahead), day-rollover jobs, heartbeat to the journal. One bot per process. |
| Backtest | `backtest.py` | Walk-forward 70/30; headline numbers are out-of-sample after costs; buy-and-hold and random-entry benchmarks share the cost model. |
| Learning | `learn.py` | Grid search on train window + champion/challenger gate (min 30 OOS trades, +0.1 Sharpe). Manual `refit`. |
| Journal | `journal.py` | SQLite per bot: events, trades, daily, params_history, llm_spend, state. |
| LLM (all advisory) | `trade_coach.py`, `coach.py`, `guardian.py`, `decider.py` | Daily review and proposals only. The in-loop decider exists but is `provider: none` (off). For the team: everything LLM stays off in the loop, per spec. |
| Reporting | `report.py`, `dashboard.py`, `notify.py` | HTML report, local dashboard, Telegram daily summaries. |
| Tests | `tests/` | 144 passing (one Windows-console-encoding flake in the dashboard test; fine under UTF-8 and in CI). |

A previous Step 0 (docs/UPGRADE_AUDIT.md, merged PR #2) audited the single-bot
self-learning upgrade. Its code-level findings were re-checked and still hold; the
ones that matter for the team build are folded into Section 5.

## 2. Bot 1 baseline, real data

Computed on this machine from real Alpaca 5-minute bars (keyless REST), walk-forward,
out-of-sample after full costs, $1,000 start, BTC/USD + ETH/USD, live config
(ORB, 30 min range, 3R target). Train 2025-10-19 to 2026-07-01; out-of-sample
2026-07-02 to 2026-10-19 (110 days, 365 days of data in all).

| Out-of-sample, after costs | Net return | Sharpe (252) | Max DD | Win rate | Profit factor | Trades | Costs paid |
|---|---|---|---|---|---|---|---|
| **Bot 1: ORB 30m / 3R (live config)** | **-32.4%** | **-10.07** | **-32.7%** | **23%** | **0.11** | **106** | **$321** |
| Buy-and-hold benchmark | +41.8% | 2.16 | -13.3% | n/a | n/a | 2 | $8 |
| Random-entry benchmark | -65.2% | -16.64 | -65.2% | 28% | 0.14 | 196 | $758 |

Train period was no better: -62.8%, Sharpe -8.88, 238 trades, win rate 22%.

What this means, plainly:

- **Bot 1's current edge is negative on real data.** The earlier audit's +40%
  synthetic ORB number was an artifact of the synthetic generator's built-in
  intraday drift, exactly as that audit warned. On a real year, the live config
  lost in-sample and out-of-sample while BTC+ETH buy-and-hold made +42%.
- Costs dominate: 106 trades paid $321 on a $1,000 account, about 0.6% per round
  trip. Any day-trading bot in the team must clear that hurdle; the coordinator's
  allocation floor (5%) is what keeps a losing control bot from burning capital.
- Bot 1 stays **unchanged as the control**, per your spec. Fixing it is the
  learning loop's job (challengers through the gate), not a hand-edit.
- The paper account's own trade history lives in the journal on the Hetzner
  server; this machine has no copy, so this is the backtest baseline. Once the
  orchestrator lands, the daily leaderboard adds live-paper metrics per bot.
- Known simulator optimism (fixed early in the build order): entries fill at the
  signal bar's close rather than the next open, and stop exits fill at the stop
  price + 5 bps, while the live stop is a 30 s software stop closed by market
  order. Real results would likely be slightly worse still.
- **Post-Section-0 update:** with next-open entry fills, pessimistic stop fills
  (stop_gap_bps 10) and 365-day annualisation, the same window reads: Bot 1
  -37.3% (Sharpe-365 -12.49, Sortino -21.99, DD -37.5%), buy-and-hold +41.6%
  (Sharpe-365 2.84), random -70.6%. The old simulator flattered the bot by
  about 5 points over 110 days. These corrected figures are the baseline the
  team is judged against.
- Quirk to document: `backtest --days N` controls how much data is fetched, but
  the run then uses the whole parquet cache (it grows forever by design), so the
  window above is the canonical one.

## 3. Repo layout (your "each bot has its own repo")

Nine private repos under `DealSkout`. The engine is **one** codebase, consumed as a
pinned dependency; a bot repo holds only what makes that bot itself.

| Repo | Contents |
|---|---|
| `Crypto-Trading-bot` (this one) | The shared core, refactored to a library: engine, data, brokers, risk, costs, journal, backtest/validation, regime detector, virtual-ledger + router primitives. Tagged releases (`core-vX.Y`). |
| `crypto-team-orchestrator` | The deployable app: supervisor that runs all 7 bots in one process, shared Alpaca websocket data service with REST backfill, order router against the one paper account, knowledge store, coordinator, reports, docker-compose + systemd units, `bots.lock.yaml` pinning each bot repo + tag. |
| `crypto-bot-01-orb` | Bot 1, the control: today's ORB strategy and params extracted unchanged. |
| `crypto-bot-02-trend` | EMA crossover filtered by ADX. |
| `crypto-bot-03-meanrev` | Bollinger + RSI extremes, enabled only in ranging regimes. |
| `crypto-bot-04-breakout` | Donchian channel breakout with volume confirmation. |
| `crypto-bot-05-rotation` | 7/30-day momentum ranking over the Alpaca universe, hold top 1-2, weekly rotation (needs the engine's new multi-day "swing" session mode). |
| `crypto-bot-06-grid` | Grid inside detected ranges; stands aside when trending. |
| `crypto-bot-07-sentiment` | Funding rates (OKX/Bybit public APIs), open interest, Fear & Greed (alternative.me); no LLM. |

Each bot repo contains: `strategy.py` (the pure plugin), `config.yaml`,
`risk.yaml` (that bot's own limits), `README.md`, a pinned `requirements.txt`
(core by git tag), its trade journal directory (gitignored; data lives on the
server), and CI that runs the core's no-look-ahead suite plus the bot's own
strategy tests. Versioning a bot = tagging its repo; the registry records every
version, its metrics and the reason it changed; `rollback` = check out the
previous tag and restart (one command in the orchestrator).

## 4. Team architecture

- **One data feed.** The orchestrator runs a single Alpaca crypto websocket
  (REST backfill on start and gap), builds bars once, stamps freshness, and fans
  them out in-process to all bots. A stale feed (no bar for 2x the timeframe, or a
  Bot-7 source past its staleness window) pauses new entries everywhere.
- **One paper account, virtual sub-accounts.** Each bot runs against a
  `VirtualAccount` ledger (its capital slice, positions, cash, P&L). The router:
  per-bot risk first, then nets opposing orders between bots at the same tick
  (internal cross at the mark price with the normal fee model, so no bot gets a
  free trade), then portfolio limits (gross exposure, per-coin cap across bots,
  kill switch), then sends the residual to Alpaca tagged
  `client_order_id = bot0N-<uuid>`, and allocates fills back pro-rata. Invariant,
  tested: sum of sub-accounts equals the real account, to the cent, fees included.
- **Shared regime detector.** Rule-based (realised-vol percentile, ADX-style trend
  strength, cross-pair breadth) labelling trending / ranging / high-vol / risk-off.
  Every bot and the coordinator read the same labels; per-bot performance is
  tracked per regime.
- **Knowledge store.** SQLite (WAL) in the orchestrator: signals, orders, fills,
  positions, per-bot daily metrics, regimes, consensus events, allocations,
  challengers, bot versions, audit trail. Each bot also keeps its own journal
  (existing `journal.py`), mirrored into the store.
- **Coordinator (weekly).** Ranks bots on risk-adjusted performance overall and in
  the current regime; reallocates virtual capital with a 5% floor and 30% cap;
  writes a weekly summary of what changed and why. Consensus: when 3+ bots agree
  on direction within a window it is logged and its hypothetical P&L tracked, so
  consensus earns influence only if it proves out.
- **Cross-learning.** A bot may adopt another bot's feature/filter/params only as a
  challenger: anchored walk-forward + purged, embargoed cross-validation with the
  full cost model, then the promotion gate (deflated Sharpe above champion after
  correcting for the number of challengers tried, >= 50 OOS trades, max drawdown no
  worse, profitable in >= 2 regimes), then 2 weeks shadow-running on live bars
  against the simulated broker, and only then replacement of the paper champion.
- **Evolution (monthly).** Worst bot with >= 50 trades is retired; a mutated variant
  of the best-performing ideas takes its slot through the same gate. Never anything
  to live automatically; going live stays a human command, per bot.

## 5. Pre-existing weaknesses fixed first (they poison team metrics otherwise)

1. Sharpe annualised at 252 days -> 365, add Sortino, keep the first day's return.
2. Entry fills at signal-bar close -> next bar open; extra stop slippage; resting
   stop-limit at Alpaca as a crash backstop.
3. ORB counts bars instead of timestamps; Alpaca omits empty 5-minute bars (ETH has
   ~4% fewer bars than BTC over the past year), shifting ranges.
4. The challenger gate reuses one fixed test window and ignores drawdown/regimes
   (replaced by the Section 4 gate).
5. Engine speed (~4 s per 30 bot-days) needs feature caching before the coordinator
   can validate many challengers weekly.
6. Skip-logging floods the journal (23k rows); deduplicate.

## 6. Risk (overrides everything)

Per bot (`risk.yaml`): max risk per trade, max daily loss, max drawdown -> pause that
bot. Portfolio (router-enforced): max total exposure, max per-coin exposure across
bots, global kill switch (file + command + Telegram alert). Automatic entry pause on
stale data. All existing guardian rules stay on. Live trading is never enabled by
code; it stays behind the existing `mode: live` + `CRYPTOBOT_LIVE_CONFIRM` and a
per-bot human command.

## 7. Reporting

Daily leaderboard (per bot: P&L, Sharpe-365, max DD, allocation, regime, consensus
signals, challenger status) written to `reports/daily/` and sent via the existing
Telegram notifier (email too once SMTP lands in `.env`). Weekly coordinator summary:
what changed, why, with numbers.

## 8. Build order (small commits/PRs; tests green at every stage)

0. Core fixes from Section 5, in this repo. Tag `core-v0.1`.
1. Core: multi-bot primitives — per-bot config/limits, swing-vs-daytrade session
   policy (Bot 5 needs multi-day holds), virtual ledger + router (simulated), store
   schema. Router netting + ledger-invariant tests.
2. Orchestrator repo: supervisor running N engines in one process, shared websocket
   data service + backfill + staleness, router wired to the one Alpaca paper
   account, kill switch. Soak with Bot 1 alone = parity check against today's bot.
3. Bot repos 01-04 scaffolded (01 = extracted control, unchanged); trend, meanrev,
   breakout strategies + per-bot backtests.
4. Bots 05-07: rotation (swing mode), grid, sentiment/derivatives + the cached,
   staleness-stamped research feed (public APIs only, no keys required).
5. Regime detector + per-regime tracking.
6. Coordinator: weekly ranking, reallocation (floor/cap), consensus logging,
   weekly summary.
7. Challenger pipeline: walk-forward + purged CV, promotion gate, 2-week shadow
   runner, monthly evolution, version registry + one-command rollback.
8. Reports (daily/weekly) to file + Telegram/email.
9. Server deployment (docker-compose/systemd), end-to-end paper soak, runbook.

Promised tests, mapped: no look-ahead (stages 0-3), router netting (1), risk limits
(1-2), promotion gate rejects noise and accepts a planted edge (7).

## 9. Decisions needed from you

1. **Repos.** OK to create the 8 new private repos under `DealSkout` with the names
   in Section 3? (You will likely get a GitHub auth prompt for repo creation.)
2. **Account and server.** The team runs on your existing Alpaca paper account and
   the Hetzner server (existing deploy pattern)? I never need the keys in chat; they
   stay in `.env` on the server.
3. **Universe for Bot 5.** BTC, ETH, SOL plus which other Alpaca-listed pairs?
   This changes what the team can trade, so it needs your yes.
4. **LLM modules.** Keep the daily Haiku trade-review coach for the team (advisory
   only, outside the loop), or fully off? The in-loop decider stays off regardless.
5. **Reports channel.** Telegram (already wired) and/or email (needs SMTP in .env)?
