# Step 0 audit: self-learning engine + research feed

Repo audited: DealSkout/Crypto-Trading-bot at `7f73d51` (main). The stock bot
(kelvora123/trade-bot) shares the same engine design, so most findings apply to both.

**Nothing has been built yet.** This is the audit and the plan for your approval.

## 1. Current architecture

| Layer | Module | What it does today |
|---|---|---|
| Data | `data.py` | Alpaca crypto bars (alpaca-py), parquet cache, CSV, synthetic |
| Strategies | `strategies/orb.py`, `intraday_momentum.py`, `vwap_revert.py`, `random_entry.py` | Pure `signal(bars, params)` functions. Live config: ORB, 30 min range, 3R target |
| Engine | `engine.py` | One bar-by-bar loop shared by backtest, paper and live. Software stop/target, daily loss stop, flatten 10 min before 00:00 UTC |
| Risk | `risk.py`, `guardian.py` | 0.5% risk per trade, 2% daily loss halt, 3 positions max, 3% max stop, 100% exposure cap, long-only. Guardian kill switch: 10% drawdown, 6% 30-day loss, 5 losing days, 3 halts, 10 errors/day |
| Costs | `costs.py` | 0.25% fee per side, 5 bps spread, 5 bps slippage per side. About 0.6% per round trip |
| Backtest | `backtest.py` | Walk-forward 70/30 split, buy-and-hold and random-entry benchmarks |
| Learning | `learn.py` | Grid search on the train window, then a champion/challenger gate on the test window. Manual `refit` command only |
| AI | `trade_coach.py`, `coach.py`, `guardian.py`, `decider.py` | Daily Haiku trade review. Optional research coach, guardian AI review (Opus), optional LLM decider (off) |
| Journal | `journal.py` | SQLite: events, trades, daily, params_history, llm_spend, state |
| Reporting | `report.py`, `dashboard.py`, `notify.py` | HTML report, dashboard, Telegram summaries |

## 2. Baseline metrics

Real-data baselines could not be computed here. This sandbox cannot reach Alpaca,
Binance, Bybit, OKX or alternative.me. The server paper journal has only a few days of trades.
The numbers below use the repo's **synthetic** data. They test the code, not the market.

Synthetic data warning: the generator adds a random daily drift with a standard deviation of
about 15% per day, against 3% of noise. That is a strong built-in intraday trend, so trend
strategies like ORB look far better than they would on real prices.

Out-of-sample (last 30% of days), after costs, $1,000 start, BTC/USD + ETH/USD, 5-minute bars:

| Days | Strategy | Net return | Sharpe (252) | Sharpe (365) | Sortino (365) | Max DD | Win rate | Profit factor | Trades |
|---|---|---|---|---|---|---|---|---|---|
| 60 | orb | +6.8% | 7.47 | 7.64 | 17.60 | -2.7% | 61% | 2.47 | 18 |
| 60 | intraday_momentum | -7.7% | -14.11 | -17.93 | -13.28 | -7.8% | 8% | 0.01 | 13 |
| 60 | vwap_revert | -20.0% | -32.30 | -39.87 | -17.32 | -20.0% | 0% | 0.00 | 26 |
| 180 | orb | +40.1% | 9.47 | 11.69 | 30.53 | -2.9% | 70% | 3.82 | 56 |
| 180 | intraday_momentum | -20.8% | -16.85 | -19.88 | -13.85 | -20.8% | 5% | 0.01 | 37 |
| 180 | vwap_revert | -36.4% | -20.27 | -24.01 | -15.01 | -36.8% | 12% | 0.01 | 67 |

Real baselines need the server. Once Section 1 lands, one command there will produce this
table from 365+ days of real Alpaca bars and from the paper journal.

### Gate test on pure noise

I generated 8 price series with no trend and no pattern, so no real edge exists, and ran the
current `refit` on ORB over 60 days. To test the Sharpe rule on its own, I turned off the
30-trade minimum.

| Result | Count |
|---|---|
| Seeds tested | 8 |
| Grid search picked different params | 7 |
| Challenger Sharpe beat the champion's | 5 |
| Challenger promoted | 0 |
| Out-of-sample trades per run | 15 to 19 |

The gate rejected everything, but only because costs made every challenger lose money. The
Sharpe rule alone would have promoted noise whenever a lucky challenger finished slightly
positive. ORB also lost 2% to 14% over 18 days on noise. That is the cost hurdle: each trade must
earn more than about 0.6% just to break even.

## 3. Bugs and weaknesses found

Look-ahead checks that **passed**: strategies only see completed bars; the live runner drops
the bar still forming; daily ATR excludes today; grid search only uses training days; the
out-of-sample run only uses history before each bar. Costs are applied in every path,
including both benchmarks, and the simulator and backtest share one cost model.

Problems, most important first:

1. **The gate reuses one fixed test window.** Every refit compares on the same last 30% of days.
   Over many weekly runs that window is peeked at again and again, so it stops being
   out-of-sample. There is no correction for testing six grid points, and the required Sharpe
   gain of 0.1 is far below the noise on 18 to 56 trades.
2. **The gate ignores drawdown and regimes.** It only checks trade count, net return and Sharpe.
3. **The learning loop cannot promote anything today.** With 60 days of data, ORB makes about
   18 test trades. The gate needs 30, and your spec needs 50. Roughly 1 year of data and walk-forward
   folds are needed.
4. **Stop fills are optimistic.** The backtest exits exactly at the stop price plus 5 bps. Live,
   the stop is a software stop checked every 30 seconds and closed with a market order, so fast
   moves fill worse. There is no resting stop at Alpaca, so a crashed process has no stop at all.
5. **Entries fill at the signal bar's close.** Live fills come later, around the next bar's open
   plus polling delay. That is slightly optimistic for breakout entries.
6. **Sharpe is annualised with 252 days.** Crypto trades 365 days, so Sharpe is understated by
   about 17%. The first day's return is also dropped. Sortino is not computed at all.
7. **ORB counts bars instead of reading timestamps.** Alpaca crypto data skips 5-minute bars
   with no trades. A missing bar shifts or skips the signal and stretches the range.
8. **Skipped-signal logging floods the journal.** VWAP revert logged 23,250 skips in 180 days
   because every blocked bar is logged again. The skip count is not meaningful.
9. **The engine is slow.** About 4 seconds per 30 days for 2 pairs. A weekly loop with many
   challengers, walk-forward folds and cross-validation would take hours. It needs feature
   caching before Section 4 is practical.
10. **Intraday momentum cannot beat costs.** It holds about 20 minutes with a 0.5% stop against
    a 0.6% round-trip cost. Not a code bug, but it should be dropped from the challenger pool.
11. **Risk limits are only checked at bar close in backtests.** A loss inside one bar can exceed
    the daily limit there. Live polling every 30 seconds limits this.
12. **The journal cannot replay decisions.** Trades lack features, regime, the signal price and
    the slippage against it. Skips store only the signal.
13. **The guardian's daily AI review uses Opus.** With the coach and the new digest that makes
    three LLM calls a day. Moving the guardian to Haiku, or folding it into the digest, would
    cut cost.

## 4. Build plan (one small commit or PR per section)

**Section 0, fixes first.** Sharpe at 365 days plus Sortino; first-day return; next-bar-open
entries; extra stop slippage and a resting stop-limit backup at Alpaca; ORB by timestamp; skip
deduplication; feature caching for speed. Tests for each.

**Section 1, journal and feature store.** New SQLite tables `decisions` (every signal, taken or
skipped, with reason) and `features` (feature vector, regime, research snapshot at decision
time), plus fills with signal price, fill price, slippage and fees. Daily Parquet export.
A `replay` command that re-runs a past day from stored bars and checks it matches.

**Section 2, research feed.** Scheduled, cached, each source with a staleness timestamp.
- Price and volume: Alpaca (existing).
- Funding and open interest: OKX and Bybit public APIs, Binance as fallback. The server is in
  Germany, so these should work there.
- Fear & Greed: alternative.me.
- News: free RSS feeds (CoinDesk, Cointelegraph, The Block) parsed with the standard library.
  CryptoPanic only if you add a free API key to `.env`.
- Sentiment scoring: VADER, locally.
- Macro calendar: a YAML file of FOMC and CPI dates, which are published a year ahead, plus
  token unlocks you add by hand. Free unlock APIs do not exist.
- One Haiku call per day writes the digest. It is stored as notes and a few numeric features.
  It has no path to orders or sizing.

**Section 3, regimes.** Rule thresholds on realised volatility, trend strength (ADX-style) and
breadth across pairs: trending, ranging, high-volatility, risk-off. Rules, not an HMM, so no new
dependency and easy to test. Performance tracked per strategy per regime.

**Section 4, weekly learning loop.**
- Challengers: parameter variations, feature filters (e.g. skip when funding is extreme or
  sentiment is risk-off), and per-regime strategy weights.
- Validation: anchored walk-forward folds plus purged, embargoed cross-validation, with the
  full cost model and the stop slippage fix.
- Promotion gate, all must hold: deflated Sharpe above the champion's (corrected for the
  number of challengers tried), at least 50 out-of-sample trades, max drawdown no worse,
  profitable in at least 2 regimes.
- Winners run 2 weeks in shadow next to the paper champion, on the same live bars with the
  simulated broker. Only then can they replace the paper champion.
- Live: never automatic. A `promote --live <version>` command needs you to run it.
- Model registry: every version, its metrics, the reason, and `rollback <version>`.

**Section 5, risk controls.** Sizing is the smallest of volatility-target size, 0.25 Kelly
and the existing 0.5% risk cap. The learning loop can only shrink size. Max-drawdown kill switch
stays in the guardian. New pauses: no entries during flagged macro windows (e.g. 30 minutes
either side of FOMC and CPI) and when any required feed is stale.

**Section 6, daily report.** P&L, metrics against baseline, regime, digest, champion against
challengers, risk events. Written to `reports/daily/YYYY-MM-DD.md`, sent to Telegram, and email
too if you add SMTP settings to `.env`.

**Tests:** no look-ahead (features at time t unchanged when future bars are appended), fee and
slippage model, gate rejects an overfit challenger on pure noise, gate promotes a challenger with
a planted real edge, kill switch, stale-feed pause, macro-window pause.

## 5. Decisions needed from you

1. **Which repo.** The spec says "Trading-bot-new, stocks + crypto". That was the old name of the
   crypto repo, and the stocks are in a different repo. Recommended: build in the crypto repo first,
   then port to the stock bot once it works.
2. **One new dependency: `vaderSentiment`.** It is small and pure Python. FinBERT would pull in
   `transformers` and `torch`, about 2 GB, too heavy for your server. Everything else (RSS, deflated
   Sharpe, regimes, email) uses the standard library or what you already have.
3. **History length.** The learning loop needs about 1 year of 5-minute bars, and more pairs help
   reach 50 trades. Recommended: 365 days and adding SOL/USD. Adding a pair changes what the bot
   trades, so it needs your yes.
4. **LLM calls.** Recommended: switch the guardian's daily AI review from Opus to Haiku. That keeps
   the total at three cheap calls a day: guardian, coach, digest.
