# How the bot learns, and why it rarely changes its mind

The bot can improve itself, but only through a gate designed to say "no" to luck. This page
explains each piece, what it can and cannot change, and how to operate it.

## What can change, and who changes it

| Thing | Who can change it |
|---|---|
| `risk:` limits, sizing method, pauses | Only you, in `config.yaml` |
| The **paper** model (strategy, params, filters) | The learning loop, after the gate and a 2-week shadow run; or you, by editing `config.yaml` or with `rollback` |
| The **live** model | Only you: `approve-live <version> --i-approve` |
| Paper or live mode | Only you (`mode:` plus `CRYPTOBOT_LIVE_CONFIRM=yes`) |

A model is a strategy, its parameters and optional filters. Filters are skip rules on decision
features (for example "no entries when Fear & Greed <= 20") and per-regime size weights. A
weight is never above 1, so the loop can make positions smaller but never bigger than your risk
limits allow.

## The pieces

1. **Feature store** (`decisions` table, `store/*.parquet`). Every signal is recorded with the
   features known at that moment, whether it was taken, skipped (and why) or closed:
   - price, volatility, trend and volume features from completed bars;
   - the regime;
   - the research feed's values.
   Each decision carries a fingerprint of its inputs, so `replay --date` can prove it again.
2. **Research feed** (`research` command). It collects:
   - funding rates and open interest from OKX, Bybit or Binance;
   - the Fear & Greed index;
   - headlines from RSS feeds, scored locally by VADER;
   - the macro calendar in `macro_calendar.yaml`.
   Each value is stored with the time it became available, and features never read a value
   earlier than that. One Haiku call a day writes a digest; it is a note, not an instruction.
3. **Regimes** (`regimes` command). Each market gets one label per UTC day, computed from
   completed days only: trending, ranging, high_vol or risk_off. Every trade carries its
   regime.
4. **Learning loop** (`learn`, weekly by systemd timer). See below.
5. **Shadow runs.** A winner trades in simulation next to a simulated mirror of the champion on
   the same live bars for 14 days. It is promoted to paper champion only if its P&L beats the
   mirror, its drawdown is no deeper and its results match the backtest.
6. **Registry** (`models`). Every version, its status, the reason and the numbers are kept, and
   `rollback <version>` brings any of them back.

## The gate

Each candidate is simulated once over the whole history, about a year. The simulation fills
entries at the next bar's open and charges spread, slippage, extra slippage on stops, and
Alpaca's fee. The days are then split 70/30.

**Selection window (first 70%).** The best challenger is chosen here, and two checks ask
whether choosing like this works at all:
- **Walk-forward.** For each later fold, take the best candidate using only earlier days. If
  that rule does worse than simply keeping the champion, the run says no.
- **Probability of backtest overfitting.** Combinatorially purged cross-validation measures how
  often the in-sample winner falls below the median out of sample. The limit is 0.5.

**Holdout (last 30%).** All of these must hold for the challenger against the champion:
- at least 50 trades;
- a positive net return after costs;
- a Sharpe ratio at least 0.25 higher (annualised over 365 days);
- a **deflated Sharpe** of the daily return difference of at least 0.95;
- a max drawdown no worse;
- profitable in at least 2 regimes.

The holdout overlaps from one week to the next, so every weekly run counts as another look at
it, and the deflated Sharpe corrects for every look in the last six months.

**How strict is it?** We tested the gate on 200 random seeds:

| Case | Promoted |
|---|---|
| 30 noise challengers, independent of the champion | 3 of 200 |
| 30 noise challengers, correlated with the champion | 3 of 200 |
| One genuinely better challenger, after 26 holdout looks | 162 of 200 |

A noise run that does pass still has to survive the 2-week shadow run.

## Operating it

```bash
python -m cryptobot research --backfill   # once: history for the research features
python -m cryptobot baseline              # once, then whenever you like: real-data baseline
python -m cryptobot learn --dry-run       # see what the loop would do, change nothing
python -m cryptobot models                # what is champion, shadow, rejected, live
python -m cryptobot rollback v3           # paper champion back to v3 (next UTC day/restart)
python -m cryptobot approve-live v5 --i-approve   # you, and only you, choose live's model
```

Reports land in `reports/learning/` (each run), `reports/research/` (each digest) and
`reports/daily/` (each day).

## Honest limits

- Backtests and the learning loop use Alpaca's crypto bars. Alpaca's own venue is thinner than
  the big exchanges, so fills on size may differ.
- Research filters are only tried once the history covers the backtest period. Headline history
  cannot be backfilled, so news filters arrive only after about a year of collecting.
- With one trade per pair per day, 50 holdout trades need about a year of data. Expect the loop
  to say "no" most weeks; that is the point.
