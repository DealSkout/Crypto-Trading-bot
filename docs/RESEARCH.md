# Alpaca crypto: what this bot assumes

These are the Alpaca crypto rules the code relies on. They come from Alpaca's documentation at
the time of writing; check them again before going live, because brokers change them.

- **Documentation:** <https://docs.alpaca.markets/docs/crypto-trading> and the crypto fees page
  linked from it.
- **Hours:** crypto trades 24 hours a day, 7 days a week, including weekends and holidays.
- **Pairs:** written `BASE/QUOTE`, e.g. `BTC/USD`. Positions are reported without the slash
  (`BTCUSD`); the bot normalises them.
- **Order types:** market, limit and stop-limit. **Time in force:** `gtc` or `ioc` (not `day`).
- **No shorting, no margin:** crypto positions are long-only and paid for in full. Crypto uses
  non-marginable buying power.
- **Fractional quantities:** allowed, subject to each asset's `min_order_size` and
  `min_trade_increment` (read from the assets endpoint).
- **Fees:** a percentage per side, tiered by 30-day volume; the lowest tier is the most
  expensive (the default `costs.fee_pct: 0.25` assumes the taker fee there). The fee is taken
  from the asset received, so fills report no fee.
- **Paper accounts:** several paper accounts can exist under one login, each with its own keys.
- **Market data:** `alpaca-py`'s `CryptoHistoricalDataClient` serves crypto bars without keys;
  keys raise the rate limit.

What has **not** been checked in this repository: live fills, real fee tiers on your account,
and whether any strategy here has an edge after costs. The backtests will tell you the last one.
