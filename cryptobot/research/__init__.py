"""Market research feed: cheap, scheduled, cached data the bot can learn from.

- feeds.py      fetchers for public APIs (Fear & Greed, funding, open interest, news)
- store.py      the SQLite cache, with an availability time on every value (no look-ahead)
- sentiment.py  local headline scoring with VADER (no LLM)
- calendar.py   the macro calendar (FOMC, CPI, token unlocks) and its high-risk windows
- digest.py     at most ONE LLM call a day: a short research digest stored as notes/features
- feed.py       the scheduler and the context provider the engine reads features from

Nothing in this package can place, size or cancel an order.
"""
