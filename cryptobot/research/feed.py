"""The research feed: refreshes each source on its own schedule (in a background thread while
the bot runs), tracks how fresh each source is, and turns the cache into decision features.

`ResearchFeed.context` is a features.py context provider: for a decision at time T it only
reads values whose availability time is <= T.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

from cryptobot.config import Config
from cryptobot.research import feeds
from cryptobot.research.calendar import MacroCalendar
from cryptobot.research.sentiment import score
from cryptobot.research.store import ResearchStore

log = logging.getLogger(__name__)

SOURCES = ("fear_greed", "funding", "open_interest", "news")


class ResearchFeed:
    def __init__(
        self,
        cfg: Config,
        store: ResearchStore,
        session=None,
        calendar: MacroCalendar | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.cfg = cfg
        self.rc = cfg.research
        self.store = store
        self.session = session or requests.Session()
        self.calendar = (
            calendar if calendar is not None else MacroCalendar.load(self.rc.calendar_file)
        )
        self.clock = clock or (lambda: datetime.now(UTC))
        self.bases = sorted({feeds.base_of(s) for s in cfg.symbols})
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ refreshing
    def due(self, source: str, now: datetime) -> bool:
        info = self.store.status().get(source) or {}
        last = info.get("last_attempt")
        if not last:
            return True
        minutes = float(self.rc.refresh_minutes.get(source, 60))
        return now - pd.Timestamp(last).to_pydatetime() >= timedelta(minutes=minutes)

    def refresh(self, now: datetime | None = None, force: bool = False) -> dict[str, str]:
        """Refresh every source that is due. Never raises: failures are recorded per source."""
        now = now or self.clock()
        out = {}
        for source in SOURCES:
            if not (force or self.due(source, now)):
                continue
            try:
                n = getattr(self, f"_refresh_{source}")()
                self.store.mark(source, True, rows=n)
                out[source] = f"ok ({n})"
            except Exception as e:  # one bad source never stops the others or the bot
                self.store.mark(source, False, error=str(e))
                out[source] = f"error: {e}"
                log.info("research %s failed: %s", source, e)
        return out

    def _refresh_fear_greed(self, limit: int = 10) -> int:
        have, _ = self.store.coverage("fear_greed", "index")
        rows = feeds.fear_greed(self.session, limit=limit if have else 0)
        return self.store.upsert("fear_greed", "index", rows)

    def _refresh_funding(self) -> int:
        n = 0
        errors = []
        for base in self.bases:
            try:
                ex, rows = feeds.funding(self.session, base, list(self.rc.exchanges))
                n += self.store.upsert("funding", f"{base}@{ex}", rows)
            except feeds.FeedError as e:
                errors.append(str(e))
        if errors and not n:
            raise feeds.FeedError("; ".join(errors))
        return n

    def _refresh_open_interest(self) -> int:
        n = 0
        errors = []
        for base in self.bases:
            try:
                ex, rows = feeds.open_interest(self.session, base, list(self.rc.exchanges))
                n += self.store.upsert("open_interest", f"{base}@{ex}", rows)
            except feeds.FeedError as e:
                errors.append(str(e))
        if errors and not n:
            raise feeds.FeedError("; ".join(errors))
        return n

    def _refresh_news(self) -> int:
        items: list[dict[str, Any]] = []
        errors = []
        for url in self.rc.rss_feeds:
            try:
                items += feeds.rss(self.session, url)
            except feeds.FeedError as e:
                errors.append(str(e))
        if self.rc.cryptopanic:
            import os

            token = os.environ.get("CRYPTOPANIC_TOKEN", "")
            if token:
                try:
                    items += feeds.cryptopanic(self.session, self.rc.cryptopanic_url, token)
                except feeds.FeedError as e:
                    errors.append(f"cryptopanic: {e}")
        if not items and errors:
            raise feeds.FeedError("; ".join(errors))
        now = self.clock()
        for it in items:
            it["score"] = score(it["title"])
            it["symbols"] = feeds.tag_symbols(it["title"], self.bases)
            # a headline counts from when we could have read it
            it["avail_ts"] = max(it["ts"], now) if it["ts"] > now - timedelta(hours=6) else it["ts"]
        return self.store.add_news(items)

    def backfill(self, days: int | None = None) -> dict[str, str]:
        """Pull as much history as the free APIs give, so backtests can use these features."""
        days = days or self.rc.backfill_days
        out = {}
        try:
            out["fear_greed"] = f"ok ({self._refresh_fear_greed(limit=0)})"
        except Exception as e:
            out["fear_greed"] = f"error: {e}"
        n, errors = 0, []
        for base in self.bases:
            try:
                ex, rows = feeds.funding_history(self.session, base, days, list(self.rc.exchanges))
                n += self.store.upsert("funding", f"{base}@{ex}", rows)
            except feeds.FeedError as e:
                errors.append(str(e))
        out["funding"] = f"ok ({n})" if n else f"error: {'; '.join(errors)}"
        out.update(self.refresh(force=True))
        return out

    # ------------------------------------------------------------------ background thread
    def start(self, poll_seconds: float = 60.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.refresh()
                except Exception as e:  # pragma: no cover - refresh never raises
                    log.warning("research loop: %s", e)
                self._stop.wait(poll_seconds)

        self._stop.clear()
        self._thread = threading.Thread(target=loop, name="research-feed", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ freshness
    def staleness_minutes(self, now: datetime | None = None) -> dict[str, float | None]:
        now = now or self.clock()
        out: dict[str, float | None] = {}
        for source in SOURCES:
            last = self.store.last_ok(source)
            out[source] = None if last is None else (now - last).total_seconds() / 60.0
        return out

    def stale_sources(
        self, now: datetime | None = None, required: list[str] | None = None
    ) -> list[str]:
        """Required sources whose last good refresh is older than stale_after_minutes."""
        ages = self.staleness_minutes(now)
        out = []
        for source in required if required is not None else self.rc.required_sources:
            if source not in ages:
                continue
            limit = float(self.rc.stale_after_minutes.get(source, 720))
            if ages[source] is None or ages[source] > limit:
                out.append(source)
        return out

    # ------------------------------------------------------------------ features
    def _first(self, source: str, base: str, when: datetime) -> tuple[str, dict] | None:
        for ex in self.rc.exchanges:
            row = self.store.asof(source, f"{base}@{ex}", when)
            if row is not None:
                return ex, row
        return None

    def context(self, symbol: str, ts: datetime, bars: pd.DataFrame | None = None) -> dict:
        """Research features for a decision at `ts` (only data available by then)."""
        base = feeds.base_of(symbol)
        out: dict[str, Any] = {}
        fg = self.store.asof("fear_greed", "index", ts)
        out["fear_greed"] = fg["value"] if fg else None
        f = self._first("funding", base, ts)
        if f is not None:
            ex, row = f
            out["funding_rate_pct"] = row["value"] * 100.0
            hist = self.store.series("funding", f"{base}@{ex}", ts - timedelta(days=3), ts)
            out["funding_3d_avg_pct"] = (
                float(hist["value"].mean()) * 100.0 if len(hist) else out["funding_rate_pct"]
            )
        else:
            out["funding_rate_pct"] = out["funding_3d_avg_pct"] = None
        o = self._first("open_interest", base, ts)
        out["oi_change_24h_pct"] = None
        if o is not None:
            ex, row = o
            prev = self.store.asof("open_interest", f"{base}@{ex}", ts - timedelta(hours=24))
            if prev and prev["value"]:
                out["oi_change_24h_pct"] = (row["value"] / prev["value"] - 1.0) * 100.0
        news = self.store.news(ts - timedelta(hours=24), ts)
        scores = news["score"].dropna() if len(news) else pd.Series(dtype=float)
        out["news_count_24h"] = int(len(news))
        out["news_sentiment_24h"] = float(scores.mean()) if len(scores) else None
        if len(news):
            mine = news[news["symbols"].fillna("").str.split(",").apply(lambda s: base in s)]
            sym_scores = mine["score"].dropna()
            out["news_sentiment_24h_symbol"] = float(sym_scores.mean()) if len(sym_scores) else None
        else:
            out["news_sentiment_24h_symbol"] = None
        ev = self.calendar.active(ts, symbol)
        out["macro_window"] = 1 if ev else 0
        out["macro_event"] = ev.name if ev else ""
        nxt = [e for e in self.calendar.events if e.at >= ts and e.affects(symbol)]
        out["hours_to_macro"] = (nxt[0].at - ts).total_seconds() / 3600 if nxt else None
        dg = self.store.asof("digest", "daily", ts)
        out["digest_risk_level"] = dg["value"] if dg else None
        return {
            k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in out.items()
        }
