"""Fetchers for free public data. Each returns plain rows; the feed decides when to call them and
the store keeps them. All network access goes through an injectable `session` (tests use fakes).

Publication lags (how long after `ts` a value can be relied on) are deliberately cautious, so
features never use a number before it was really out.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import requests

TIMEOUT = 10
USER_AGENT = "cryptobot-research/1.0 (+https://github.com/DealSkout/Crypto-Trading-bot)"

FEAR_GREED_URL = "https://api.alternative.me/fng/"
FEAR_GREED_LAG = timedelta(hours=1)  # the daily index is stamped 00:00 UTC, published soon after
FUNDING_LAG = timedelta(minutes=1)  # a funding rate is known at its settlement time
OI_LAG = timedelta(hours=1)  # an hourly open-interest bucket is complete an hour after it starts

DEFAULT_RSS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://www.theblock.co/rss.xml",
]


class FeedError(Exception):
    pass


Row = tuple[datetime, datetime, float | None, dict[str, Any] | None]


def _get(session, url: str, params: dict[str, Any] | None = None) -> Any:
    try:
        r = session.get(
            url, params=params or {}, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}
        )
    except requests.RequestException as e:
        raise FeedError(f"{url}: {e}") from e
    if getattr(r, "status_code", 200) != 200:
        raise FeedError(f"{url}: HTTP {r.status_code}")
    return r


def _json(session, url: str, params: dict[str, Any] | None = None) -> Any:
    r = _get(session, url, params)
    try:
        return r.json()
    except ValueError as e:
        raise FeedError(f"{url}: not JSON") from e


def _ms(x) -> datetime:
    return datetime.fromtimestamp(int(x) / 1000, tz=UTC)


def base_of(symbol: str) -> str:
    return symbol.split("/")[0].upper()


# ----------------------------------------------------------------------------- Fear & Greed


def fear_greed(session, limit: int = 10) -> list[Row]:
    """The Crypto Fear & Greed index (alternative.me), one value per UTC day, 0..100.
    limit=0 returns the full history (for the backfill)."""
    js = _json(session, FEAR_GREED_URL, {"limit": limit, "format": "json"})
    rows: list[Row] = []
    for it in js.get("data") or []:
        ts = datetime.fromtimestamp(int(it["timestamp"]), tz=UTC)
        rows.append(
            (ts, ts + FEAR_GREED_LAG, float(it["value"]), {"class": it.get("value_classification")})
        )
    if not rows:
        raise FeedError("fear & greed: empty response")
    return rows


# ----------------------------------------------------------------------------- funding rates


def funding_okx(session, base: str, limit: int = 100) -> list[Row]:
    js = _json(
        session,
        "https://www.okx.com/api/v5/public/funding-rate-history",
        {"instId": f"{base}-USDT-SWAP", "limit": limit},
    )
    if str(js.get("code", "0")) != "0":
        raise FeedError(f"okx funding: {js.get('msg')}")
    rows = []
    for it in js.get("data") or []:
        ts = _ms(it["fundingTime"])
        rate = it.get("realizedRate") or it.get("fundingRate")
        rows.append((ts, ts + FUNDING_LAG, float(rate), {"exchange": "okx"}))
    return rows


def funding_bybit(session, base: str, limit: int = 200, end: datetime | None = None) -> list[Row]:
    params: dict[str, Any] = {"category": "linear", "symbol": f"{base}USDT", "limit": limit}
    if end is not None:
        params["endTime"] = int(end.timestamp() * 1000)
    js = _json(session, "https://api.bybit.com/v5/market/funding/history", params)
    if int(js.get("retCode", 0)) != 0:
        raise FeedError(f"bybit funding: {js.get('retMsg')}")
    rows = []
    for it in (js.get("result") or {}).get("list") or []:
        ts = _ms(it["fundingRateTimestamp"])
        rows.append((ts, ts + FUNDING_LAG, float(it["fundingRate"]), {"exchange": "bybit"}))
    return rows


def funding_binance(
    session, base: str, limit: int = 1000, start: datetime | None = None
) -> list[Row]:
    params: dict[str, Any] = {"symbol": f"{base}USDT", "limit": limit}
    if start is not None:
        params["startTime"] = int(start.timestamp() * 1000)
    js = _json(session, "https://fapi.binance.com/fapi/v1/fundingRate", params)
    if isinstance(js, dict):
        raise FeedError(f"binance funding: {js.get('msg', js)}")
    return [
        (
            _ms(it["fundingTime"]),
            _ms(it["fundingTime"]) + FUNDING_LAG,
            float(it["fundingRate"]),
            {"exchange": "binance"},
        )
        for it in js
    ]


FUNDING = {"okx": funding_okx, "bybit": funding_bybit, "binance": funding_binance}


def funding(session, base: str, exchanges: list[str]) -> tuple[str, list[Row]]:
    """Recent funding rates from the first exchange that answers."""
    errors = []
    for ex in exchanges:
        try:
            rows = FUNDING[ex](session, base)
            if rows:
                return ex, rows
        except (FeedError, KeyError, ValueError, TypeError) as e:
            errors.append(f"{ex}: {e}")
    raise FeedError("funding: " + "; ".join(errors or ["no exchange configured"]))


def funding_history(session, base: str, days: int, exchanges: list[str]) -> tuple[str, list[Row]]:
    """Backfill: page back through Binance (or Bybit) funding history for `days` days."""
    since = datetime.now(UTC) - timedelta(days=days)
    errors = []
    for ex in exchanges:
        try:
            rows: list[Row] = []
            if ex == "binance":
                start = since
                for _ in range(10):
                    page = funding_binance(session, base, start=start)
                    rows += page
                    if len(page) < 1000:
                        break
                    start = page[-1][0] + timedelta(seconds=1)
            elif ex == "bybit":
                end = datetime.now(UTC)
                for _ in range(20):
                    page = funding_bybit(session, base, end=end)
                    if not page:
                        break
                    rows += page
                    oldest = min(r[0] for r in page)
                    if oldest <= since:
                        break
                    end = oldest - timedelta(seconds=1)
            else:
                rows = FUNDING[ex](session, base)
            if rows:
                return ex, [r for r in rows if r[0] >= since]
        except (FeedError, KeyError, ValueError, TypeError) as e:
            errors.append(f"{ex}: {e}")
    raise FeedError("funding history: " + "; ".join(errors or ["no exchange configured"]))


# ----------------------------------------------------------------------------- open interest


def oi_bybit(session, base: str, limit: int = 200) -> list[Row]:
    js = _json(
        session,
        "https://api.bybit.com/v5/market/open-interest",
        {"category": "linear", "symbol": f"{base}USDT", "intervalTime": "1h", "limit": limit},
    )
    if int(js.get("retCode", 0)) != 0:
        raise FeedError(f"bybit oi: {js.get('retMsg')}")
    return [
        (
            _ms(it["timestamp"]),
            _ms(it["timestamp"]) + OI_LAG,
            float(it["openInterest"]),
            {"exchange": "bybit", "unit": "coins"},
        )
        for it in (js.get("result") or {}).get("list") or []
    ]


def oi_okx(session, base: str) -> list[Row]:
    js = _json(
        session,
        "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume",
        {"ccy": base, "period": "1H"},
    )
    if str(js.get("code", "0")) != "0":
        raise FeedError(f"okx oi: {js.get('msg')}")
    return [
        (_ms(it[0]), _ms(it[0]) + OI_LAG, float(it[1]), {"exchange": "okx", "unit": "usd"})
        for it in js.get("data") or []
    ]


def oi_binance(session, base: str, limit: int = 500) -> list[Row]:
    js = _json(
        session,
        "https://fapi.binance.com/futures/data/openInterestHist",
        {"symbol": f"{base}USDT", "period": "1h", "limit": limit},
    )
    if isinstance(js, dict):
        raise FeedError(f"binance oi: {js.get('msg', js)}")
    return [
        (
            _ms(it["timestamp"]),
            _ms(it["timestamp"]) + OI_LAG,
            float(it["sumOpenInterest"]),
            {"exchange": "binance", "unit": "coins"},
        )
        for it in js
    ]


OPEN_INTEREST = {"okx": oi_okx, "bybit": oi_bybit, "binance": oi_binance}


def open_interest(session, base: str, exchanges: list[str]) -> tuple[str, list[Row]]:
    """Hourly open interest from the first exchange that answers. Units differ by exchange,
    so features only ever compare values from the same exchange (percentage changes)."""
    errors = []
    for ex in exchanges:
        try:
            rows = OPEN_INTEREST[ex](session, base)
            if rows:
                return ex, rows
        except (FeedError, KeyError, ValueError, TypeError, IndexError) as e:
            errors.append(f"{ex}: {e}")
    raise FeedError("open interest: " + "; ".join(errors or ["no exchange configured"]))


# ----------------------------------------------------------------------------- news


SYMBOL_WORDS = {
    "BTC": ("bitcoin", "btc"),
    "ETH": ("ethereum", "ether", "eth"),
    "SOL": ("solana", "sol"),
}


def tag_symbols(title: str, bases: list[str]) -> list[str]:
    words = {w.strip(".,:;!?()'\"").lower() for w in title.split()}
    out = []
    for base in bases:
        names = SYMBOL_WORDS.get(base, (base.lower(),))
        if any(n in words for n in names):
            out.append(base)
    return out


def _parse_time(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        ts = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        try:
            ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def rss(session, url: str) -> list[dict[str, Any]]:
    """Headlines from an RSS 2.0 or Atom feed (standard library XML parser)."""
    r = _get(session, url)
    try:
        root = ET.fromstring(r.content if hasattr(r, "content") else r.text)
    except ET.ParseError as e:
        raise FeedError(f"{url}: bad XML: {e}") from e
    atom = "{http://www.w3.org/2005/Atom}"
    items = []
    for it in root.iter("item"):
        items.append(
            (
                it.findtext("title"),
                it.findtext("link"),
                it.findtext("pubDate") or it.findtext("{http://purl.org/dc/elements/1.1/}date"),
            )
        )
    for it in root.iter(f"{atom}entry"):
        link = it.find(f"{atom}link")
        items.append(
            (
                it.findtext(f"{atom}title"),
                link.get("href") if link is not None else "",
                it.findtext(f"{atom}published") or it.findtext(f"{atom}updated"),
            )
        )
    out = []
    host = url.split("/")[2] if "//" in url else url
    for title, link, when in items:
        ts = _parse_time(when)
        if not title or ts is None:
            continue
        title = " ".join(title.split())
        out.append(
            {
                "id": hashlib.sha1(f"{link or ''}|{title}".encode()).hexdigest(),
                "ts": ts.astimezone(UTC),
                "title": title,
                "url": link or "",
                "source": host,
            }
        )
    return out


def cryptopanic(session, url: str, token: str) -> list[dict[str, Any]]:
    """CryptoPanic posts (needs a free API token in CRYPTOPANIC_TOKEN)."""
    js = _json(session, url, {"auth_token": token, "public": "true", "kind": "news"})
    out = []
    for it in js.get("results") or []:
        ts = _parse_time(it.get("published_at") or it.get("created_at"))
        title = it.get("title")
        if not title or ts is None:
            continue
        out.append(
            {
                "id": hashlib.sha1(f"cp|{it.get('id')}|{title}".encode()).hexdigest(),
                "ts": ts.astimezone(UTC),
                "title": " ".join(title.split()),
                "url": it.get("url") or "",
                "source": "cryptopanic",
            }
        )
    return out
