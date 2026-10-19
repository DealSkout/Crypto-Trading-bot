"""Trading "sessions" for crypto. Crypto trades 24 hours a day, 7 days a week, so the bot's
trading day is the UTC calendar day: it opens at 00:00 UTC and closes at the next 00:00 UTC.

The day boundary is what the risk rules hang on: the daily loss limit resets each UTC day,
and everything is flattened `risk.flatten_minutes_before_close` minutes before midnight UTC
(the bot is a day trader, like its stock sibling). Everything is timezone-aware.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Market:
    name: str
    tz: str
    open: time

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)

    @property
    def session_minutes(self) -> int:
        return 24 * 60


CRYPTO = Market("CRYPTO", "UTC", time(0, 0))
MARKETS = {"CRYPTO": CRYPTO}
QUOTES = ("USD", "USDT", "USDC", "BTC")


def is_pair(symbol: str) -> bool:
    """'BTC/USD' style: a base and a supported quote currency separated by a slash."""
    parts = symbol.upper().split("/")
    return len(parts) == 2 and bool(parts[0]) and parts[1] in QUOTES


def normalize(symbol: str) -> str:
    """Alpaca reports crypto positions as 'BTCUSD'; the bot always uses 'BTC/USD'."""
    s = symbol.upper().replace("-", "/")
    if "/" in s:
        return s
    for quote in sorted(QUOTES, key=len, reverse=True):
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[: -len(quote)]}/{quote}"
    return s


def market_for(symbol: str) -> Market:
    return CRYPTO


def market_from_tz(tz) -> Market:
    return CRYPTO


def currency_for(symbol: str) -> str:
    """The quote currency: what prices and P&L are measured in."""
    return normalize(symbol).split("/")[-1]


def session_open(market: Market, day: date) -> datetime:
    return datetime.combine(day, market.open, tzinfo=market.zone)


def session_close(market: Market, day: date) -> datetime:
    return session_open(market, day) + timedelta(days=1)


def session_date(market: Market, ts: datetime) -> date:
    return ts.astimezone(market.zone).date()


def is_in_session(market: Market, ts: datetime) -> bool:
    return True  # crypto never closes


def minutes_to_close(market: Market, ts: datetime) -> float:
    local = ts.astimezone(market.zone)
    return (session_close(market, local.date()) - local).total_seconds() / 60.0


def minutes_since_open(market: Market, ts: datetime) -> float:
    local = ts.astimezone(market.zone)
    return (local - session_open(market, local.date())).total_seconds() / 60.0


def next_session_open(market: Market, ts: datetime) -> datetime:
    local = ts.astimezone(market.zone)
    return session_open(market, local.date() + timedelta(days=1))
