"""Section 2: the market research feed. Fake HTTP responses stand in for the public APIs."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cryptobot.journal import Journal
from cryptobot.llm import LLMResult
from cryptobot.research import feeds
from cryptobot.research.calendar import MacroCalendar, parse_events
from cryptobot.research.digest import run_daily_digest
from cryptobot.research.feed import ResearchFeed
from cryptobot.research.sentiment import score
from cryptobot.research.store import ResearchStore

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


class Resp:
    def __init__(self, body, status=200):
        self.status_code = status
        self._body = body
        self.content = body.encode() if isinstance(body, str) else b""
        self.text = body if isinstance(body, str) else ""

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


def ms(dt):
    return int(dt.timestamp() * 1000)


RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Bitcoin rally lifts crypto market</title><link>https://x/1</link>
<pubDate>Mon, 19 Oct 2026 09:00:00 +0000</pubDate></item>
<item><title>Solana exchange hack drains funds</title><link>https://x/2</link>
<pubDate>Mon, 19 Oct 2026 10:00:00 +0000</pubDate></item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Ethereum upgrade approved</title><link href="https://y/1"/>
<published>2026-10-19T08:00:00Z</published></entry></feed>"""


class FakeSession:
    """Routes URLs to canned answers; `down` lists hosts that fail."""

    def __init__(self, down=()):
        self.down = set(down)
        self.calls = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append(url)
        host = url.split("/")[2]
        if any(d in host for d in self.down):
            return Resp({"msg": "unavailable"}, status=503)
        base = (params or {}).get(
            "instId", (params or {}).get("symbol", (params or {}).get("ccy", ""))
        )
        if "alternative.me" in url:
            days = [NOW.replace(hour=0) - timedelta(days=i) for i in range(3)]
            return Resp(
                {
                    "data": [
                        {
                            "value": str(30 + i),
                            "value_classification": "Fear",
                            "timestamp": str(int(d.timestamp())),
                        }
                        for i, d in enumerate(days)
                    ]
                }
            )
        if "okx.com" in url and "funding" in url:
            t = NOW - timedelta(hours=4)
            return Resp(
                {
                    "code": "0",
                    "data": [
                        {
                            "fundingTime": str(ms(t)),
                            "fundingRate": "0.0001",
                            "realizedRate": "0.0001",
                            "instId": base,
                        }
                    ],
                }
            )
        if "bybit.com" in url and "funding" in url:
            t = NOW - timedelta(hours=4)
            return Resp(
                {
                    "retCode": 0,
                    "result": {
                        "list": [{"fundingRateTimestamp": str(ms(t)), "fundingRate": "0.0003"}]
                    },
                }
            )
        if "bybit.com" in url and "open-interest" in url:
            rows = [
                {
                    "timestamp": str(ms(NOW - timedelta(hours=h))),
                    "openInterest": str(1000 + 10 * (30 - h)),
                }
                for h in range(1, 30)
            ]
            return Resp({"retCode": 0, "result": {"list": rows}})
        if "okx.com" in url and "open-interest" in url:
            return Resp({"code": "1", "msg": "nope"})
        if url.endswith("atom.xml"):
            return Resp(ATOM)
        if "rss" in url:
            return Resp(RSS)
        return Resp({"msg": "unknown"}, status=404)


@pytest.fixture
def feed(cfg, tmp_path):
    cfg.symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    cfg.research.rss_feeds = ["https://news.example/rss", "https://other.example/atom.xml"]
    cfg.research.calendar_file = str(tmp_path / "none.yaml")
    store = ResearchStore(":memory:")
    cal = MacroCalendar(
        parse_events(
            {
                "events": [
                    {"name": "FOMC decision", "kind": "fomc", "time": "2026-10-19 08:30"},
                ]
            }
        )
    )
    return ResearchFeed(cfg, store, session=FakeSession(), calendar=cal, clock=lambda: NOW)


def test_fetchers_parse_each_api(feed):
    s = FakeSession()
    fg = feeds.fear_greed(s)
    assert len(fg) == 3 and fg[0][2] == 30.0 and fg[0][1] - fg[0][0] == feeds.FEAR_GREED_LAG
    ex, rows = feeds.funding(s, "BTC", ["okx", "bybit"])
    assert ex == "okx" and rows[0][2] == pytest.approx(0.0001)
    news = feeds.rss(s, "https://news.example/rss") + feeds.rss(s, "https://x.example/atom.xml")
    assert {n["title"] for n in news} >= {
        "Bitcoin rally lifts crypto market",
        "Ethereum upgrade approved",
    }
    assert feeds.tag_symbols("Solana exchange hack drains funds", ["BTC", "SOL"]) == ["SOL"]


def test_exchanges_fall_back_in_order():
    s = FakeSession(down=["okx.com"])
    ex, rows = feeds.funding(s, "ETH", ["okx", "bybit", "binance"])
    assert ex == "bybit" and rows[0][2] == pytest.approx(0.0003)
    with pytest.raises(feeds.FeedError):
        feeds.funding(
            FakeSession(down=["okx.com", "bybit.com", "binance.com"]),
            "ETH",
            ["okx", "bybit", "binance"],
        )


def test_vader_scores_crypto_words_the_way_a_trader_reads_them():
    assert score("Bitcoin rally lifts crypto market") > 0.3
    assert score("Solana exchange hack drains funds") < -0.3
    assert score("") is None


def test_store_never_returns_a_value_before_it_was_available():
    store = ResearchStore(":memory:")
    t = datetime(2026, 10, 19, 0, 0, tzinfo=UTC)
    store.upsert("fear_greed", "index", [(t, t + timedelta(hours=1), 42.0, {})])
    assert store.asof("fear_greed", "index", t + timedelta(minutes=30)) is None  # not out yet
    assert store.asof("fear_greed", "index", t + timedelta(hours=2))["value"] == 42.0


def test_refresh_records_failures_without_raising(feed):
    feed.session = FakeSession(down=["okx.com", "bybit.com", "binance.com"])
    out = feed.refresh(force=True)
    assert out["fear_greed"].startswith("ok") and out["funding"].startswith("error")
    status = feed.store.status()
    assert status["funding"]["last_error"] and status["fear_greed"]["last_ok"]
    assert "funding" in feed.stale_sources(NOW, required=["funding", "fear_greed"])
    assert "fear_greed" not in feed.stale_sources(NOW, required=["funding", "fear_greed"])


def test_context_features_at_a_decision_time(feed):
    feed.refresh(force=True)
    ctx = feed.context("BTC/USD", NOW)
    assert ctx["fear_greed"] == 30.0
    assert ctx["funding_rate_pct"] == pytest.approx(0.01)
    assert ctx["oi_change_24h_pct"] is not None and ctx["oi_change_24h_pct"] > 0  # bybit
    assert ctx["news_count_24h"] == 3 and ctx["news_sentiment_24h_symbol"] > 0
    assert ctx["macro_window"] == 1 and ctx["macro_event"] == "FOMC decision"
    sol = feed.context("SOL/USD", NOW)
    assert sol["news_sentiment_24h_symbol"] < 0
    # a decision before the data was published sees none of it
    early = feed.context("BTC/USD", NOW - timedelta(days=5))
    assert early["fear_greed"] is None and early["funding_rate_pct"] is None
    assert early["news_count_24h"] == 0


def test_calendar_handles_time_zones_and_daylight_saving():
    events = parse_events(
        {
            "events": [
                {"name": "FOMC", "kind": "fomc", "time": "2026-10-28 14:00"},  # EDT: UTC-4
                {"name": "FOMC", "kind": "fomc", "time": "2026-12-09 14:00"},  # EST: UTC-5
                {
                    "name": "SOL unlock",
                    "kind": "unlock",
                    "time": "2026-11-01 00:00",
                    "tz": "UTC",
                    "symbols": ["SOL"],
                },
            ]
        }
    )
    by_name = {(e.name, e.at.month): e for e in events}
    assert by_name[("FOMC", 10)].at == datetime(2026, 10, 28, 18, 0, tzinfo=UTC)
    assert by_name[("FOMC", 12)].at == datetime(2026, 12, 9, 19, 0, tzinfo=UTC)
    cal = MacroCalendar(events)
    assert cal.active(datetime(2026, 10, 28, 17, 30, tzinfo=UTC)).name == "FOMC"
    assert cal.active(datetime(2026, 10, 28, 21, 0, tzinfo=UTC)) is None  # window over
    unlock = datetime(2026, 11, 1, 0, 30, tzinfo=UTC)
    assert cal.active(unlock, "SOL/USD") is not None and cal.active(unlock, "BTC/USD") is None


def test_the_shipped_calendar_loads_and_covers_the_months_ahead():
    cal = MacroCalendar.load(Path(__file__).resolve().parent.parent / "macro_calendar.yaml")
    kinds = {e.kind for e in cal.events}
    assert {"fomc", "cpi"} <= kinds
    assert cal.days_of_cover(datetime(2026, 10, 19, tzinfo=UTC)) > 300


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self.prompts = []

    def complete(self, system, user, schema, effort=None):
        self.calls += 1
        self.prompts.append((system, user))
        return LLMResult(
            data={
                "summary": "Calm market.",
                "risk_level": "elevated",
                "themes": ["FOMC"],
                "watch": ["funding"],
            },
            input_tokens=2000,
            output_tokens=150,
            cost_usd=0.0003,
            model="claude-haiku-5-5",
        )


def test_digest_is_one_llm_call_a_day_and_stored_as_a_feature(cfg, feed):
    feed.refresh(force=True)
    journal = Journal(":memory:")
    llm = FakeLLM()
    out = run_daily_digest(cfg, journal, feed, NOW.date(), llm=llm, now=NOW)
    assert out["status"] == "done" and out["digest"]["risk_level"] == "elevated"
    again = run_daily_digest(cfg, journal, feed, NOW.date(), llm=llm, now=NOW)
    assert again["status"] == "exists" and llm.calls == 1  # never a second call that day
    system, user = llm.prompts[0]
    assert "never give trade instructions" in system and "untrusted" in system
    assert "Solana exchange hack" in user  # headlines go in as data
    later = feed.context("BTC/USD", NOW + timedelta(minutes=1))
    assert later["digest_risk_level"] == 2.0
    assert journal.llm_spend_today("digest", NOW.date()) == pytest.approx(0.0003)
    assert (Path(cfg.reports_dir) / "research" / f"{NOW.date()}.md").exists()


def test_digest_falls_back_to_numbers_without_a_key_or_over_budget(cfg, feed):
    feed.refresh(force=True)
    journal = Journal(":memory:")
    out = run_daily_digest(cfg, journal, feed, NOW.date(), llm=None, now=NOW)
    assert out["source"] == "rules" and out["digest"]["risk_level"] in ("elevated", "high")
    nxt = NOW + timedelta(days=1)
    journal.add_llm_spend("digest", 1.0, nxt.date())
    llm = FakeLLM()
    out2 = run_daily_digest(cfg, journal, feed, nxt.date(), llm=llm, now=nxt)
    assert out2["source"] == "rules" and llm.calls == 0


def test_research_code_cannot_trade():
    pkg = Path(__file__).resolve().parent.parent / "cryptobot" / "research"
    text = "".join(p.read_text() for p in pkg.glob("*.py"))
    for word in ("place_order", "flatten", "position_size", "brokers"):
        assert word not in text, word


def test_digest_record_is_valid_json(cfg, feed):
    feed.refresh(force=True)
    run_daily_digest(cfg, Journal(":memory:"), feed, NOW.date(), llm=None, now=NOW)
    row = feed.store.series("digest", "daily").iloc[-1]
    assert json.loads(row["data"])["digest"]["summary"]
