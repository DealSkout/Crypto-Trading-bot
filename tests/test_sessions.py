from datetime import UTC, date, datetime

from cryptobot import sessions


def test_pairs_and_normalization():
    assert sessions.is_pair("BTC/USD") and sessions.is_pair("eth/usdt")
    assert not sessions.is_pair("SPY") and not sessions.is_pair("BTC/EUR")
    assert sessions.normalize("BTCUSD") == "BTC/USD"
    assert sessions.normalize("ethusdt") == "ETH/USDT"
    assert sessions.normalize("SOL-USD") == "SOL/USD"
    assert sessions.currency_for("BTC/USD") == "USD"


def test_the_trading_day_is_the_utc_day_and_never_closes():
    m = sessions.CRYPTO
    ts = datetime(2026, 10, 3, 23, 50, tzinfo=UTC)  # a Saturday
    assert sessions.is_in_session(m, ts)
    assert sessions.minutes_to_close(m, ts) == 10
    assert sessions.minutes_since_open(m, ts) == 23 * 60 + 50
    assert sessions.session_date(m, ts) == date(2026, 10, 3)
    assert sessions.next_session_open(m, ts) == datetime(2026, 10, 4, tzinfo=UTC)
    assert m.session_minutes == 1440
