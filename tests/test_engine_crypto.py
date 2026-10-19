"""Engine behaviour specific to the Alpaca crypto setup: limit entries, fees taken from the
coins received, the account-wide exposure cap, long-only and asset_class tagging."""

from cryptobot.backtest import cost_model
from cryptobot.brokers.base import Broker, Order, Position
from cryptobot.brokers.paper import PaperBroker
from cryptobot.engine import Engine
from cryptobot.journal import Journal
from cryptobot.risk import RiskManager
from tests.conftest import StubStrategy, bars_from_closes
from tests.test_engine import feed, make_engine


class FakeAlpaca(Broker):
    """Behaves like Alpaca crypto: no fee on the order, fewer coins delivered than bought."""

    name = "alpaca"
    supports_fractional = True
    qty_decimals = 6
    supports_limit = True

    def __init__(self, fill=True, fee_coins=0.0025):
        self.cash, self.fill, self.fee_coins = 10_000.0, fill, fee_coins
        self.pos: dict[str, float] = {}
        self.last: dict[str, float] = {}
        self.placed: list[dict] = []
        self.cancelled: list[str] = []

    def mark(self, symbol, price, ts=None):
        self.last[symbol] = price

    def get_cash(self):
        return self.cash

    def get_equity(self):
        return self.cash + sum(q * self.last.get(s, 0) for s, q in self.pos.items())

    def get_positions(self):
        return [Position(s, q, "long", self.last.get(s, 0)) for s, q in self.pos.items() if q]

    def held(self, symbol):
        return self.pos.get(symbol, 0.0)

    def place_order(self, symbol, side, qty, order_type="market", limit_price=None):
        self.placed.append({"side": side, "qty": qty, "type": order_type, "limit": limit_price})
        return Order(id=f"o{len(self.placed)}", symbol=symbol, side=side, qty=qty)

    def await_fill(self, order, timeout=30.0):
        if not self.fill:
            return order  # still resting
        price = self.last[order.symbol]
        if order.side == "buy":
            self.pos[order.symbol] = self.pos.get(order.symbol, 0) + order.qty * (
                1 - self.fee_coins
            )
            self.cash -= order.qty * price
        else:
            self.pos[order.symbol] = self.pos.get(order.symbol, 0) - order.qty
            self.cash += order.qty * price
        order.status, order.filled_qty, order.avg_fill_price = "filled", order.qty, price
        return order

    def cancel(self, order_id):
        self.cancelled.append(order_id)

    def open_orders(self):
        return []


def engine_with(cfg, broker, strategy):
    cm = cost_model(cfg)
    risk = RiskManager(cfg.risk, cfg.allow_short, cfg.max_entries_per_day, cost_model=cm)
    journal = Journal(":memory:")
    return Engine(cfg, strategy, {}, broker, risk, journal), journal


def test_fees_are_booked_and_the_held_quantity_is_tracked(cfg):
    df = bars_from_closes([100, 100, 100, 100, 103, 103], wick=0.0)
    b = FakeAlpaca()
    engine, journal = engine_with(cfg, b, StubStrategy(fire_at=2, stop_pct=1.0, target_pct=2.0))
    feed(engine, df)
    assert len(engine.trades) == 1
    t = engine.trades[0]
    bought = b.placed[0]["qty"]
    assert abs(t.qty - bought * (1 - 0.0025)) < 1e-9  # we only sell what we actually got
    assert b.placed[1]["qty"] == t.qty and b.pos["BTC/USD"] < 1e-9
    gross = (t.exit_price - t.entry_price) * t.qty
    assert t.costs > 0 and t.pnl < gross  # the 0.25% fee per side is charged
    assert journal.trades()["asset_class"].tolist() == ["crypto"]


def test_limit_entries_and_cancel_when_not_filled(cfg):
    cfg.orders.entry_type = "limit"
    df = bars_from_closes([100, 100, 100, 100], wick=0.0)
    b = FakeAlpaca()
    engine, _ = engine_with(cfg, b, StubStrategy(fire_at=2, stop_pct=1.0))
    feed(engine, df)
    entry = b.placed[0]
    assert entry["type"] == "limit" and abs(entry["limit"] - 100 * 1.001) < 1e-9

    slow = FakeAlpaca(fill=False)
    engine, journal = engine_with(cfg, slow, StubStrategy(fire_at=2, stop_pct=1.0))
    feed(engine, df)
    assert slow.cancelled == ["o1"] and not engine.open
    msgs = journal.events(kinds=["order_rejected"])["message"].tolist()
    assert any("not filled in time" in m for m in msgs)


def test_exposure_cap_limits_the_second_position(cfg):
    cfg.risk.risk_per_trade_pct = 5.0
    cfg.risk.max_total_exposure_pct = 30.0
    df = bars_from_closes([100] * 6, wick=0.0)
    b = FakeAlpaca()
    engine, journal = engine_with(cfg, b, StubStrategy(fire_at=2, stop_pct=1.0))
    feed(engine, df)
    pos_value = engine.open["BTC/USD"].qty * 100
    assert pos_value <= 0.30 * 10_000 + 1e-6
    reasons = journal.events(kinds=["skip"])["message"].tolist()
    assert not reasons  # first entry fits; it is just sized down


def test_short_signals_are_refused(cfg):
    df = bars_from_closes([100, 100, 100, 100], wick=0.0)
    engine, journal = make_engine(cfg, StubStrategy(fire_at=2, side="short", stop_pct=1.0))
    feed(engine, df)
    assert not engine.open and not engine.trades
    assert any("long-only" in m for m in journal.events(kinds=["skip"])["message"].tolist())


def test_paper_simulator_ignores_limit_entries(cfg):
    cfg.orders.entry_type = "limit"
    df = bars_from_closes([100, 100, 100, 100], wick=0.0)
    engine, _ = make_engine(cfg, StubStrategy(fire_at=2, stop_pct=1.0))
    feed(engine, df)
    assert "BTC/USD" in engine.open and isinstance(engine.broker, PaperBroker)
