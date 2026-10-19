from datetime import UTC, datetime, timedelta

import pytest

from cryptobot.brokers.paper import PaperBroker
from cryptobot.costs import CostModel
from cryptobot.team.ledger import VirtualAccount
from cryptobot.team.router import PortfolioLimits, Router

T0 = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)

NO_COSTS = CostModel(commission=0.0, fee_pct=0.0, spread_bps=0.0, slippage_bps=0.0)
FEES_ONLY = CostModel(commission=0.0, fee_pct=0.1, spread_bps=0.0, slippage_bps=0.0)


class TaggingBroker(PaperBroker):
    """Records the router's bot tag at the moment each order is placed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tags = []
        self.router = None

    def place_order(self, symbol, side, qty, order_type="market", limit_price=None):
        if self.router is not None:
            self.tags.append(self.router.last_tag)
        return super().place_order(symbol, side, qty, order_type, limit_price)


def make_team(
    tmp_path, cash_each=10_000.0, bots=("a", "b"), cm=FEES_ONLY, limits=None, broker_cash=None
):
    broker = TaggingBroker(broker_cash if broker_cash is not None else cash_each * len(bots), cm)
    accounts = {b: VirtualAccount(b, cash_each) for b in bots}
    limits = limits or PortfolioLimits(kill_switch_file=str(tmp_path / "team.KILLSWITCH"))
    router = Router(broker, accounts, limits, cm, clock=lambda: T0)
    broker.router = router
    return broker, accounts, router


def test_internal_netting_sends_nothing_to_broker(tmp_path):
    broker, accounts, router = make_team(tmp_path)
    accounts["b"].apply_fill("BTC/USD", "buy", 2, 95.0, 0.0)  # seed the long b will sell
    router.mark("BTC/USD", 100.0, T0)
    router.submit("a", "BTC/USD", "buy", 2, 100.0)
    router.submit("b", "BTC/USD", "sell", 2, 100.0)
    reports = router.flush(T0)
    assert broker.orders == [] and broker.tags == []
    assert len(reports) == 1
    r = reports[0]
    assert r["internal"] is True and r["bots"] == ["a", "b"]
    assert r["qty"] == pytest.approx(2) and r["price"] == pytest.approx(100.0)
    fee_side = FEES_ONLY.fees(2 * 100.0)
    assert accounts["a"].cash == pytest.approx(10_000 - 200 - fee_side)
    assert accounts["a"].positions["BTC/USD"] == (2, 100.0)
    assert accounts["b"].cash == pytest.approx(10_000 - 190 + 200 - fee_side)
    assert "BTC/USD" not in accounts["b"].positions
    assert accounts["a"].fees_paid == pytest.approx(fee_side)
    assert accounts["b"].fees_paid == pytest.approx(fee_side)
    assert router.internal_fees == pytest.approx(2 * fee_side)


def test_partial_netting_residual_keeps_tag(tmp_path):
    broker, accounts, router = make_team(tmp_path)
    accounts["b"].apply_fill("BTC/USD", "buy", 1, 95.0, 0.0)
    router.mark("BTC/USD", 100.0, T0)
    router.submit("a", "BTC/USD", "buy", 3, 100.0)
    router.submit("b", "BTC/USD", "sell", 1, 100.0)
    reports = router.flush(T0)
    internal = [r for r in reports if r.get("internal")]
    external = [r for r in reports if r.get("internal") is False]
    assert len(internal) == 1 and internal[0]["qty"] == pytest.approx(1)
    assert internal[0]["bots"] == ["a", "b"]
    assert len(external) == 1
    assert external[0]["bot_id"] == "a" and external[0]["qty"] == pytest.approx(2)
    assert len(broker.orders) == 1
    assert broker.orders[0].side == "buy" and broker.orders[0].qty == pytest.approx(2)
    assert broker.tags == ["a"]
    assert accounts["a"].positions["BTC/USD"][0] == pytest.approx(3)
    assert "BTC/USD" not in accounts["b"].positions


def test_tags_preserved_per_bot_on_residuals(tmp_path):
    broker, accounts, router = make_team(tmp_path, bots=("a", "b", "c"))
    router.mark("BTC/USD", 100.0, T0)
    router.mark("ETH/USD", 20.0, T0)
    router.submit("a", "BTC/USD", "buy", 1, 100.0)
    router.submit("c", "ETH/USD", "buy", 5, 20.0)
    reports = router.flush(T0)
    assert broker.tags == ["a", "c"]
    assert [r["bot_id"] for r in reports] == ["a", "c"]
    assert all(r.get("internal") is False for r in reports)
    assert len(broker.orders) == 2
    assert accounts["a"].positions["BTC/USD"][0] == pytest.approx(1)
    assert accounts["c"].positions["ETH/USD"][0] == pytest.approx(5)


def test_gross_exposure_cap_across_bots(tmp_path):
    limits = PortfolioLimits(
        max_gross_exposure_pct=50.0,
        max_symbol_exposure_pct=100.0,
        kill_switch_file=str(tmp_path / "team.KILLSWITCH"),
    )
    broker, accounts, router = make_team(tmp_path, cash_each=1_000.0, cm=NO_COSTS, limits=limits)
    router.mark("BTC/USD", 100.0, T0)
    router.mark("ETH/USD", 100.0, T0)
    router.submit("a", "BTC/USD", "buy", 6, 100.0)  # 600 of the 1000 cap (equity 2000)
    router.submit("b", "ETH/USD", "buy", 5, 100.0)  # would take gross to 1100
    reports = router.flush(T0)
    assert reports[0].get("internal") is False and reports[0]["bot_id"] == "a"
    assert reports[1]["rejected"] is True and reports[1]["bot_id"] == "b"
    assert "gross" in reports[1]["reason"]
    assert len(broker.orders) == 1
    assert accounts["b"].positions == {} and accounts["b"].cash == pytest.approx(1_000)


def test_symbol_exposure_cap_across_bots(tmp_path):
    limits = PortfolioLimits(
        max_gross_exposure_pct=100.0,
        max_symbol_exposure_pct=30.0,
        kill_switch_file=str(tmp_path / "team.KILLSWITCH"),
    )
    broker, accounts, router = make_team(tmp_path, cash_each=1_000.0, cm=NO_COSTS, limits=limits)
    router.mark("BTC/USD", 100.0, T0)
    router.submit("a", "BTC/USD", "buy", 4, 100.0)  # 400 of the 600 symbol cap
    router.submit("b", "BTC/USD", "buy", 3, 100.0)  # 400 + 300 breaches it across bots
    reports = router.flush(T0)
    assert reports[0].get("internal") is False and reports[0]["bot_id"] == "a"
    assert reports[1]["rejected"] is True and reports[1]["bot_id"] == "b"
    assert "symbol" in reports[1]["reason"]
    assert len(broker.orders) == 1


def test_kill_switch_rejects_buys_allows_reducing_sells(tmp_path):
    broker, accounts, router = make_team(tmp_path)
    router.mark("BTC/USD", 100.0, T0)
    order = broker.place_order("BTC/USD", "buy", 2)  # seed a real long for bot a
    accounts["a"].apply_fill("BTC/USD", "buy", 2, order.avg_fill_price, order.fees)
    (tmp_path / "team.KILLSWITCH").write_text("halt")
    router.submit("a", "BTC/USD", "buy", 1, 100.0)
    router.submit("a", "BTC/USD", "sell", 1, 100.0)
    reports = router.flush(T0)
    rejected = [r for r in reports if r.get("rejected")]
    filled = [r for r in reports if r.get("internal") is False and not r.get("rejected")]
    assert len(rejected) == 1 and "kill" in rejected[0]["reason"]
    assert len(filled) == 1 and filled[0]["side"] == "sell"
    assert accounts["a"].positions["BTC/USD"][0] == pytest.approx(1)


def test_stale_marks_reject_buys_only(tmp_path):
    broker, accounts, router = make_team(tmp_path)
    router.mark("BTC/USD", 100.0, T0)
    order = broker.place_order("BTC/USD", "buy", 2)
    accounts["a"].apply_fill("BTC/USD", "buy", 2, order.avg_fill_price, order.fees)
    later = T0 + timedelta(seconds=601)  # just past the 600 s freshness window
    router.submit("a", "BTC/USD", "buy", 1, 100.0)
    router.submit("a", "BTC/USD", "sell", 1, 100.0)
    reports = router.flush(later)
    rejected = [r for r in reports if r.get("rejected")]
    filled = [r for r in reports if r.get("internal") is False and not r.get("rejected")]
    assert len(rejected) == 1 and "stale" in rejected[0]["reason"]
    assert len(filled) == 1 and filled[0]["side"] == "sell"
    assert accounts["a"].positions["BTC/USD"][0] == pytest.approx(1)


def test_broker_rejection_books_nothing(tmp_path):
    limits = PortfolioLimits(
        max_gross_exposure_pct=10_000.0,
        max_symbol_exposure_pct=10_000.0,
        kill_switch_file=str(tmp_path / "team.KILLSWITCH"),
    )
    broker, accounts, router = make_team(
        tmp_path, cash_each=100.0, cm=NO_COSTS, limits=limits, broker_cash=100.0
    )
    router.mark("BTC/USD", 100.0, T0)
    router.submit("a", "BTC/USD", "buy", 5, 100.0)  # 500 notional, broker only has 100
    reports = router.flush(T0)
    assert len(reports) == 1
    assert reports[0]["rejected"] is True and "cash" in reports[0]["reason"]
    assert broker.orders[0].status == "rejected"
    assert accounts["a"].cash == pytest.approx(100.0) and accounts["a"].positions == {}


def test_team_books_track_real_account(tmp_path):
    """The ledger invariant: after a mixed sequence of real fills, internal
    crosses and rejections, the sum of virtual equities plus the internal fees
    retained in the real account matches the broker's equity."""
    cm = CostModel(commission=0.0, fee_pct=0.25, spread_bps=5.0, slippage_bps=5.0)
    broker, accounts, router = make_team(
        tmp_path, cash_each=5_000.0, bots=("a", "b", "c"), cm=cm, broker_cash=15_000.0
    )
    t = T0
    router.mark("BTC/USD", 100.0, t)
    router.mark("ETH/USD", 20.0, t)
    router.submit("a", "BTC/USD", "buy", 10, 100.0)
    router.submit("b", "ETH/USD", "buy", 50, 20.0)
    router.flush(t)

    t += timedelta(minutes=5)
    router.mark("BTC/USD", 104.0, t)
    router.mark("ETH/USD", 19.0, t)
    router.submit("a", "BTC/USD", "sell", 4, 104.0)
    router.submit("c", "BTC/USD", "buy", 4, 104.0)  # crossed internally against a's sell
    router.submit("b", "ETH/USD", "sell", 20, 19.0)  # a real sell
    router.flush(t)

    t += timedelta(minutes=5)
    router.mark("BTC/USD", 101.0, t)
    router.submit("c", "BTC/USD", "sell", 2, 101.0)  # 1 crossed with b, 1 sold at the broker
    router.submit("b", "BTC/USD", "buy", 1, 101.0)
    router.flush(t)

    assert router.internal_fees > 0  # at least one internal cross happened
    assert len(broker.orders) >= 3  # and real orders went out too
    marks = {"BTC/USD": 101.0, "ETH/USD": 19.0}
    total_virtual = sum(acc.equity(marks) for acc in accounts.values())
    assert abs(broker.get_equity() - (total_virtual + router.internal_fees)) < 0.01
