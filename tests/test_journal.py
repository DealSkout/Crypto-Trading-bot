from datetime import UTC, date, datetime

from cryptobot.journal import Journal


def test_journal_roundtrip(tmp_path):
    j = Journal(str(tmp_path / "j.sqlite"), mode="paper")
    j.log("entry", "long 1 QQQ", symbol="BTC/USD", data={"qty": 1})
    j.log("skip", "max positions", symbol="ETH/USD")
    ev = j.events()
    assert len(ev) == 2 and set(ev["kind"]) == {"entry", "skip"}
    assert len(j.events(kinds=["skip"])) == 1
    j.record_trade(
        {
            "symbol": "BTC/USD",
            "side": "long",
            "qty": 1,
            "entry_ts": "t0",
            "entry_price": 1,
            "exit_ts": "t1",
            "exit_price": 2,
            "stop": 0.5,
            "target": None,
            "pnl": 1,
            "r_multiple": 2,
            "costs": 0.1,
            "exit_reason": "target",
        },
        strategy="orb",
    )
    assert len(j.trades()) == 1 and j.trades(mode="live").empty
    j.record_daily(date(2026, 10, 5), 1000, 1010, 10, 1, 0.5, False)
    j.record_daily(date(2026, 10, 5), 1000, 1020, 20, 2, 0.5, True)  # upsert
    d = j.daily()
    assert len(d) == 1 and d["pnl"].iloc[0] == 20 and d["halted"].iloc[0] == 1
    j.add_params_change("orb", {"a": 1}, {"a": 2}, {"reason": "x"}, "refit", True)
    assert j.params_history()[0]["new_params"] == {"a": 2} and j.last_applied_change()["applied"]
    j.add_coach_note({"review": "hi"})
    assert j.coach_notes()[0]["review"] == "hi"
    day = datetime.now(UTC).date()
    j.add_coach_spend(0.2)
    j.add_coach_spend(0.3)
    assert abs(j.coach_spend_today(day) - 0.5) < 1e-9
    j.close()
