from boat_watch.results import RecordedBet, settle_bet


def race_with_result(combination="1-2-3", amount=1240):
    return {
        "result": {
            "payouts": {
                "trifecta": [{"combination": combination, "amount": amount}],
                "exacta": [{"combination": "1-2", "amount": 660}],
            }
        }
    }


def test_settle_hit_uses_payout_per_100_yen():
    record = RecordedBet("20260921-01-01", 1, 1, {"1-2-3": 500, "1-3-2": 700})
    settlement = settle_bet(record, race_with_result())

    assert settlement is not None
    assert settlement.hit
    assert settlement.stake == 1200
    assert settlement.return_amount == 6200
    assert settlement.profit == 5000


def test_settle_miss_loses_recorded_stake():
    record = RecordedBet("20260921-01-01", 1, 1, {"1-3-2": 1200})
    settlement = settle_bet(record, race_with_result())

    assert settlement is not None
    assert not settlement.hit
    assert settlement.return_amount == 0
    assert settlement.profit == -1200


def test_unsettled_race_returns_none():
    record = RecordedBet("20260921-01-01", 1, 1, {"1-2-3": 1200})
    assert settle_bet(record, {"result": None}) is None


def test_settle_exacta_uses_exacta_payout():
    record = RecordedBet("20260922-01-12", 1, 12, {"2T:1-2": 500, "2T:1-3": 1500})

    settlement = settle_bet(record, race_with_result())

    assert settlement is not None
    assert settlement.hit
    assert settlement.winning_payouts == {"2T:1-2": 660}
    assert settlement.return_amount == 3300
    assert settlement.profit == 1300
