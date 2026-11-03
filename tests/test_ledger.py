from datetime import datetime
from zoneinfo import ZoneInfo

from boat_watch.ledger import Ledger
from boat_watch.model import Bet, Decision
from boat_watch.results import Settlement


JST = ZoneInfo("Asia/Tokyo")


def test_prediction_and_result_are_saved_with_cumulative_report(tmp_path):
    path = tmp_path / "data" / "ledger.json"
    report = tmp_path / "PERFORMANCE.md"
    ledger = Ledger(path, report)
    now = datetime(2026, 10, 26, 12, 0, tzinfo=JST)
    decision = Decision(
        "BUY",
        ["期待値基準1.15以上"],
        [Bet("1-2", 0.3, 5.0, 1.5, 2000, "2連単")],
        {},
    )

    ledger.record_prediction("20261026-01-01", decision, True, now)
    ledger.record_settlement(
        Settlement("20261026-01-01", {"2T:1-2": 500}, 2000, 10000, 8000),
        now,
    )

    assert ledger.save() is True
    assert Ledger(path, report).summary() == {
        "settled": 1,
        "pending": 0,
        "stake": 2000,
        "return_amount": 10000,
        "profit": 8000,
        "hits": 1,
        "losses": 0,
        "recovery_rate": 500.0,
        "hit_rate": 100.0,
    }
    text = report.read_text(encoding="utf-8")
    assert "**+8,000円**" in text
    assert "500.0%" in text


def test_ntfy_history_recovers_pending_and_settled_records(tmp_path):
    path = tmp_path / "ledger.json"
    report = tmp_path / "PERFORMANCE.md"
    ledger = Ledger(path, report)
    now = datetime(2026, 10, 26, 16, 0, tzinfo=JST)
    history = [
        {
            "time": int(now.timestamp()) - 600,
            "message": "管理ID:20261026-06-12;予定額:2000;買い目:3T:1-2-3@1200,3T:1-3-2@800;検証",
        },
        {
            "time": int(now.timestamp()),
            "message": "仮想払戻：6,000円\n結果ID:20261026-06-12;収支:4000",
        },
    ]

    ledger.reconcile_history(history, now)

    record = ledger.records["20261026-06-12"]
    assert record["status"] == "settled"
    assert record["return_amount"] == 6000
    assert record["profit"] == 4000
    assert record["validation"] is True
    assert ledger.save() is True


def test_pending_records_and_used_amounts_survive_ntfy_expiry(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json", tmp_path / "PERFORMANCE.md")
    now = datetime(2026, 10, 26, 12, 0, tzinfo=JST)
    decision = Decision(
        "BUY",
        [],
        [Bet("1-2-3", 0.1, 15.0, 1.5, 1500, "3連単")],
        {},
    )
    ledger.record_prediction("20261026-01-01", decision, False, now)

    assert ledger.used_amounts(now) == {"20261026-01-01": 1500}
    assert ledger.pending_records(now)["20261026-01-01"].bets == {"3T:1-2-3": 1500}
