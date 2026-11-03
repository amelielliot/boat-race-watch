import copy
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from boat_watch.backtest import evaluate, paired_day_interval, split_periods
from boat_watch.comparison import ComparisonBook
from boat_watch.validation_model import (
    FEATURES, MODEL_VERSION, PAIRS, features, fit, outcome, strategies, top_pair,
)

JST = ZoneInfo("Asia/Tokyo")


def race(day="2026-10-30", first=1, second=2):
    item = {
        "date": day, "stadium_number": 1, "race_number": 1,
        "closed_at": f"{day} 12:10:00",
        "racers": {str(i):dict(national_win_rate=6 if i == 1 else 5,
            local_win_rate=5, motor_top_2_percent=30, average_start_timing=.17)
            for i in range(1, 7)},
        "preview": {"wind_speed": 2, "wave_height": 2, "racers": {
            str(i):dict(course_number=i, start_timing=.1, exhibition_time=6.6+i*.02)
            for i in range(1, 7)}},
    }
    order = [first, second] + [i for i in range(1, 7) if i not in {first, second}]
    item["result"] = {
        "racers": {str(i): {"place_number": place} for place,i in enumerate(order, 1)},
        "payouts": {"exacta": [{"combination": f"{first}-{second}", "amount": 500}]},
        "refunds": [],
    }
    return item


def model():
    counts = np.ones((6, 6))
    np.fill_diagonal(counts, 0)
    return {
        "version": MODEL_VERSION, "features": FEATURES,
        "calibration_end": "2026-10-28", "first_weights": [0.] * len(FEATURES),
        "second_weights": [0.] * len(FEATURES), "first_temperature": 1.,
        "second_temperature": 1., "course_counts": counts.tolist(),
    }


def book(tmp_path):
    root = tmp_path / "validation"
    root.mkdir()
    (root / "model.json").write_text(json.dumps(model()), encoding="utf-8")
    return ComparisonBook(root, tmp_path / "LIVE_COMPARISON.md")


def payload(item):
    return {"programs": {"stadiums": {"1": {"races": {"1": item}}}}}


def test_input_features_cannot_read_result_or_payouts():
    a = race()
    b = copy.deepcopy(a)
    b["result"] = {"racers": {"1": {"start_timing": -100}}, "payouts": {"exacta": [999]}}
    assert np.array_equal(features(a), features(b))


@pytest.mark.parametrize("change", ["missing_entry", "course", "nan", "missing_exhibition"])
def test_incomplete_features_are_rejected(change):
    item = race()
    if change == "missing_entry":
        del item["racers"]["6"]
    elif change == "course":
        item["preview"]["racers"]["6"]["course_number"] = 1
    elif change == "nan":
        item["racers"]["1"]["national_win_rate"] = float("nan")
    else:
        del item["preview"]["racers"]["1"]["exhibition_time"]
    with pytest.raises((ValueError, KeyError)):
        features(item)


def test_probabilities_are_valid_for_all_strategies():
    for probabilities in strategies(race(), model()).values():
        assert set(probabilities) == set(PAIRS)
        assert sum(probabilities.values()) == pytest.approx(1.)
        assert all(0 <= p <= 1 for p in probabilities.values())
    assert top_pair(dict.fromkeys(reversed(PAIRS), 1/30)) == "1-2"


def test_dates_are_split_before_missing_days_are_removed():
    days = [f"2026-10-{i:02d}" for i in range(1, 29)]
    train, calibration, test = split_periods(days)
    assert (len(train),len(calibration),len(test)) == (18,5,5)
    assert max(train) < min(calibration) < min(test)
    assert not (train & calibration or train & test or calibration & test)


def test_fit_and_final_evaluation_reject_date_leakage():
    with pytest.raises(ValueError, match="strictly precede"):
        fit([race()], [race()])
    with pytest.raises(ValueError, match="without overlap"):
        evaluate([race("2026-10-28")], model())


def test_models_fit_only_on_train_and_calibration():
    fitted = fit([race("2026-10-01"), race("2026-10-02", 2, 1)], [race("2026-10-03")])
    assert fitted["train_end"] == "2026-10-02"
    assert fitted["calibration_end"] == "2026-10-03"
    results = evaluate([race("2026-10-04"),race("2026-10-05",2,1)], fitted)
    assert len({m["races"] for m in results.values()}) == 1
    assert all(m["races"] == 2 and np.isfinite(m["win_log_loss"]) for m in results.values())
    assert all("profit" not in m for m in results.values())
    low, high = paired_day_interval(results)
    assert low <= high


@pytest.mark.parametrize("kind", ["refund", "tie", "mismatched_payout", "incomplete"])
def test_irregular_results_do_not_become_losses(kind):
    item = race()
    if kind == "refund":
        item["result"]["refunds"] = [1]
    elif kind == "tie":
        item["result"]["racers"]["2"]["place_number"] = 1
    elif kind == "incomplete":
        del item["result"]["racers"]["6"]
    else:
        item["result"]["payouts"]["exacta"][0]["combination"] = "2-1"
    assert outcome(item) is None


def test_future_snapshot_is_immutable_and_equal_stake(tmp_path):
    b = book(tmp_path)
    item = race()
    del item["result"]
    observed = datetime(2026,10,30,12,tzinfo=JST)
    odds = dict.fromkeys(PAIRS,30.)
    assert b.capture(item, odds, observed)
    key = "20261030-01-01"
    original = copy.deepcopy(b.data["records"][key])
    assert not b.capture(item,dict.fromkeys(PAIRS,10.),observed)
    assert b.data["records"][key] == original
    assert len(original["strategies"]) == 4
    assert all(e["stake"] == 100 for e in original["strategies"].values())
    b.save()
    reloaded = ComparisonBook(b.root, b.report)
    assert reloaded.pending_dates() == {"2026-10-30"}
    assert not reloaded.summary()
    reloaded.settle(payload(race()),datetime(2026,10,30,13,tzinfo=JST))
    assert all(m["races"] == 1 and m["stake"] == 100 for m in reloaded.summary().values())
    saved = copy.deepcopy(reloaded.data)
    reloaded.settle(payload(race()),datetime(2026,10,30,14,tzinfo=JST))
    assert saved == reloaded.data


@pytest.mark.parametrize("kind", ["late", "result_known", "old_date", "missing_odds", "nan_odds", "bad_weather"])
def test_no_backfilled_or_unfair_live_comparisons(tmp_path,kind):
    b = book(tmp_path)
    item = race()
    del item["result"]
    now = datetime(2026,10,30,12,tzinfo=JST)
    odds = dict.fromkeys(PAIRS,10.)
    if kind == "late":
        now = now.replace(minute=11)
    elif kind == "result_known":
        item["result"] = race()["result"]
    elif kind == "old_date":
        now = datetime(2026,10,28,12,tzinfo=JST)
    elif kind == "missing_odds":
        del odds["1-2"]
    elif kind == "nan_odds":
        odds["1-2"] = float("nan")
    else:
        item["preview"]["wind_speed"] = 6
    assert not b.capture(item, odds, now)
    assert not b.data["records"]


def test_refunds_are_excluded_for_every_strategy_and_partial_results_wait(tmp_path):
    b = book(tmp_path)
    item = race()
    del item["result"]
    now = datetime(2026,10,30,12,tzinfo=JST)
    b.capture(item, dict.fromkeys(PAIRS,10.), now)
    item["result"] = {"racers": {"1": {"place_number": 1}}}
    b.settle(payload(item), now)
    assert b.pending_dates()
    item["result"]["refunds"] = [1]
    b.settle(payload(item), now)
    assert not b.pending_dates()
    assert not b.summary()
    assert next(iter(b.data["records"].values()))["status"] == "excluded"


def test_max_drawdown_and_losing_streak_follow_close_time(tmp_path):
    b = book(tmp_path)
    # Store in reverse insertion order to exercise chronological ordering.
    for i, profit in reversed(list(enumerate([400,-100,-100,400,-100]))):
        b.data["records"][str(i)] = {
            "status": "settled", "closed_at": f"2026-10-30T12:{i:02d}:00+09:00",
            "strategies": {"test": {"hit": profit > 0,"return_amount":profit+100,"profit":profit}},
        }
    m = b.summary()["test"]
    assert m["max_drawdown"] == 200
    assert m["max_losing_streak"] == 2
    assert m["profit"] == 500


def test_comparison_storage_failure_does_not_stop_existing_notifications(tmp_path, monkeypatch):
    from boat_watch import main
    from boat_watch.ledger import Ledger
    now = datetime.now(JST)
    item = race(now.date().isoformat())
    item["closed_at"] = (now + timedelta(minutes=10)).isoformat()
    del item["result"]
    published = []

    class BrokenComparison:
        data = {"records": {}}
        def settle(self, *args):
            raise OSError("storage unavailable")
        def save(self):
            raise OSError("storage unavailable")
        def capture(self, *args):
            raise OSError("storage unavailable")

    class FakeNotifier:
        def __init__(self, *args):
            pass
        def recent(self):
            return []
        def used_amounts(self, *args):
            return {}
        def recorded_bets(self, *args):
            return {}
        def settled_results(self, *args):
            return set(), 0
        def publish(self, *args):
            published.append(args)

    monkeypatch.setattr(main, "ComparisonBook", BrokenComparison)
    monkeypatch.setattr(main, "Ntfy", FakeNotifier)
    monkeypatch.setattr(main, "Ledger", lambda:Ledger(tmp_path/"ledger.json",tmp_path/"PERFORMANCE.md"))
    monkeypatch.setattr(main, "fetch_programs", lambda *args:payload(item))
    monkeypatch.setattr(main, "fetch_trifecta_odds", lambda *args:{})
    monkeypatch.setattr(main, "fetch_exacta_odds", lambda *args:{})
    assert main.run(now) == 0
    assert len(published) == 1
