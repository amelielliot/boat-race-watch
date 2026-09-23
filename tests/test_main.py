from types import SimpleNamespace

from boat_watch.main import _budget_for_decision


def config(shadow_mode: bool):
    return SimpleNamespace(shadow_mode=shadow_mode, max_per_race=2000, max_per_day=5000)


def test_shadow_mode_always_keeps_full_validation_budget():
    assert _budget_for_decision(config(True), 9000) == (2000, True)


def test_live_mode_keeps_daily_limit_then_switches_to_validation():
    assert _budget_for_decision(config(False), 3500) == (1500, False)
    assert _budget_for_decision(config(False), 5000) == (2000, True)
