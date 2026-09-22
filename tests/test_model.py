from types import SimpleNamespace

from boat_watch.model import decide, order_probabilities


def sample_race():
    racers = {}
    preview_racers = {}
    for n in range(1, 7):
        racers[str(n)] = {
            "national_win_rate": 7.0 if n == 1 else 5.0,
            "local_win_rate": 6.8 if n == 1 else 5.0,
            "motor_top_2_percent": 38 if n == 1 else 30,
            "average_start_timing": 0.14 if n == 1 else 0.17,
        }
        preview_racers[str(n)] = {
            "course_number": n,
            "start_timing": 0.12 if n == 1 else 0.17,
            "exhibition_time": 6.60 + n * 0.02,
        }
    return {
        "racers": racers,
        "preview": {"wind_speed": 2, "wave_height": 2, "racers": preview_racers},
    }


def config():
    return SimpleNamespace(max_wind_speed=6, max_wave_height=10, min_expected_value=1.15, max_per_race=2000)


def test_probabilities_sum_to_one():
    probabilities = order_probabilities(sample_race())
    assert len(probabilities) == 120
    assert abs(sum(probabilities.values()) - 1) < 1e-9


def test_buy_is_capped():
    race = sample_race()
    probabilities = order_probabilities(race)
    odds = {combo: max(1.1, 1.25 / probability) for combo, probability in probabilities.items()}
    decision = decide(race, odds, 2000, config())
    assert decision.action == "BUY"
    assert sum(bet.stake for bet in decision.bets) == 2000
    assert len(decision.bets) <= 3


def test_selects_exacta_when_its_best_expected_value_is_higher():
    race = sample_race()
    probabilities = order_probabilities(race)
    trifecta_odds = {combo: 1.2 / probability for combo, probability in probabilities.items()}
    exacta_probabilities = {}
    for combo, probability in probabilities.items():
        pair = "-".join(combo.split("-")[:2])
        exacta_probabilities[pair] = exacta_probabilities.get(pair, 0) + probability
    exacta_odds = {combo: 1.8 / probability for combo, probability in exacta_probabilities.items()}

    decision = decide(race, trifecta_odds, 2000, config(), exacta_odds=exacta_odds)

    assert decision.action == "BUY"
    assert decision.bets
    assert all(bet.bet_type == "2連単" for bet in decision.bets)
    assert "選択券種2連単" in decision.reasons


def test_high_wind_skips():
    race = sample_race()
    race["preview"]["wind_speed"] = 7
    decision = decide(race, {}, 2000, config())
    assert decision.action == "SKIP"
    assert "風速" in decision.reasons[0]
