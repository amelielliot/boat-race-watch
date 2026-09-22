from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from statistics import median


@dataclass(frozen=True)
class Bet:
    combination: str
    probability: float
    odds: float
    expected_value: float
    stake: int
    bet_type: str = "3連単"


@dataclass(frozen=True)
class Decision:
    action: str
    reasons: list[str]
    bets: list[Bet]
    probabilities: dict[str, float]


def _number(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _racers(race: dict) -> list[dict]:
    program = race.get("racers") or {}
    preview = (race.get("preview") or {}).get("racers") or {}
    merged = []
    for entry in range(1, 7):
        base = dict(program.get(str(entry), program.get(entry, {})))
        base.update({f"preview_{key}": value for key, value in (preview.get(str(entry), preview.get(entry, {})) or {}).items()})
        base["entry_number"] = entry
        merged.append(base)
    return merged


def _strengths(race: dict) -> tuple[list[float], list[dict]]:
    racers = _racers(race)
    exhibition_values = [_number(r.get("preview_exhibition_time"), 99) for r in racers]
    valid_exhibition = [x for x in exhibition_values if x < 20]
    exhibition_mid = median(valid_exhibition) if valid_exhibition else 6.8
    strengths = []
    for racer, exhibition in zip(racers, exhibition_values):
        course = int(_number(racer.get("preview_course_number"), racer["entry_number"]))
        lane_bonus = {1: 1.15, 2: 0.38, 3: 0.12, 4: -0.12, 5: -0.45, 6: -0.75}.get(course, -0.3)
        national = _number(racer.get("national_win_rate"), 5.0)
        local = _number(racer.get("local_win_rate"), national)
        motor = _number(racer.get("motor_top_2_percent"), 30.0)
        avg_start = _number(racer.get("average_start_timing"), 0.17)
        preview_start = _number(racer.get("preview_start_timing"), avg_start)
        exhibition_effect = 0.0 if exhibition >= 20 else (exhibition_mid - exhibition) * 3.2
        score = (
            lane_bonus
            + 0.24 * (national - 5.0)
            + 0.08 * (local - 5.0)
            + 0.012 * (motor - 30.0)
            + exhibition_effect
            - 1.3 * (avg_start - 0.17)
            - 0.7 * (preview_start - 0.17)
        )
        strengths.append(math.exp(max(-4.0, min(4.0, score))))
    return strengths, racers


def order_probabilities(race: dict) -> dict[str, float]:
    strengths, _ = _strengths(race)
    result: dict[str, float] = {}
    for order in permutations(range(6), 3):
        first, second, third = order
        p1 = strengths[first] / sum(strengths)
        remaining_1 = sum(strengths[i] for i in range(6) if i != first)
        p2 = strengths[second] / remaining_1
        remaining_2 = sum(strengths[i] for i in range(6) if i not in {first, second})
        p3 = strengths[third] / remaining_2
        result[f"{first + 1}-{second + 1}-{third + 1}"] = p1 * p2 * p3
    return result


def decide(
    race: dict,
    trifecta_odds: dict[str, float] | None,
    remaining_budget: int,
    config,
    exacta_odds: dict[str, float] | None = None,
) -> Decision:
    preview = race.get("preview") or {}
    if not preview:
        return Decision("SKIP", ["直前情報が未反映"], [], {})
    racers = _racers(race)
    courses = [int(_number(r.get("preview_course_number"), r["entry_number"])) for r in racers]
    if len(set(courses)) != 6:
        return Decision("SKIP", ["展示進入を確定できない"], [], {})
    if courses[0] != 1:
        return Decision("SKIP", ["1号艇がイン進入ではない"], [], {})
    wind = _number(preview.get("wind_speed"), 0)
    wave = _number(preview.get("wave_height"), 0)
    if wind >= config.max_wind_speed:
        return Decision("SKIP", [f"風速{wind:.0f}mで不確実性が高い"], [], {})
    if wave >= config.max_wave_height:
        return Decision("SKIP", [f"波高{wave:.0f}cmで不確実性が高い"], [], {})
    if trifecta_odds is None and exacta_odds is None:
        return Decision("SKIP", ["直前オッズを取得できない"], [], {})

    probabilities = order_probabilities(race)
    p1_win = sum(prob for combo, prob in probabilities.items() if combo.startswith("1-"))
    exhibition = [_number(r.get("preview_exhibition_time"), 99) for r in racers]
    exhibition_rank = sorted(exhibition).index(exhibition[0]) + 1
    if p1_win < 0.48:
        return Decision("SKIP", [f"1号艇の推定1着率が低い（{p1_win:.0%}）"], [], probabilities)
    if exhibition_rank > 3:
        return Decision("SKIP", [f"1号艇の展示タイムが{exhibition_rank}位"], [], probabilities)

    candidates: list[tuple[float, float, str, float, str]] = []
    for combination, probability in probabilities.items():
        current_odds = (trifecta_odds or {}).get(combination)
        if not current_odds:
            continue
        ev = probability * current_odds
        if combination.startswith("1-") and ev >= config.min_expected_value and current_odds <= 80:
            candidates.append((ev, probability, combination, current_odds, "3連単"))

    exacta_probabilities: dict[str, float] = {}
    for combination, probability in probabilities.items():
        pair = "-".join(combination.split("-")[:2])
        exacta_probabilities[pair] = exacta_probabilities.get(pair, 0.0) + probability
    for combination, probability in exacta_probabilities.items():
        current_odds = (exacta_odds or {}).get(combination)
        if not current_odds:
            continue
        ev = probability * current_odds
        if combination.startswith("1-") and ev >= config.min_expected_value and current_odds <= 80:
            candidates.append((ev, probability, combination, current_odds, "2連単"))

    if candidates:
        selected_type = max(candidates, key=lambda item: item[0])[4]
        candidates = [candidate for candidate in candidates if candidate[4] == selected_type]
    candidates.sort(reverse=True)
    candidates = candidates[:3]
    budget = min(config.max_per_race, remaining_budget)
    if not candidates:
        return Decision("SKIP", [f"期待値{config.min_expected_value:.2f}以上の買い目なし"], [], probabilities)
    if budget < 100:
        return Decision("SKIP", ["1日の上限5,000円に到達"], [], probabilities)

    weights = [max(0.01, ev - 1) for ev, *_ in candidates]
    unit_count = budget // 100
    raw_units = [unit_count * weight / sum(weights) for weight in weights]
    units = [max(1, int(value)) for value in raw_units]
    while sum(units) > unit_count:
        index = max(range(len(units)), key=lambda i: units[i])
        units[index] -= 1
    while sum(units) < unit_count:
        index = max(range(len(units)), key=lambda i: raw_units[i] - units[i])
        units[index] += 1

    bets = [
        Bet(combo, probability, current_odds, ev, units[i] * 100, bet_type)
        for i, (ev, probability, combo, current_odds, bet_type) in enumerate(candidates)
        if units[i] > 0
    ]
    reasons = [
        f"1号艇推定1着率{p1_win:.0%}",
        f"1号艇展示{exhibition_rank}位",
        f"選択券種{candidates[0][4]}",
        f"期待値基準{config.min_expected_value:.2f}以上",
    ]
    return Decision("BUY", reasons, bets, probabilities)
