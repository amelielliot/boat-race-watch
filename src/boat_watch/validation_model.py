"""Frozen, research-only conditional-logit models. Never used for real bets."""
from __future__ import annotations

import math
from itertools import permutations

import numpy as np

from .model import order_probabilities

PAIRS = [f"{a}-{b}" for a, b in permutations(range(1, 7), 2)]
TRIPLES = [f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)]
FEATURES = (
    [f"course_{i}" for i in range(1, 7)]
    + [f"venue_{i}_inside" for i in range(1, 25)]
    + ["national", "local", "motor", "exhibition", "average_start", "preview_start",
       "flying_count", "inside_wind", "inside_wave"]
)
MODEL_VERSION = "conditional-logit-v1"
TRIFECTA_MODEL_VERSION = "conditional-logit-v2-3t"


def finite(value) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite input")
    return number


def features(race: dict) -> np.ndarray:
    """Explicit allowlist: result, payouts and actual race ST are never read."""
    racers = race.get("racers") or {}
    preview = race.get("preview") or {}
    exhibition = preview.get("racers") or {}
    if set(map(str, racers)) != set("123456") or set(map(str, exhibition)) != set("123456"):
        raise ValueError("six complete entries required")
    venue = int(race["stadium_number"])
    if not 1 <= venue <= 24:
        raise ValueError("invalid venue")
    courses = [int(exhibition[str(i)]["course_number"]) for i in range(1, 7)]
    if set(courses) != set(range(1, 7)):
        raise ValueError("ambiguous courses")
    times = [finite(exhibition[str(i)]["exhibition_time"]) for i in range(1, 7)]
    if not all(5 < x < 10 for x in times):
        raise ValueError("invalid exhibition times")
    mid = float(np.median(times))
    wind, wave = finite(preview["wind_speed"]), finite(preview["wave_height"])
    rows = []
    for i, course in enumerate(courses, 1):
        base, pre = racers[str(i)], exhibition[str(i)]
        national = finite(base["national_win_rate"])
        values = [
            (national - 5) / 2,
            (finite(base.get("local_win_rate", national)) - 5) / 2,
            (finite(base["motor_top_2_percent"]) - 30) / 20,
            (mid - times[i - 1]) / .1,
            (.17 - finite(base["average_start_timing"])) / .05,
            (.17 - finite(pre["start_timing"])) / .1,
            finite(base.get("flying_count", 0)),
            wind / 6 if course == 1 else 0,
            wave / 10 if course == 1 else 0,
        ]
        rows.append(
            [float(course == k) for k in range(1, 7)]
            + [float(venue == k and course == 1) for k in range(1, 25)]
            + [max(-5, min(5, x)) for x in values]
        )
    return np.asarray(rows, dtype=float)


def outcome(race: dict) -> tuple[int, int] | None:
    """Only ordinary, complete races; refunds, ties and irregular results excluded."""
    result = race.get("result") or {}
    if result.get("refunds"):
        return None
    racers = result.get("racers") or {}
    try:
        places = {int(k): int(v["place_number"]) for k, v in racers.items()}
    except (ValueError, TypeError, KeyError):
        return None
    if set(places) != set(range(1, 7)) or set(places.values()) != set(range(1, 7)):
        return None
    order = sorted(places, key=places.get)
    payouts = (result.get("payouts") or {}).get("exacta") or []
    if len(payouts) != 1 or payouts[0].get("combination") != f"{order[0]}-{order[1]}":
        return None
    return order[0] - 1, order[1] - 1


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def triple_outcome(race: dict) -> tuple[int, int, int] | None:
    pair = outcome(race)
    if pair is None:
        return None
    rows = race["result"]["racers"]
    third = next(int(k)-1 for k,v in rows.items() if int(v["place_number"]) == 3)
    payouts = (race["result"].get("payouts") or {}).get("trifecta") or []
    if len(payouts) != 1 or payouts[0].get("combination") != f"{pair[0]+1}-{pair[1]+1}-{third+1}":
        return None
    return *pair, third


def fit_weights(x: np.ndarray, y: np.ndarray, excluded: np.ndarray | None = None) -> np.ndarray:
    weights = np.zeros(x.shape[-1])
    target = np.eye(6)[y]
    for _ in range(250):
        scores = x @ weights
        if excluded is not None:
            rows = np.arange(len(y))[:, None] if excluded.ndim == 2 else np.arange(len(y))
            scores[rows, excluded] = -1e9
        probabilities = softmax(scores)
        gradient = np.einsum("ncf,nc->f", x, probabilities - target) / len(y)
        # Fixed hyperparameters, not selected using the final evaluation period.
        weights -= .15 * (gradient + .01 * weights)
    return weights


def choose_temperature(scores: np.ndarray, y: np.ndarray) -> float:
    candidates = [.65, .8, 1., 1.25, 1.5, 2., 3.]
    return min(candidates, key=lambda t: float(-np.log(
        np.maximum(softmax(scores / t)[np.arange(len(y)), y], 1e-12)
    ).mean()))


def fit(train: list[dict], calibration: list[dict]) -> dict:
    if not train or not calibration:
        raise ValueError("separate nonempty training and calibration periods required")
    if max(r["date"] for r in train) >= min(r["date"] for r in calibration):
        raise ValueError("training must strictly precede calibration")
    x = np.stack([features(r) for r in train])
    labels = np.asarray([outcome(r) for r in train])
    first = fit_weights(x, labels[:, 0])
    second = fit_weights(x, labels[:, 1], labels[:, 0])
    cal_x = np.stack([features(r) for r in calibration])
    cal_y = np.asarray([outcome(r) for r in calibration])
    scores2 = cal_x @ second
    scores2[np.arange(len(cal_y)), cal_y[:, 0]] = -1e9
    course_counts = np.ones((6, 6))
    np.fill_diagonal(course_counts, 0)
    for race, (winner, runner_up) in zip(train, labels):
        courses = race["preview"]["racers"]
        a = int(courses[str(winner + 1)]["course_number"]) - 1
        b = int(courses[str(runner_up + 1)]["course_number"]) - 1
        course_counts[a, b] += 1
    return {
        "version": MODEL_VERSION, "features": FEATURES,
        "train_start": min(r["date"] for r in train),
        "train_end": max(r["date"] for r in train),
        "calibration_start": min(r["date"] for r in calibration),
        "calibration_end": max(r["date"] for r in calibration),
        "train_races": len(train), "calibration_races": len(calibration),
        "first_weights": first.tolist(), "second_weights": second.tolist(),
        "first_temperature": choose_temperature(cal_x @ first, cal_y[:, 0]),
        "second_temperature": choose_temperature(scores2, cal_y[:, 1]),
        "course_counts": course_counts.tolist(),
    }


def learned_probabilities(race: dict, model: dict) -> dict[str, float]:
    if model["features"] != FEATURES or model["version"] not in {MODEL_VERSION, TRIFECTA_MODEL_VERSION}:
        raise ValueError("model feature/version mismatch")
    x = features(race)
    first = softmax(x @ np.asarray(model["first_weights"]) / model["first_temperature"])
    second_scores = x @ np.asarray(model["second_weights"])
    result = {}
    for a in range(6):
        conditional = second_scores.copy() / model["second_temperature"]
        conditional[a] = -1e9
        second = softmax(conditional)
        for b in range(6):
            if a != b:
                result[f"{a + 1}-{b + 1}"] = float(first[a] * second[b])
    return result


def learned_trifecta_probabilities(race: dict, model: dict) -> dict[str, float]:
    if model["version"] != TRIFECTA_MODEL_VERSION:
        raise ValueError("third-place model required; no heuristic fallback")
    pairs = learned_probabilities(race, model)
    scores = features(race) @ np.asarray(model["third_weights"]) / model["third_temperature"]
    result = {}
    for pair,p in pairs.items():
        a,b = (int(v)-1 for v in pair.split("-"))
        conditional = scores.copy()
        conditional[[a,b]] = -1e9
        third = softmax(conditional)
        for c in range(6):
            if c not in {a,b}:
                result[f"{a+1}-{b+1}-{c+1}"] = float(p*third[c])
    return result


def trifecta_strategies(race: dict, model: dict) -> dict[str, dict[str, float]]:
    features(race)
    courses = race["preview"]["racers"]
    counts = np.asarray(model["course_triple_counts"])
    total = float(counts.sum())
    course = {triple: float(counts[tuple(int(courses[k]["course_number"])-1 for k in triple.split("-"))] / total)
              for triple in TRIPLES}
    return {"現行スコア": order_probabilities(race), "コース頻度": course,
            "学習版3着対応": learned_trifecta_probabilities(race, model)}


def course_probabilities(race: dict, model: dict) -> dict[str, float]:
    courses = race["preview"]["racers"]
    counts = np.asarray(model["course_counts"])
    total = float(counts.sum())
    return {pair: float(counts[
        int(courses[pair[0]]["course_number"]) - 1,
        int(courses[pair[2]]["course_number"]) - 1,
    ] / total) for pair in PAIRS}


def heuristic_probabilities(race: dict) -> dict[str, float]:
    result = dict.fromkeys(PAIRS, 0.)
    for triple, probability in order_probabilities(race).items():
        result[triple[:3]] += probability
    return result


def win_probabilities(pairs: dict[str, float]) -> list[float]:
    return [sum(p for pair, p in pairs.items() if pair.startswith(f"{a}-")) for a in range(1, 7)]


def top_pair(probabilities: dict[str, float]) -> str:
    # Stable common tiebreak for retrospective and prospective evaluation.
    return min(probabilities, key=lambda pair: (-probabilities[pair], pair))


def strategies(race: dict, model: dict) -> dict[str, dict[str, float]]:
    features(race)  # Identical completeness filter for all strategies.
    return {
        "現行スコア": heuristic_probabilities(race),
        "コース頻度": course_probabilities(race, model),
        "学習版v1": learned_probabilities(race, model),
    }
