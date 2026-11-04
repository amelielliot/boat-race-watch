"""Learned-model notifications. No silent fallback to heuristic probabilities."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from .model import Bet, Decision
from .validation_model import (TRIFECTA_MODEL_VERSION, PAIRS, TRIPLES,
    learned_probabilities, learned_trifecta_probabilities, win_probabilities)


def load_model(path="data/validation/notification-model.json") -> dict:
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    model["notification_id"] = model["version"] + "@" + hashlib.sha256(
        json.dumps(model, sort_keys=True).encode()
    ).hexdigest()[:16]
    return model


def decide_learned(race: dict, odds: dict[str, float] | None, budget: int,
                   config, model: dict | None, now: datetime,
                   trifecta_odds: dict[str, float] | None = None) -> Decision:
    version = model.get("notification_id", TRIFECTA_MODEL_VERSION) if model else "learned-model-unavailable"

    def skip(reason, probabilities=None):
        return Decision("SKIP", [reason], [], probabilities or {}, version)

    if model is None:
        return skip("学習モデルを読み込めないため停止（旧方式への自動切替なし）")
    try:
        cutoff = datetime.fromisoformat(model["calibration_end"]).date()
        if now.date() <= cutoff:
            return skip("学習・確率補正期間内のレースには通知モデルを適用しない")
        probabilities = learned_probabilities(race, model)
        triples = learned_trifecta_probabilities(race, model)
        if set(probabilities) != set(PAIRS) or any(not math.isfinite(p) or p < 0 for p in probabilities.values()):
            return skip("学習モデルの確率を検証できない")
        if not math.isclose(sum(probabilities.values()), 1., abs_tol=1e-6):
            return skip("学習モデルの確率合計が不正")
        if set(triples) != set(TRIPLES) or any(not math.isfinite(p) or p < 0 for p in triples.values()) or not math.isclose(sum(triples.values()),1.,abs_tol=1e-6):
            return skip("学習モデルの3連単確率が不正")
    except (ValueError, TypeError, KeyError, IndexError):
        return skip("展示・進入・選手情報または学習モデルが不完全")
    preview = race["preview"]
    wind, wave = float(preview["wind_speed"]), float(preview["wave_height"])
    if wind < 0 or wave < 0 or wind >= config.max_wind_speed or wave >= config.max_wave_height:
        return skip(f"風速{wind:g}m／波高{wave:g}cmが安全基準外", probabilities)
    result = race.get("result") or {}
    if result.get("refunds") or any((result.get("payouts") or {}).values()) or any(
        r.get("place_number") is not None or r.get("start_timing") is not None
        for r in (result.get("racers") or {}).values()
    ):
        return skip("本番スタート・結果が既に反映されたレース", probabilities)
    def valid(quotes, combinations):
        return bool(quotes) and set(quotes)==set(combinations) and all(
            isinstance(v,(int,float)) and math.isfinite(v) and v >= 1 for v in quotes.values())
    candidates = []
    available = []
    for quotes, ps, combinations, label in ((odds, probabilities, PAIRS, "2連単"),
                                          (trifecta_odds, triples, TRIPLES, "3連単")):
        if valid(quotes,combinations):
            available.append(label)
            candidates += [(p*quotes[combo],combo,p,quotes[combo],label) for combo,p in ps.items()
                if p*quotes[combo] >= config.min_expected_value and quotes[combo] <= 80]
    if not available:
        return skip("直前オッズを確認できない（2連単30通り・3連単120通り）",probabilities)
    # Keep the existing policy: one ticket family per race, within one total budget.
    candidates.sort(key=lambda item: (-item[0],item[1],item[4]))
    if candidates:
        selected_type = candidates[0][4]
        candidates = [c for c in candidates if c[4]==selected_type][:3]
    if not candidates:
        return skip(f"学習版の推定期待値{config.min_expected_value:.2f}以上の買い目なし（確認券種：{'・'.join(available)}）", probabilities)
    units_available = min(config.max_per_race, budget) // 100
    if units_available < 1:
        return skip("仮想予算の残額が100円未満", probabilities)
    candidates = candidates[:units_available]
    weights = [max(.01, ev-1) for ev, *_ in candidates]
    raw = [units_available*w/sum(weights) for w in weights]
    units = [max(1,int(x)) for x in raw]
    while sum(units) > units_available:
        i = max(range(len(units)), key=lambda i: units[i])
        units[i] -= 1
    while sum(units) < units_available:
        i = max(range(len(units)), key=lambda i: raw[i]-units[i])
        units[i] += 1
    bets = [Bet(pair, p, quote, ev, units[i]*100, label)
            for i,(ev,pair,p,quote,label) in enumerate(candidates) if units[i] > 0]
    win = win_probabilities(probabilities)
    leader = max(range(6),key=lambda i:win[i])
    reasons = [f"学習版・{leader+1}号艇推定1着率{win[leader]:.0%}",
               "1着・条件付き2着・条件付き3着を別モデルで推定", f"選択券種{candidates[0][4]}",
               f"確認券種：{'・'.join(available)}／共通の仮想予算内",
               f"推定期待値基準{config.min_expected_value:.2f}以上（利益保証なし）"]
    return Decision("BUY", reasons, bets, probabilities, version)
