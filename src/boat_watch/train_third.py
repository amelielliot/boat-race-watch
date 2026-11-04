"""Add a separately fitted third-place model without retuning the frozen pair model."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .data import fetch_programs, iter_races
from .validation_model import (TRIFECTA_MODEL_VERSION, features, fit_weights,
    choose_temperature, triple_outcome, trifecta_strategies, top_pair)


def run(cache: Path):
    root = Path("data/validation")
    base = json.loads((root/"model.json").read_text(encoding="utf-8"))
    manifests = json.loads((root/"evaluation.json").read_text(encoding="utf-8"))["manifest"]
    manifest_by_day = {m["date"]:m for m in manifests}
    days = [date.fromisoformat(m["date"]) for m in manifests]
    races, sources = [], []
    for day in days:
        path = cache/f"{day}.json"
        if not path.exists():
            cache.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(fetch_programs(day),ensure_ascii=False,sort_keys=True),encoding="utf-8")
        raw = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(raw.encode()).hexdigest()
        if digest != manifest_by_day[day.isoformat()]["sha256"]:
            raise ValueError(f"source archive changed: {day}; do not silently retrain")
        sources.append({"date":str(day),"sha256":digest})
        for race in iter_races(json.loads(raw)):
            if race.get("date") != str(day) or triple_outcome(race) is None:
                continue
            try:
                features(race)
            except (ValueError,TypeError,KeyError):
                continue
            races.append(race)
    train = [r for r in races if base["train_start"] <= r["date"] <= base["train_end"]]
    calibration = [r for r in races if base["calibration_start"] <= r["date"] <= base["calibration_end"]]
    test = [r for r in races if r["date"] > base["calibration_end"]]
    if min(len(train),len(calibration),len(test)) < 100:
        raise ValueError("insufficient third-place data")
    x = np.stack([features(r) for r in train])
    y = np.asarray([triple_outcome(r) for r in train])
    weights = fit_weights(x,y[:,2],y[:,:2])
    cx = np.stack([features(r) for r in calibration])
    cy = np.asarray([triple_outcome(r) for r in calibration])
    scores = cx @ weights
    scores[np.arange(len(cy))[:,None],cy[:,:2]] = -1e9
    counts = np.zeros((6,6,6))
    for a in range(6):
        for b in range(6):
            for c in range(6):
                if len({a,b,c}) == 3:
                    counts[a,b,c] = 1
    for race,order in zip(train,y):
        counts[tuple(int(race["preview"]["racers"][str(i+1)]["course_number"])-1 for i in order)] += 1
    model = dict(base, version=TRIFECTA_MODEL_VERSION, third_weights=weights.tolist(),
        third_temperature=choose_temperature(scores,cy[:,2]),course_triple_counts=counts.tolist(),
        third_train_races=len(train),third_calibration_races=len(calibration),
        base_pair_model_sha256=hashlib.sha256((root/"model.json").read_bytes()).hexdigest(),
        third_sources=sources)
    results = {}
    for race in test:
        actual = "-".join(str(i+1) for i in triple_outcome(race))
        for name,p in trifecta_strategies(race,model).items():
            m=results.setdefault(name,{"races":0,"hits":0,"log_loss_sum":0.})
            m["races"] += 1
            m["hits"] += int(top_pair(p)==actual)
            m["log_loss_sum"] -= float(np.log(max(p[actual],1e-12)))
    (root/"notification-model.json").write_text(json.dumps(model,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# 3連単の学習モデル・参考評価","",
        "通知用モデル：conditional-logit-v2-3t。1・2着モデルは既存の学習済み重みを固定し、条件付き3着だけ追加学習。", "",
        f"- 3着の学習：{base['train_start']}〜{base['train_end']}（{len(train)}レース）",
        f"- 3着の確率補正：{base['calibration_start']}〜{base['calibration_end']}（{len(calibration)}レース）",
        f"- 参考評価：{min(r['date'] for r in test)}〜{max(r['date'] for r in test)}（{len(test)}レース）",
        "- 既に2連単評価に使った期間の再利用です。新しい独立検証ではありません。ここで重みや閾値は調整しません。",
        "- 実際の本番ST・結果・払戻を予測入力に含めません。確定後アーカイブの訂正リスクは残ります。", "",
        "| 方式 | 3連単1点的中率 | 3連単Log loss↓ |","|---|---:|---:|"]
    for name,m in results.items():
        lines.append(f"| {name} | {m['hits']/m['races']:.1%} | {m['log_loss_sum']/m['races']:.4f} |")
    lines += ["","過去の締切前オッズがないため損益・回収率は算出しません。通知への採用はユーザー指定による検証運用であり、優位性や利益が確認できたためではありません。",
        "将来の比較は2連単・3連単を別々に各1点100円で保存します。両券種で共通に取得できたレース以外は、券種間の優劣を直接比較しないでください。", ""]
    Path("TRIFECTA_VALIDATION.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,required=True)
    run(parser.parse_args().cache)
