"""Chronological retrospective evaluation; no invented historical odds/returns."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .data import fetch_programs, iter_races, PROGRAM_URL
from .validation_model import features, fit, outcome, strategies, win_probabilities, top_pair


def split_periods(days: list[str]) -> tuple[set[str], set[str], set[str]]:
    days = sorted(set(days))
    if len(days) < 14:
        raise ValueError("at least 14 calendar days required")
    evaluation_days = max(3, len(days) // 5)
    calibration_days = max(3, len(days) // 5)
    return (set(days[:-evaluation_days-calibration_days]),
            set(days[-evaluation_days-calibration_days:-evaluation_days]),
            set(days[-evaluation_days:]))


def evaluate(races: list[dict], model: dict) -> dict:
    if any(r["date"] <= model["calibration_end"] for r in races):
        raise ValueError("evaluation must follow calibration, without overlap")
    results = {}
    for race in races:
        winner, second = outcome(race)
        actual_pair = f"{winner + 1}-{second + 1}"
        for name, probabilities in strategies(race, model).items():
            win = win_probabilities(probabilities)
            metrics = results.setdefault(name, {"rows": []})
            metrics["rows"].append({
                "key": f"{race['date']}-{race['stadium_number']:02d}-{race['race_number']:02d}",
                "date": race["date"], "venue": race["stadium_number"],
                "actual_first": winner + 1, "actual_pair": actual_pair,
                "win_probabilities": [round(p, 8) for p in win],
                "win_hit": int(max(range(6), key=lambda i: win[i]) == winner),
                "pair_hit": int(top_pair(probabilities) == actual_pair),
                "win_log_loss": -float(np.log(max(win[winner], 1e-12))),
                "pair_log_loss": -float(np.log(max(probabilities[actual_pair], 1e-12))),
                "win_brier": sum((p - int(i == winner)) ** 2 for i, p in enumerate(win)),
            })
    for metrics in results.values():
        rows = metrics["rows"]
        metrics["races"] = len(rows)
        for field in ("win_hit", "pair_hit", "win_log_loss", "pair_log_loss", "win_brier"):
            metrics[field] = sum(r[field] for r in rows) / len(rows)
        metrics["calibration"] = []
        for lo in range(10):
            selected = [r for r in rows if lo / 10 <= r["win_probabilities"][0] < (lo + 1) / 10
                        or (lo == 9 and r["win_probabilities"][0] == 1.)]
            if selected:
                metrics["calibration"].append({
                    "bin": f"{lo*10}〜{(lo+1)*10}%", "n": len(selected),
                    "predicted": sum(r["win_probabilities"][0] for r in selected) / len(selected),
                    "observed": sum(r["actual_first"] == 1 for r in selected) / len(selected),
                })
    return results


def paired_day_interval(results: dict) -> tuple[float, float]:
    # Day-block bootstrap: do not treat correlated races as independent trials.
    old = results["現行スコア"]["rows"]
    new = results["学習版v1"]["rows"]
    if [r["key"] for r in old] != [r["key"] for r in new]:
        raise ValueError("paired cohorts differ")
    by_day = {}
    for a, b in zip(old, new):
        by_day.setdefault(a["date"], []).append(b["win_log_loss"] - a["win_log_loss"])
    groups = list(by_day.values())
    rng = np.random.default_rng(20261103)
    draws = []
    for _ in range(1000):
        chosen = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        draws.append(sum(map(sum, chosen)) / sum(map(len, chosen)))
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def render_report(model: dict, results: dict, manifest: list[dict], exclusions: Counter, periods) -> str:
    low, high = paired_day_interval(results)
    train, calibration, test = periods
    lines = [
        "# 予測ロジック比較・過去データ検証", "",
        "研究用。実購入の推奨ではありません。現行通知は変更せず、学習版は比較専用です。", "",
        "## 条件", "",
        f"- 学習：{min(train)}〜{max(train)}（{model['train_races']:,}レース）",
        f"- 確率補正：{min(calibration)}〜{max(calibration)}（{model['calibration_races']:,}レース）",
        f"- 最終評価：{min(test)}〜{max(test)}（{results['学習版v1']['races']:,}レース）",
        "- 日付順で完全分離。重みは学習期間のみ、確率補正は補正期間のみで決定。評価期間で調整しない。",
        "- 取得できた全場のうち、出走表・展示が揃い、返還・同着等のない通常決着を全方式共通で評価。現行の買い判断・見送り条件とは別の『予測の部品』比較。",
        "- コース頻度：学習期間のコース別1・2着頻度のみ。学習版：場別イン特性・選手成績・モーター・展示・ST・風波を使う1着／条件付き2着の別モデル。",
        "- 人気順方式は締切前オッズの保存開始後に比較。過去の確定払戻から人気順や期待値を作らない。", "",
        "## 最終評価（全方式で同じレース）", "",
        "| 方式 | レース | 1着的中率 | 2連単1点的中率 | 1着Log loss↓ | 2連単Log loss↓ | 1着Brier↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in results.items():
        lines.append(f"| {name} | {m['races']:,} | {m['win_hit']:.1%} | {m['pair_hit']:.1%} | "
                     f"{m['win_log_loss']:.4f} | {m['pair_log_loss']:.4f} | {m['win_brier']:.4f} |")
    lines += [
        "", "Log lossは外れた予測への過信を、Brierは確率と結果のずれを測ります。どちらも小さいほど良いですが、確率の信頼性は下の表でも別に確認します。",
        f"学習版−現行の1着Log loss差：日単位再標本化による参考95%区間 [{low:+.4f}, {high:+.4f}]（負なら改善）。",
        "評価日数が短く、別開催・別季節への再現性や利益は保証しません。良かった方式への自動切替はしません。", "",
        "## 1号艇の予測確率と実際の1着率", "",
        "| 方式 | 予測帯 | 件数 | 平均予測 | 実際の1着率 |", "|---|---|---:|---:|---:|",
    ]
    for name, m in results.items():
        for b in m["calibration"]:
            lines.append(f"| {name} | {b['bin']} | {b['n']} | {b['predicted']:.1%} | {b['observed']:.1%} |")
    lines += ["", "## 読み違えないために", "",
        "- 締切前オッズの過去スナップショットがないため、過去の回収率・損益・最大損失は算出しません。的中率の改善だけで利益が出るとは判定しません。",
        "- 非公式APIの確定後アーカイブを使用。特徴量は出走表・展示のみで、結果・本番ST・払戻を入力しません。ただし当時の取得時刻を証明できず、訂正情報が含まれる可能性があります。",
        "- 将来の比較は締切前に保存した同一レース・同一オッズ・2連単1点100円で実施。記録できなかったレースは補わず、未確定や返還を負けに数えません。",
        f"- データ取得：{sum(x['status']=='ok' for x in manifest)}/{len(manifest)}日。失敗日は学習・補正・評価の境界を詰めず欠損として扱います。",
        f"- 除外内訳：{dict(exclusions)}",
        "- 元URL・SHA256・分割・モデル重み・評価レースごとの予測は `data/validation/` に保存。",
        "- 次の判断：別期間と将来データで、単純方式を上回るか、確率の過信がないか、仮想回収率と最大下落がどうかを確認。現段階は実購入不可。", ""]
    return "\n".join(lines)


def run(start: date, end: date, cache: Path, output: Path) -> dict:
    if end >= datetime.now(ZoneInfo("Asia/Tokyo")).date():
        raise ValueError("only completed calendar days may be used")
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    periods = split_periods([d.isoformat() for d in days])
    cache.mkdir(parents=True, exist_ok=True)

    def load(day):
        path = cache / f"{day}.json"
        try:
            if path.exists():
                raw = path.read_text(encoding="utf-8")
                payload = json.loads(raw)
            else:
                payload = fetch_programs(day)
                raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                path.write_text(raw, encoding="utf-8")
            return day, payload, {"date": day.isoformat(), "status": "ok",
                "url": PROGRAM_URL.format(year=day.year, yyyymmdd=day.strftime("%Y%m%d")),
                "sha256": hashlib.sha256(raw.encode()).hexdigest()}
        except Exception as exc:
            return day, {}, {"date": day.isoformat(), "status": "error", "error": str(exc)}

    races, manifest, exclusions = [], [], Counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for day, payload, entry in pool.map(load, days):
            manifest.append(entry)
            day_count = 0
            for race in iter_races(payload):
                if race.get("date") != day.isoformat():
                    exclusions["date_mismatch"] += 1
                    continue
                if outcome(race) is None:
                    exclusions["no_normal_result"] += 1
                    continue
                try:
                    features(race)
                except (ValueError, TypeError, KeyError):
                    exclusions["incomplete_features"] += 1
                    continue
                races.append(race)
                day_count += 1
            entry["eligible_races"] = day_count
            print(f"{day}: {entry['status']}, eligible={day_count}", flush=True)
    train = [r for r in races if r["date"] in periods[0]]
    calibration = [r for r in races if r["date"] in periods[1]]
    test = [r for r in races if r["date"] in periods[2]]
    if min(len(train), len(calibration), len(test)) < 100:
        raise ValueError("insufficient data in a chronological partition (<100 races)")
    model = fit(train, calibration)
    results = evaluate(test, model)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "evaluation.json").write_text(json.dumps({
        "manifest": manifest, "excluded": dict(exclusions),
        "partitions": [sorted(p) for p in periods], "results": results,
    }, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    Path("VALIDATION.md").write_text(render_report(model, results, manifest, exclusions, periods), encoding="utf-8")
    return {name: {k:v for k,v in m.items() if k not in {"rows", "calibration"}} for name,m in results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".validation-cache"))
    parser.add_argument("--output", type=Path, default=Path("data/validation"))
    args = parser.parse_args()
    print(json.dumps(run(args.start, args.end, args.cache, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
