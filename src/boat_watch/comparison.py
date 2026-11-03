"""Prospective, equal-stake comparison, isolated from the notification strategy."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .data import parse_closed_at, iter_races
from .validation_model import PAIRS, features, outcome, strategies, top_pair


class ComparisonBook:
    def __init__(self, root="data/validation", report="LIVE_COMPARISON.md"):
        self.root = Path(root)
        self.path = self.root / "live.json"
        self.report = Path(report)
        self.model = json.loads((self.root / "model.json").read_text(encoding="utf-8"))
        self.model_hash = hashlib.sha256(json.dumps(self.model, sort_keys=True).encode()).hexdigest()[:16]
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "version": 1, "stake_per_strategy": 100, "records": {},
        }
        self.dirty = not self.path.exists()

    def capture(self, race: dict, odds: dict[str, float] | None, observed: datetime) -> bool:
        """The same complete, before-close snapshot is shared by all four methods."""
        if observed.tzinfo is None:
            raise ValueError("timezone-aware observation required")
        if observed.date().isoformat() <= self.model["calibration_end"]:
            return False
        remaining = (parse_closed_at(race["closed_at"]) - observed).total_seconds() / 60
        if not 7 <= remaining <= 13 or race.get("result"):
            return False
        if not odds or set(odds) != set(PAIRS):
            return False
        if any(not isinstance(x, (int, float)) or not 1 <= x < float("inf") for x in odds.values()):
            return False
        try:
            matrix = features(race)
            preview = race["preview"]
            if float(preview["wind_speed"]) >= 6 or float(preview["wave_height"]) >= 10:
                return False
            forecasts = strategies(race, self.model)
        except (ValueError, TypeError, KeyError):
            return False
        key = f"{observed:%Y%m%d}-{int(race['stadium_number']):02d}-{int(race['race_number']):02d}"
        if key in self.data["records"]:
            return False
        implied = {pair: 1 / odds[pair] for pair in PAIRS}
        total = sum(implied.values())
        forecasts["人気順"] = {pair:p / total for pair,p in implied.items()}
        entries = {}
        for name, probabilities in forecasts.items():
            pick = top_pair(probabilities)
            entries[name] = {
                "pick": pick, "stake": 100, "quoted_odds": odds[pick],
                "probabilities": {p:round(v, 9) for p,v in probabilities.items()},
            }
        self.data["records"][key] = {
            "date": observed.date().isoformat(), "closed_at": parse_closed_at(race["closed_at"]).isoformat(),
            "observed_at": observed.isoformat(), "status": "pending",
            "model_hash": self.model_hash, "model_version": self.model["version"],
            "odds": odds, "features": matrix.tolist(), "strategies": entries,
        }
        self.dirty = True
        return True

    def pending_dates(self) -> set[str]:
        return {r["date"] for r in self.data["records"].values() if r["status"] == "pending"}

    def settle(self, payload: dict, now: datetime) -> None:
        for race in iter_races(payload):
            day = race.get("date", "").replace("-", "")
            key = f"{day}-{int(race['stadium_number']):02d}-{int(race['race_number']):02d}"
            record = self.data["records"].get(key)
            if record is None or record["status"] != "pending":
                continue
            result = race.get("result") or {}
            if not result:
                continue
            labels = outcome(race)
            if labels is None:
                # Do not turn partially published results into exclusions or losses.
                rows = (result.get("racers") or {}).values()
                all_finished = len(result.get("racers") or {}) == 6 and all(
                    str(r.get("place_number", "")) not in {"", "None"} for r in rows)
                if result.get("refunds") or (all_finished and (result.get("payouts") or {}).get("exacta")):
                    record.update(status="excluded", reason="返還・同着・不成立等の非通常決着")
                    self.dirty = True
                continue
            payouts = result["payouts"]["exacta"]
            amount = int(payouts[0].get("amount") or 0)
            if amount <= 0:
                continue
            winning = payouts[0]["combination"]
            for entry in record["strategies"].values():
                returned = amount if entry["pick"] == winning else 0
                entry.update(return_amount=returned, profit=returned - 100, hit=returned > 0)
            record.update(status="settled", actual_pair=winning, settled_at=now.isoformat())
            self.dirty = True

    def summary(self) -> dict:
        records = sorted((r for r in self.data["records"].values() if r["status"] == "settled"),
                         key=lambda r:r["closed_at"])
        result = {}
        for record in records:
            for name, entry in record["strategies"].items():
                m = result.setdefault(name, dict(races=0, hits=0, stake=0, returned=0, profit=0,
                                                max_drawdown=0, max_losing_streak=0, peak=0, streak=0))
                m["races"] += 1
                m["hits"] += int(entry["hit"])
                m["stake"] += 100
                m["returned"] += entry["return_amount"]
                m["profit"] += entry["profit"]
                m["peak"] = max(m["peak"], m["profit"])
                m["max_drawdown"] = max(m["max_drawdown"], m["peak"] - m["profit"])
                m["streak"] = 0 if entry["hit"] else m["streak"] + 1
                m["max_losing_streak"] = max(m["max_losing_streak"], m["streak"])
        return result

    def save(self) -> None:
        if not self.dirty:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        states = [r["status"] for r in self.data["records"].values()]
        lines = ["# 締切前スナップショットによる将来比較", "",
            "比較専用。実購入なし。全方式が同じレースで2連単1点・100円を仮想購入します。", "",
            f"保存済み {len(states)}レース／確定 {states.count('settled')}／未確定 {states.count('pending')}／非通常決着で除外 {states.count('excluded')}", "",
            "| 方式 | 確定 | 的中率 | 仮想投資 | 仮想払戻 | 収支 | 回収率 | 最大下落 | 最大連敗 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, m in self.summary().items():
            lines.append(f"| {name} | {m['races']} | {m['hits']/m['races']:.1%} | {m['stake']:,}円 | "
                         f"{m['returned']:,}円 | {m['profit']:+,}円 | {m['returned']/m['stake']:.1%} | "
                         f"{m['max_drawdown']:,}円 | {m['max_losing_streak']} |")
        if not self.summary():
            lines += ["", "まだ比較結果はありません。成績を推測して埋めません。"]
        lines += ["", "## 条件・限界", "",
            "- 共通対象：締切7〜13分前、6艇の展示・選手情報と2連単30通りのオッズが揃い、風速6m未満・波高10cm未満。1号艇1着には限定しません。",
            "- 現行スコアもこの比較では確率最大の2連単1点を選びます。従来の期待値選別・最大3点の通知成績とは別物です。",
            "- 人気順は取得時オッズの最小の組合せ。逆オッズの正規化は市場比較用で、真の確率と保証するものではありません。",
            "- 取得時刻・特徴量・30通りのオッズ・モデル識別子・全方式の予測を保存。最初の有効スナップショットを固定し、結果判明後に選び直しません。",
            "- 保存時点で結果があるレースは不採用。オッズの元サイト更新時刻までは保証できません。",
            "- 払戻は確定払戻を使用。未確定は損益に含めず、返還・同着等は全方式で除外。取り逃したレースは後付けで予想しません。",
            "- 最大下落は締切時刻順の累計損益のピークからの下落額。別期間での再現性や将来利益は保証しません。",
            "- GitHubの実行遅延・データ欠損のあるレースは記録されません。全開催レースの成績ではありません。", ""]
        self.report.write_text("\n".join(lines), encoding="utf-8")
        self.dirty = False
