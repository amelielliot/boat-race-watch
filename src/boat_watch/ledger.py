from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .model import Decision
from .notify import BUY_RE, RESULT_RE
from .results import RecordedBet, Settlement


RETURN_RE = re.compile(r"仮想払戻：([\d,]+)円")
TYPE_CODES = {"2連単": "2T", "3連単": "3T"}


class Ledger:
    """Durable paper-trading records stored in the repository."""

    def __init__(
        self,
        path: str | Path = "data/ledger.json",
        report_path: str | Path = "PERFORMANCE.md",
    ):
        self.path = Path(path)
        self.report_path = Path(report_path)
        self.data = self._load()
        self.dirty = False

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {
            "version": 1,
            "complete_from": datetime.now().date().isoformat(),
            "records": {},
        }

    @property
    def records(self) -> dict[str, dict]:
        return self.data.setdefault("records", {})

    @staticmethod
    def _day(now: datetime) -> str:
        return now.strftime("%Y%m%d")

    @staticmethod
    def _parts(key: str) -> tuple[int, int]:
        parts = key.split("-")
        return int(parts[1]), int(parts[2])

    @staticmethod
    def _encoded_bets(decision: Decision) -> dict[str, dict]:
        return {
            f"{TYPE_CODES[bet.bet_type]}:{bet.combination}": {
                "stake": bet.stake,
                "odds": bet.odds,
                "probability": round(bet.probability, 8),
                "expected_value": round(bet.expected_value, 4),
            }
            for bet in decision.bets
        }

    def record_prediction(
        self,
        key: str,
        decision: Decision,
        validation: bool,
        now: datetime,
    ) -> None:
        if decision.action != "BUY" or key in self.records:
            return
        stadium, race_number = self._parts(key)
        bets = self._encoded_bets(decision)
        self.records[key] = {
            "date": f"{key[:4]}-{key[4:6]}-{key[6:8]}",
            "stadium_number": stadium,
            "race_number": race_number,
            "status": "pending",
            "validation": validation,
            "bets": bets,
            "stake": sum(item["stake"] for item in bets.values()),
            "return_amount": None,
            "profit": None,
            "hit": None,
            "recorded_at": now.isoformat(),
            "settled_at": None,
            "source": "automatic",
            "model_version": decision.model_version,
        }
        self.dirty = True

    def reconcile_history(self, items: list[dict], now: datetime) -> None:
        """Recover records if ntfy succeeded but a previous git save did not."""
        for item in items:
            message = item.get("message", "")
            for key, amount, raw_bets, validation_marker in BUY_RE.findall(message):
                if key in self.records or not raw_bets:
                    continue
                stadium, race_number = self._parts(key)
                bets = {}
                for encoded in raw_bets.split(","):
                    selection, separator, stake = encoded.partition("@")
                    if separator and stake.isdigit():
                        bets[selection] = {
                            "stake": int(stake),
                            "odds": None,
                            "probability": None,
                            "expected_value": None,
                        }
                if not bets:
                    continue
                recorded_at = datetime.fromtimestamp(item.get("time", now.timestamp()), now.tzinfo)
                self.records[key] = {
                    "date": f"{key[:4]}-{key[4:6]}-{key[6:8]}",
                    "stadium_number": stadium,
                    "race_number": race_number,
                    "status": "pending",
                    "validation": bool(validation_marker),
                    "bets": bets,
                    "stake": int(amount),
                    "return_amount": None,
                    "profit": None,
                    "hit": None,
                    "recorded_at": recorded_at.isoformat(),
                    "settled_at": None,
                    "source": "ntfy-recovery",
                    "model_version": (re.search(r"モデル：([^\s]+)", message).group(1)
                                      if re.search(r"モデル：([^\s]+)", message) else "未記録"),
                }
                self.dirty = True

        for item in items:
            message = item.get("message", "")
            return_match = RETURN_RE.search(message)
            for key, raw_profit in RESULT_RE.findall(message):
                record = self.records.get(key)
                if record is None or record.get("status") == "settled":
                    continue
                profit = int(raw_profit)
                returned = (
                    int(return_match.group(1).replace(",", ""))
                    if return_match
                    else int(record["stake"]) + profit
                )
                settled_at = datetime.fromtimestamp(item.get("time", now.timestamp()), now.tzinfo)
                self._settle(record, returned, profit, settled_at)
                self.dirty = True

    @staticmethod
    def _settle(record: dict, returned: int, profit: int, now: datetime) -> None:
        record.update(
            status="settled",
            return_amount=returned,
            profit=profit,
            hit=returned > 0,
            settled_at=now.isoformat(),
        )

    def record_settlement(self, settlement: Settlement, now: datetime) -> None:
        record = self.records.get(settlement.key)
        if record is None:
            stadium, race_number = self._parts(settlement.key)
            record = {
                "date": f"{settlement.key[:4]}-{settlement.key[4:6]}-{settlement.key[6:8]}",
                "stadium_number": stadium,
                "race_number": race_number,
                "validation": True,
                "bets": {},
                "stake": settlement.stake,
                "recorded_at": None,
                "source": "settlement-recovery",
            }
            self.records[settlement.key] = record
        if record.get("status") == "settled":
            return
        self._settle(record, settlement.return_amount, settlement.profit, now)
        self.dirty = True

    def pending_records(self, now: datetime) -> dict[str, RecordedBet]:
        day = self._day(now)
        records = {}
        for key, record in self.records.items():
            if not key.startswith(day) or record.get("status") != "pending":
                continue
            bets = {selection: int(details["stake"]) for selection, details in record["bets"].items()}
            records[key] = RecordedBet(
                key,
                int(record["stadium_number"]),
                int(record["race_number"]),
                bets,
            )
        return records

    def used_amounts(self, now: datetime) -> dict[str, int]:
        day = self._day(now)
        return {
            key: 0 if record.get("validation") else int(record["stake"])
            for key, record in self.records.items()
            if key.startswith(day)
        }

    def settled_keys(self, now: datetime) -> set[str]:
        day = self._day(now)
        return {
            key
            for key, record in self.records.items()
            if key.startswith(day) and record.get("status") == "settled"
        }

    def daily_profit(self, now: datetime) -> int:
        day = self._day(now)
        return sum(
            int(record["profit"])
            for key, record in self.records.items()
            if key.startswith(day) and record.get("status") == "settled"
        )

    def summary(self) -> dict[str, float | int]:
        settled = [record for record in self.records.values() if record.get("status") == "settled"]
        stake = sum(int(record["stake"]) for record in settled)
        returned = sum(int(record["return_amount"]) for record in settled)
        hits = sum(bool(record["hit"]) for record in settled)
        return {
            "settled": len(settled),
            "pending": sum(record.get("status") == "pending" for record in self.records.values()),
            "stake": stake,
            "return_amount": returned,
            "profit": returned - stake,
            "hits": hits,
            "losses": len(settled) - hits,
            "recovery_rate": (returned / stake * 100) if stake else 0.0,
            "hit_rate": (hits / len(settled) * 100) if settled else 0.0,
        }

    def _report(self) -> str:
        summary = self.summary()
        by_day: dict[str, dict[str, int]] = {}
        for record in self.records.values():
            if record.get("status") != "settled":
                continue
            daily = by_day.setdefault(record["date"], {"races": 0, "stake": 0, "return": 0, "hits": 0})
            daily["races"] += 1
            daily["stake"] += int(record["stake"])
            daily["return"] += int(record["return_amount"])
            daily["hits"] += int(bool(record["hit"]))

        lines = [
            "# BOAT RACE 検証成績",
            "",
            f"最終更新：{self.data.get('updated_at', '-')}",
            "",
            "| 指標 | 累計 |",
            "|---|---:|",
            f"| 結果確定 | {summary['settled']:,}レース |",
            f"| 的中 / ハズレ | {summary['hits']:,} / {summary['losses']:,} |",
            f"| 仮想投資額 | {summary['stake']:,}円 |",
            f"| 仮想払戻 | {summary['return_amount']:,}円 |",
            f"| **収支** | **{summary['profit']:+,}円** |",
            f"| 回収率 | {summary['recovery_rate']:.1f}% |",
            f"| 的中率 | {summary['hit_rate']:.1f}% |",
            f"| 未確定 | {summary['pending']:,}レース |",
            "",
            "## 日別成績",
            "",
            "| 日付 | レース | 的中 | 投資額 | 払戻 | 収支 | 回収率 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for day in sorted(by_day, reverse=True):
            daily = by_day[day]
            profit = daily["return"] - daily["stake"]
            rate = daily["return"] / daily["stake"] * 100 if daily["stake"] else 0
            lines.append(
                f"| {day} | {daily['races']} | {daily['hits']} | {daily['stake']:,}円 | "
                f"{daily['return']:,}円 | {profit:+,}円 | {rate:.1f}% |"
            )
        lines += ["", "## モデル別成績", "",
                  "| モデル・選定方針 | 確定 | 的中率 | 未確定 | 仮想投資（確定分） | 払戻 | 収支 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        grouped = {}
        for record in self.records.values():
            version = record.get("model_version","旧方式・モデル未記録")
            group = grouped.setdefault(version,dict(settled=0,hits=0,pending=0,stake=0,returned=0))
            if record.get("status")=="settled":
                group["settled"] += 1
                group["hits"] += int(bool(record.get("hit")))
                group["stake"] += int(record["stake"])
                group["returned"] += int(record["return_amount"])
            elif record.get("status")=="pending":
                group["pending"] += 1
        for version,g in sorted(grouped.items()):
            hit_rate = g["hits"] / g["settled"] * 100 if g["settled"] else 0
            lines.append(f"| {version} | {g['settled']} | {hit_rate:.1f}% | {g['pending']} | {g['stake']:,}円 | "
                         f"{g['returned']:,}円 | {g['returned']-g['stake']:+,}円 |")
        lines += [
            "",
            "## データの範囲",
            "",
            f"- {self.data.get('complete_from', '-')}以降、正常に実行・保存できた買い候補と結果を自動保存。停止中・未取得のレースは含みません。",
            "- それ以前は、内容を確認できたレースだけを集計。",
            "- これは検証用の仮想成績で、実際の購入履歴ではありません。",
            "",
        ]
        return "\n".join(lines)

    def save(self) -> bool:
        if not self.dirty:
            return False
        self.data["updated_at"] = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.report_path.write_text(self._report(), encoding="utf-8")
        self.dirty = False
        return True
