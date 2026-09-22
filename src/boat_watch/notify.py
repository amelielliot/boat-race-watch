from __future__ import annotations

import json
import re
from datetime import datetime

import requests

from .results import RecordedBet


BUY_RE = re.compile(r"管理ID:([0-9-]+);予定額:(\d+)(?:;買い目:([0-9A-Z:@,\-]+))?")
SKIP_RE = re.compile(r"管理ID:([0-9-]+);見送り")
RESULT_RE = re.compile(r"結果ID:([0-9-]+);収支:([+-]?\d+)")


class Ntfy:
    def __init__(self, topic: str, session: requests.Session | None = None):
        self.topic = topic
        self.session = session or requests.Session()

    @property
    def url(self) -> str:
        return f"https://ntfy.sh/{self.topic}"

    def recent(self) -> list[dict]:
        if not self.topic:
            return []
        response = self.session.get(f"{self.url}/json", params={"poll": "1", "since": "24h"}, timeout=20)
        response.raise_for_status()
        return [json.loads(line) for line in response.text.splitlines() if line.strip()]

    def used_keys_and_budget(self, now: datetime, items: list[dict] | None = None) -> tuple[set[str], int]:
        keys: set[str] = set()
        amounts: dict[str, int] = {}
        day = now.strftime("%Y%m%d")
        for item in self.recent() if items is None else items:
            message = item.get("message", "")
            for key, amount, _ in BUY_RE.findall(message):
                if key.startswith(day):
                    keys.add(key)
                    amounts[key] = int(amount)
            for key in SKIP_RE.findall(message):
                if key.startswith(day):
                    keys.add(key)
        return keys, sum(amounts.values())

    def recorded_bets(self, now: datetime, items: list[dict] | None = None) -> dict[str, RecordedBet]:
        day = now.strftime("%Y%m%d")
        records: dict[str, RecordedBet] = {}
        for item in self.recent() if items is None else items:
            for key, _, raw_bets in BUY_RE.findall(item.get("message", "")):
                if not key.startswith(day) or not raw_bets:
                    continue
                parts = key.split("-")
                if len(parts) != 3:
                    continue
                bets: dict[str, int] = {}
                for encoded in raw_bets.split(","):
                    combination, separator, stake = encoded.partition("@")
                    if separator and stake.isdigit():
                        bets[combination] = int(stake)
                if bets:
                    records[key] = RecordedBet(key, int(parts[1]), int(parts[2]), bets)
        return records

    def settled_results(self, now: datetime, items: list[dict] | None = None) -> tuple[set[str], int]:
        day = now.strftime("%Y%m%d")
        profits: dict[str, int] = {}
        for item in self.recent() if items is None else items:
            for key, profit in RESULT_RE.findall(item.get("message", "")):
                if key.startswith(day):
                    profits[key] = int(profit)
        return set(profits), sum(profits.values())

    def publish(self, title: str, message: str, priority: int = 3) -> None:
        if not self.topic:
            print(title)
            print(message)
            return
        response = self.session.post(
            "https://ntfy.sh",
            json={
                "topic": self.topic,
                "title": title,
                "message": message,
                "priority": priority,
                "tags": ["boat"],
            },
            timeout=20,
        )
        response.raise_for_status()
