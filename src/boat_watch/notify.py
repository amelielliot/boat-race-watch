from __future__ import annotations

import json
import re
from datetime import datetime

import requests


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

    def used_keys_and_budget(self, now: datetime) -> tuple[set[str], int]:
        keys: set[str] = set()
        total = 0
        day = now.strftime("%Y%m%d")
        for item in self.recent():
            message = item.get("message", "")
            for key, amount in re.findall(r"管理ID:([0-9-]+);予定額:(\d+)", message):
                if key.startswith(day):
                    keys.add(key)
                    total += int(amount)
            for key in re.findall(r"管理ID:([0-9-]+);見送り", message):
                if key.startswith(day):
                    keys.add(key)
        return keys, total

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
