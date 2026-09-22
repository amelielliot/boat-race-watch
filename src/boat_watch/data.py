from __future__ import annotations

import re
from datetime import date, datetime
from itertools import permutations
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

JST = ZoneInfo("Asia/Tokyo")
PROGRAM_URL = "https://boatraceopenapi.github.io/api/v1/{year}/{yyyymmdd}.json"
TRIFECTA_ODDS_URL = "https://www.boatrace.jp/owpc/pc/race/odds3t"
EXACTA_ODDS_URL = "https://www.boatrace.jp/owpc/pc/race/odds2tf"
USER_AGENT = "boat-race-watch/0.1 (+personal advisory; low-frequency candidate requests)"


class DataError(RuntimeError):
    pass


def fetch_programs(day: date, session: requests.Session | None = None) -> dict:
    client = session or requests.Session()
    url = PROGRAM_URL.format(year=day.year, yyyymmdd=day.strftime("%Y%m%d"))
    response = client.get(url, timeout=20, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.json()


def iter_races(payload: dict):
    stadiums = payload.get("programs", {}).get("stadiums", {})
    for stadium_key, stadium in stadiums.items():
        for race_key, race in stadium.get("races", {}).items():
            race.setdefault("stadium_number", int(stadium_key))
            race.setdefault("race_number", int(race_key))
            yield race


def parse_closed_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("/", "-"))
    return parsed.replace(tzinfo=JST) if parsed.tzinfo is None else parsed.astimezone(JST)


def races_in_window(payload: dict, now: datetime, minimum: float, maximum: float) -> list[dict]:
    selected = []
    for race in iter_races(payload):
        closed_at = parse_closed_at(race["closed_at"])
        minutes = (closed_at - now).total_seconds() / 60
        if minimum <= minutes <= maximum:
            race["minutes_to_close"] = minutes
            selected.append(race)
    return selected


def _expand_table(table) -> list[list[str]]:
    grid: list[list[str]] = []
    spans: dict[tuple[int, int], tuple[str, int]] = {}
    rows = table.find_all("tr")
    for r_index, row in enumerate(rows):
        values: list[str] = []
        c_index = 0

        def consume_spans() -> None:
            nonlocal c_index
            while (r_index, c_index) in spans:
                text, _ = spans[(r_index, c_index)]
                values.append(text)
                c_index += 1

        consume_spans()
        for cell in row.find_all(["th", "td"], recursive=False):
            consume_spans()
            text = " ".join(cell.get_text(" ", strip=True).split())
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            for dc in range(colspan):
                values.append(text)
                if rowspan > 1:
                    for dr in range(1, rowspan):
                        spans[(r_index + dr, c_index + dc)] = (text, rowspan - dr)
            c_index += colspan
        consume_spans()
        grid.append(values)
    width = max((len(row) for row in grid), default=0)
    return [row + [""] * (width - len(row)) for row in grid]


def parse_trifecta_odds(html: str) -> dict[str, float]:
    """Parse the official six-block trifecta table and validate all 120 combinations."""
    soup = BeautifulSoup(html, "html.parser")
    possible = {"-".join(map(str, p)) for p in permutations(range(1, 7), 3)}
    best: dict[str, float] = {}

    for table in soup.find_all("table"):
        grid = _expand_table(table)
        width = max((len(row) for row in grid), default=0)
        if width < 18:
            continue
        parsed: dict[str, float] = {}
        # Official layout: six horizontal blocks. The block position is first place;
        # each block then contains second, third, and odds columns.
        for row in grid:
            for first in range(1, 7):
                offset = (first - 1) * 3
                if offset + 2 >= len(row):
                    continue
                second, third, raw_odds = row[offset : offset + 3]
                if second not in "123456" or third not in "123456":
                    continue
                combo = f"{first}-{second}-{third}"
                if combo not in possible:
                    continue
                match = re.search(r"\d+(?:\.\d+)?", raw_odds.replace(",", ""))
                if match:
                    parsed[combo] = float(match.group())
        if len(parsed) > len(best):
            best = parsed

    if len(best) != 120:
        raise DataError(f"3連単オッズを120通り取得できませんでした（{len(best)}通り）")
    return best


def parse_exacta_odds(html: str) -> dict[str, float]:
    """Parse the official six-block exacta table and validate all 30 combinations."""
    soup = BeautifulSoup(html, "html.parser")
    possible = {f"{first}-{second}" for first in range(1, 7) for second in range(1, 7) if first != second}
    best: dict[str, float] = {}

    for table in soup.find_all("table"):
        parsed: dict[str, float] = {}
        body = table.find("tbody")
        if body is None:
            continue
        for row in body.find_all("tr", recursive=False):
            cells = row.find_all("td", recursive=False)
            if len(cells) < 12:
                continue
            # Official layout: six horizontal blocks. The block position is first
            # place; each pair of cells contains second place and its odds.
            for first in range(1, 7):
                offset = (first - 1) * 2
                second = cells[offset].get_text(" ", strip=True)
                raw_odds = cells[offset + 1].get_text(" ", strip=True)
                combination = f"{first}-{second}"
                if combination not in possible:
                    continue
                match = re.search(r"\d+(?:\.\d+)?", raw_odds.replace(",", ""))
                if match and float(match.group()) > 0:
                    parsed[combination] = float(match.group())
        if len(parsed) > len(best):
            best = parsed

    if len(best) != 30:
        raise DataError(f"2連単オッズを30通り取得できませんでした（{len(best)}通り）")
    return best


def fetch_trifecta_odds(day: date, stadium: int, race_number: int, session=None) -> dict[str, float]:
    client = session or requests.Session()
    response = client.get(
        TRIFECTA_ODDS_URL,
        params={"hd": day.strftime("%Y%m%d"), "jcd": f"{stadium:02d}", "rno": race_number},
        timeout=20,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return parse_trifecta_odds(response.text)


def fetch_exacta_odds(day: date, stadium: int, race_number: int, session=None) -> dict[str, float]:
    client = session or requests.Session()
    response = client.get(
        EXACTA_ODDS_URL,
        params={"hd": day.strftime("%Y%m%d"), "jcd": f"{stadium:02d}", "rno": race_number},
        timeout=20,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return parse_exacta_odds(response.text)
