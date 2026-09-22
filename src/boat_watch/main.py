from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from .config import Config
from .data import (
    DataError,
    fetch_exacta_odds,
    fetch_programs,
    fetch_trifecta_odds,
    iter_races,
    races_in_window,
)
from .model import decide
from .notify import Ntfy
from .results import RecordedBet, Settlement, selection_label, settle_bet

JST = ZoneInfo("Asia/Tokyo")
STADIUMS = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}


def _race_key(now: datetime, race: dict) -> str:
    return f"{now:%Y%m%d}-{int(race['stadium_number']):02d}-{int(race['race_number']):02d}"


def _format(race: dict, decision, key: str, shadow: bool) -> tuple[str, str, int]:
    stadium = STADIUMS.get(int(race["stadium_number"]), str(race["stadium_number"]))
    race_no = int(race["race_number"])
    prefix = "【検証中・買わない】" if shadow and decision.action == "BUY" else "【買い候補】"
    if decision.action == "SKIP":
        return (
            f"【見送り】{stadium}{race_no}R",
            "\n".join([f"理由：{'／'.join(decision.reasons)}", f"管理ID:{key};見送り"]),
            2,
        )
    total = sum(bet.stake for bet in decision.bets)
    lines = [f"{bet.bet_type} {bet.combination}　{bet.stake:,}円（オッズ{bet.odds:.1f}／期待値{bet.expected_value:.2f}）" for bet in decision.bets]
    type_codes = {"2連単": "2T", "3連単": "3T"}
    encoded_bets = ",".join(
        f"{type_codes[bet.bet_type]}:{bet.combination}@{bet.stake}" for bet in decision.bets
    )
    lines += [
        f"理由：{'／'.join(decision.reasons)}",
        "投票前に公式サイトで欠場・進入・最新オッズを再確認",
        f"管理ID:{key};予定額:{total};買い目:{encoded_bets}",
    ]
    return f"{prefix}{stadium}{race_no}R", "\n".join(lines), 4


def _format_result(record: RecordedBet, settlement: Settlement, daily_profit: int) -> tuple[str, str]:
    stadium = STADIUMS.get(record.stadium_number, str(record.stadium_number))
    label = "的中" if settlement.hit else "ハズレ"
    payout_text = "／".join(
        f"{selection_label(selection)} {amount:,}円"
        for selection, amount in settlement.winning_payouts.items()
    )
    bet_text = "／".join(
        f"{selection_label(selection)} {stake:,}円" for selection, stake in record.bets.items()
    )
    lines = [
        f"結果：{payout_text}（100円あたり）",
        f"通知買い目：{bet_text}",
        f"仮想払戻：{settlement.return_amount:,}円",
        f"レース収支：{settlement.profit:+,}円",
        f"本日累計：{daily_profit:+,}円",
        "検証成績（実購入ではありません）",
        f"結果ID:{record.key};収支:{settlement.profit}",
    ]
    return f"【{label}】{stadium}{record.race_number}R", "\n".join(lines)


def _notify_results(payload: dict, now: datetime, notifier: Ntfy, history: list[dict]) -> None:
    records = notifier.recorded_bets(now, history)
    settled_keys, daily_profit = notifier.settled_results(now, history)
    races = {
        (int(race["stadium_number"]), int(race["race_number"])): race
        for race in iter_races(payload)
    }
    for key, record in records.items():
        if key in settled_keys:
            continue
        race = races.get((record.stadium_number, record.race_number))
        if race is None:
            continue
        settlement = settle_bet(record, race)
        if settlement is None:
            continue
        daily_profit += settlement.profit
        title, message = _format_result(record, settlement, daily_profit)
        notifier.publish(title, message, 4 if settlement.hit else 3)
        settled_keys.add(key)


def run(now: datetime | None = None) -> int:
    config = Config()
    now = (now or datetime.now(JST)).astimezone(JST)
    session = requests.Session()
    notifier = Ntfy(config.ntfy_topic, session)
    try:
        payload = fetch_programs(now.date(), session)
    except Exception as exc:
        print(f"開催データ取得失敗: {exc}")
        return 1
    try:
        history = notifier.recent()
        used_keys, used_budget = notifier.used_keys_and_budget(now, history)
        _notify_results(payload, now, notifier, history)
    except Exception as exc:
        print(f"通知履歴取得失敗（安全のため買い候補を停止）: {exc}")
        used_keys, used_budget = set(), config.max_per_day

    races = races_in_window(payload, now, config.minutes_before_min, config.minutes_before_max)
    if not races:
        return 0

    for race in races:
        key = _race_key(now, race)
        if key in used_keys:
            continue
        trifecta_odds = None
        exacta_odds = None
        preview = race.get("preview") or {}
        # Only request the official odds page when the preview exists and severe-weather
        # safety gates are not already known to fail.
        if preview and float(preview.get("wind_speed") or 0) < config.max_wind_speed and float(preview.get("wave_height") or 0) < config.max_wave_height:
            try:
                trifecta_odds = fetch_trifecta_odds(now.date(), int(race["stadium_number"]), int(race["race_number"]), session)
            except (requests.RequestException, DataError, ValueError) as exc:
                print(f"{key} 3連単オッズ取得失敗: {exc}")
            try:
                exacta_odds = fetch_exacta_odds(now.date(), int(race["stadium_number"]), int(race["race_number"]), session)
            except (requests.RequestException, DataError, ValueError) as exc:
                print(f"{key} 2連単オッズ取得失敗: {exc}")
        remaining = max(0, config.max_per_day - used_budget)
        decision = decide(race, trifecta_odds, remaining, config, exacta_odds=exacta_odds)
        if decision.action == "SKIP" and not config.notify_skips:
            continue
        title, message, priority = _format(race, decision, key, config.shadow_mode)
        notifier.publish(title, message, priority)
        if decision.action == "BUY":
            used_budget += sum(bet.stake for bet in decision.bets)
        used_keys.add(key)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
