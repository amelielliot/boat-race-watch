from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecordedBet:
    key: str
    stadium_number: int
    race_number: int
    bets: dict[str, int]

    @property
    def stake(self) -> int:
        return sum(self.bets.values())


@dataclass(frozen=True)
class Settlement:
    key: str
    winning_payouts: dict[str, int]
    stake: int
    return_amount: int
    profit: int

    @property
    def hit(self) -> bool:
        return self.return_amount > 0


BET_TYPES = {
    "2T": ("2連単", "exacta"),
    "3T": ("3連単", "trifecta"),
}


def split_selection(selection: str) -> tuple[str, str]:
    code, separator, combination = selection.partition(":")
    if separator and code in BET_TYPES:
        return code, combination
    # Backward compatibility: records created before ticket-type support were trifecta.
    return "3T", selection


def selection_label(selection: str) -> str:
    code, combination = split_selection(selection)
    return f"{BET_TYPES[code][0]} {combination}"


def settle_bet(record: RecordedBet, race: dict) -> Settlement | None:
    """Settle recorded exacta/trifecta bets once their result payouts are present."""
    result = race.get("result") or {}
    result_payouts = result.get("payouts") or {}
    winning_payouts: dict[str, int] = {}
    selected_codes = {split_selection(selection)[0] for selection in record.bets}
    for code in selected_codes:
        payout_key = BET_TYPES[code][1]
        payouts = result_payouts.get(payout_key) or []
        if not payouts:
            return None
        for payout in payouts:
            combination = str(payout.get("combination") or "").strip()
            try:
                amount = int(payout.get("amount") or 0)
            except (TypeError, ValueError):
                continue
            if combination and amount > 0:
                winning_payouts[f"{code}:{combination}"] = amount
    if not winning_payouts:
        return None

    returned = sum(
        winning_payouts.get(
            selection if ":" in selection else f"3T:{selection}", 0
        ) * stake // 100
        for selection, stake in record.bets.items()
    )
    return Settlement(
        key=record.key,
        winning_payouts=winning_payouts,
        stake=record.stake,
        return_amount=returned,
        profit=returned - record.stake,
    )
