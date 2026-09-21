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


def settle_bet(record: RecordedBet, race: dict) -> Settlement | None:
    """Settle a recorded trifecta bet once official result payouts are present."""
    result = race.get("result") or {}
    payouts = (result.get("payouts") or {}).get("trifecta") or []
    if not payouts:
        return None

    winning_payouts: dict[str, int] = {}
    for payout in payouts:
        combination = str(payout.get("combination") or "").strip()
        try:
            amount = int(payout.get("amount") or 0)
        except (TypeError, ValueError):
            continue
        if combination and amount > 0:
            winning_payouts[combination] = amount
    if not winning_payouts:
        return None

    returned = sum(
        winning_payouts.get(combination, 0) * stake // 100
        for combination, stake in record.bets.items()
    )
    return Settlement(
        key=record.key,
        winning_payouts=winning_payouts,
        stake=record.stake,
        return_amount=returned,
        profit=returned - record.stake,
    )
