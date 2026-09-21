from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    ntfy_topic: str = os.getenv("NTFY_TOPIC", "")
    shadow_mode: bool = _bool("SHADOW_MODE", True)
    notify_skips: bool = _bool("NOTIFY_SKIPS", True)
    minutes_before_min: float = float(os.getenv("MINUTES_BEFORE_MIN", "7"))
    minutes_before_max: float = float(os.getenv("MINUTES_BEFORE_MAX", "13"))
    max_per_race: int = int(os.getenv("MAX_PER_RACE", "2000"))
    max_per_day: int = int(os.getenv("MAX_PER_DAY", "5000"))
    min_expected_value: float = float(os.getenv("MIN_EXPECTED_VALUE", "1.15"))
    max_wind_speed: float = float(os.getenv("MAX_WIND_SPEED", "6"))
    max_wave_height: float = float(os.getenv("MAX_WAVE_HEIGHT", "10"))
