"""
features/era.py

Era bands for the win-prediction feature builder, per FEATURE_SPEC.md section 5.

Band boundaries (all season strings are "YYYY-YY" and sort correctly as text):
    A0: 1946-47 .. 1972-73   sparse box score only, BPM/VORP not available
    A1: 1973-74 .. 1978-79   BPM/VORP + rates available, no 3PT line yet
    A2: 1979-80 .. 1995-96   BPM/VORP + full box incl. 3PAr
    B:  1996-97 .. 2006-07   BPM/VORP + box, play-by-play derived stats exist
    C:  2007-08 ..           BPM/VORP + box, lineup net rating available
"""

from enum import Enum
from functools import lru_cache

import db


class EraBand(str, Enum):
    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    B = "B"
    C = "C"


_BAND_THRESHOLDS = [
    (EraBand.C, "2007-08"),
    (EraBand.B, "1996-97"),
    (EraBand.A2, "1979-80"),
    (EraBand.A1, "1973-74"),
    (EraBand.A0, "1946-47"),
]


def band_for_season(season: str) -> EraBand:
    """Return the EraBand a given season string falls into."""
    for band, threshold in _BAND_THRESHOLDS:
        if season >= threshold:
            return band
    raise ValueError(f"Season {season!r} predates the tracked league history (1946-47).")


@lru_cache(maxsize=1)
def _thresholds(db_path_str: str) -> dict:
    """One-time load of stat_availability thresholds (dataset builds call
    gate() hundreds of thousands of times; opening a fresh sqlite
    connection per call dominated runtime, so cache the whole table)."""
    conn = db.get_connection(db_path_str)
    try:
        rows = conn.execute("SELECT stat_category, season FROM stat_availability").fetchall()
    finally:
        conn.close()
    return dict(rows)


def gate(stat_category: str, season: str, db_path=db.DB_PATH) -> bool:
    """True if `stat_category` is available for `season` (see db.is_stat_available)."""
    threshold_season = _thresholds(str(db_path)).get(stat_category)
    if threshold_season is None:
        return False
    return season >= threshold_season
