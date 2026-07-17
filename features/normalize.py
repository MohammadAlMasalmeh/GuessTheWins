"""
features/normalize.py

Cross-era value normalization.

The game composes fictional rosters from *any* era ("any 5 players, any era,
any team, thrown together"), so a roster can mix Band A0 players (no
BPM/VORP — value falls back to PER) with Band A1-C players (BPM/VORP). PER
(league-average ~15) and BPM (league-average ~0, SD ~2) are not on
comparable scales, so summing them directly for a mixed-era roster's
"overall value" would be wrong — a strong PER (e.g. 25) would swamp strong
BPM values (e.g. 8) despite being roughly the same level of dominance.

We fix this by z-scoring each band's primary value metric against that
band's own population (players with meaningful minutes, so bench garbage
time doesn't skew the mean/std), then rescaling every band onto the BPM
scale (mean 0, SD = the empirical BPM SD from the richest/most reliable
bands). This is a modeling approximation, not a real conversion — PER and
BPM measure different things — but it's the least-bad way to make
composition features coherent when eras are mixed, and it only affects the
composite "overall value" fields, not the raw per-metric diagnostics.
"""

from functools import lru_cache

import db
from features.era import EraBand

MIN_MINUTES_FOR_BAND_STATS = 500.0

# Which raw column is the "overall value" signal for each band, per compose.py.
BAND_VALUE_METRIC = {
    EraBand.A0: "per",
    EraBand.A1: "bpm",
    EraBand.A2: "bpm",
    EraBand.B: "bpm",
    EraBand.C: "bpm",
}

_BAND_SEASON_RANGES = {
    EraBand.A0: ("1946-47", "1972-73"),
    EraBand.A1: ("1973-74", "1978-79"),
    EraBand.A2: ("1979-80", "1995-96"),
    EraBand.B: ("1996-97", "2006-07"),
    EraBand.C: ("2007-08", "9999-99"),
}


@lru_cache(maxsize=1)
def _band_stats(db_path_str: str) -> dict:
    """mean/std of each band's primary value metric, qualified by minutes played."""
    conn = db.get_connection(db_path_str)
    try:
        stats = {}
        for band, (lo, hi) in _BAND_SEASON_RANGES.items():
            metric = BAND_VALUE_METRIC[band]
            rows = conn.execute(
                f"""
                SELECT {metric} FROM player_seasons
                WHERE season >= ? AND season <= ? AND minutes_total >= ? AND {metric} IS NOT NULL
                """,
                (lo, hi, MIN_MINUTES_FOR_BAND_STATS),
            ).fetchall()
            vals = [r[0] for r in rows]
            if len(vals) < 2:
                stats[band] = (0.0, 1.0)
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            stats[band] = (mean, var ** 0.5)
        return stats
    finally:
        conn.close()


@lru_cache(maxsize=1)
def _target_std(db_path_str: str) -> float:
    """Reference SD (BPM scale) computed from the richest bands, used as the common unit."""
    stats = _band_stats(db_path_str)
    bpm_stds = [std for band, (mean, std) in stats.items() if BAND_VALUE_METRIC[band] == "bpm"]
    return sum(bpm_stds) / len(bpm_stds) if bpm_stds else 2.0


def normalize_value(raw_value, band: EraBand, db_path=db.DB_PATH):
    """Rescale a band's raw overall-value metric onto a common BPM-like unit (mean 0)."""
    if raw_value is None:
        return None
    db_path_str = str(db_path)
    mean, std = _band_stats(db_path_str)[band]
    if std == 0:
        return 0.0
    z = (raw_value - mean) / std
    return z * _target_std(db_path_str)
