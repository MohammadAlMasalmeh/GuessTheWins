"""
features/atomic.py

Per-player, per-season atomic feature extraction, gated by era availability.

Every value returned is either the real column value or None (never a fake
0) when the underlying stat category doesn't exist yet for that season, per
FEATURE_SPEC.md section 5. Callers (features/compose.py) are responsible for
aggregating only over the non-None values across a 5-player roster.
"""

from dataclasses import dataclass, field
from typing import Optional

import db
from features.era import EraBand, band_for_season, gate

# column -> stat_availability category that gates it, or None if the column
# is simply read as-is (NULL in the db already means "missing"), per spec
# section 5's per-band gate notes (e.g. `per` in Band A0 has "no" formal
# gate — it's partial-fill and NULL when absent).
ATOMIC_GATES = {
    "pts_per_game": "FIELD_GOALS_BASIC",
    "fg_pct": "FIELD_GOALS_BASIC",
    "ts_pct": "FIELD_GOALS_BASIC",
    "ft_rate": "THREE_PAR_FT_RATE",
    "per": None,
    "obpm": "ADVANCED_BPM_VORP",
    "dbpm": "ADVANCED_BPM_VORP",
    "bpm": "ADVANCED_BPM_VORP",
    "vorp": "ADVANCED_BPM_VORP",
    "usg_pct": "ADVANCED_RATE_STATS",
    "ast_pct": "ASSISTS",
    "trb_pct": "REBOUNDS_OFF_DEF_SPLIT",
    "orb_pct": "REBOUNDS_OFF_DEF_SPLIT",
    "drb_pct": "REBOUNDS_OFF_DEF_SPLIT",
    "stl_pct": "STEALS_BLOCKS",
    "blk_pct": "STEALS_BLOCKS",
    "tov_pct": "TURNOVERS",
    "ast_to": "AST_TO_RATIO",
    "three_par": "THREE_POINT",
    "fg3_pct": "THREE_POINT",
}

# Which atomic columns are in-scope for each band. Per-column gates above
# still apply on top of this (e.g. ast_to sits inside A1's season range but
# its own gate threshold is 1977-78, so it's None for 1973-76 rows anyway).
BAND_COLUMNS = {
    EraBand.A0: ["pts_per_game", "ts_pct", "ft_rate", "fg_pct", "per"],
    EraBand.A1: [
        "obpm", "dbpm", "bpm", "vorp", "per", "ts_pct", "usg_pct",
        "ast_pct", "trb_pct", "orb_pct", "drb_pct", "stl_pct", "blk_pct",
        "tov_pct", "ft_rate", "ast_to", "pts_per_game",
    ],
}
BAND_COLUMNS[EraBand.A2] = BAND_COLUMNS[EraBand.A1] + ["three_par", "fg3_pct"]
BAND_COLUMNS[EraBand.B] = BAND_COLUMNS[EraBand.A2]
BAND_COLUMNS[EraBand.C] = BAND_COLUMNS[EraBand.A2]


@dataclass
class AtomicPlayerSeason:
    player_id: int
    season: str
    team_id: Optional[int]
    band: EraBand
    minutes_total: Optional[float]
    minutes_per_game: Optional[float]
    games_played: Optional[int]
    has_estimated_stats: bool
    values: dict = field(default_factory=dict)  # column -> float or None


_SELECT_COLUMNS = sorted(set(ATOMIC_GATES.keys()))


def _row_for_player_season(player_id: int, season: str, team_id: Optional[int], conn) -> Optional[tuple]:
    """
    A player can have multiple rows for one season if traded mid-year
    (PRIMARY KEY includes team_id). Pick the team_id row if given, else the
    stint with the most minutes_total (their primary team that season).
    """
    cols = ", ".join(["team_id", "minutes_total", "minutes_per_game", "games_played", "has_estimated_stats"] + _SELECT_COLUMNS)
    if team_id is not None:
        row = conn.execute(
            f"SELECT {cols} FROM player_seasons WHERE player_id=? AND season=? AND team_id=?",
            (player_id, season, team_id),
        ).fetchone()
        if row is not None:
            return row
    rows = conn.execute(
        f"SELECT {cols} FROM player_seasons WHERE player_id=? AND season=?",
        (player_id, season),
    ).fetchall()
    if not rows:
        return None
    rows.sort(key=lambda r: (r[1] if r[1] is not None else -1), reverse=True)
    return rows[0]


def get_atomic_features(
    player_id: int,
    season: str,
    team_id: Optional[int] = None,
    db_path=db.DB_PATH,
) -> Optional[AtomicPlayerSeason]:
    """
    Fetch the era-gated atomic feature set for one player-season.

    Returns None if the player has no row for that season at all. Otherwise
    returns an AtomicPlayerSeason whose `values` dict only contains columns
    legal for that season's era band, each gated individually to None if the
    underlying stat category doesn't exist yet.
    """
    band = band_for_season(season)
    legal_columns = set(BAND_COLUMNS[band])

    conn = db.get_connection(db_path)
    try:
        row = _row_for_player_season(player_id, season, team_id, conn)
    finally:
        conn.close()

    if row is None:
        return None

    resolved_team_id, minutes_total, minutes_per_game, games_played, has_estimated_stats = row[:5]
    raw_values = dict(zip(_SELECT_COLUMNS, row[5:]))

    values = {}
    for col in _SELECT_COLUMNS:
        if col not in legal_columns:
            values[col] = None
            continue
        gate_category = ATOMIC_GATES[col]
        if gate_category is not None and not gate(gate_category, season, db_path=db_path):
            values[col] = None
            continue
        values[col] = raw_values[col]

    return AtomicPlayerSeason(
        player_id=player_id,
        season=season,
        team_id=resolved_team_id,
        band=band,
        minutes_total=minutes_total,
        minutes_per_game=minutes_per_game,
        games_played=games_played,
        has_estimated_stats=bool(has_estimated_stats),
        values=values,
    )
