"""
features/compose.py

Aggregate 5 players' era-gated atomic features (features/atomic.py) into one
roster feature vector, per FEATURE_SPEC.md section 6.

Value core selection by band (never a team-outcome field, never Win Shares):
    A1, A2, B, C: offensive=obpm, defensive=dbpm, overall=bpm, replacement=vorp
    A0:           no BPM/VORP exists; overall value falls back to `per`
                  (partial-fill, treated as low confidence), no split
                  offense/defense is possible, so defensive_value is left None.

Composition/fit features (spacing, usage collision, rebounding coverage,
role redundancy, top-heaviness) prevent 5 high-usage/ball-dominant players
from naively summing to an unrealistic win total.
"""

from dataclasses import dataclass
from statistics import pstdev
from typing import List, Optional

import db
from features.atomic import AtomicPlayerSeason, get_atomic_features
from features.era import EraBand
from features.normalize import normalize_value

# Open calibration knobs, per FEATURE_SPEC.md section 9.
SPACER_THREE_PAR_THRESHOLD = 0.35
PRIMARY_CREATOR_USG_THRESHOLD = 24.0
PRIMARY_CREATOR_AST_PCT_THRESHOLD = 18.0
ESTIMATED_ROW_SAMPLE_WEIGHT = 0.5
REBOUND_COVERAGE_DEFICIENT_THRESHOLD = 80.0  # sum of trb_pct across the 5

_HAS_BPM_BANDS = {EraBand.A1, EraBand.A2, EraBand.B, EraBand.C}


def _mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _sum(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) if vals else None


def _max(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _min(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return min(vals) if vals else None


def _std(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return pstdev(vals) if len(vals) >= 2 else (0.0 if len(vals) == 1 else None)


def _get_roles(atomic_list: List[AtomicPlayerSeason], db_path) -> List[Optional[str]]:
    conn = db.get_connection(db_path)
    try:
        roles = []
        for a in atomic_list:
            row = conn.execute(
                "SELECT primary_role FROM player_roles WHERE player_id=? AND season=?",
                (a.player_id, a.season),
            ).fetchone()
            roles.append(row[0] if row else None)
        return roles
    finally:
        conn.close()


@dataclass
class RosterFeatures:
    features: dict
    band_mix: List[EraBand]
    n_estimated: int
    n_missing_value: int
    sample_weight: float


def compose_roster(atomic_list: List[AtomicPlayerSeason], db_path=db.DB_PATH) -> RosterFeatures:
    if len(atomic_list) != 5:
        raise ValueError(f"compose_roster requires exactly 5 players, got {len(atomic_list)}")

    bands = [a.band for a in atomic_list]

    # -- value core: offensive / defensive / overall / replacement --------
    offensive_vals, defensive_vals, overall_vals_raw, vorp_vals = [], [], [], []
    for a in atomic_list:
        if a.band in _HAS_BPM_BANDS:
            offensive_vals.append(a.values.get("obpm"))
            defensive_vals.append(a.values.get("dbpm"))
            overall_vals_raw.append(a.values.get("bpm"))
            vorp_vals.append(a.values.get("vorp"))
        else:  # A0 — no BPM/VORP; fall back to PER as sole low-confidence overall value
            offensive_vals.append(None)
            defensive_vals.append(None)
            overall_vals_raw.append(a.values.get("per"))
            vorp_vals.append(None)

    # Rescale each player's overall value onto a common cross-era unit (see
    # features/normalize.py) so a mixed-era roster (e.g. a pre-BPM legend
    # alongside a modern player) aggregates coherently instead of mixing
    # PER and BPM on their native, incompatible scales.
    overall_vals = [
        normalize_value(raw, a.band, db_path=db_path)
        for raw, a in zip(overall_vals_raw, atomic_list)
    ]

    n_missing_value = sum(1 for v in overall_vals if v is None)

    f = {}
    f["sum_off_value"] = _sum(offensive_vals)
    f["mean_off_value"] = _mean(offensive_vals)
    f["sum_def_value"] = _sum(defensive_vals)
    f["mean_def_value"] = _mean(defensive_vals)
    f["sum_overall_value"] = _sum(overall_vals)
    f["mean_overall_value"] = _mean(overall_vals)
    f["max_overall_value"] = _max(overall_vals)
    f["min_overall_value"] = _min(overall_vals)
    f["overall_value_std"] = _std(overall_vals)
    top1 = f["max_overall_value"]
    total = f["sum_overall_value"]
    f["value_top1_share"] = (top1 / total) if (top1 is not None and total not in (None, 0)) else None
    f["sum_vorp"] = _sum(vorp_vals)
    f["mean_vorp"] = _mean(vorp_vals)

    # A0-only low-confidence substitutes (PPG concentration, TS, FTr)
    ppg_vals = [a.values.get("pts_per_game") for a in atomic_list]
    ts_vals = [a.values.get("ts_pct") for a in atomic_list]
    ftr_vals = [a.values.get("ft_rate") for a in atomic_list]
    f["sum_ppg"] = _sum(ppg_vals)
    f["max_ppg"] = _max(ppg_vals)
    f["ppg_top1_share"] = (f["max_ppg"] / f["sum_ppg"]) if (f["max_ppg"] is not None and f["sum_ppg"] not in (None, 0)) else None
    f["mean_ts_pct"] = _mean(ts_vals)
    f["mean_ft_rate"] = _mean(ftr_vals)

    # -- composition / fit --------------------------------------------------
    usg_vals = [a.values.get("usg_pct") for a in atomic_list]
    ast_pct_vals = [a.values.get("ast_pct") for a in atomic_list]
    three_par_vals = [a.values.get("three_par") for a in atomic_list]
    trb_pct_vals = [a.values.get("trb_pct") for a in atomic_list]
    ast_to_vals = [a.values.get("ast_to") for a in atomic_list]

    f["sum_usg"] = _sum(usg_vals)
    f["max_usg"] = _max(usg_vals)
    f["usg_overflow"] = max(0.0, f["sum_usg"] - 100.0) if f["sum_usg"] is not None else None

    f["n_primary_creators"] = sum(
        1
        for usg, ast in zip(usg_vals, ast_pct_vals)
        if usg is not None and ast is not None
        and usg >= PRIMARY_CREATOR_USG_THRESHOLD and ast >= PRIMARY_CREATOR_AST_PCT_THRESHOLD
    )
    # How many creators past a workable two-creator offense. Real NBA fives
    # almost never field 3+ true primary creators; fictional mashups do.
    f["creator_excess"] = float(max(0, f["n_primary_creators"] - 2))

    n_three_par_known = sum(1 for v in three_par_vals if v is not None)
    f["n_spacers"] = (
        sum(1 for v in three_par_vals if v is not None and v >= SPACER_THREE_PAR_THRESHOLD)
        if n_three_par_known > 0 else None
    )
    f["mean_three_par"] = _mean(three_par_vals)
    f["min_three_par"] = _min(three_par_vals)
    f["spacing_void"] = (
        1.0 if (f["n_spacers"] is not None and f["n_spacers"] == 0) else 0.0
    ) if f["n_spacers"] is not None else None

    f["sum_trb_pct"] = _sum(trb_pct_vals)
    f["mean_trb_pct"] = _mean(trb_pct_vals)
    f["rebound_deficient"] = (
        1.0 if (f["sum_trb_pct"] is not None and f["sum_trb_pct"] < REBOUND_COVERAGE_DEFICIENT_THRESHOLD) else 0.0
    ) if f["sum_trb_pct"] is not None else None

    # Defensive / discipline aggregates — repeatedly selected by Lasso/Ridge
    # NBA win models (DREB, STL, TOV matter more than raw FG%; TS + AST/TO
    # capture offense). Atomic columns already exist; previously only TRB
    # and AST/TO were composed into the roster vector.
    drb_pct_vals = [a.values.get("drb_pct") for a in atomic_list]
    stl_pct_vals = [a.values.get("stl_pct") for a in atomic_list]
    blk_pct_vals = [a.values.get("blk_pct") for a in atomic_list]
    tov_pct_vals = [a.values.get("tov_pct") for a in atomic_list]
    f["mean_drb_pct"] = _mean(drb_pct_vals)
    f["sum_drb_pct"] = _sum(drb_pct_vals)
    f["mean_stl_pct"] = _mean(stl_pct_vals)
    f["sum_stl_pct"] = _sum(stl_pct_vals)
    f["mean_blk_pct"] = _mean(blk_pct_vals)
    f["mean_tov_pct"] = _mean(tov_pct_vals)

    f["mean_ast_to"] = _mean(ast_to_vals)

    roles = _get_roles(atomic_list, db_path)
    known_roles = [r for r in roles if r is not None]
    n_unique_roles = len(set(known_roles)) if known_roles else None
    f["role_redundancy"] = (5 - n_unique_roles) if n_unique_roles is not None else None
    f["role_overlap_flag"] = (1.0 if (f["role_redundancy"] is not None and f["role_redundancy"] >= 2) else 0.0) \
        if f["role_redundancy"] is not None else None

    # Star / support shape — used by fit adjustment (model/fit_adjustment.py)
    # and as candidate Stage 1 features. top2 captures "two alphas";
    # support_value is everyone except the best player.
    sorted_overall = sorted((v for v in overall_vals if v is not None), reverse=True)
    f["top2_overall_value"] = (
        sum(sorted_overall[:2]) if len(sorted_overall) >= 2
        else (sorted_overall[0] if sorted_overall else None)
    )
    if f["max_overall_value"] is not None and f["sum_overall_value"] is not None:
        f["support_value"] = f["sum_overall_value"] - f["max_overall_value"]
    else:
        f["support_value"] = None

    # Offense piled up without team defense (iso-scorer mashups).
    if f["sum_off_value"] is not None and f["sum_def_value"] is not None:
        f["off_def_gap"] = max(0.0, f["sum_off_value"] - f["sum_def_value"])
    else:
        f["off_def_gap"] = None

    n_estimated = sum(1 for a in atomic_list if a.has_estimated_stats)
    f["n_estimated"] = float(n_estimated)
    f["n_missing_value"] = float(n_missing_value)

    # Sample weight: down-weight rosters leaning on estimated stats, and
    # down-weight A0 (sparse, low-confidence era) relative to richer bands.
    band_weight = _band_weight(bands)
    estimated_frac = n_estimated / 5.0
    sample_weight = band_weight * (1.0 - estimated_frac * (1.0 - ESTIMATED_ROW_SAMPLE_WEIGHT))

    return RosterFeatures(
        features=f,
        band_mix=bands,
        n_estimated=n_estimated,
        n_missing_value=n_missing_value,
        sample_weight=sample_weight,
    )


def _band_weight(bands: List[EraBand]) -> float:
    """Coarsest (lowest-confidence) band present in the roster sets the weight."""
    weights = {EraBand.A0: 0.35, EraBand.A1: 0.7, EraBand.A2: 0.85, EraBand.B: 0.9, EraBand.C: 1.0}
    return min(weights[b] for b in bands)


def compose_roster_by_ids(
    player_seasons: List[tuple],  # list of (player_id, season, team_id_or_None)
    db_path=db.DB_PATH,
) -> RosterFeatures:
    """Convenience wrapper: fetch atomic features for 5 (player_id, season, team_id) tuples and compose."""
    atomic_list = []
    for player_id, season, team_id in player_seasons:
        a = get_atomic_features(player_id, season, team_id, db_path=db_path)
        if a is None:
            raise ValueError(f"No player_seasons row for player_id={player_id}, season={season}")
        atomic_list.append(a)
    return compose_roster(atomic_list, db_path=db_path)
