"""
model/simulate.py

Monte Carlo season simulator: given a 5-player roster (feature vector or
5 player IDs), returns a point estimate plus a full 82-game-season win-total
distribution, exposed for the (separate, not-built-here) game layer.

Pipeline:
  1. Stage 1 (model/stage1.py, GBM) -> point-estimate net rating.
  2. Stage 2 (model/stage2.py)      -> point-estimate expected wins (Pythagorean).
  3. Monte Carlo:
     For each of N simulated seasons:
       a. Redraw the roster's "true" net rating from Normal(point_estimate,
          residual_std) — residual_std is the Stage 1 model's own held-out
          (leave-one-era-out) error, so a season's outcome isn't pinned to a
          possibly-wrong point estimate (model uncertainty).
       b. For each of 82 games: bootstrap-sample a real opponent net rating
          from historical team_seasons, convert the strength differential
          into a win probability via the logistic calibrated in
          model/game_logistic.py, flip a weighted coin.
     10,000 season replicates by default; this is cheap (each game is one
     coin flip against a precomputed probability, not a possession sim), so
     it runs in well under a second.
  4. Report point estimate + distribution (mean, median, std, percentiles).
"""

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

import db
from features.atomic import get_atomic_features
from features.compose import compose_roster
from model.stage2 import Stage2Calibration
from model.game_logistic import GameLogisticCalibration
from model.fit_adjustment import fit_net_rating_adjustment

N_GAMES = 82
DEFAULT_N_SIMS = 10000

# How far a roster's aggregate value can exceed the max ever observed in
# training before we start blending in Ridge (see predict_net_rating_blended).
# A ratio of 1.0 means "as far past the max as the whole observed range is
# wide" gets full blend weight.
MAX_BLEND_WEIGHT = 0.5

# Public runtime: Stage 1 (Ridge + GBM trees) evaluated in numpy from
# exported model files — no sklearn/pandas/scipy.
@dataclass
class SeasonDistribution:
    point_estimate_net_rating: float
    point_estimate_wins: float
    model_residual_std: float
    mean: float
    median: float
    std: float
    p10: float
    p25: float
    p75: float
    p90: float
    min: float
    max: float
    n_sims: int
    gbm_net_rating: float
    ridge_net_rating: Optional[float]
    ood_score: float
    blend_weight: float
    extrapolation_warning: bool

    def as_dict(self) -> dict:
        return asdict(self)


class WinPredictor:
    def __init__(self, db_path=db.DB_PATH):
        from model.runtime_predict import Stage1Runtime

        self.db_path = db_path
        self.runtime = Stage1Runtime()
        self.ridge_medians = {"global": self.runtime.medians_global}
        self.feature_names = self.runtime.feature_names
        self.residual_std = self.runtime.residual_std
        self.residual_std_by_band = self.runtime.residual_std_by_band
        self.training_value_ranges = self.runtime.training_value_ranges
        self.disattenuation_by_band = self.runtime.disattenuation_by_band
        self.disattenuation_slope = self.runtime.disattenuation_slope
        self.disattenuation_intercept = self.runtime.disattenuation_intercept
        self.primary_by_band = self.runtime.primary_by_band

        self.stage2 = Stage2Calibration.load()
        self.game_logistic = GameLogisticCalibration.load()
        self._opponent_pool = self._load_opponent_pool()

    def _load_opponent_pool(self) -> np.ndarray:
        conn = db.get_connection(self.db_path)
        try:
            rows = conn.execute(
                "SELECT net_rating, mov FROM team_seasons WHERE net_rating IS NOT NULL OR mov IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()
        return np.array([nr if nr is not None else mov for nr, mov in rows], dtype=float)

    def _ood_score(self, features: dict) -> float:
        """
        How far this roster's aggregate value features fall outside anything
        ever observed in training, as a fraction of the observed range (0 =
        in-distribution, 1 = as far past the max as the whole training range
        is wide). GBM (a tree ensemble) cannot extrapolate its output past
        leaf values learned from the training range, so a high score here
        means its prediction is likely to understate this roster's strength.
        """
        max_excess = 0.0
        for col, (lo, hi) in self.training_value_ranges.items():
            val = features.get(col)
            if val is None:
                continue
            span = hi - lo
            if span <= 0:
                continue
            if val > hi:
                max_excess = max(max_excess, (val - hi) / span)
            elif val < lo:
                max_excess = max(max_excess, (lo - val) / span)
        return max_excess

    def _disattenuation_for_bands(self, bands: Optional[List] = None) -> tuple:
        """
        Per-band (slope, intercept) for undoing GBM's mean-compression, per
        MODELING_DECISIONS.md "Disattenuation correction". A pooled slope
        was tried first and rejected: it was dominated by Band A0/C's severe
        compression and over-corrected A1/A2/B, which individually are
        barely compressed at all (slope~1.0). Uses the band with the most
        compression (smallest slope) present in the roster — same
        "least-reliable-band-wins" logic as _residual_std_for_bands.
        """
        if not bands or not self.disattenuation_by_band:
            return self.disattenuation_slope, self.disattenuation_intercept
        candidates = [
            self.disattenuation_by_band.get(b.value if hasattr(b, "value") else b)
            for b in bands
        ]
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            return self.disattenuation_slope, self.disattenuation_intercept
        worst = min(candidates, key=lambda c: c["slope"])
        return worst["slope"], worst["intercept"]

    def _primary_model_for_bands(self, bands: Optional[List] = None) -> str:
        """
        Pick ridge vs gbm from the coarsest (least-reliable) band present —
        same worst-band-wins logic as residual_std. Defaults to ridge (the
        Lasso-selected linear model) when bands are unknown.
        """
        if not bands:
            return "ridge"
        keys = []
        for b in bands:
            key = b.value if hasattr(b, "value") else b
            keys.append(self.primary_by_band.get(key, "ridge"))
        # Prefer gbm if any band requires it (A0).
        return "gbm" if "gbm" in keys else "ridge"

    def _raw_predictions(self, features: dict) -> tuple:
        """Return (gbm_pred_raw, ridge_pred_raw) via the numpy runtime."""
        return (
            self.runtime.gbm_predict_raw(features),
            self.runtime.ridge_predict_raw(features),
        )

    def predict_net_rating_blended(self, features: dict, bands: Optional[List] = None) -> dict:
        """
        Point estimate from the band-appropriate primary model (Ridge on
        Lasso-selected features for A1–C; GBM for A0), with OOD blending:
        when GBM is primary and roster value is out-of-distribution, blend
        toward Ridge so tree flatlining isn't silently trusted alone.

        """
        gbm_pred_raw, ridge_pred_raw = self._raw_predictions(features)

        slope, intercept = self._disattenuation_for_bands(bands)
        gbm_pred = (gbm_pred_raw - intercept) / slope
        # Ridge LOEO residuals are already near-unbiased on A1/A2/B; apply
        # the same per-band correction table (often slope≈1 / skipped).
        ridge_pred = (ridge_pred_raw - intercept) / slope

        primary = self._primary_model_for_bands(bands)
        primary_pred = ridge_pred if primary == "ridge" else gbm_pred
        secondary_pred = gbm_pred if primary == "ridge" else ridge_pred

        ood_score = self._ood_score(features)
        if primary == "gbm":
            blend_weight = min(MAX_BLEND_WEIGHT, ood_score)
            net_rating = (1 - blend_weight) * primary_pred + blend_weight * secondary_pred
        else:
            blend_weight = 0.0
            net_rating = primary_pred

        return {
            "net_rating": net_rating,
            "gbm_net_rating": gbm_pred,
            "ridge_net_rating": ridge_pred,
            "primary_model": primary,
            "ood_score": ood_score,
            "blend_weight": blend_weight,
            "extrapolation_warning": ood_score > 0,
        }

    def features_for_player_ids(self, player_seasons: List[tuple]) -> dict:
        """player_seasons: list of 5 (player_id, season, team_id_or_None)."""
        atomic_list = []
        for player_id, season, team_id in player_seasons:
            a = get_atomic_features(player_id, season, team_id, db_path=self.db_path)
            if a is None:
                raise ValueError(f"No player_seasons row for player_id={player_id}, season={season}")
            atomic_list.append(a)
        rf = compose_roster(atomic_list, db_path=self.db_path)
        return rf.features, rf.band_mix

    def _residual_std_for_bands(self, bands: Optional[List] = None) -> float:
        """Use the least-reliable band present in the roster (band_weight logic
        in features/compose.py already treats a roster's confidence this way) —
        a single pooled std would understate uncertainty for a roster leaning on
        the noisiest band (Band C's lineup labels) and overstate it for an
        all-A2/B roster. Falls back to the pooled std if bands are unknown or
        a band is missing from the LOEO table."""
        if not bands or not self.residual_std_by_band:
            return self.residual_std
        stds = [self.residual_std_by_band.get(b.value if hasattr(b, "value") else b) for b in bands]
        stds = [s for s in stds if s is not None]
        return max(stds) if stds else self.residual_std

    def simulate(
        self,
        features: dict,
        n_sims: int = DEFAULT_N_SIMS,
        n_games: int = N_GAMES,
        seed: Optional[int] = None,
        bands: Optional[List] = None,
    ) -> SeasonDistribution:
        pred = self.predict_net_rating_blended(features, bands=bands)
        # Game answer = how tough are these five. No bench/depth discount —
        # a murderer's row of peaks should land near 81–82, not get pulled
        # toward a real franchise's season record.
        #
        # Fit adjustment: Stage 1 under-penalizes usage/spacing/defense
        # pathologies that real training rosters almost never exhibit
        # (see model/fit_adjustment.py, MODELING_DECISIONS.md).
        fit_delta = fit_net_rating_adjustment(features)
        point_net_rating = pred["net_rating"] + fit_delta
        point_wins = self.stage2.expected_wins(point_net_rating, games=n_games)

        # Widen model uncertainty under extrapolation: the further out of
        # distribution the roster is, the less either GBM or Ridge should be
        # trusted, so the "true" net rating drawn per simulated season should
        # spread wider than a normal in-distribution roster's residual std.
        residual_std = self._residual_std_for_bands(bands) * (1.0 + pred["ood_score"])

        rng = np.random.default_rng(seed)

        # (a) model uncertainty: redraw the roster's "true" net rating once per season.
        true_net_ratings = rng.normal(point_net_rating, residual_std, size=n_sims)

        # (b) game-to-game variance: bootstrap real opponents, flip weighted coins.
        opponent_draws = rng.choice(self._opponent_pool, size=(n_sims, n_games), replace=True)
        our_ratings = true_net_ratings[:, None]  # broadcast over games
        win_probs = self.game_logistic.win_prob(our_ratings, opponent_draws)
        outcomes = rng.random((n_sims, n_games)) < win_probs
        season_wins = outcomes.sum(axis=1)

        percentiles = np.percentile(season_wins, [10, 25, 50, 75, 90])
        return SeasonDistribution(
            point_estimate_net_rating=point_net_rating,
            point_estimate_wins=point_wins,
            model_residual_std=residual_std,
            mean=float(season_wins.mean()),
            median=float(percentiles[2]),
            std=float(season_wins.std()),
            p10=float(percentiles[0]),
            p25=float(percentiles[1]),
            p75=float(percentiles[3]),
            p90=float(percentiles[4]),
            min=float(season_wins.min()),
            max=float(season_wins.max()),
            n_sims=n_sims,
            gbm_net_rating=pred["gbm_net_rating"],
            ridge_net_rating=pred["ridge_net_rating"],
            ood_score=pred["ood_score"],
            blend_weight=pred["blend_weight"],
            extrapolation_warning=pred["extrapolation_warning"],
        )

    def simulate_by_player_ids(
        self,
        player_seasons: List[tuple],
        n_sims: int = DEFAULT_N_SIMS,
        n_games: int = N_GAMES,
        seed: Optional[int] = None,
    ) -> SeasonDistribution:
        features, bands = self.features_for_player_ids(player_seasons)
        return self.simulate(features, n_sims=n_sims, n_games=n_games, seed=seed, bands=bands)
