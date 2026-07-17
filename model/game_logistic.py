"""
model/game_logistic.py

Calibrates the single-game win-probability logistic used by the Monte Carlo
season simulator: P(win a given game) = sigmoid(k * (our_net_rating -
opponent_net_rating)).

Runtime load/predict is numpy-only. Fitting (`calibrate`) still needs scipy.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import db

CALIBRATION_PATH = Path(__file__).parent / "game_logistic_calibration.json"


@dataclass
class GameLogisticCalibration:
    k: float
    mae_win_pct: float
    n_team_seasons: int

    def win_prob(self, our_net_rating, opponent_net_rating):
        diff = np.asarray(our_net_rating, dtype=float) - np.asarray(opponent_net_rating, dtype=float)
        return 1.0 / (1.0 + np.exp(-self.k * diff))

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "GameLogisticCalibration":
        return cls(**json.loads(path.read_text()))


def _load_season_pools(db_path=db.DB_PATH):
    """season -> array of team net_ratings (mov fallback) that season."""
    conn = db.get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT season, team_id, net_rating, mov, win_pct FROM team_seasons WHERE win_pct IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    by_season = {}
    for season, team_id, net_rating, mov, win_pct in rows:
        x = net_rating if net_rating is not None else mov
        if x is None:
            continue
        by_season.setdefault(season, []).append((team_id, x, win_pct))
    return by_season


def _implied_win_pct_for_k(k: float, by_season: dict):
    preds, actuals = [], []
    for season, teams in by_season.items():
        if len(teams) < 3:
            continue
        xs = np.array([t[1] for t in teams])
        for i, (team_id, x, win_pct) in enumerate(teams):
            opponents = np.delete(xs, i)
            p = 1.0 / (1.0 + np.exp(-k * (x - opponents)))
            preds.append(p.mean())
            actuals.append(win_pct)
    return np.array(preds), np.array(actuals)


def calibrate(db_path=db.DB_PATH) -> GameLogisticCalibration:
    from scipy.optimize import minimize_scalar

    by_season = _load_season_pools(db_path)

    def loss(k):
        preds, actuals = _implied_win_pct_for_k(k, by_season)
        return float(np.mean(np.abs(preds - actuals)))

    result = minimize_scalar(loss, bounds=(0.01, 1.0), method="bounded")
    k = float(result.x)
    preds, actuals = _implied_win_pct_for_k(k, by_season)
    mae = float(np.mean(np.abs(preds - actuals)))
    return GameLogisticCalibration(k=k, mae_win_pct=mae, n_team_seasons=len(actuals))


if __name__ == "__main__":
    cal = calibrate()
    print(f"Calibrated single-game logistic k={cal.k:.5f}, MAE(win_pct)={cal.mae_win_pct:.4f}, n={cal.n_team_seasons}")
    print(f"Sanity: net_rating diff=+5 -> P(win)={cal.win_prob(5.0, 0.0):.3f}")
    print(f"Sanity: net_rating diff=-5 -> P(win)={cal.win_prob(-5.0, 0.0):.3f}")
    cal.save()
    print(f"Saved to {CALIBRATION_PATH}")
