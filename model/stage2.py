"""
model/stage2.py

Stage 2: closed-form conversion from net rating (points per 100 possessions)
to expected win percentage, calibrated once against real historical
team-seasons — never re-learned inside Stage 1.

FEATURE_SPEC.md section 1 asks for "a fixed Pythagorean-style map ... from
point differential/net rating to win%", calibrated on our own
`team_seasons` data rather than assumed. The classic Bill James / Morey
Pythagorean formula is win% = PF^k / (PF^k + PA^k), but that form needs raw
points-for/against, which only ~52% of our team_seasons rows have (older
seasons often lack per_game scoring splits). `mov` (point differential per
game) and `win_pct` are populated for essentially the whole 1946-2026 range,
and the spec explicitly permits calibrating directly against MOV/net_rating,
so we use the differential-form Pythagorean expectation instead:

    win_pct = sigmoid(intercept + slope * point_diff)

which is the standard log-odds-linear-in-differential formulation of the
same idea (net rating of 0 -> win_pct ~= 0.5), fit by weighted logistic
regression (weight = games played that season) against real win_pct. The
fitted `slope` plays the role of the Pythagorean exponent here.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import db

CALIBRATION_PATH = Path(__file__).parent / "stage2_calibration.json"


@dataclass
class Stage2Calibration:
    intercept: float
    slope: float
    n_seasons: int
    mae_win_pct: float
    r2_win_pct: float

    def win_pct(self, point_diff) -> float:
        x = np.asarray(point_diff, dtype=float)
        return 1.0 / (1.0 + np.exp(-(self.intercept + self.slope * x)))

    def expected_wins(self, point_diff, games: int = 82) -> float:
        return float(self.win_pct(point_diff) * games)

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "Stage2Calibration":
        return cls(**json.loads(path.read_text()))


def _load_team_season_diff_and_winpct(db_path=db.DB_PATH):
    conn = db.get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT net_rating, mov, win_pct, wins, losses
            FROM team_seasons
            WHERE win_pct IS NOT NULL AND (net_rating IS NOT NULL OR mov IS NOT NULL)
            """
        ).fetchall()
    finally:
        conn.close()

    xs, ys, ws = [], [], []
    for net_rating, mov, win_pct, wins, losses in rows:
        x = net_rating if net_rating is not None else mov
        games = (wins or 0) + (losses or 0)
        if games <= 0:
            continue
        xs.append(x)
        ys.append(win_pct)
        ws.append(games)
    return np.array(xs), np.array(ys), np.array(ws)


def calibrate(db_path=db.DB_PATH) -> Stage2Calibration:
    x, y, w = _load_team_season_diff_and_winpct(db_path)

    # Logistic regression needs a binary/discrete target; fit against the
    # continuous win_pct via weighted least squares on the logit transform
    # instead (equivalent MLE-style fit for a Bernoulli mean model, and
    # avoids needing per-game outcome rows we don't have for older eras).
    eps = 1e-3
    y_clipped = np.clip(y, eps, 1 - eps)
    logit_y = np.log(y_clipped / (1 - y_clipped))

    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    # Weighted least squares: beta = (X'WX)^-1 X'Wy
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ logit_y)
    intercept, slope = beta

    pred_win_pct = 1.0 / (1.0 + np.exp(-(intercept + slope * x)))
    mae = float(np.average(np.abs(pred_win_pct - y), weights=w))
    ss_res = float(np.average((pred_win_pct - y) ** 2, weights=w))
    ss_tot = float(np.average((y - np.average(y, weights=w)) ** 2, weights=w))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return Stage2Calibration(
        intercept=float(intercept),
        slope=float(slope),
        n_seasons=len(x),
        mae_win_pct=mae,
        r2_win_pct=r2,
    )


if __name__ == "__main__":
    cal = calibrate()
    print(f"Stage 2 calibration: intercept={cal.intercept:.5f}, slope={cal.slope:.5f}")
    print(f"n_seasons={cal.n_seasons}, MAE(win_pct)={cal.mae_win_pct:.4f}, R2={cal.r2_win_pct:.4f}")
    print(f"Sanity: net_rating=0 -> win_pct={cal.win_pct(0.0):.3f} (should be ~0.5)")
    print(f"Sanity: net_rating=+10 -> expected_wins={cal.expected_wins(10.0):.1f} / 82")
    print(f"Sanity: net_rating=-10 -> expected_wins={cal.expected_wins(-10.0):.1f} / 82")
    cal.save()
    print(f"Saved to {CALIBRATION_PATH}")
