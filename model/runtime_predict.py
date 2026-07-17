"""
model/runtime_predict.py

Numpy-only Stage 1 inference for slim deploys (Vercel). Reproduces the full
trained models with no sklearn/pandas/scipy:

  - Ridge: exported coefficients (stage1_runtime.json)
  - GBM:   exported HistGradientBoosting trees (stage1_gbm_runtime.npz),
           evaluated as plain numeric decision trees

Predictions match the sklearn models to within float tolerance, so the web
runtime keeps GBM-for-A0 + OOD blending + Ridge behavior without the heavy
dependency bundle. Regenerate the exports with model/export_runtime.py after
retraining.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

RUNTIME_PATH = Path(__file__).parent / "stage1_runtime.json"
GBM_NPZ_PATH = Path(__file__).parent / "stage1_gbm_runtime.npz"


class _GBMRuntime:
    """Numeric HistGradientBoosting inference in pure numpy."""

    def __init__(self, path: Path = GBM_NPZ_PATH):
        data = np.load(path)
        self.baseline = float(data["baseline"][0])
        self.feature_idx = data["feature_idx"]
        self.threshold = data["threshold"]
        self.missing_go_to_left = data["missing_go_to_left"]
        self.left = data["left"]
        self.right = data["right"]
        self.is_leaf = data["is_leaf"]
        self.value = data["value"]
        self.tree_offsets = data["tree_offsets"]

    def predict_raw(self, x: np.ndarray) -> float:
        """x: 1-D feature vector (NaN allowed for missing)."""
        total = self.baseline
        offsets = self.tree_offsets
        feat = self.feature_idx
        thr = self.threshold
        miss_left = self.missing_go_to_left
        left = self.left
        right = self.right
        is_leaf = self.is_leaf
        value = self.value

        for t in range(len(offsets) - 1):
            base = int(offsets[t])
            node = 0
            while not is_leaf[base + node]:
                gi = base + node
                fv = x[feat[gi]]
                if np.isnan(fv):
                    node = left[gi] if miss_left[gi] else right[gi]
                elif fv <= thr[gi]:
                    node = left[gi]
                else:
                    node = right[gi]
            total += value[base + node]
        return float(total)


class Stage1Runtime:
    """
    Full Stage 1 point estimate from exported models — numpy only.

    Mirrors WinPredictor.predict_net_rating_blended: Ridge for A1–C, GBM for
    A0, with OOD blending toward Ridge when GBM is primary.
    """

    MAX_BLEND_WEIGHT = 0.5

    def __init__(self, path: Path = RUNTIME_PATH, gbm_path: Path = GBM_NPZ_PATH):
        data = json.loads(path.read_text())
        self.feature_names: List[str] = list(data["feature_names"])
        self.coef = np.asarray(data["coef"], dtype=float)
        self.intercept = float(data["intercept"])
        self.medians_global: Dict[str, float] = {
            k: (0.0 if v is None else float(v))
            for k, v in data["medians"]["global"].items()
        }
        self.residual_std = float(data["residual_std"])
        self.residual_std_by_band = data.get("residual_std_by_band", {})
        self.training_value_ranges = data.get("training_value_ranges", {})
        self.disattenuation_by_band = data.get("disattenuation_by_band", {})
        self.disattenuation_slope = float(data.get("disattenuation_slope", 1.0))
        self.disattenuation_intercept = float(data.get("disattenuation_intercept", 0.0))
        self.primary_by_band = data.get("primary_by_band", {
            "A0": "gbm", "A1": "ridge", "A2": "ridge", "B": "ridge", "C": "ridge",
        })
        self.gbm = _GBMRuntime(gbm_path)

    def _vector(self, features: dict, fill_median: bool) -> np.ndarray:
        x = np.empty(len(self.feature_names), dtype=float)
        for i, name in enumerate(self.feature_names):
            v = features.get(name)
            if v is None:
                x[i] = self.medians_global.get(name, 0.0) if fill_median else np.nan
            else:
                x[i] = float(v)
        return x

    def ridge_predict_raw(self, features: dict) -> float:
        # Ridge uses median-imputed inputs (matches sklearn X.fillna(medians)).
        x = self._vector(features, fill_median=True)
        return float(np.dot(x, self.coef) + self.intercept)

    def gbm_predict_raw(self, features: dict) -> float:
        # GBM keeps NaN so the trees' missing-value routing applies.
        x = self._vector(features, fill_median=False)
        return self.gbm.predict_raw(x)
