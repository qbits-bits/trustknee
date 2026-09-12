"""Feature-bridge adapter for the canonical XGBoost artifact.

Loads the trained Phase-3 XGBoost artifact and exposes it behind Nihal's
common InferenceEngine contract. Because the engineered feature set is
temporal (lags, rolling statistics, EMA, deltas) and uses per-subject causal
calibration, inference is performed at the trial/subject level rather than on
isolated single windows.

This module does not modify any existing inference code; it adds a new adapter
that reloads the saved artifact with prediction parity and rejects missing or
mis-ordered features.
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from src.inference.base import InferenceEngine, WindowPrediction
from src.models.xgboost_pipeline import build_features

logger = logging.getLogger("trustknee.inference")


class XGBoostArtifactAdapter(InferenceEngine):
    """Adapter that runs the saved XGBoost artifact behind the inference contract."""

    def __init__(
        self,
        artifact_path: str,
        num_classes: int = 2,
        uncertainty_threshold: float = 0.60,
    ) -> None:
        super().__init__(
            model_name="xgboost_canonical",
            num_classes=num_classes,
            uncertainty_threshold=uncertainty_threshold,
        )
        self.artifact = joblib.load(artifact_path)
        self.model = self.artifact["model"]
        self.feature_order = list(self.artifact["feature_order"])
        self.clip = self.artifact["clip"]
        self.threshold = float(self.artifact["threshold"])

    # ------------------------------------------------------------- contract
    def predict_window(
        self,
        imu_window: np.ndarray,
        emg_window: np.ndarray,
        window_index: int = 0,
        start_time_s: float = 0.0,
        end_time_s: float = 0.2,
        subject_id: int | None = None,
        trial_num: int | None = None,
    ) -> WindowPrediction:
        raise NotImplementedError(
            "The canonical XGBoost model uses temporal engineered features "
            "(lags, rolling stats, EMA) and per-subject causal calibration, "
            "which require trial-level context. Use predict_trial_features()."
        )

    def predict_batch(
        self,
        imu_windows: np.ndarray,
        emg_windows: np.ndarray,
        window_indices: list[int] | None = None,
        start_times_s: list[float] | None = None,
        end_times_s: list[float] | None = None,
        subject_id: int | None = None,
        trial_num: int | None = None,
    ) -> list[WindowPrediction]:
        raise NotImplementedError(
            "Raw-array batch prediction requires the raw-to-feature bridge. "
            "Use predict_trial_features() with a DataFrame of sensor columns."
        )

    # ------------------------------------------------------ feature bridge
    def predict_trial_features(self, raw_df: pd.DataFrame) -> list[WindowPrediction]:
        """Build engineered features from raw sensor columns and predict.

        Args:
            raw_df: DataFrame of raw sensor columns for one or more subjects,
                sorted by [subject_id, trial_num, window_index]. Must include
                the full window history for each subject so causal calibration
                and temporal features match training.

        Returns:
            A list of WindowPrediction objects, one per input row.
        """
        df, _ = build_features(raw_df.copy())

        # Reject missing features; enforce the artifact's feature order.
        missing = [c for c in self.feature_order if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required features: {missing[:10]}")

        X = df[self.feature_order].copy()

        # Apply training-derived clipping bounds.
        for c in self.feature_order:
            lo, hi = self.clip[c]
            X[c] = X[c].clip(lo, hi)

        proba = self.model.predict_proba(X)
        preds = (proba[:, 1] >= self.threshold).astype(int)

        predictions = []
        for i in range(len(df)):
            label_id = int(preds[i])
            confidence = float(proba[i][label_id])
            is_uncertain = confidence < self.uncertainty_threshold
            execution = "Uncertain" if is_uncertain else ("Correct" if label_id == 0 else "Wrong")
            row = df.iloc[i]
            predictions.append(
                WindowPrediction(
                    window_index=int(row.get("window_index", i)),
                    start_time_s=float(row.get("start_time_s", i * 0.2)),
                    end_time_s=float(row.get("end_time_s", (i + 1) * 0.2)),
                    probabilities=proba[i],
                    predicted_label_id=label_id,
                    confidence_score=confidence,
                    is_flagged_uncertain=is_uncertain,
                    execution=execution,
                    exercise=None,
                    model_name=self.model_name,
                    subject_id=(int(row["subject_id"]) if "subject_id" in row else None),
                    trial_num=int(row["trial_num"]) if "trial_num" in row else None,
                )
            )
        return predictions
