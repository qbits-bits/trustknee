"""XGBoost tabular inference engine implementing the common InferenceEngine contract."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.features.extract import extract_window_features
from src.inference.base import InferenceEngine, WindowPrediction

logger = logging.getLogger("trustknee.inference")


class XGBoostInferenceEngine(InferenceEngine):
    """Inference engine for tabular XGBoost classifiers with feature extraction."""

    def __init__(
        self,
        model: Any,
        feature_names: list[str] | None = None,
        num_classes: int = 2,
        uncertainty_threshold: float = 0.60,
    ) -> None:
        super().__init__(
            model_name="xgboost",
            num_classes=num_classes,
            uncertainty_threshold=uncertainty_threshold,
        )
        self.model = model
        self.feature_names = feature_names

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
        # Extract the 432 domain features from the window
        features_dict = extract_window_features(imu_window, emg_window)
        feature_df = pd.DataFrame([features_dict]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if self.feature_names is not None:
            # Enforce ordered subset columns
            missing = set(self.feature_names) - set(feature_df.columns)
            if missing:
                raise ValueError(f"Extracted features missing expected columns: {missing}")
            feature_df = feature_df[self.feature_names]

        probs = np.asarray(self.model.predict_proba(feature_df), dtype=np.float32)[0]

        predicted_label_id = int(np.argmax(probs))
        confidence = float(probs[predicted_label_id])
        is_uncertain = confidence < self.uncertainty_threshold

        if self.num_classes == 2:
            execution = (
                "Uncertain" if is_uncertain else ("Correct" if predicted_label_id == 0 else "Wrong")
            )
            exercise = None
        else:
            info = config.LABELS.get(predicted_label_id)
            execution = "Uncertain" if is_uncertain else (info.execution if info else "Unknown")
            exercise = info.exercise if info else None

        return WindowPrediction(
            window_index=window_index,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            probabilities=probs,
            predicted_label_id=predicted_label_id,
            confidence_score=confidence,
            is_flagged_uncertain=is_uncertain,
            execution=execution,
            exercise=exercise,
            model_name=self.model_name,
            subject_id=subject_id,
            trial_num=trial_num,
        )
