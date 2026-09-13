"""Common inference contract for TrustKnee models.

Defines unified data models and abstract interfaces for window-level and
trial-level inference across both Transformer sequence models and tabular XGBoost.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from src import config

logger = logging.getLogger("trustknee.inference")


@dataclass(frozen=True)
class WindowPrediction:
    """Standardized prediction for a single 200 ms sensor window."""

    window_index: int
    start_time_s: float
    end_time_s: float
    probabilities: np.ndarray  # Shape: (num_classes,)
    predicted_label_id: int
    confidence_score: float  # Posterior probability of the predicted class
    is_flagged_uncertain: bool  # True when confidence_score < uncertainty_threshold
    execution: str  # "Correct" | "Wrong" | "Uncertain"
    exercise: str | None
    model_name: str
    subject_id: int | None = None
    trial_num: int | None = None
    feature_attributions: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_index": self.window_index,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "probabilities": self.probabilities.tolist(),
            "predicted_label_id": self.predicted_label_id,
            "confidence_score": self.confidence_score,
            "is_flagged_uncertain": self.is_flagged_uncertain,
            "execution": self.execution,
            "exercise": self.exercise,
            "model_name": self.model_name,
            "subject_id": self.subject_id,
            "trial_num": self.trial_num,
            "feature_attributions": self.feature_attributions,
        }


@dataclass(frozen=True)
class TrialPrediction:
    """Aggregated prediction for an entire exercise trial."""

    subject_id: int | None
    label_id: int | None
    trial_num: int | None
    n_windows: int
    mean_probabilities: np.ndarray
    predicted_label_id: int
    confidence_score: float
    is_flagged_uncertain: bool
    execution: str
    exercise: str | None
    model_name: str
    window_predictions: list[WindowPrediction]
    top_contributing_sensors: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "label_id": self.label_id,
            "trial_num": self.trial_num,
            "n_windows": self.n_windows,
            "mean_probabilities": self.mean_probabilities.tolist(),
            "predicted_label_id": self.predicted_label_id,
            "confidence_score": self.confidence_score,
            "is_flagged_uncertain": self.is_flagged_uncertain,
            "execution": self.execution,
            "exercise": self.exercise,
            "model_name": self.model_name,
            "window_count": len(self.window_predictions),
            "top_contributing_sensors": self.top_contributing_sensors,
        }


class InferenceEngine(ABC):
    """Abstract base class for all TrustKnee inference models."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        uncertainty_threshold: float = 0.60,
    ) -> None:
        self.model_name = str(model_name)
        self.num_classes = int(num_classes)
        self.uncertainty_threshold = float(uncertainty_threshold)

    @abstractmethod
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
        """Run inference on a single synchronized (imu_window, emg_window) pair."""

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
        """Run inference on a sequence of windows, returning a list of WindowPredictions."""
        n_windows = len(imu_windows)
        predictions = []
        for i in range(n_windows):
            win_idx = window_indices[i] if window_indices is not None else i
            start_t = start_times_s[i] if start_times_s is not None else i * 0.1
            end_t = end_times_s[i] if end_times_s is not None else start_t + 0.2

            pred = self.predict_window(
                imu_window=imu_windows[i],
                emg_window=emg_windows[i],
                window_index=win_idx,
                start_time_s=start_t,
                end_time_s=end_t,
                subject_id=subject_id,
                trial_num=trial_num,
            )
            predictions.append(pred)
        return predictions

    def aggregate_trial(
        self,
        window_predictions: list[WindowPrediction],
        method: str = "mean_probability",
    ) -> TrialPrediction:
        """Aggregate window-level predictions into a single trial-level decision.

        Supported aggregation methods:
        - "mean_probability": Average posterior class probabilities across windows.
        - "majority_vote": Plurality of window predicted labels.
        - "confidence_weighted": Weighted probability average by confidence.
        """
        if not window_predictions:
            raise ValueError("Cannot aggregate an empty list of window predictions")

        first = window_predictions[0]
        subject_id = first.subject_id
        trial_num = first.trial_num
        model_name = self.model_name
        n_windows = len(window_predictions)

        probs_matrix = np.stack([w.probabilities for w in window_predictions], axis=0)

        if method == "mean_probability":
            agg_probs = np.mean(probs_matrix, axis=0)
        elif method == "confidence_weighted":
            confidences = np.array([w.confidence_score for w in window_predictions])
            conf_sum = np.sum(confidences)
            weights = confidences / conf_sum if conf_sum > 0 else np.ones(n_windows) / n_windows
            agg_probs = np.average(probs_matrix, axis=0, weights=weights)
        elif method == "majority_vote":
            labels = [w.predicted_label_id for w in window_predictions]
            counts = np.bincount(labels, minlength=self.num_classes)
            agg_probs = counts / np.sum(counts)
        else:
            raise ValueError(f"Unknown aggregation method '{method}'")

        predicted_label_id = int(np.argmax(agg_probs))
        confidence = float(agg_probs[predicted_label_id])
        is_uncertain = confidence < self.uncertainty_threshold

        # Map to human-readable taxonomy
        if self.num_classes == 2:
            if is_uncertain:
                execution = "Uncertain"
            else:
                execution = "Correct" if predicted_label_id == 0 else "Wrong"
            exercise = None
        else:
            info = config.LABELS.get(predicted_label_id)
            execution = "Uncertain" if is_uncertain else (info.execution if info else "Unknown")
            exercise = info.exercise if info else None

        sensor_totals: dict[str, float] = {}
        for prediction in window_predictions:
            for sensor, attribution in (prediction.feature_attributions or {}).items():
                sensor_totals[sensor] = sensor_totals.get(sensor, 0.0) + abs(float(attribution))
        top_sensors = [
            sensor
            for sensor, _ in sorted(sensor_totals.items(), key=lambda item: item[1], reverse=True)[
                :3
            ]
        ]

        return TrialPrediction(
            subject_id=subject_id,
            label_id=predicted_label_id,
            trial_num=trial_num,
            n_windows=n_windows,
            mean_probabilities=agg_probs,
            predicted_label_id=predicted_label_id,
            confidence_score=confidence,
            is_flagged_uncertain=is_uncertain,
            execution=execution,
            exercise=exercise,
            model_name=model_name,
            window_predictions=window_predictions,
            top_contributing_sensors=top_sensors or None,
        )
