"""Mock inference engine for fast unit testing and CI without heavy model dependencies."""

from __future__ import annotations

import numpy as np

from src import config
from src.inference.base import InferenceEngine, WindowPrediction


class MockInferenceEngine(InferenceEngine):
    """Deterministic mock inference engine for pipeline validation."""

    def __init__(
        self,
        predicted_class: int = 0,
        confidence: float = 0.85,
        num_classes: int = 2,
        uncertainty_threshold: float = 0.60,
    ) -> None:
        super().__init__(
            model_name="mock_model",
            num_classes=num_classes,
            uncertainty_threshold=uncertainty_threshold,
        )
        self.predicted_class = int(predicted_class)
        self.confidence = float(confidence)

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
        probs = np.full(
            self.num_classes,
            (1.0 - self.confidence) / max(1, self.num_classes - 1),
            dtype=np.float32,
        )
        probs[self.predicted_class] = self.confidence

        is_uncertain = self.confidence < self.uncertainty_threshold
        if self.num_classes == 2:
            execution = (
                "Uncertain"
                if is_uncertain
                else ("Correct" if self.predicted_class == 0 else "Wrong")
            )
            exercise = None
        else:
            info = config.LABELS.get(self.predicted_class)
            execution = "Uncertain" if is_uncertain else (info.execution if info else "Unknown")
            exercise = info.exercise if info else None

        return WindowPrediction(
            window_index=window_index,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            probabilities=probs,
            predicted_label_id=self.predicted_class,
            confidence_score=self.confidence,
            is_flagged_uncertain=is_uncertain,
            execution=execution,
            exercise=exercise,
            model_name=self.model_name,
            subject_id=subject_id,
            trial_num=trial_num,
        )
