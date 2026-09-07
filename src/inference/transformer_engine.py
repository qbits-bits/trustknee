"""Transformer sequence inference engine implementing the common InferenceEngine contract."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src import config
from src.inference.base import InferenceEngine, WindowPrediction
from src.models.model_data import make_transformer_sequences
from src.models.normalization import SequenceNormalizer

logger = logging.getLogger("trustknee.inference")


class TransformerInferenceEngine(InferenceEngine):
    """Real-time inference wrapper for PyTorch TransformerEncoderClassifier."""

    def __init__(
        self,
        model: Any,
        normalizer: SequenceNormalizer,
        num_classes: int = 2,
        temperature: float = 1.0,
        uncertainty_threshold: float = 0.60,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            model_name="transformer",
            num_classes=num_classes,
            uncertainty_threshold=uncertainty_threshold,
        )
        self.model = model
        self.normalizer = normalizer
        self.temperature = max(1e-4, float(temperature))
        self.device = device
        self.model.eval()

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
        import torch

        # Format into (1, 8, 6, 30) and (1, 8, 252)
        imu_batch = imu_window[np.newaxis, ...] if imu_window.ndim == 3 else imu_window
        emg_batch = emg_window[np.newaxis, ...] if emg_window.ndim == 2 else emg_window

        seq = make_transformer_sequences(imu_batch, emg_batch)
        norm_seq = self.normalizer.transform(seq)

        tensor_x = torch.from_numpy(norm_seq).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_x)
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()[0]

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
