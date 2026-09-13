"""Portable Transformer checkpoint bundles for calibrated inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.models.calibration import TemperatureScaler
from src.models.normalization import SequenceNormalizer
from src.models.transformer import TransformerEncoderClassifier


@dataclass
class LoadedTransformerArtifact:
    """Validated objects and metadata restored from one checkpoint."""

    model: TransformerEncoderClassifier
    normalizer: SequenceNormalizer
    temperature_scaler: TemperatureScaler
    uncertainty_threshold: float
    label_mapping: dict[int, str]
    sensor_mask: tuple[int, ...] | None
    metadata: dict[str, Any]

    def build_inference_engine(
        self,
        device: str = "cpu",
        *,
        explain: bool = False,
        explanation_steps: int = 32,
    ):
        from src.inference.transformer_engine import TransformerInferenceEngine

        self.model.to(device)
        return TransformerInferenceEngine(
            model=self.model,
            normalizer=self.normalizer,
            num_classes=self.model.model_config["num_classes"],
            temperature=self.temperature_scaler.temperature,
            uncertainty_threshold=self.uncertainty_threshold,
            device=device,
            active_sensors=self.sensor_mask,
            explain=explain,
            explanation_steps=explanation_steps,
        )


def save_transformer_artifact(
    path: str | Path,
    model: TransformerEncoderClassifier,
    normalizer: SequenceNormalizer,
    temperature_scaler: TemperatureScaler,
    uncertainty_threshold: float,
    label_mapping: dict[int, str],
    *,
    sensor_mask: tuple[int, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save weights, preprocessing, calibration, and contract metadata together."""
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not 0 <= uncertainty_threshold <= 1:
        raise ValueError("uncertainty_threshold must be in [0, 1]")
    if sensor_mask is not None and (
        not sensor_mask or any(sensor < 1 or sensor > 8 for sensor in sensor_mask)
    ):
        raise ValueError("sensor_mask must contain one-based sensor IDs from 1 through 8")
    payload = {
        "format_version": 1,
        "model_config": dict(model.model_config),
        "model_state_dict": model.state_dict(),
        "normalizer_mean": torch.from_numpy(np.asarray(normalizer.mean, dtype=np.float32)),
        "normalizer_scale": torch.from_numpy(np.asarray(normalizer.scale, dtype=np.float32)),
        "temperature": float(temperature_scaler.temperature),
        "uncertainty_threshold": float(uncertainty_threshold),
        "label_mapping": {int(key): str(value) for key, value in label_mapping.items()},
        "sensor_mask": list(sensor_mask) if sensor_mask is not None else None,
        "metadata": dict(metadata or {}),
    }
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_transformer_artifact(
    path: str | Path,
    *,
    device: str = "cpu",
) -> LoadedTransformerArtifact:
    """Load and validate a versioned Transformer artifact."""
    import torch

    source = Path(path)
    payload = torch.load(source, map_location=device, weights_only=True)
    required = {
        "format_version",
        "model_config",
        "model_state_dict",
        "normalizer_mean",
        "normalizer_scale",
        "temperature",
        "uncertainty_threshold",
        "label_mapping",
        "sensor_mask",
        "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Transformer artifact is missing fields: {sorted(missing)}")
    if payload["format_version"] != 1:
        raise ValueError(f"Unsupported Transformer artifact version: {payload['format_version']}")
    model = TransformerEncoderClassifier(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    mean = payload["normalizer_mean"].cpu().numpy().astype(np.float32)
    scale = payload["normalizer_scale"].cpu().numpy().astype(np.float32)
    if mean.shape != scale.shape or mean.shape != (1, 1, model.model_config["input_dim"]):
        raise ValueError("Transformer artifact contains incompatible normalization arrays")
    sensor_values = payload["sensor_mask"]
    sensor_mask = (
        tuple(int(value) for value in sensor_values) if sensor_values is not None else None
    )
    return LoadedTransformerArtifact(
        model=model,
        normalizer=SequenceNormalizer(mean=mean, scale=scale),
        temperature_scaler=TemperatureScaler(float(payload["temperature"])),
        uncertainty_threshold=float(payload["uncertainty_threshold"]),
        label_mapping={int(key): str(value) for key, value in payload["label_mapping"].items()},
        sensor_mask=sensor_mask,
        metadata=dict(payload["metadata"]),
    )
