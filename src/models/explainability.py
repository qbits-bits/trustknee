"""Integrated Gradients explanations for synchronized Transformer inputs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from src import config
from src.models.model_data import MODEL_INPUT_DIM, MODEL_SEQUENCE_LENGTH


def sensor_feature_mask(active_sensors: Iterable[int]) -> np.ndarray:
    """Return a 56-channel mask for explicit one-based sensor identifiers."""
    sensors = sorted({int(sensor) for sensor in active_sensors})
    if not sensors or any(sensor < 1 or sensor > config.N_SENSORS for sensor in sensors):
        raise ValueError("active_sensors must contain one-based IDs from 1 through 8")
    mask = np.zeros(MODEL_INPUT_DIM, dtype=bool)
    for sensor in sensors:
        start = (sensor - 1) * config.IMU_CHANNELS_PER_SENSOR
        mask[start : start + config.IMU_CHANNELS_PER_SENSOR] = True
        mask[48 + sensor - 1] = True
    return mask


def apply_sensor_mask(sequences: np.ndarray, active_sensors: Iterable[int]) -> np.ndarray:
    """Replace inactive normalized channels with the zero (training-mean) baseline."""
    values = np.asarray(sequences, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (MODEL_SEQUENCE_LENGTH, MODEL_INPUT_DIM):
        raise ValueError("sequences must have shape (batch, 30, 56)")
    result = values.copy()
    result[..., ~sensor_feature_mask(active_sensors)] = 0.0
    return result


def integrated_gradients(
    model: Any,
    inputs: np.ndarray,
    targets: np.ndarray | None = None,
    *,
    baseline: np.ndarray | None = None,
    steps: int = 32,
    feature_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Attribute target logits to normalized sequence inputs.

    The default all-zero baseline represents the training-channel mean after
    normalization. The returned array has exactly the same shape as ``inputs``.
    """
    import torch

    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (MODEL_SEQUENCE_LENGTH, MODEL_INPUT_DIM):
        raise ValueError("inputs must have shape (batch, 30, 56)")
    if not len(values):
        raise ValueError("inputs must not be empty")
    if steps <= 0:
        raise ValueError("steps must be positive")

    base = np.zeros_like(values) if baseline is None else np.asarray(baseline, dtype=np.float32)
    if base.shape == values.shape[1:]:
        base = np.broadcast_to(base, values.shape).copy()
    if base.shape != values.shape:
        raise ValueError("baseline must match inputs or one input sample")

    mask = None
    if feature_mask is not None:
        mask = np.asarray(feature_mask, dtype=bool)
        if mask.shape != (MODEL_INPUT_DIM,):
            raise ValueError("feature_mask must contain 56 boolean values")
        values = values.copy()
        base = base.copy()
        values[..., ~mask] = 0.0
        base[..., ~mask] = 0.0

    try:
        model_device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        model_device = torch.device("cpu")
    input_tensor = torch.as_tensor(values, device=model_device)
    baseline_tensor = torch.as_tensor(base, device=model_device)

    model.eval()
    if targets is None:
        with torch.no_grad():
            target_tensor = model(input_tensor).argmax(dim=1)
    else:
        target_values = np.asarray(targets, dtype=np.int64)
        if target_values.shape != (len(values),):
            raise ValueError("targets must contain one class index per input")
        target_tensor = torch.as_tensor(target_values, dtype=torch.long, device=model_device)

    gradient_sum = torch.zeros_like(input_tensor)
    difference = input_tensor - baseline_tensor
    for step in range(steps + 1):
        alpha = step / steps
        interpolated = (baseline_tensor + alpha * difference).detach().requires_grad_(True)
        logits = model(interpolated)
        if torch.any(target_tensor < 0) or torch.any(target_tensor >= logits.shape[1]):
            raise ValueError("targets fall outside the model class range")
        selected = logits.gather(1, target_tensor.unsqueeze(1)).sum()
        gradient = torch.autograd.grad(selected, interpolated)[0]
        weight = 0.5 if step in (0, steps) else 1.0
        gradient_sum += weight * gradient

    attributions = difference * gradient_sum / steps
    result = attributions.detach().cpu().numpy().astype(np.float32)
    if mask is not None:
        result[..., ~mask] = 0.0
    if not np.all(np.isfinite(result)):
        raise ValueError("Integrated Gradients produced non-finite attributions")
    return result


def summarize_sensor_attributions(attribution: np.ndarray) -> list[dict[str, float | int | str]]:
    """Aggregate one ``(30, 56)`` explanation into eight sensor summaries."""
    values = np.asarray(attribution, dtype=np.float64)
    if values.shape != (MODEL_SEQUENCE_LENGTH, MODEL_INPUT_DIM):
        raise ValueError("attribution must have shape (30, 56)")
    summaries = []
    for sensor in range(1, config.N_SENSORS + 1):
        imu_start = (sensor - 1) * config.IMU_CHANNELS_PER_SENSOR
        indices = list(range(imu_start, imu_start + config.IMU_CHANNELS_PER_SENSOR)) + [
            48 + sensor - 1
        ]
        channel_names = [*config.IMU_CHANNEL_NAMES, "emg_activity"]
        sensor_values = values[:, indices]
        temporal_strength = np.abs(sensor_values).sum(axis=1)
        channel_strength = np.abs(sensor_values).sum(axis=0)
        top_channel_index = int(np.argmax(channel_strength))
        peak_time_index = int(np.argmax(temporal_strength))
        signed = float(sensor_values.sum())
        summaries.append(
            {
                "sensor_id": sensor,
                "sensor_name": config.SENSOR_PLACEMENT[sensor],
                "absolute_attribution": float(np.abs(sensor_values).sum()),
                "signed_attribution": signed,
                "direction": "supports" if signed >= 0 else "opposes",
                "top_modality": "sEMG" if top_channel_index == 6 else "IMU",
                "top_channel": channel_names[top_channel_index],
                "peak_time_index": peak_time_index,
                "peak_time_ms": float(
                    (peak_time_index + 0.5) * config.WINDOW_MS / MODEL_SEQUENCE_LENGTH
                ),
            }
        )
    total = sum(float(row["absolute_attribution"]) for row in summaries)
    for row in summaries:
        row["attribution_fraction"] = (
            float(row["absolute_attribution"]) / total if total > 0 else 0.0
        )
    return sorted(summaries, key=lambda row: float(row["absolute_attribution"]), reverse=True)
