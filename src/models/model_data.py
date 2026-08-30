"""Convert synchronized TrustKnee windows into model-ready representations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.features.extract import extract_window_features
from src.models.labels import labels_to_binary
from src.preprocessing.windowing import WindowedTrial, generate_windows_from_manifest

MODEL_SEQUENCE_LENGTH = 30
MODEL_INPUT_DIM = 56


@dataclass
class ModelDataset:
    """Aligned sequence, tabular, target, and grouping data for an experiment."""

    transformer_sequences: np.ndarray  # (windows, 30, 56), model input only
    xgboost_features: pd.DataFrame  # numeric handcrafted features only
    labels_9: np.ndarray
    labels_binary: np.ndarray
    metadata: pd.DataFrame  # IDs/timestamps used for grouping and reporting only

    def __post_init__(self) -> None:
        n_samples = len(self.transformer_sequences)
        if self.transformer_sequences.ndim != 3 or self.transformer_sequences.shape[1:] != (
            MODEL_SEQUENCE_LENGTH,
            MODEL_INPUT_DIM,
        ):
            raise ValueError(
                "transformer_sequences must have shape "
                f"(n, {MODEL_SEQUENCE_LENGTH}, {MODEL_INPUT_DIM})"
            )
        if len(self.xgboost_features) != n_samples:
            raise ValueError("Sequence and XGBoost feature rows are not aligned")
        for name, values in (
            ("labels_9", self.labels_9),
            ("labels_binary", self.labels_binary),
            ("metadata", self.metadata),
        ):
            if len(values) != n_samples:
                raise ValueError(f"{name} is not aligned with the model rows")

    @property
    def n_samples(self) -> int:
        return len(self.transformer_sequences)

    @property
    def feature_names(self) -> list[str]:
        """Names passed to XGBoost; no metadata names are included."""
        return list(self.xgboost_features.columns)

    # Short aliases make the object convenient in notebooks without changing
    # the explicit names used by the evaluator.
    @property
    def sequences(self) -> np.ndarray:
        return self.transformer_sequences

    @property
    def features(self) -> pd.DataFrame:
        return self.xgboost_features

    @property
    def subject_ids(self) -> np.ndarray:
        return self.metadata["subject_id"].to_numpy()

    @property
    def trial_ids(self) -> np.ndarray:
        return self.metadata["trial_id"].to_numpy()

    def subset(self, indices: np.ndarray | list[int]) -> ModelDataset:
        """Return an aligned copy containing only ``indices``."""
        indices = np.asarray(indices, dtype=int)
        return ModelDataset(
            transformer_sequences=self.transformer_sequences[indices],
            xgboost_features=self.xgboost_features.iloc[indices].reset_index(drop=True),
            labels_9=self.labels_9[indices],
            labels_binary=self.labels_binary[indices],
            metadata=self.metadata.iloc[indices].reset_index(drop=True),
        )


def _mean_equal_time_bins(values: np.ndarray, sequence_length: int) -> np.ndarray:
    """Average the last axis into equal-duration bins, including every sample."""
    n_samples = values.shape[-1]
    if n_samples == 0:
        raise ValueError("Cannot create an envelope from an empty EMG window")

    edges = np.linspace(0, n_samples, sequence_length + 1)
    result = np.empty(values.shape[:-1] + (sequence_length,), dtype=np.float64)
    for bin_index, (start_edge, end_edge) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        start = int(np.floor(start_edge))
        end = int(np.floor(end_edge))
        if end <= start:
            # This only occurs when a caller requests more bins than samples.
            # Reusing the nearest sample keeps the public conversion finite.
            sample_index = min(start, n_samples - 1)
            result[..., bin_index] = values[..., sample_index]
        else:
            result[..., bin_index] = values[..., start:end].mean(axis=-1)
    return result


def create_emg_activity_envelope(
    emg_windows: np.ndarray, sequence_length: int = MODEL_SEQUENCE_LENGTH
) -> np.ndarray:
    """Create synchronized low-rate sEMG activity from filtered EMG windows.

    The filtered waveform is rectified and averaged into equal-duration bins;
    it is not resampled point-by-point to the IMU rate.  A single input has
    shape ``(8, samples_emg)`` and returns ``(8, sequence_length)``.  A batch
    has shape ``(n, 8, samples_emg)`` and returns ``(n, 8, sequence_length)``.
    """
    values = np.asarray(emg_windows, dtype=np.float64)
    if values.ndim not in (2, 3):
        raise ValueError("emg_windows must have shape (8, samples) or (n, 8, samples)")
    if values.shape[-2] != config.N_SENSORS:
        raise ValueError(f"Expected {config.N_SENSORS} EMG sensor rows, got {values.shape[-2]}")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    envelope = _mean_equal_time_bins(np.abs(values), sequence_length)
    return envelope.astype(np.float32)


def _convert_imu_windows(imu_windows: np.ndarray, sequence_length: int) -> np.ndarray:
    values = np.asarray(imu_windows)
    if values.ndim != 4:
        raise ValueError("imu_windows must have shape (n, 8, 6, samples)")
    expected_shape = (config.N_SENSORS, config.IMU_CHANNELS_PER_SENSOR)
    if values.shape[1:3] != expected_shape:
        raise ValueError(
            f"Expected IMU shape (n, {expected_shape[0]}, {expected_shape[1]}, samples)"
        )
    if values.shape[-1] != sequence_length:
        raise ValueError(
            f"IMU windows must contain {sequence_length} samples for the Transformer, "
            f"got {values.shape[-1]}"
        )
    # Sensor-major flattening preserves the existing raw row layout at each
    # time point: 8 sensors x 6 IMU channels = 48 movement values.
    return values.transpose(0, 3, 1, 2).reshape(len(values), sequence_length, -1).astype(np.float32)


def make_transformer_sequences(
    imu_windows: np.ndarray,
    emg_windows: np.ndarray,
    sequence_length: int = MODEL_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Combine 48 IMU channels and 8 binned sEMG activity channels."""
    imu_sequence = _convert_imu_windows(imu_windows, sequence_length)
    emg_envelope = create_emg_activity_envelope(emg_windows, sequence_length)
    if emg_envelope.ndim != 3 or len(imu_sequence) != len(emg_envelope):
        raise ValueError("IMU and EMG window batches must contain the same number of windows")
    emg_sequence = emg_envelope.transpose(0, 2, 1)
    combined = np.concatenate([imu_sequence, emg_sequence], axis=-1)
    if combined.shape[-1] != MODEL_INPUT_DIM:
        raise AssertionError(f"Expected {MODEL_INPUT_DIM} model channels, got {combined.shape[-1]}")
    return np.nan_to_num(combined, copy=False).astype(np.float32)


def _trial_id(subject_id: int, label_id: int, trial_num: int) -> str:
    # Trial numbers restart under every subject/label directory in KneE-PAD.
    return f"subject_{int(subject_id)}_label_{int(label_id)}_trial_{int(trial_num)}"


def build_model_inputs_from_trials(
    windowed_trials: Iterable[WindowedTrial],
) -> ModelDataset:
    """Build aligned Transformer sequences and XGBoost rows from windowed trials."""
    sequence_batches: list[np.ndarray] = []
    feature_frames: list[pd.DataFrame] = []
    metadata_frames: list[pd.DataFrame] = []
    labels_9: list[np.ndarray] = []
    labels_binary: list[np.ndarray] = []

    for windowed_trial in windowed_trials:
        if windowed_trial.n_windows == 0:
            continue
        sequences = make_transformer_sequences(
            windowed_trial.imu_windows,
            windowed_trial.emg_windows,
            sequence_length=MODEL_SEQUENCE_LENGTH,
        )
        records = [
            extract_window_features(imu_window, emg_window)
            for imu_window, emg_window in zip(
                windowed_trial.imu_windows, windowed_trial.emg_windows, strict=True
            )
        ]
        features = pd.DataFrame(records).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        metadata = windowed_trial.metadata.copy().reset_index(drop=True)
        if "synthetic" not in metadata.columns:
            metadata["synthetic"] = False
        else:
            metadata["synthetic"] = metadata["synthetic"].astype(bool)
        metadata["trial_id"] = [
            _trial_id(windowed_trial.subject_id, windowed_trial.label_id, windowed_trial.trial_num)
        ] * len(metadata)
        metadata_frames.append(metadata)
        sequence_batches.append(sequences)
        feature_frames.append(features.astype(np.float32))

        n = windowed_trial.n_windows
        label_ids = np.full(n, windowed_trial.label_id, dtype=np.int64)
        execution_values = [windowed_trial.execution] * n
        labels_9.append(label_ids)
        labels_binary.append(labels_to_binary(label_ids, execution_values))

    if not sequence_batches:
        return ModelDataset(
            transformer_sequences=np.empty(
                (0, MODEL_SEQUENCE_LENGTH, MODEL_INPUT_DIM), dtype=np.float32
            ),
            xgboost_features=pd.DataFrame(),
            labels_9=np.empty(0, dtype=np.int64),
            labels_binary=np.empty(0, dtype=np.int64),
            metadata=pd.DataFrame(
                columns=[
                    "window_index",
                    "start_time_s",
                    "end_time_s",
                    "subject_id",
                    "label_id",
                    "trial_num",
                    "execution",
                    "exercise",
                    "trial_id",
                    "synthetic",
                ]
            ),
        )

    return ModelDataset(
        transformer_sequences=np.concatenate(sequence_batches, axis=0),
        xgboost_features=pd.concat(feature_frames, ignore_index=True),
        labels_9=np.concatenate(labels_9),
        labels_binary=np.concatenate(labels_binary),
        metadata=pd.concat(metadata_frames, ignore_index=True),
    )


def build_model_inputs(
    manifest: pd.DataFrame,
    window_ms: float = config.WINDOW_MS,
    overlap: float = config.WINDOW_OVERLAP,
    preprocess: bool = True,
    augment_minority: bool = False,
    target_labels: tuple[int, ...] = (6, 7, 8),
    aug_methods: tuple[str, ...] = ("jitter", "magnitude_scale", "time_warp"),
    aug_multiplier: int = 3,
) -> ModelDataset:
    """Build aligned Transformer sequences and XGBoost rows from a manifest.

    The two model views are generated from exactly the same windows.  Subject,
    trial, window, and timestamp columns are stored in ``metadata`` and never
    copied into either model's feature matrix.
    """
    if augment_minority:
        from src.augmentation.pipeline import augment_minority_classes

        trials = augment_minority_classes(
            manifest,
            target_labels=target_labels,
            methods=aug_methods,
            multiplier=aug_multiplier,
            window_ms=window_ms,
            overlap=overlap,
            preprocess=preprocess,
        )
    else:
        trials = generate_windows_from_manifest(
            manifest,
            window_ms=window_ms,
            overlap=overlap,
            preprocess=preprocess,
            skip_corrupt=True,
        )
    return build_model_inputs_from_trials(trials)
