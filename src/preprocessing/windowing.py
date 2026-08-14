"""Sliding-window segmentation for multi-modal IMU and EMG signals.

Provides time-aligned sliding window extraction across differing sampling rates
(IMU ~148 Hz and EMG ~1259 Hz). Each window retains synchronized start/end timestamps
and produces 3D/4D tensors for Deep Learning (LSTM/Transformer) alongside structured
metadata for tabular Feature Extraction (Random Forest/XGBoost).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from src import config
from src.ingestion.ingest import Trial, load_trial_from_manifest_row
from src.preprocessing.filters import preprocess_emg, preprocess_imu


@dataclass
class WindowedTrial:
    """A trial's signals segmented into synchronized sliding windows."""

    subject_id: int
    label_id: int
    trial_num: int
    execution: str | None
    exercise: str | None
    imu_windows: np.ndarray  # Shape: (n_windows, N_SENSORS, IMU_CHANNELS_PER_SENSOR, samples_imu)
    emg_windows: np.ndarray  # Shape: (n_windows, N_SENSORS, samples_emg)
    metadata: pd.DataFrame   # Aligned per-window metadata (timestamps, labels, etc.)

    @property
    def n_windows(self) -> int:
        return len(self.imu_windows)


def calculate_window_sample_counts(
    window_ms: float = config.WINDOW_MS,
    overlap: float = config.WINDOW_OVERLAP,
    fs_imu: float = config.IMU_SAMPLING_RATE_HZ,
    fs_emg: float = config.EMG_SAMPLING_RATE_HZ,
) -> tuple[int, int, int, int]:
    """Calculate exact integer sample lengths and step strides for IMU and EMG.

    Returns:
        (imu_win_samples, imu_step_samples, emg_win_samples, emg_step_samples)
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"Overlap must be in range [0, 1), got {overlap}")
    if window_ms <= 0:
        raise ValueError(f"Window duration must be positive, got {window_ms}")

    window_sec = window_ms / 1000.0
    step_sec = window_sec * (1.0 - overlap)

    imu_win_samples = max(1, int(round(window_sec * fs_imu)))
    imu_step_samples = max(1, int(round(step_sec * fs_imu)))

    emg_win_samples = max(1, int(round(window_sec * fs_emg)))
    emg_step_samples = max(1, int(round(step_sec * fs_emg)))

    return imu_win_samples, imu_step_samples, emg_win_samples, emg_step_samples


def slice_arrays_to_windows(
    imu: np.ndarray,
    emg: np.ndarray,
    window_ms: float = config.WINDOW_MS,
    overlap: float = config.WINDOW_OVERLAP,
    fs_imu: float = config.IMU_SAMPLING_RATE_HZ,
    fs_emg: float = config.EMG_SAMPLING_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slice raw/filtered IMU and EMG continuous arrays into synchronized windows.

    Args:
        imu: Array of shape (N_SENSORS, 6, T_imu) or (48, T_imu).
        emg: Array of shape (N_SENSORS, T_emg) or (8, T_emg).
        window_ms: Duration of each window in milliseconds.
        overlap: Overlap fraction between consecutive windows [0, 1).
        fs_imu: IMU sampling rate in Hz.
        fs_emg: EMG sampling rate in Hz.

    Returns:
        imu_windows: (n_windows, N_SENSORS, 6, imu_win_samples)
        emg_windows: (n_windows, N_SENSORS, emg_win_samples)
        start_times_s: (n_windows,) array of window start times in seconds
        end_times_s: (n_windows,) array of window end times in seconds
    """
    # Ensure standard shapes
    if imu.ndim == 2:
        imu = imu.reshape(config.N_SENSORS, config.IMU_CHANNELS_PER_SENSOR, -1)
    if emg.ndim == 1:
        emg = emg.reshape(config.N_SENSORS, -1)

    imu_t = imu.shape[-1]
    emg_t = emg.shape[-1]

    imu_win_samples, _, emg_win_samples, _ = calculate_window_sample_counts(
        window_ms=window_ms, overlap=overlap, fs_imu=fs_imu, fs_emg=fs_emg
    )

    window_sec = window_ms / 1000.0
    step_sec = window_sec * (1.0 - overlap)

    imu_slices = []
    emg_slices = []
    start_times = []
    end_times = []

    k = 0
    while True:
        t_start = k * step_sec
        t_end = t_start + window_sec

        # Anchor start samples to exact timestamp to prevent cumulative rounding drift
        imu_start = int(round(t_start * fs_imu))
        imu_end = imu_start + imu_win_samples

        emg_start = int(round(t_start * fs_emg))
        emg_end = emg_start + emg_win_samples

        if imu_end > imu_t or emg_end > emg_t:
            break

        imu_slices.append(imu[:, :, imu_start:imu_end])
        emg_slices.append(emg[:, emg_start:emg_end])
        start_times.append(t_start)
        end_times.append(t_end)

        k += 1

    if not imu_slices:
        empty_imu = np.empty((0, config.N_SENSORS, config.IMU_CHANNELS_PER_SENSOR, imu_win_samples), dtype=imu.dtype)
        empty_emg = np.empty((0, config.N_SENSORS, emg_win_samples), dtype=emg.dtype)
        return empty_imu, empty_emg, np.array([], dtype=float), np.array([], dtype=float)

    return (
        np.stack(imu_slices, axis=0),
        np.stack(emg_slices, axis=0),
        np.array(start_times, dtype=float),
        np.array(end_times, dtype=float),
    )


def slice_trial_windows(
    trial: Trial,
    window_ms: float = config.WINDOW_MS,
    overlap: float = config.WINDOW_OVERLAP,
    preprocess: bool = True,
    execution: str | None = None,
    exercise: str | None = None,
) -> WindowedTrial:
    """Slice a Trial into filtered, synchronized sliding windows with metadata."""
    imu_data = preprocess_imu(trial.imu) if preprocess else trial.imu
    emg_data = preprocess_emg(trial.emg) if preprocess else trial.emg

    imu_wins, emg_wins, start_times, end_times = slice_arrays_to_windows(
        imu=imu_data,
        emg=emg_data,
        window_ms=window_ms,
        overlap=overlap,
    )

    n_wins = len(start_times)
    meta_df = pd.DataFrame(
        {
            "window_index": np.arange(n_wins, dtype=int),
            "start_time_s": start_times,
            "end_time_s": end_times,
            "subject_id": trial.subject_id,
            "label_id": trial.label_id,
            "trial_num": trial.trial_num,
            "execution": execution,
            "exercise": exercise,
        }
    )

    return WindowedTrial(
        subject_id=trial.subject_id,
        label_id=trial.label_id,
        trial_num=trial.trial_num,
        execution=execution,
        exercise=exercise,
        imu_windows=imu_wins,
        emg_windows=emg_wins,
        metadata=meta_df,
    )


def generate_windows_from_manifest(
    manifest: pd.DataFrame,
    window_ms: float = config.WINDOW_MS,
    overlap: float = config.WINDOW_OVERLAP,
    preprocess: bool = True,
) -> Iterator[WindowedTrial]:
    """Yield WindowedTrial objects one-by-one from a build_manifest() DataFrame."""
    for _, row in manifest.iterrows():
        trial = load_trial_from_manifest_row(row)
        yield slice_trial_windows(
            trial=trial,
            window_ms=window_ms,
            overlap=overlap,
            preprocess=preprocess,
            execution=row.get("execution"),
            exercise=row.get("exercise"),
        )
