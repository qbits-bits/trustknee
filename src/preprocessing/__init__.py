"""Public API for the TrustKnee preprocessing package."""

from src.preprocessing.filters import (
    bandpass_emg,
    correct_drift_imu,
    lowpass_imu,
    notch_emg,
    preprocess_emg,
    preprocess_imu,
)
from src.preprocessing.windowing import (
    WindowedTrial,
    calculate_window_sample_counts,
    generate_windows_from_manifest,
    slice_arrays_to_windows,
    slice_trial_windows,
)

__all__ = [
    "WindowedTrial",
    "bandpass_emg",
    "calculate_window_sample_counts",
    "correct_drift_imu",
    "generate_windows_from_manifest",
    "lowpass_imu",
    "notch_emg",
    "preprocess_emg",
    "preprocess_imu",
    "slice_arrays_to_windows",
    "slice_trial_windows",
]
