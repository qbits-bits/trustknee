"""Public API for the TrustKnee features package."""

from src.features.extract import (
    build_feature_matrix,
    extract_emg_window_features,
    extract_imu_window_features,
    extract_trial_features,
    extract_window_features,
)

__all__ = [
    "build_feature_matrix",
    "extract_emg_window_features",
    "extract_imu_window_features",
    "extract_trial_features",
    "extract_window_features",
]
