"""Feature extraction from windowed IMU and sEMG signals.

Computes time-domain, frequency-domain, and kinematic features (ROM, jerk, peak angular
velocity, signal power, RMS, MAV, etc.) from windowed multimodal sensor data to produce
the standardized feature matrix (WindowFeatures) for traditional ML baselines (Random Forest, XGBoost).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.signal import welch

from src import config
from src.preprocessing.windowing import WindowedTrial, generate_windows_from_manifest


def compute_rms(signal: np.ndarray) -> float:
    """Root Mean Square."""
    return float(np.sqrt(np.mean(signal**2)))


def compute_jerk(accel: np.ndarray, fs: float = config.IMU_SAMPLING_RATE_HZ) -> float:
    """RMS jerk (derivative of acceleration over time)."""
    if accel.shape[-1] < 2:
        return 0.0
    dt = 1.0 / fs
    jerk = np.diff(accel, axis=-1) / dt
    return float(np.sqrt(np.mean(jerk**2)))


def compute_range_of_motion(gyro: np.ndarray, fs: float = config.IMU_SAMPLING_RATE_HZ):
    """Range of motion (integral of angular velocity over time)."""
    if gyro.shape[-1] < 2:
        return 0.0, 0.0, 0.0
    dt = 1.0 / fs
    angle = cumulative_trapezoid(gyro, dx=dt, axis=1, initial=0)  # replace rect. w/ trapezoid
    range_of_motion = np.ptp(angle, axis=1)
    return float(range_of_motion[0]), float(range_of_motion[1]), float(range_of_motion[2])


def compute_waveform_length(emg: np.ndarray) -> float:
    """Waveform length (cumulative absolute amplitude change)."""
    if emg.shape[-1] < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(emg, axis=-1))))


def compute_spectral_density(emg: np.ndarray) -> float:
    """Total Spectral Power of an EMG signal (Welch's method)."""
    if emg.shape[-1] < 2:
        return 0.0
    # Ensure nperseg doesn't exceed signal length
    nperseg = min(emg.shape[-1], 128)
    freq, psd = welch(emg, fs=config.EMG_SAMPLING_RATE_HZ, nperseg=nperseg)
    return float(trapezoid(psd, freq))


def compute_mnf(emg: np.ndarray) -> float:
    """Mean Frequency (MNF / MPF) of an EMG signal."""
    if emg.shape[-1] < 2:
        return 0.0
    nperseg = min(emg.shape[-1], 128)
    freq, psd = welch(emg, fs=config.EMG_SAMPLING_RATE_HZ, nperseg=nperseg)
    # Weighted average of frequencies (MNF formula)
    return float(
        np.sum(freq * psd) / (np.sum(psd) + 1e-8)
    )  # added 1e-8, to avoid arithematic error


def extract_imu_window_features(
    imu_win: np.ndarray, fs: float = config.IMU_SAMPLING_RATE_HZ
) -> dict[str, float]:
    """Extract kinematic and statistical features from a single IMU window (8, 6, T_imu)."""
    features: dict[str, float] = {}

    for s_idx in range(config.N_SENSORS):
        sensor_id = s_idx + 1
        prefix = f"s{sensor_id}"

        # Channels: 0:acc_x, 1:acc_y, 2:acc_z, 3:gyro_x, 4:gyro_y, 5:gyro_z
        for c_idx, ch_name in enumerate(config.IMU_CHANNEL_NAMES):
            sig = imu_win[s_idx, c_idx, :]

            features[f"{prefix}_{ch_name}_mean"] = float(np.mean(sig))
            features[f"{prefix}_{ch_name}_std"] = float(np.std(sig))
            features[f"{prefix}_{ch_name}_rms"] = compute_rms(sig)
            features[f"{prefix}_{ch_name}_peak"] = float(np.max(np.abs(sig)))
            features[f"{prefix}_{ch_name}_rom"] = float(np.ptp(sig))  # peak-to-peak range
            features[f"{prefix}_{ch_name}_power"] = float(np.mean(sig**2))

            if "acc" in ch_name:
                features[f"{prefix}_{ch_name}_jerk"] = compute_jerk(sig, fs=fs)

        # Multi-axis kinematic composites
        acc_mag = np.linalg.norm(imu_win[s_idx, 0:3, :], axis=0)
        gyro_mag = np.linalg.norm(imu_win[s_idx, 3:6, :], axis=0)

        features[f"{prefix}_acc_mag_mean"] = float(np.mean(acc_mag))
        features[f"{prefix}_acc_mag_peak"] = float(np.max(acc_mag))
        features[f"{prefix}_gyro_peak_angular_vel"] = float(np.max(gyro_mag))
        features[f"{prefix}_gyro_angular_vel_mean"] = float(np.mean(gyro_mag))

        # 3. Integrated Range of Motion features, defines the extension angle of limb
        gyro_block = imu_win[s_idx, 3:6, :]  # gyro channels (3, T_imu)
        (
            features[f"{prefix}_range_of_motion_x"],
            features[f"{prefix}_range_of_motion_y"],
            features[f"{prefix}_range_of_motion_z"],
        ) = compute_range_of_motion(gyro_block)

    return features


def extract_emg_window_features(
    emg_win: np.ndarray, fs: float = config.EMG_SAMPLING_RATE_HZ
) -> dict[str, float]:
    """Extract electromyographic features from a single EMG window (8, T_emg)."""
    features: dict[str, float] = {}

    for s_idx in range(config.N_SENSORS):
        sensor_id = s_idx + 1
        prefix = f"s{sensor_id}_emg"
        sig = emg_win[s_idx, :]

        features[f"{prefix}_mav"] = float(np.mean(np.abs(sig)))  # Mean Absolute Value
        features[f"{prefix}_rms"] = compute_rms(sig)
        features[f"{prefix}_std"] = float(np.std(sig))
        features[f"{prefix}_iemg"] = float(np.sum(np.abs(sig)))  # Integrated EMG
        features[f"{prefix}_wl"] = compute_waveform_length(sig)  # Waveform Length
        features[f"{prefix}_power"] = float(np.mean(sig**2))
        features[f"{prefix}_spectral_power"] = compute_spectral_density(sig)
        features[f"{prefix}_mnf"] = compute_mnf(sig)  # Mean Frequency

    return features


def extract_window_features(
    imu_win: np.ndarray,
    emg_win: np.ndarray,
    fs_imu: float = config.IMU_SAMPLING_RATE_HZ,
    fs_emg: float = config.EMG_SAMPLING_RATE_HZ,
) -> dict[str, float]:
    """Extract full multimodal feature set for one window."""
    feat_dict = extract_imu_window_features(imu_win, fs=fs_imu)
    feat_dict.update(extract_emg_window_features(emg_win, fs=fs_emg))
    return feat_dict


def extract_trial_features(windowed_trial: WindowedTrial) -> pd.DataFrame:
    """Extract a complete DataFrame of window features from a WindowedTrial."""
    if windowed_trial.n_windows == 0:
        return pd.DataFrame()

    records = []
    for i in range(windowed_trial.n_windows):
        imu_w = windowed_trial.imu_windows[i]
        emg_w = windowed_trial.emg_windows[i]
        w_feats = extract_window_features(imu_w, emg_w)
        records.append(w_feats)

    feats_df = pd.DataFrame(records)
    # Concatenate metadata on the left
    return pd.concat([windowed_trial.metadata.reset_index(drop=True), feats_df], axis=1)


def build_feature_matrix(
    manifest: pd.DataFrame,
    window_ms: float = config.WINDOW_MS,
    overlap: float = config.WINDOW_OVERLAP,
    preprocess: bool = True,
) -> pd.DataFrame:
    """Aggregate all trials in a manifest into a single unified features DataFrame."""
    trial_dfs = []
    for windowed_trial in generate_windows_from_manifest(
        manifest=manifest, window_ms=window_ms, overlap=overlap, preprocess=preprocess
    ):
        trial_df = extract_trial_features(windowed_trial)
        if not trial_df.empty:
            trial_dfs.append(trial_df)

    if not trial_dfs:
        return pd.DataFrame()

    return pd.concat(trial_dfs, axis=0, ignore_index=True)
