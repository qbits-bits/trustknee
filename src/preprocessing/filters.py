"""Zero-phase filtering for IMU drift/noise and EMG conditioning.

IMU low-pass (6 Hz, order 4): movement signal power sits below roughly
6-10 Hz for non-ballistic exercises like ours (squat, leg extension,
walking). See Yu, Gabriel, Noble & An (1999, J. Appl. Biomech. 15(3)) and
Winter's Biomechanics and Motor Control of Human Movement. Nyquist is
74 Hz at our 148 Hz sampling rate, so there's plenty of headroom to raise
this later if needed.

IMU drift high-pass (0.3 Hz, order 2): removes baseline wander and mounting
offset, not gyroscope-integration drift. That kind of drift needs Kalman or
complementary sensor fusion and doesn't apply here since these are short,
pre-segmented trials, not one long continuous stream.

EMG bandpass (20-450 Hz) + 50 Hz notch: matches the Trigno Avanti sensor's
own published bandwidth and standard EMG pattern-recognition convention.
Nyquist is 630 Hz at 1259 Hz sampling, well above 450 Hz. The notch removes
mains hum (use 60 Hz for a 60 Hz mains region).

All filters run along the last axis, so they work the same on a 2D
(channels, T) array or the reshaped 3D (sensors, channels, T) array.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from src import config


def _butter_filtfilt(data: np.ndarray, cutoff, fs: float, order: int, btype: str) -> np.ndarray:
    nyq = fs / 2.0
    if btype == "bandpass":
        wn = [c / nyq for c in cutoff]
    else:
        wn = cutoff / nyq
    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, data, axis=-1)


def lowpass_imu(imu: np.ndarray, fs: float = config.IMU_SAMPLING_RATE_HZ) -> np.ndarray:
    """Low-pass filter IMU data at config.FILTERS.imu_lowpass_hz."""
    return _butter_filtfilt(
        imu, config.FILTERS.imu_lowpass_hz, fs, config.FILTERS.imu_lowpass_order, "low"
    )


def correct_drift_imu(
    imu: np.ndarray, fs: float = config.IMU_SAMPLING_RATE_HZ, method: str = "highpass"
) -> np.ndarray:
    """Correct IMU baseline wander / mounting offset.

    method="highpass" (default): high-pass at config.FILTERS.imu_drift_highpass_hz.
    method="mean_subtract": per-channel mean subtraction, a lighter debug fallback.
    """
    if method == "highpass":
        return _butter_filtfilt(
            imu,
            config.FILTERS.imu_drift_highpass_hz,
            fs,
            config.FILTERS.imu_drift_highpass_order,
            "high",
        )
    elif method == "mean_subtract":
        return imu - imu.mean(axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unknown drift correction method: {method!r}")


def preprocess_imu(
    imu: np.ndarray, fs: float = config.IMU_SAMPLING_RATE_HZ, drift_method: str = "highpass"
) -> np.ndarray:
    """Drift-correct then low-pass smooth IMU data."""
    corrected = correct_drift_imu(imu, fs=fs, method=drift_method)
    return lowpass_imu(corrected, fs=fs)


def bandpass_emg(emg: np.ndarray, fs: float = config.EMG_SAMPLING_RATE_HZ) -> np.ndarray:
    """Bandpass filter EMG data at config.FILTERS.emg_bandpass_hz."""
    return _butter_filtfilt(
        emg, config.FILTERS.emg_bandpass_hz, fs, config.FILTERS.emg_bandpass_order, "bandpass"
    )


def notch_emg(emg: np.ndarray, fs: float = config.EMG_SAMPLING_RATE_HZ) -> np.ndarray:
    """Notch filter EMG data at config.FILTERS.emg_notch_hz to remove mains hum."""
    b, a = iirnotch(config.FILTERS.emg_notch_hz, config.FILTERS.emg_notch_q, fs=fs)
    return filtfilt(b, a, emg, axis=-1)


def preprocess_emg(emg: np.ndarray, fs: float = config.EMG_SAMPLING_RATE_HZ) -> np.ndarray:
    """Bandpass then notch-filter EMG data."""
    return notch_emg(bandpass_emg(emg, fs=fs), fs=fs)
