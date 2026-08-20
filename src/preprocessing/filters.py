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


def _filt_filt(b, a, data, axis=-1):
    samples = data.shape[axis]
    ntaps = max(len(a), len(b))  # for 4th order butterworth, number of taps are 9;
    required_padlen = 3 * ntaps  # standard pad len, 27;

    if samples <= 1:
        return data

    if samples <= required_padlen:
        # Build pad_width for N-dimensional arrays (pads ONLY along target axis);
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (required_padlen, required_padlen)

        # Reflect-pad boundaries so signal length safely exceeds required_padlen;
        padded_data = np.pad(data, pad_width, mode="reflect")

        # Run filtfilt on the padded array
        filtered_padded = filtfilt(b, a, padded_data, axis=axis)

        # Slice target axis back to original bounds: [required_padlen : required_padlen + samples];
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(required_padlen, required_padlen + samples)
        return filtered_padded[tuple(slices)]

    return filtfilt(b, a, data, axis=axis)  # ample samples/time-steps;


def _butter_filtfilt(data: np.ndarray, cutoff, fs: float, order: int, btype: str) -> np.ndarray:
    nyq = fs / 2.0
    wn = [c / nyq for c in cutoff] if btype == "bandpass" else cutoff / nyq
    b, a = butter(order, wn, btype=btype)  # type: ignore
    return _filt_filt(b, a, data, axis=-1)


def lowpass_imu(imu: np.ndarray, fs: float = config.IMU_SAMPLING_RATE_HZ) -> np.ndarray:
    """Low-pass filter IMU data at config.FILTERS.imu_lowpass_hz."""
    return _butter_filtfilt(
        imu, config.FILTERS.imu_lowpass_hz, fs, config.FILTERS.imu_lowpass_order, "low"
    )


def correct_drift_imu(
    imu: np.ndarray, fs: float = config.IMU_SAMPLING_RATE_HZ, method: str = "mean_subtract"
) -> np.ndarray:
    """Correct IMU baseline wander / mounting offset.

    method="mean_subtract" (default): per-channel mean subtraction.
    method="highpass": high-pass at config.FILTERS.imu_drift_highpass_hz across all channels.
    method="gyro_only": high-pass at config.FILTERS.imu_drift_highpass_hz ONLY on gyroscope channels,
    preserving the 1g DC gravity vector on accelerometer channels.
    """
    if method == "highpass":
        return _butter_filtfilt(
            imu,
            config.FILTERS.imu_drift_highpass_hz,
            fs,
            config.FILTERS.imu_drift_highpass_order,
            "high",
        )
    elif method == "gyro_only":
        # Check if 3D (sensors, channels, T) or 2D (48, T)
        res = imu.copy()
        if imu.ndim == 3 and imu.shape[1] >= 6:
            # Gyro channels are indices 3, 4, 5
            res[:, 3:6, :] = _butter_filtfilt(
                res[:, 3:6, :],
                config.FILTERS.imu_drift_highpass_hz,
                fs,
                config.FILTERS.imu_drift_highpass_order,
                "high",
            )
        elif imu.ndim == 2 and imu.shape[0] == config.N_SENSORS * config.IMU_CHANNELS_PER_SENSOR:
            for s in range(config.N_SENSORS):
                gyro_start = s * 6 + 3
                gyro_end = s * 6 + 6
                res[gyro_start:gyro_end, :] = _butter_filtfilt(
                    res[gyro_start:gyro_end, :],
                    config.FILTERS.imu_drift_highpass_hz,
                    fs,
                    config.FILTERS.imu_drift_highpass_order,
                    "high",
                )
        else:
            res = _butter_filtfilt(
                res,
                config.FILTERS.imu_drift_highpass_hz,
                fs,
                config.FILTERS.imu_drift_highpass_order,
                "high",
            )
        return res
    elif method == "mean_subtract":
        return imu - imu.mean(axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unknown drift correction method: {method!r}")


def preprocess_imu(
    imu: np.ndarray, fs: float = config.IMU_SAMPLING_RATE_HZ, drift_method: str | None = None
) -> np.ndarray:
    """Filter IMU data with zero-phase low-pass smoothing (and optional drift correction).

    By default, drift_method=None so that only low-pass filtering (config.FILTERS.imu_lowpass_hz)
    is applied. This preserves the 1g DC gravity component on accelerometer channels, which is
    essential for limb inclination and Range of Motion (ROM) calculations.
    """
    if drift_method is not None and drift_method != "none":
        imu = correct_drift_imu(imu, fs=fs, method=drift_method)
    return lowpass_imu(imu, fs=fs)


def bandpass_emg(emg: np.ndarray, fs: float = config.EMG_SAMPLING_RATE_HZ) -> np.ndarray:
    """Bandpass filter EMG data at config.FILTERS.emg_bandpass_hz."""
    return _butter_filtfilt(
        emg, config.FILTERS.emg_bandpass_hz, fs, config.FILTERS.emg_bandpass_order, "bandpass"
    )


def notch_emg(emg: np.ndarray, fs: float = config.EMG_SAMPLING_RATE_HZ) -> np.ndarray:
    """Notch filter EMG data at config.FILTERS.emg_notch_hz to remove mains hum."""
    b, a = iirnotch(config.FILTERS.emg_notch_hz, config.FILTERS.emg_notch_q, fs=fs)
    return _filt_filt(b, a, emg, axis=-1)


def preprocess_emg(emg: np.ndarray, fs: float = config.EMG_SAMPLING_RATE_HZ) -> np.ndarray:
    """Bandpass then notch-filter EMG data."""
    return notch_emg(bandpass_emg(emg, fs=fs), fs=fs)
