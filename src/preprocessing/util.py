"""Signal alignment, resampling, normalization, and length utilities."""

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.preprocessing.filters import preprocess_emg, preprocess_imu  # noqa: E402


def upsampling_imu(imu, imu_fs=config.IMU_SAMPLING_RATE_HZ, target=config.EMG_SAMPLING_RATE_HZ):
    """Resample an IMU signal to a target sampling rate.

    Args:
        imu: IMU array shaped ``(channels, time)`` or ``(time, channels)``.
        imu_fs: Original IMU sampling rate in Hz.
        target: Target sampling rate in Hz.

    Returns:
        The centered IMU signal resampled along its time axis.
    """
    # IF orientation is (T, Channels), then transpose to (Channels, T);
    if imu.ndim == 2 and imu.shape[0] > imu.shape[1]:
        imu = imu.T

    # Baseline correction;
    imu_centered = imu - np.mean(
        imu[:, :20], keepdims=True, axis=-1
    )  # can be used to average out the first 20 samples, no physical motion;

    frac = Fraction(int(target), int(imu_fs))
    up_factor = frac.numerator
    down_factor = frac.denominator
    # upsample_factor = int(emg_fs//imu_fs); # can be used but gives floor for sampling factor;

    # Resampling `resample_poly`: insert 0s b/w the samples, uses internal FIR filter to smooth the waveform;
    return resample_poly(
        imu_centered, up=up_factor, down=down_factor, axis=-1
    )  # up: upsample factor (sample in b/w), down: downsample factor (samples to drop);


def combine_signals(
    emg, imu, emg_fs=config.EMG_SAMPLING_RATE_HZ, imu_fs=config.IMU_SAMPLING_RATE_HZ
):
    """Filter, align, and vertically concatenate EMG and IMU signals.

    Inputs may be arrays or paths to ``.npy`` files. The shorter filtered
    stream determines the output duration.
    """
    if isinstance(emg, str | bytes):
        emg = np.load(emg)  # (8, T)
    if isinstance(imu, str | bytes):
        imu = np.load(imu)  # (48, T)

    # IF orientation is (T, Channels), then transpose to (Channels, T);
    if emg.ndim == 2 and emg.shape[0] > emg.shape[1]:
        emg = emg.T
    if imu.ndim == 2 and imu.shape[0] > imu.shape[1]:
        imu = imu.T

    # Upsampling IMU stream;
    imu_upsampled = upsampling_imu(imu, imu_fs=imu_fs, target=emg_fs)

    # Filtering;
    imu_filtered = preprocess_imu(imu_upsampled, fs=emg_fs)
    emg_filtered = preprocess_emg(emg, fs=emg_fs)

    # Alignment;
    min_length = min(
        emg_filtered.shape[-1], imu_filtered.shape[-1]
    )  # spoting minimum samples, in each stream for alignment;
    emg_mod = emg_filtered[:, :min_length]
    imu_mod = imu_filtered[:, :min_length]

    # Combining streams;
    trial = np.vstack((emg_mod, imu_mod))  # (8 + 48, T)

    return trial


def normalize_signals(train_list, val_list, test_list):
    """Z-score signal splits using statistics calculated from training data.

    Returns normalized train, validation, and test arrays followed by the
    training mean and standard deviation used for all three transformations.
    """
    train_concat = np.concatenate(train_list, axis=1)
    global_mean = train_concat.mean(axis=-1, keepdims=True)
    global_std = train_concat.std(axis=-1, keepdims=True)

    train_norm = [(trial - global_mean) / (global_std + 1e-8) for trial in train_list]
    val_norm = [(trial - global_mean) / (global_std + 1e-8) for trial in val_list]
    test_norm = [(trial - global_mean) / (global_std + 1e-8) for trial in test_list]

    return train_norm, val_norm, test_norm, global_mean, global_std
