"""SensorSource abstractions and recorded-trial streaming replay.

Provides real-time and simulated streaming abstractions for multi-modal IMU and
sEMG signals without altering or duplicating signal definitions.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src import config
from src.ingestion.ingest import Trial, load_trial

logger = logging.getLogger("trustknee.streaming")


class MalformedFrameError(ValueError):
    """Raised when incoming sensor frames violate hardware channel or shape constraints."""


@dataclass(frozen=True)
class SensorFrame:
    """A multi-rate sensor chunk received at a specific point in time."""

    timestamp_s: float
    imu_data: np.ndarray  # Shape: (N_SENSORS, 6, T_imu) or (48, T_imu)
    emg_data: np.ndarray  # Shape: (N_SENSORS, T_emg) or (8, T_emg)
    subject_id: int | None = None
    label_id: int | None = None
    trial_num: int | None = None
    is_last: bool = False

    def __post_init__(self) -> None:
        if self.is_last and (self.imu_data.size == 0 or self.emg_data.size == 0):
            return

        if self.imu_data.ndim not in (2, 3):
            raise MalformedFrameError(
                f"IMU data must be 2D (48, T) or 3D (8, 6, T), got ndim={self.imu_data.ndim}"
            )
        if self.emg_data.ndim not in (1, 2):
            raise MalformedFrameError(
                f"EMG data must be 1D (8,) or 2D (8, T), got ndim={self.emg_data.ndim}"
            )

        # Validate channel dimensions
        imu_channels = (
            self.imu_data.shape[0]
            if self.imu_data.ndim == 2
            else self.imu_data.shape[0] * self.imu_data.shape[1]
        )
        expected_imu_channels = config.N_SENSORS * config.IMU_CHANNELS_PER_SENSOR
        if imu_channels != expected_imu_channels:
            raise MalformedFrameError(
                f"Expected {expected_imu_channels} IMU channels, got {imu_channels}"
            )

        emg_channels = self.emg_data.shape[0]
        if emg_channels != config.N_SENSORS:
            raise MalformedFrameError(
                f"Expected {config.N_SENSORS} EMG channels, got {emg_channels}"
            )

        if not np.all(np.isfinite(self.imu_data)):
            raise MalformedFrameError("IMU frame contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(self.emg_data)):
            raise MalformedFrameError("EMG frame contains non-finite values (NaN or Inf)")


class SensorSource(ABC):
    """Abstract base class for all streaming signal sources."""

    @abstractmethod
    def stream_frames(self) -> Iterator[SensorFrame]:
        """Yield time-stamped sensor frames continuously or until stream completion."""

    def close(self) -> None:
        """Release underlying hardware resources or file descriptors."""
        return None


class RecordedTrialSource(SensorSource):
    """Replays a recorded trial from KneE-PAD files or an in-memory Trial instance."""

    def __init__(
        self,
        trial: Trial | None = None,
        imu_path: str | Path | None = None,
        emg_path: str | Path | None = None,
        subject_id: int | None = None,
        label_id: int | None = None,
        trial_num: int | None = None,
        chunk_duration_ms: float = 50.0,
        pacing_realtime: bool = False,
    ) -> None:
        if trial is not None:
            self._trial = trial
        elif imu_path is not None and emg_path is not None:
            self._trial = load_trial(
                imu_path=imu_path,
                emg_path=emg_path,
                subject_id=subject_id or 0,
                label_id=label_id or 0,
                trial_num=trial_num or 0,
            )
        else:
            raise ValueError("Must provide either a Trial instance or imu_path and emg_path")

        if chunk_duration_ms <= 0:
            raise ValueError(f"chunk_duration_ms must be positive, got {chunk_duration_ms}")

        self.chunk_duration_ms = float(chunk_duration_ms)
        self.pacing_realtime = bool(pacing_realtime)

        # Standardize arrays to 3D IMU (8, 6, T) and 2D EMG (8, T)
        imu = self._trial.imu
        if imu.ndim == 2:
            imu = imu.reshape(config.N_SENSORS, config.IMU_CHANNELS_PER_SENSOR, -1)
        self._imu = imu

        emg = self._trial.emg
        if emg.ndim == 1:
            emg = emg.reshape(config.N_SENSORS, -1)
        self._emg = emg

        self._t_imu_total = self._imu.shape[-1]
        self._t_emg_total = self._emg.shape[-1]

        self.fs_imu = config.IMU_SAMPLING_RATE_HZ
        self.fs_emg = config.EMG_SAMPLING_RATE_HZ

    @property
    def trial_info(self) -> dict[str, Any]:
        return {
            "subject_id": self._trial.subject_id,
            "label_id": self._trial.label_id,
            "trial_num": self._trial.trial_num,
            "duration_s": self._trial.duration_s,
            "n_imu_samples": self._t_imu_total,
            "n_emg_samples": self._t_emg_total,
        }

    def stream_frames(self) -> Iterator[SensorFrame]:
        """Stream the recorded trial in temporal chunks with zero data loss."""
        chunk_sec = self.chunk_duration_ms / 1000.0
        imu_chunk_samples = max(1, int(round(chunk_sec * self.fs_imu)))
        emg_chunk_samples = max(1, int(round(chunk_sec * self.fs_emg)))

        imu_idx = 0
        emg_idx = 0
        chunk_idx = 0

        while imu_idx < self._t_imu_total or emg_idx < self._t_emg_total:
            imu_next = min(self._t_imu_total, imu_idx + imu_chunk_samples)
            emg_next = min(self._t_emg_total, emg_idx + emg_chunk_samples)

            imu_slice = self._imu[:, :, imu_idx:imu_next]
            emg_slice = self._emg[:, emg_idx:emg_next]

            timestamp_s = chunk_idx * chunk_sec
            is_last = (imu_next >= self._t_imu_total) and (emg_next >= self._t_emg_total)

            frame = SensorFrame(
                timestamp_s=timestamp_s,
                imu_data=imu_slice,
                emg_data=emg_slice,
                subject_id=self._trial.subject_id,
                label_id=self._trial.label_id,
                trial_num=self._trial.trial_num,
                is_last=is_last,
            )

            if self.pacing_realtime and chunk_idx > 0:
                time.sleep(chunk_sec)

            yield frame

            imu_idx = imu_next
            emg_idx = emg_next
            chunk_idx += 1
