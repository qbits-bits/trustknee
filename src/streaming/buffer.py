"""Stateful synchronized multi-modal window buffer for live/replayed streams.

Accurately segments arriving IMU and sEMG data into synchronized 200 ms windows
with 50% overlap, matching offline windowing bitwise without cumulative drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src import config
from src.preprocessing.windowing import calculate_window_sample_counts

logger = logging.getLogger("trustknee.streaming")


@dataclass(frozen=True)
class WindowFrame:
    """A synchronized, multi-modal window segment emitted by the streaming buffer."""

    window_index: int
    imu: np.ndarray  # Shape: (config.N_SENSORS, config.IMU_CHANNELS_PER_SENSOR, imu_win_samples)
    emg: np.ndarray  # Shape: (config.N_SENSORS, emg_win_samples)
    start_time_s: float
    end_time_s: float
    subject_id: int | None = None
    label_id: int | None = None
    trial_num: int | None = None


class StreamingWindowBuffer:
    """Rolling multi-rate sample buffer emitting synchronized windows with zero cumulative drift."""

    def __init__(
        self,
        window_ms: float = config.WINDOW_MS,
        overlap: float = config.WINDOW_OVERLAP,
        fs_imu: float = config.IMU_SAMPLING_RATE_HZ,
        fs_emg: float = config.EMG_SAMPLING_RATE_HZ,
    ) -> None:
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"Overlap must be in range [0, 1), got {overlap}")
        if window_ms <= 0:
            raise ValueError(f"Window duration must be positive, got {window_ms}")

        self.window_ms = float(window_ms)
        self.overlap = float(overlap)
        self.fs_imu = float(fs_imu)
        self.fs_emg = float(fs_emg)

        (
            self.imu_win_samples,
            self.imu_step_samples,
            self.emg_win_samples,
            self.emg_step_samples,
        ) = calculate_window_sample_counts(
            window_ms=window_ms,
            overlap=overlap,
            fs_imu=fs_imu,
            fs_emg=fs_emg,
        )

        self.window_sec = self.window_ms / 1000.0
        self.step_sec = self.window_sec * (1.0 - self.overlap)

        self._k = 0
        self._total_imu_received = 0
        self._total_emg_received = 0
        self._imu_global_offset = 0
        self._emg_global_offset = 0

        self._imu_buffer: np.ndarray | None = None
        self._emg_buffer: np.ndarray | None = None

        self._subject_id: int | None = None
        self._label_id: int | None = None
        self._trial_num: int | None = None

    @property
    def window_index(self) -> int:
        """The index of the next window to emit."""
        return self._k

    @property
    def total_imu_samples(self) -> int:
        return self._total_imu_received

    @property
    def total_emg_samples(self) -> int:
        return self._total_emg_received

    def reset(self) -> None:
        """Reset the buffer state for a fresh trial or recording session."""
        self._k = 0
        self._total_imu_received = 0
        self._total_emg_received = 0
        self._imu_global_offset = 0
        self._emg_global_offset = 0
        self._imu_buffer = None
        self._emg_buffer = None
        self._subject_id = None
        self._label_id = None
        self._trial_num = None

    def push(
        self,
        imu_chunk: np.ndarray,
        emg_chunk: np.ndarray,
        subject_id: int | None = None,
        label_id: int | None = None,
        trial_num: int | None = None,
    ) -> list[WindowFrame]:
        """Ingest new sensor chunks and return all newly ready synchronized windows."""
        if subject_id is not None:
            self._subject_id = subject_id
        if label_id is not None:
            self._label_id = label_id
        if trial_num is not None:
            self._trial_num = trial_num

        # Ensure standard shapes: IMU (8, 6, T), EMG (8, T)
        if imu_chunk.ndim == 2:
            imu_chunk = imu_chunk.reshape(config.N_SENSORS, config.IMU_CHANNELS_PER_SENSOR, -1)
        if emg_chunk.ndim == 1:
            emg_chunk = emg_chunk.reshape(config.N_SENSORS, -1)

        imu_t = imu_chunk.shape[-1]
        emg_t = emg_chunk.shape[-1]

        if imu_t > 0:
            if self._imu_buffer is None:
                self._imu_buffer = imu_chunk
            else:
                self._imu_buffer = np.concatenate([self._imu_buffer, imu_chunk], axis=-1)
            self._total_imu_received += imu_t

        if emg_t > 0:
            if self._emg_buffer is None:
                self._emg_buffer = emg_chunk
            else:
                self._emg_buffer = np.concatenate([self._emg_buffer, emg_chunk], axis=-1)
            self._total_emg_received += emg_t

        emitted: list[WindowFrame] = []

        while True:
            t_start = self._k * self.step_sec
            t_end = t_start + self.window_sec

            imu_start = int(round(t_start * self.fs_imu))
            imu_end = imu_start + self.imu_win_samples

            emg_start = int(round(t_start * self.fs_emg))
            emg_end = emg_start + self.emg_win_samples

            # Check if we have enough global samples for this window
            if self._total_imu_received < imu_end or self._total_emg_received < emg_end:
                break

            assert self._imu_buffer is not None
            assert self._emg_buffer is not None

            buf_imu_start = imu_start - self._imu_global_offset
            buf_imu_end = imu_end - self._imu_global_offset

            buf_emg_start = emg_start - self._emg_global_offset
            buf_emg_end = emg_end - self._emg_global_offset

            imu_window = self._imu_buffer[:, :, buf_imu_start:buf_imu_end]
            emg_window = self._emg_buffer[:, buf_emg_start:buf_emg_end]

            frame = WindowFrame(
                window_index=self._k,
                imu=imu_window,
                emg=emg_window,
                start_time_s=t_start,
                end_time_s=t_end,
                subject_id=self._subject_id,
                label_id=self._label_id,
                trial_num=self._trial_num,
            )
            emitted.append(frame)

            self._k += 1

            # Determine earliest sample needed for next window
            next_t_start = self._k * self.step_sec
            next_imu_start = int(round(next_t_start * self.fs_imu))
            next_emg_start = int(round(next_t_start * self.fs_emg))

            # Prune obsolete samples to prevent unbounded memory growth
            discard_imu = next_imu_start - self._imu_global_offset
            if discard_imu > 0 and self._imu_buffer.shape[-1] >= discard_imu:
                self._imu_buffer = self._imu_buffer[:, :, discard_imu:]
                self._imu_global_offset = next_imu_start

            discard_emg = next_emg_start - self._emg_global_offset
            if discard_emg > 0 and self._emg_buffer.shape[-1] >= discard_emg:
                self._emg_buffer = self._emg_buffer[:, discard_emg:]
                self._emg_global_offset = next_emg_start

        return emitted
