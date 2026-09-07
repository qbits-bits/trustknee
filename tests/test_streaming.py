"""Tests for canonical Phase 3 contracts, SensorSource, and StreamingWindowBuffer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.contracts import CANONICAL_PHASE3_CONTRACT, ExperimentContract, get_canonical_contract
from src.ingestion.ingest import Trial
from src.preprocessing.windowing import slice_arrays_to_windows
from src.streaming.buffer import StreamingWindowBuffer
from src.streaming.source import MalformedFrameError, RecordedTrialSource, SensorFrame


class TestCanonicalContracts:
    """Validate canonical experiment contract definitions and leakage guards."""

    def test_canonical_contract_structure(self) -> None:
        contract = get_canonical_contract()
        assert len(contract.train_subjects) == 21
        assert len(contract.val_subjects) == 4
        assert len(contract.test_subjects) == 5
        assert len(contract.held_out_subjects) == 1
        assert contract.total_subject_count == 31
        assert contract.seed == 42
        assert contract.window_ms == 200.0
        assert contract.window_overlap == 0.5

    def test_pairwise_disjointness_no_leakage(self) -> None:
        contract = CANONICAL_PHASE3_CONTRACT
        train_s = set(contract.train_subjects)
        val_s = set(contract.val_subjects)
        test_s = set(contract.test_subjects)
        held_s = set(contract.held_out_subjects)

        assert train_s.isdisjoint(val_s)
        assert train_s.isdisjoint(test_s)
        assert train_s.isdisjoint(held_s)
        assert val_s.isdisjoint(test_s)
        assert val_s.isdisjoint(held_s)
        assert test_s.isdisjoint(held_s)

        all_union = train_s | val_s | test_s | held_s
        assert all_union == set(range(1, 32))

    def test_leakage_assertion_on_invalid_contract(self) -> None:
        with pytest.raises(ValueError, match="Subject leakage detected"):
            ExperimentContract(
                name="Leaky",
                seed=42,
                train_subjects=(2, 3, 4),
                val_subjects=(4, 5),  # 4 is leaked
                test_subjects=(6,),
                held_out_subjects=(1,),
                window_ms=200.0,
                window_overlap=0.5,
                expected_imu_samples=30,
                expected_emg_samples=252,
                tasks=("binary",),
                reporting_levels=("window",),
                required_metrics=("accuracy",),
            )

    def test_filter_manifest(self) -> None:
        contract = get_canonical_contract()
        df = pd.DataFrame({"subject_id": [1, 2, 4, 3, 99], "trial_id": [1, 2, 3, 4, 5]})
        train_df = contract.filter_manifest(df, split="train")
        assert list(train_df["subject_id"]) == [2]

        val_df = contract.filter_manifest(df, split="val")
        assert list(val_df["subject_id"]) == [4]

        test_df = contract.filter_manifest(df, split="test")
        assert list(test_df["subject_id"]) == [3]

        held_df = contract.filter_manifest(df, split="held_out")
        assert list(held_df["subject_id"]) == [1]

        bench_df = contract.filter_manifest(df, split="benchmark")
        assert set(bench_df["subject_id"]) == {2, 3, 4}


class TestSensorSource:
    """Validate streaming frames and RecordedTrialSource."""

    def test_sensor_frame_validation(self) -> None:
        # Valid frame
        imu = np.zeros((8, 6, 15), dtype=np.float32)
        emg = np.zeros((8, 126), dtype=np.float32)
        frame = SensorFrame(timestamp_s=0.0, imu_data=imu, emg_data=emg)
        assert frame.timestamp_s == 0.0

        # Non-finite values
        with pytest.raises(MalformedFrameError, match="non-finite"):
            bad_imu = imu.copy()
            bad_imu[0, 0, 0] = np.nan
            SensorFrame(timestamp_s=0.0, imu_data=bad_imu, emg_data=emg)

        # Invalid IMU channels
        with pytest.raises(MalformedFrameError, match="IMU channels"):
            bad_imu_shape = np.zeros((7, 6, 15), dtype=np.float32)
            SensorFrame(timestamp_s=0.0, imu_data=bad_imu_shape, emg_data=emg)

        # Invalid EMG channels
        with pytest.raises(MalformedFrameError, match="EMG channels"):
            bad_emg_shape = np.zeros((7, 126), dtype=np.float32)
            SensorFrame(timestamp_s=0.0, imu_data=imu, emg_data=bad_emg_shape)

    def test_recorded_trial_source_replay(self) -> None:
        t_imu = 300
        t_emg = int(round(t_imu / config.IMU_SAMPLING_RATE_HZ * config.EMG_SAMPLING_RATE_HZ))
        imu = np.random.randn(8, 6, t_imu).astype(np.float32)
        emg = np.random.randn(8, t_emg).astype(np.float32)

        trial = Trial(
            subject_id=5,
            label_id=2,
            trial_num=1,
            imu=imu,
            emg=emg,
            duration_s=t_imu / config.IMU_SAMPLING_RATE_HZ,
        )

        source = RecordedTrialSource(trial=trial, chunk_duration_ms=40.0, pacing_realtime=False)
        frames = list(source.stream_frames())
        assert len(frames) > 0
        assert frames[-1].is_last

        # Reconstructed signals match original
        reconstructed_imu = np.concatenate([f.imu_data for f in frames], axis=-1)
        reconstructed_emg = np.concatenate([f.emg_data for f in frames], axis=-1)

        np.testing.assert_array_equal(reconstructed_imu, imu)
        np.testing.assert_array_equal(reconstructed_emg, emg)


class TestStreamingWindowBufferParity:
    """Verify that streaming windowing has 100% bitwise parity with offline windowing."""

    @pytest.mark.parametrize("chunk_ms", [25.0, 50.0, 100.0, 133.0])
    def test_offline_streaming_window_parity(self, chunk_ms: float) -> None:
        rng = np.random.default_rng(12345)
        # Create a ~3 second synthetic trial
        n_imu_samples = 450
        n_emg_samples = int(
            round(n_imu_samples / config.IMU_SAMPLING_RATE_HZ * config.EMG_SAMPLING_RATE_HZ)
        )

        imu = rng.standard_normal((8, 6, n_imu_samples), dtype=np.float32)
        emg = rng.standard_normal((8, n_emg_samples), dtype=np.float32)

        # 1. Offline slicing reference
        offline_imu, offline_emg, offline_start_t, offline_end_t = slice_arrays_to_windows(
            imu=imu,
            emg=emg,
            window_ms=200.0,
            overlap=0.5,
        )

        # 2. Streaming buffer
        buffer = StreamingWindowBuffer(window_ms=200.0, overlap=0.5)
        trial = Trial(
            subject_id=7,
            label_id=0,
            trial_num=1,
            imu=imu,
            emg=emg,
            duration_s=n_imu_samples / config.IMU_SAMPLING_RATE_HZ,
        )
        source = RecordedTrialSource(trial=trial, chunk_duration_ms=chunk_ms)

        streamed_windows = []
        for frame in source.stream_frames():
            new_windows = buffer.push(
                imu_chunk=frame.imu_data,
                emg_chunk=frame.emg_data,
                subject_id=frame.subject_id,
                label_id=frame.label_id,
                trial_num=frame.trial_num,
            )
            streamed_windows.extend(new_windows)

        # 3. Assert exact parity
        assert len(streamed_windows) == len(offline_imu)
        assert len(streamed_windows) > 0

        streamed_imu = np.stack([w.imu for w in streamed_windows], axis=0)
        streamed_emg = np.stack([w.emg for w in streamed_windows], axis=0)
        streamed_start_t = np.array([w.start_time_s for w in streamed_windows])
        streamed_end_t = np.array([w.end_time_s for w in streamed_windows])

        np.testing.assert_allclose(streamed_imu, offline_imu, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(streamed_emg, offline_emg, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(streamed_start_t, offline_start_t, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(streamed_end_t, offline_end_t, rtol=1e-6, atol=1e-6)

        # Check metadata
        for idx, win in enumerate(streamed_windows):
            assert win.window_index == idx
            assert win.subject_id == 7
            assert win.label_id == 0
            assert win.trial_num == 1

    def test_buffer_reset(self) -> None:
        buffer = StreamingWindowBuffer()
        imu = np.zeros((8, 6, 50), dtype=np.float32)
        emg = np.zeros((8, 400), dtype=np.float32)
        wins = buffer.push(imu, emg)
        assert len(wins) > 0
        assert buffer.window_index > 0

        buffer.reset()
        assert buffer.window_index == 0
        assert buffer.total_imu_samples == 0
        assert buffer.total_emg_samples == 0
