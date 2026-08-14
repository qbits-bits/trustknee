"""End-to-end smoke tests for TrustKnee pipeline:
- Manifest construction and metadata validation
- Preprocessing and low-pass filtering (preserving gravity DC)
- Sliding-window segmentation across multi-rate IMU and EMG
- Feature extraction producing standardized ML feature matrix
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import build_feature_matrix, extract_trial_features, extract_window_features
from src.ingestion import build_manifest, load_trial, load_trial_from_manifest_row
from src.preprocessing import (
    calculate_window_sample_counts,
    correct_drift_imu,
    preprocess_emg,
    preprocess_imu,
    slice_arrays_to_windows,
    slice_trial_windows,
)

RNG = np.random.default_rng(42)


def _write_csvs(data_root):
    pd.DataFrame(
        [
            {
                "Participant ID": 1,
                "Gender (M/F)": "M",
                "Height (CM)": 1.82,  # matches real file: mislabeled, actually meters
                "Weight(KG)": 78.0,
                "Age (Years)": 24,
                "Leg": "Right",
                "Pathology": "ACL",
            },
            {
                "Participant ID": 2,
                "Gender (M/F)": "F",
                "Height (CM)": 1.60,  # matches real file: mislabeled, actually meters
                "Weight(KG)": 55.0,
                "Age (Years)": 29,
                "Leg": "Left",
                "Pathology": "None",
            },
        ]
    ).to_csv(data_root / "participants.csv", index=False)

    pd.DataFrame(
        [
            {"Label ID": 0, "Execution": "Correct", "Details": "Squat"},
            {"Label ID": 3, "Execution": "Correct", "Details": "Seated leg extension"},
        ]
    ).to_csv(data_root / "labels.csv", index=False)

    pd.DataFrame(
        [{"Sensor ID": sid, "Muscle": muscle} for sid, muscle in config.SENSOR_PLACEMENT.items()]
    ).to_csv(data_root / "placement.csv", index=False)

    with open(data_root / "sensors.csv", "w") as f:
        f.write("n_sensors,8\n")
        f.write("emg_sampling_rate_hz,1259.2592592592594\n")
        f.write("imu_sampling_rate_hz,148.14814814814815\n")


def _write_trial(trial_dir, imu_t, emg_t, offset=0.0):
    trial_dir.mkdir(parents=True, exist_ok=True)
    imu = RNG.normal(size=(48, imu_t)) + offset
    emg = RNG.normal(size=(8, emg_t))
    np.save(trial_dir / "imu.npy", imu)
    np.save(trial_dir / "emg.npy", emg)


def _build_fake_dataset(data_root):
    _write_csvs(data_root)
    dataset = data_root / "dataset"
    # Subject 1, label 0, trial 1: exact real observed shapes.
    _write_trial(dataset / "Subject_1" / "0" / "Trial_1", imu_t=1440, emg_t=12237, offset=5.0)
    # Subject 1, label 3, trial 1: smaller matching trial.
    _write_trial(dataset / "Subject_1" / "3" / "Trial_1", imu_t=300, emg_t=2550)
    # Subject 2, label 0, trial 1: second subject, same label.
    _write_trial(dataset / "Subject_2" / "0" / "Trial_1", imu_t=300, emg_t=2550)


def test_duration_alignment_matches_expected():
    imu_duration = 1440 / config.IMU_SAMPLING_RATE_HZ
    emg_duration = 12237 / config.EMG_SAMPLING_RATE_HZ
    assert imu_duration == pytest.approx(9.72, abs=0.01)
    assert emg_duration == pytest.approx(9.72, abs=0.01)
    assert abs(imu_duration - emg_duration) < 0.05


def test_manifest_and_pipeline_end_to_end(tmp_path):
    _build_fake_dataset(tmp_path)

    manifest = build_manifest(tmp_path)
    assert len(manifest) == 3
    assert set(manifest["subject_id"]) == {1, 2}
    assert set(manifest["label_id"]) == {0, 3}
    assert set(manifest.columns) == {
        "subject_id",
        "label_id",
        "trial_num",
        "imu_path",
        "emg_path",
        "execution",
        "exercise",
        "description",
        "gender",
        "height_cm",
        "weight_kg",
        "age_years",
        "injured_leg",
        "pathology",
    }
    # demographics joined correctly
    subj1_row = manifest[manifest["subject_id"] == 1].iloc[0]
    # 1.82 raw (meters) should come out as 182.0 (true cm), see ingest.py.
    assert subj1_row["height_cm"] == pytest.approx(182.0)
    assert subj1_row["exercise"] == "Squat"

    row = manifest[(manifest["subject_id"] == 1) & (manifest["label_id"] == 0)].iloc[0]
    trial = load_trial_from_manifest_row(row)
    assert trial.imu.shape == (config.N_SENSORS, config.IMU_CHANNELS_PER_SENSOR, 1440)
    assert trial.emg.shape == (config.N_SENSORS, 12237)
    assert trial.duration_s == pytest.approx(9.72, abs=0.01)

    imu_filtered = preprocess_imu(trial.imu)
    emg_filtered = preprocess_emg(trial.emg)

    assert imu_filtered.shape == trial.imu.shape
    assert emg_filtered.shape == trial.emg.shape
    assert np.all(np.isfinite(imu_filtered))
    assert np.all(np.isfinite(emg_filtered))

    # By default, preprocess_imu preserves the constant offset (DC gravity component)
    raw_mean_offset = trial.imu.mean()
    assert np.isclose(imu_filtered.mean(), raw_mean_offset, atol=0.1)

    # When explicit mean subtraction drift correction is requested, offset is centered to ~0
    drift_corrected = correct_drift_imu(trial.imu, method="mean_subtract")
    assert np.isclose(drift_corrected.mean(), 0.0, atol=1e-6)


def test_gyro_only_drift_correction():
    # Shape: (8 sensors, 6 channels, 1000 samples)
    imu = RNG.normal(size=(8, 6, 1000)) + 3.0
    corrected = correct_drift_imu(imu, method="gyro_only")
    # Accelerometer channels (0:3) retain mean ~3.0
    assert np.isclose(corrected[:, 0:3, :].mean(), 3.0, atol=0.2)
    # Gyro channels (3:6) are highpass-filtered around 0.0
    assert np.isclose(corrected[:, 3:6, :].mean(), 0.0, atol=0.2)


def test_windowing_and_sample_counts():
    imu_win_samp, imu_step_samp, emg_win_samp, emg_step_samp = calculate_window_sample_counts(
        window_ms=200.0, overlap=0.5
    )
    # 200 ms at ~148.15 Hz -> ~30 samples
    assert imu_win_samp == 30
    assert imu_step_samp == 15
    # 200 ms at ~1259.26 Hz -> ~252 samples
    assert emg_win_samp == 252
    assert emg_step_samp == 126


def test_windowing_slices_and_metadata_alignment(tmp_path):
    _build_fake_dataset(tmp_path)
    manifest = build_manifest(tmp_path)
    row = manifest[(manifest["subject_id"] == 1) & (manifest["label_id"] == 0)].iloc[0]
    trial = load_trial_from_manifest_row(row)

    windowed_trial = slice_trial_windows(
        trial=trial,
        window_ms=200.0,
        overlap=0.5,
        preprocess=True,
        execution=row["execution"],
        exercise=row["exercise"],
    )

    assert windowed_trial.n_windows > 0
    assert windowed_trial.imu_windows.shape == (
        windowed_trial.n_windows,
        config.N_SENSORS,
        config.IMU_CHANNELS_PER_SENSOR,
        30,
    )
    assert windowed_trial.emg_windows.shape == (
        windowed_trial.n_windows,
        config.N_SENSORS,
        252,
    )
    assert len(windowed_trial.metadata) == windowed_trial.n_windows
    assert list(windowed_trial.metadata.columns) == [
        "window_index",
        "start_time_s",
        "end_time_s",
        "subject_id",
        "label_id",
        "trial_num",
        "execution",
        "exercise",
    ]
    # Check start / end duration
    first_row = windowed_trial.metadata.iloc[0]
    assert first_row["start_time_s"] == 0.0
    assert first_row["end_time_s"] == pytest.approx(0.200)


def test_multimodal_window_synchronization_drift():
    # 9.72s trial (1440 IMU samples, 12237 EMG samples)
    imu_raw = np.zeros((config.N_SENSORS, config.IMU_CHANNELS_PER_SENSOR, 1440))
    emg_raw = np.zeros((config.N_SENSORS, 12237))

    imu_wins, emg_wins, start_times, _ = slice_arrays_to_windows(
        imu=imu_raw,
        emg=emg_raw,
        window_ms=200.0,
        overlap=0.5,
    )

    n_wins = len(start_times)
    assert n_wins > 90

    step_sec = 0.200 * (1.0 - 0.5)
    for k in range(n_wins):
        nominal_t = start_times[k]
        assert nominal_t == pytest.approx(k * step_sec)

        # Derived actual start times from sample indices
        imu_sample_idx = int(round(nominal_t * config.IMU_SAMPLING_RATE_HZ))
        emg_sample_idx = int(round(nominal_t * config.EMG_SAMPLING_RATE_HZ))

        actual_imu_t = imu_sample_idx / config.IMU_SAMPLING_RATE_HZ
        actual_emg_t = emg_sample_idx / config.EMG_SAMPLING_RATE_HZ

        # Timing offset from nominal must stay within sub-sample tolerance (< 4 ms)
        assert abs(actual_imu_t - nominal_t) < (0.5 / config.IMU_SAMPLING_RATE_HZ + 1e-6)
        assert abs(actual_emg_t - nominal_t) < (0.5 / config.EMG_SAMPLING_RATE_HZ + 1e-6)

        # Relative desynchronization between IMU and EMG must never accumulate or exceed ~4 ms
        modality_offset_ms = abs(actual_imu_t - actual_emg_t) * 1000.0
        assert modality_offset_ms < 4.0


def test_feature_extraction_produces_valid_matrix(tmp_path):
    _build_fake_dataset(tmp_path)
    manifest = build_manifest(tmp_path)

    # 1. Test single trial feature extraction
    row = manifest.iloc[0]
    trial = load_trial_from_manifest_row(row)
    windowed = slice_trial_windows(trial, window_ms=200.0, overlap=0.5, execution="Correct", exercise="Squat")
    trial_feats_df = extract_trial_features(windowed)

    assert not trial_feats_df.empty
    assert len(trial_feats_df) == windowed.n_windows
    # Check that ROM, jerk, peak angular velocity, RMS, and EMG stats exist
    assert "s1_acc_x_rom" in trial_feats_df.columns
    assert "s1_acc_x_jerk" in trial_feats_df.columns
    assert "s1_gyro_peak_angular_vel" in trial_feats_df.columns
    assert "s1_emg_mav" in trial_feats_df.columns
    assert "s1_emg_wl" in trial_feats_df.columns
    assert np.all(np.isfinite(trial_feats_df.select_dtypes(include=[np.number]).values))

    # 2. Test full manifest aggregation
    full_matrix_df = build_feature_matrix(manifest, window_ms=200.0, overlap=0.5)
    assert len(full_matrix_df) > len(trial_feats_df)
    assert "subject_id" in full_matrix_df.columns
    assert "label_id" in full_matrix_df.columns


def test_load_trial_raises_on_duration_mismatch(tmp_path):
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    # emg truncated far below what 1440 IMU samples' duration implies.
    _write_trial(trial_dir, imu_t=1440, emg_t=100)
    with pytest.raises(ValueError, match="duration mismatch"):
        load_trial(trial_dir / "imu.npy", trial_dir / "emg.npy")


def test_correct_drift_imu_unknown_method_raises():
    imu = RNG.normal(size=(8, 6, 100))
    with pytest.raises(ValueError, match="Unknown drift correction method"):
        correct_drift_imu(imu, method="bogus")
