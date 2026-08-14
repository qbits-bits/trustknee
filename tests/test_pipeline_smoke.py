"""End-to-end smoke test: fabricate a tiny fake KneE-PAD dataset on disk,
then run build_manifest -> load_trial_from_manifest_row -> preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.ingestion import build_manifest, load_trial, load_trial_from_manifest_row
from src.preprocessing import correct_drift_imu, preprocess_emg, preprocess_imu

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

    # Drift correction should reduce the injected constant-offset mean magnitude.
    raw_mean_mag = np.abs(trial.imu.mean(axis=-1)).mean()
    drift_corrected = correct_drift_imu(trial.imu)
    corrected_mean_mag = np.abs(drift_corrected.mean(axis=-1)).mean()
    assert corrected_mean_mag < raw_mean_mag


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
