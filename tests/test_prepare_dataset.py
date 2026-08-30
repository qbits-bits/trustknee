"""Tests for the explicit metadata preparation step."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.models.prepare_dataset import prepare_dataset


def _metadata_text() -> str:
    participant_header = (
        "Participant ID\tGender (M/F)\tHeight (CM)\tWeight(KG)\tAge (Years)\tLeg\tPathology"
    )
    participants = [f"{i}\tMale\t1.80\t80\t30\tRight\tNone" for i in range(1, 32)]
    labels = [f"{i}\t{'Correct' if i in {0, 3, 6} else 'Wrong'}\tlabel {i}" for i in range(9)]
    placement = [f"{i}\tmuscle {i}" for i in range(1, 9)]
    return "\n".join(
        [
            participant_header,
            *participants,
            "",
            "Label ID\tExecution\tDetails",
            *labels,
            "",
            "Sensor ID\tMuscle",
            *placement,
            "",
        ]
    )


def _write_valid_trial(trial_dir):
    trial_dir.mkdir(parents=True)
    np.save(trial_dir / "imu.npy", np.zeros((48, 30)))
    np.save(trial_dir / "emg.npy", np.zeros((8, 252)))


def test_prepare_dataset_writes_reader_layout_and_report(tmp_path):
    source = tmp_path / "source"
    source_dataset = source / "dataset"
    for subject in range(1, 32):
        _write_valid_trial(source_dataset / f"Subject_{subject}" / "0" / "Trial_1")
    for label in range(1, 9):
        _write_valid_trial(source_dataset / "Subject_1" / str(label) / "Trial_1")
    # The missing EMG file is reported but does not hide the valid trials.
    missing = source_dataset / "Subject_1" / "0" / "Trial_2"
    missing.mkdir(parents=True)
    np.save(missing / "imu.npy", np.zeros((48, 30)))
    metadata_file = tmp_path / "metadata.txt"
    metadata_file.write_text(_metadata_text(), encoding="utf-8")

    report = prepare_dataset(source, metadata_file, tmp_path / "prepared")

    assert report["valid_trials"] == 39
    assert len(report["missing_files"]) == 1
    assert (tmp_path / "prepared" / "dataset" / "Subject_1" / "0" / "Trial_1" / "imu.npy").exists()
    assert pd.read_csv(tmp_path / "prepared" / "participants.csv").shape == (31, 7)
    assert pd.read_csv(tmp_path / "prepared" / "labels.csv").shape == (9, 3)
    assert (
        json.loads((tmp_path / "prepared" / "preparation_report.json").read_text())["valid_trials"]
        == 39
    )


def test_prepare_dataset_refresh_removes_trials_absent_from_source(tmp_path):
    source = tmp_path / "source"
    source_dataset = source / "dataset"
    for subject in range(1, 32):
        _write_valid_trial(source_dataset / f"Subject_{subject}" / "0" / "Trial_1")
    for label in range(1, 9):
        _write_valid_trial(source_dataset / "Subject_1" / str(label) / "Trial_1")
    metadata_file = tmp_path / "metadata.txt"
    metadata_file.write_text(_metadata_text(), encoding="utf-8")
    prepared = tmp_path / "prepared"

    prepare_dataset(source, metadata_file, prepared)
    obsolete_trial = prepared / "dataset" / "Subject_1" / "0" / "Trial_99"
    _write_valid_trial(obsolete_trial)

    prepare_dataset(source, metadata_file, prepared)

    assert not obsolete_trial.exists()


def test_prepare_dataset_failure_preserves_previous_dataset_and_metadata(tmp_path):
    source = tmp_path / "source"
    source_dataset = source / "dataset"
    for subject in range(1, 32):
        _write_valid_trial(source_dataset / f"Subject_{subject}" / "0" / "Trial_1")
    for label in range(1, 9):
        _write_valid_trial(source_dataset / "Subject_1" / str(label) / "Trial_1")
    metadata_file = tmp_path / "metadata.txt"
    metadata_file.write_text(_metadata_text(), encoding="utf-8")
    prepared = tmp_path / "prepared"

    # Initial successful preparation
    prepare_dataset(source, metadata_file, prepared)
    original_participants = (prepared / "participants.csv").read_text(encoding="utf-8")
    sentinel_trial = prepared / "dataset" / "Subject_1" / "0" / "Trial_1" / "imu.npy"
    assert sentinel_trial.exists()

    # Create new source with an additional trial
    new_source = tmp_path / "new_source"
    new_dataset = new_source / "dataset"
    for subject in range(1, 32):
        _write_valid_trial(new_dataset / f"Subject_{subject}" / "0" / "Trial_1")
    for label in range(1, 9):
        _write_valid_trial(new_dataset / "Subject_1" / str(label) / "Trial_1")
    _write_valid_trial(new_dataset / "Subject_1" / "0" / "Trial_999")

    # Simulate a failure during metadata writing
    with (
        patch("pandas.DataFrame.to_csv", side_effect=OSError("Disk write failed")),
        pytest.raises(OSError, match="Disk write failed"),
    ):
        prepare_dataset(new_source, metadata_file, prepared)

    # Verify that the prepared directory still has the original metadata and dataset intact
    assert (prepared / "participants.csv").read_text(encoding="utf-8") == original_participants
    assert not (prepared / "dataset" / "Subject_1" / "0" / "Trial_999").exists()
    assert (prepared / "dataset" / "Subject_1" / "0" / "Trial_1" / "imu.npy").exists()
