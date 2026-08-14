"""Ingestion for raw IMU/EMG trials and dataset metadata.

Layout:
    <data_root>/participants.csv
    <data_root>/labels.csv
    <data_root>/placement.csv
    <data_root>/sensors.csv
    <data_root>/dataset/Subject_<id>/<label_id>/Trial_<n>/imu.npy   (48, T_imu)
    <data_root>/dataset/Subject_<id>/<label_id>/Trial_<n>/emg.npy   (8, T_emg)

label_id has no prefix, unlike Subject_/Trial_. IMU and EMG are already
time-aligned (same real duration, different sampling rates), so we never
resample, only check the two durations match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger("trustknee.ingestion")


def load_participants(data_root: Path | str) -> pd.DataFrame:
    """Load participants.csv, indexed by Participant ID."""
    path = Path(data_root) / "participants.csv"
    df = pd.read_csv(path)
    return df.set_index("Participant ID")


def load_labels(data_root: Path | str) -> pd.DataFrame:
    """Load labels.csv, indexed by Label ID."""
    path = Path(data_root) / "labels.csv"
    df = pd.read_csv(path)
    return df.set_index("Label ID")


def load_placement(data_root: Path | str) -> pd.DataFrame:
    """Load placement.csv, indexed by Sensor ID."""
    path = Path(data_root) / "placement.csv"
    df = pd.read_csv(path)
    return df.set_index("Sensor ID")


def load_sensor_config(data_root: Path | str) -> pd.Series:
    """Read sensors.csv as a two-column (name, value) table, indexed by name."""
    path = Path(data_root) / "sensors.csv"
    df = pd.read_csv(path, header=None, names=["parameter", "value"])
    return df.set_index("parameter")["value"]


@dataclass
class Trial:
    subject_id: int
    label_id: int
    trial_num: int
    imu: np.ndarray  # (N_SENSORS, IMU_CHANNELS_PER_SENSOR, T_imu)
    emg: np.ndarray  # (N_SENSORS, T_emg)
    duration_s: float


def load_trial(
    imu_path: Path | str,
    emg_path: Path | str,
    subject_id: int = -1,
    label_id: int = -1,
    trial_num: int = -1,
    duration_tolerance_s: float = 0.05,
) -> Trial:
    """Load one trial's imu.npy/emg.npy, validate shapes, check durations match."""
    imu_raw = np.load(imu_path)
    emg_raw = np.load(emg_path)

    if imu_raw.size == 0 or emg_raw.size == 0 or imu_raw.shape[-1] == 0 or emg_raw.shape[-1] == 0:
        raise ValueError(
            f"Empty signal in trial subject={subject_id} label={label_id} trial={trial_num}: "
            f"imu shape={imu_raw.shape}, emg shape={emg_raw.shape}"
        )

    expected_imu_rows = config.N_SENSORS * config.IMU_CHANNELS_PER_SENSOR
    if imu_raw.shape[0] != expected_imu_rows:
        raise ValueError(
            f"{imu_path}: expected {expected_imu_rows} IMU rows "
            f"({config.N_SENSORS} sensors x {config.IMU_CHANNELS_PER_SENSOR} channels), "
            f"got shape {imu_raw.shape}"
        )
    if emg_raw.shape[0] != config.N_SENSORS:
        raise ValueError(
            f"{emg_path}: expected {config.N_SENSORS} EMG rows, got shape {emg_raw.shape}"
        )

    imu = imu_raw.reshape(config.N_SENSORS, config.IMU_CHANNELS_PER_SENSOR, -1)
    emg = emg_raw

    imu_duration_s = imu.shape[-1] / config.IMU_SAMPLING_RATE_HZ
    emg_duration_s = emg.shape[-1] / config.EMG_SAMPLING_RATE_HZ
    if abs(imu_duration_s - emg_duration_s) > duration_tolerance_s:
        raise ValueError(
            f"IMU/EMG duration mismatch for subject={subject_id} label={label_id} "
            f"trial={trial_num}: imu={imu_duration_s:.3f}s vs emg={emg_duration_s:.3f}s "
            f"(tolerance={duration_tolerance_s}s)"
        )

    return Trial(
        subject_id=subject_id,
        label_id=label_id,
        trial_num=trial_num,
        imu=imu,
        emg=emg,
        duration_s=imu_duration_s,
    )


def load_trial_from_manifest_row(row) -> Trial:
    """Convenience wrapper: load a Trial from a build_manifest() DataFrame row."""
    return load_trial(
        imu_path=row["imu_path"],
        emg_path=row["emg_path"],
        subject_id=row["subject_id"],
        label_id=row["label_id"],
        trial_num=row["trial_num"],
    )


def build_manifest(data_root: Path | str) -> pd.DataFrame:
    """Walk <data_root>/dataset/Subject_*/<label_id>/Trial_*/ and build a trial manifest."""
    data_root = Path(data_root)
    dataset_dir = data_root / "dataset"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    participants = load_participants(data_root)
    labels = load_labels(data_root)

    rows = []
    for subject_dir in sorted(dataset_dir.glob("Subject_*")):
        if not subject_dir.is_dir():
            continue
        try:
            subject_id = int(subject_dir.name.removeprefix("Subject_"))
        except ValueError:
            logger.warning("Skipping unparseable subject dir: %s", subject_dir)
            continue

        for label_dir in sorted(subject_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            try:
                label_id = int(label_dir.name)
            except ValueError:
                logger.warning("Skipping unparseable label dir: %s", label_dir)
                continue

            for trial_dir in sorted(label_dir.glob("Trial_*")):
                if not trial_dir.is_dir():
                    continue
                try:
                    trial_num = int(trial_dir.name.removeprefix("Trial_"))
                except ValueError:
                    logger.warning("Skipping unparseable trial dir: %s", trial_dir)
                    continue

                imu_path = trial_dir / "imu.npy"
                emg_path = trial_dir / "emg.npy"
                if not imu_path.exists() or not emg_path.exists():
                    logger.warning("Missing imu.npy/emg.npy in %s, skipping", trial_dir)
                    continue

                participant = (
                    participants.loc[subject_id] if subject_id in participants.index else None
                )
                label_row = labels.loc[label_id] if label_id in labels.index else None
                label_cfg = config.LABELS.get(label_id)

                rows.append(
                    {
                        "subject_id": subject_id,
                        "label_id": label_id,
                        "trial_num": trial_num,
                        "imu_path": imu_path,
                        "emg_path": emg_path,
                        "execution": label_row["Execution"] if label_row is not None else None,
                        "exercise": label_cfg.exercise if label_cfg is not None else None,
                        "description": label_row["Details"] if label_row is not None else None,
                        "gender": participant["Gender (M/F)"] if participant is not None else None,
                        # participants.csv header says CM but the values are
                        # meters (e.g. 1.82). Convert so height_cm is real cm.
                        "height_cm": participant["Height (CM)"] * 100
                        if participant is not None
                        else None,
                        "weight_kg": participant["Weight(KG)"] if participant is not None else None,
                        "age_years": participant["Age (Years)"]
                        if participant is not None
                        else None,
                        "injured_leg": participant["Leg"] if participant is not None else None,
                        "pathology": participant["Pathology"] if participant is not None else None,
                    }
                )

    if not rows:
        raise RuntimeError(f"No trials found under {dataset_dir}")

    return pd.DataFrame(rows)
