"""Prepare local KneE-PAD metadata beside an extracted trial directory."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger("trustknee.prepare_dataset")

EXPECTED_SUBJECT_IDS = set(range(1, 32))
EXPECTED_LABEL_IDS = set(config.LABELS)


def _split_metadata_row(line: str, expected_columns: int) -> list[str] | None:
    parts = [part.strip() for part in line.split("\t")]
    if len(parts) == expected_columns:
        return parts
    # Some text extraction tools replace tabs with runs of spaces.  The source
    # fields themselves may contain single spaces, so only split on 2+ spaces.
    parts = [part.strip() for part in re.split(r"\s{2,}", line.strip())]
    return parts if len(parts) == expected_columns else None


def _extract_table(text: str, header_prefix: str, expected_columns: int) -> list[list[str]]:
    lines = text.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.strip().startswith(header_prefix)), None
    )
    if header_index is None:
        raise ValueError(f"Metadata table header not found: {header_prefix!r}")
    rows = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if stripped.startswith("Table "):
            break
        row = _split_metadata_row(stripped, expected_columns)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError(f"Metadata table contains no rows: {header_prefix!r}")
    return rows


def parse_metadata_file(metadata_file: Path | str) -> dict[str, pd.DataFrame]:
    """Parse the tabular text distributed with the dataset."""
    text = Path(metadata_file).read_text(encoding="utf-8")
    participant_rows = _extract_table(text, "Participant ID", 7)
    label_rows = _extract_table(text, "Label ID", 3)
    placement_rows = _extract_table(text, "Sensor ID", 2)

    participants = pd.DataFrame(
        participant_rows,
        columns=[
            "Participant ID",
            "Gender (M/F)",
            "Height (CM)",
            "Weight(KG)",
            "Age (Years)",
            "Leg",
            "Pathology",
        ],
    )
    participants["Participant ID"] = pd.to_numeric(participants["Participant ID"])
    for column in ["Height (CM)", "Weight(KG)", "Age (Years)"]:
        participants[column] = pd.to_numeric(participants[column])

    labels = pd.DataFrame(label_rows, columns=["Label ID", "Execution", "Details"])
    labels["Label ID"] = pd.to_numeric(labels["Label ID"])
    placement = pd.DataFrame(placement_rows, columns=["Sensor ID", "Muscle"])
    placement["Sensor ID"] = pd.to_numeric(placement["Sensor ID"])
    return {"participants": participants, "labels": labels, "placement": placement}


def _validate_metadata(metadata: dict[str, pd.DataFrame]) -> None:
    subject_ids = set(metadata["participants"]["Participant ID"].astype(int))
    label_ids = set(metadata["labels"]["Label ID"].astype(int))
    placement_ids = set(metadata["placement"]["Sensor ID"].astype(int))
    if subject_ids != EXPECTED_SUBJECT_IDS:
        raise ValueError(
            f"Metadata must contain subjects 1-31; missing={sorted(EXPECTED_SUBJECT_IDS - subject_ids)}, "
            f"extra={sorted(subject_ids - EXPECTED_SUBJECT_IDS)}"
        )
    if label_ids != EXPECTED_LABEL_IDS:
        raise ValueError(
            f"Metadata must contain label IDs 0-8; missing={sorted(EXPECTED_LABEL_IDS - label_ids)}, "
            f"extra={sorted(label_ids - EXPECTED_LABEL_IDS)}"
        )
    if placement_ids != set(config.SENSOR_PLACEMENT):
        raise ValueError("Metadata must contain sensor placement IDs 1-8")


def _resolve_dataset_dir(dataset_root: Path) -> Path:
    nested = dataset_root / "dataset"
    if nested.is_dir():
        return nested
    if any(dataset_root.glob("Subject_*")):
        return dataset_root
    raise FileNotFoundError(
        f"No extracted dataset found under {dataset_root}; expected dataset/Subject_<id>/..."
    )


def _replace_dataset_tree(
    source_dataset: Path,
    output_dir: Path,
    metadata: dict[str, pd.DataFrame],
    report: dict[str, object],
) -> None:
    """Stage dataset and all metadata files in a temporary directory, then atomically commit them."""
    output_dataset = output_dir / "dataset"
    same_dataset = source_dataset == output_dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".trustknee-prepare-", dir=output_dir.parent
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        staged_dataset = temporary_root / "dataset"
        staged_participants = temporary_root / "participants.csv"
        staged_labels = temporary_root / "labels.csv"
        staged_placement = temporary_root / "placement.csv"
        staged_sensors = temporary_root / "sensors.csv"
        staged_report = temporary_root / "preparation_report.json"

        # Stage metadata files and report first
        metadata["participants"].to_csv(staged_participants, index=False)
        metadata["labels"].to_csv(staged_labels, index=False)
        metadata["placement"].to_csv(staged_placement, index=False)
        pd.DataFrame(
            [
                ["n_sensors", config.N_SENSORS],
                ["emg_channels_per_sensor", config.EMG_CHANNELS_PER_SENSOR],
                ["imu_channels_per_sensor", config.IMU_CHANNELS_PER_SENSOR],
                ["emg_sampling_rate_hz", config.EMG_SAMPLING_RATE_HZ],
                ["imu_sampling_rate_hz", config.IMU_SAMPLING_RATE_HZ],
                ["emg_unit", config.EMG_UNIT],
                ["accel_unit", config.ACCEL_UNIT],
                ["gyro_unit", config.GYRO_UNIT],
            ]
        ).to_csv(staged_sensors, index=False, header=False)
        staged_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Stage dataset tree if not operating in-place
        if not same_dataset:
            shutil.copytree(source_dataset, staged_dataset)

        dest_participants = output_dir / "participants.csv"
        dest_labels = output_dir / "labels.csv"
        dest_placement = output_dir / "placement.csv"
        dest_sensors = output_dir / "sensors.csv"
        dest_report = output_dir / "preparation_report.json"

        backup_dir = temporary_root / "backup"
        backup_dir.mkdir(exist_ok=True)

        backups: dict[Path, Path] = {}
        files_to_swap = [
            (staged_participants, dest_participants, backup_dir / "participants.csv"),
            (staged_labels, dest_labels, backup_dir / "labels.csv"),
            (staged_placement, dest_placement, backup_dir / "placement.csv"),
            (staged_sensors, dest_sensors, backup_dir / "sensors.csv"),
            (staged_report, dest_report, backup_dir / "preparation_report.json"),
        ]
        if not same_dataset:
            files_to_swap.append((staged_dataset, output_dataset, backup_dir / "dataset"))

        for _, dest, backup in files_to_swap:
            if dest.exists():
                dest.replace(backup)
                backups[dest] = backup

        installed: list[Path] = []
        try:
            for staged, dest, _ in files_to_swap:
                staged.replace(dest)
                installed.append(dest)
        except Exception:
            for _, dest, backup in files_to_swap:
                if dest in backups:
                    with contextlib.suppress(Exception):
                        if dest.exists():
                            if dest.is_dir():
                                shutil.rmtree(dest)
                            else:
                                dest.unlink()
                        backup.replace(dest)
                elif dest in installed:
                    with contextlib.suppress(Exception):
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
            raise


def _trial_validation(dataset_dir: Path) -> dict[str, object]:
    valid = 0
    missing_files: list[str] = []
    malformed: list[dict[str, str]] = []
    observed_subjects: set[int] = set()
    observed_labels: set[int] = set()
    expected_imu_rows = config.N_SENSORS * config.IMU_CHANNELS_PER_SENSOR

    for subject_dir in sorted(dataset_dir.glob("Subject_*")):
        if not subject_dir.is_dir():
            continue
        try:
            subject_id = int(subject_dir.name.removeprefix("Subject_"))
        except ValueError:
            continue
        observed_subjects.add(subject_id)
        for label_dir in sorted(subject_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            try:
                label_id = int(label_dir.name)
            except ValueError:
                continue
            observed_labels.add(label_id)
            for trial_dir in sorted(label_dir.glob("Trial_*")):
                if not trial_dir.is_dir():
                    continue
                trial_name = str(trial_dir.relative_to(dataset_dir))
                imu_path, emg_path = trial_dir / "imu.npy", trial_dir / "emg.npy"
                if not imu_path.exists() or not emg_path.exists():
                    missing_files.append(trial_name)
                    continue
                try:
                    imu = np.load(imu_path, mmap_mode="r")
                    emg = np.load(emg_path, mmap_mode="r")
                    if imu.ndim != 2 or imu.shape[0] != expected_imu_rows:
                        raise ValueError(
                            f"IMU shape {imu.shape}, expected ({expected_imu_rows}, T)"
                        )
                    if emg.ndim != 2 or emg.shape[0] != config.N_SENSORS:
                        raise ValueError(f"EMG shape {emg.shape}, expected ({config.N_SENSORS}, T)")
                    if imu.shape[-1] == 0 or emg.shape[-1] == 0:
                        raise ValueError("empty signal")
                    imu_duration = imu.shape[-1] / config.IMU_SAMPLING_RATE_HZ
                    emg_duration = emg.shape[-1] / config.EMG_SAMPLING_RATE_HZ
                    if abs(imu_duration - emg_duration) > 0.05:
                        raise ValueError(
                            f"duration mismatch ({imu_duration:.3f}s vs {emg_duration:.3f}s)"
                        )
                except (OSError, ValueError) as exc:
                    malformed.append({"trial": trial_name, "reason": str(exc)})
                    continue
                valid += 1

    missing_subjects = sorted(EXPECTED_SUBJECT_IDS - observed_subjects)
    missing_labels = sorted(EXPECTED_LABEL_IDS - observed_labels)
    return {
        "valid_trials": valid,
        "missing_files": missing_files,
        "malformed_trials": malformed,
        "observed_subjects": sorted(observed_subjects),
        "observed_labels": sorted(observed_labels),
        "missing_subjects": missing_subjects,
        "missing_labels": missing_labels,
    }


def prepare_dataset(
    dataset_root: Path | str,
    metadata_file: Path | str,
    output_dir: Path | str,
) -> dict[str, object]:
    """Copy trial data and write the four metadata files expected by ingestion."""
    dataset_root = Path(dataset_root).resolve()
    metadata_file = Path(metadata_file).resolve()
    output_dir = Path(output_dir).resolve()
    metadata = parse_metadata_file(metadata_file)
    _validate_metadata(metadata)
    source_dataset = _resolve_dataset_dir(dataset_root)
    trial_report = _trial_validation(source_dataset)
    if not trial_report["valid_trials"]:
        raise ValueError("No valid trials found; see the dataset source and metadata format")
    if trial_report["missing_subjects"] or trial_report["missing_labels"]:
        raise ValueError(
            "Dataset does not represent all expected subjects/labels: "
            f"missing subjects={trial_report['missing_subjects']}, "
            f"missing labels={trial_report['missing_labels']}"
        )

    report = {
        "source_dataset": str(source_dataset),
        "metadata_file": str(metadata_file),
        "output_dir": str(output_dir),
        **trial_report,
    }
    _replace_dataset_tree(source_dataset, output_dir, metadata, report)

    if trial_report["missing_files"] or trial_report["malformed_trials"]:
        logger.warning(
            "Prepared data with %d missing-file trials and %d malformed trials; ingestion will skip them.",
            len(trial_report["missing_files"]),
            len(trial_report["malformed_trials"]),
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare local KneE-PAD data for TrustKnee")
    parser.add_argument(
        "--dataset-root", type=Path, required=True, help="Extracted dataset or its parent"
    )
    parser.add_argument(
        "--metadata-file", type=Path, required=True, help="KneE-PAD metadata text file"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Prepared data root, e.g. data/raw"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        report = prepare_dataset(args.dataset_root, args.metadata_file, args.output_dir)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"Dataset preparation blocked: {exc}") from exc
    logger.info(
        "Prepared %d valid trials for %d subjects at %s",
        report["valid_trials"],
        len(report["observed_subjects"]),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    main()
