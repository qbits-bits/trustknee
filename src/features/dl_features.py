"""Load raw IMU and EMG trials for sequence-model feature preparation."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingestion import build_manifest  # noqa: E402
from src.preprocessing.util import combine_signals  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trustknee.features")


def fetch_signal_features(path: Path, mode: str = "emg") -> pd.DataFrame:
    """Load signal arrays from the dataset manifest.

    Args:
        path: Dataset root containing the metadata CSV files and ``dataset`` directory.
        mode: Signal representation to load: ``"imu"``, ``"emg"``, or ``"combined"``.

    Returns:
        A DataFrame sorted by subject and trial, with one signal array per row.
    """
    rows = []
    logger.info("Scanning raw data under: %s", path)
    manifest = build_manifest(path)
    logger.info("Loaded manifest with %d trials.", len(manifest))

    for row_dict in manifest.to_dict(orient="records"):
        subject_id = row_dict["subject_id"]
        label_id = row_dict["label_id"]
        trial_num = row_dict["trial_num"]

        imu_path = Path(str(row_dict["imu_path"]))
        emg_path = Path(str(row_dict["emg_path"]))
        if mode == "emg" and emg_path.is_file():
            try:
                emg = np.load(emg_path)
                if emg.shape[-1] == 0:
                    logger.warning(
                        "Skipping trial (subject=%s, label=%s, trial=%s): Empty signal",
                        row_dict.get("subject_id"),
                        row_dict.get("label_id"),
                        row_dict.get("trial_num"),
                    )
                    continue
                rows.append(
                    {
                        "subject_id": subject_id,
                        "label_id": label_id,
                        "trial_num": trial_num,
                        "emg": emg,
                    }
                )
            except Exception as e:
                logger.error("Failed to load numpy array at %s: %s", emg_path, e)

        elif mode == "imu" and imu_path.is_file():
            try:
                imu = np.load(imu_path)
                if imu.shape[-1] == 0:
                    logger.warning(
                        "Skipping trial (subject=%s, label=%s, trial=%s): Empty signal",
                        row_dict.get("subject_id"),
                        row_dict.get("label_id"),
                        row_dict.get("trial_num"),
                    )
                    continue
                rows.append(
                    {
                        "subject_id": subject_id,
                        "label_id": label_id,
                        "trial_num": trial_num,
                        "imu": imu,
                    }
                )
            except Exception as e:
                logger.error("Failed to load numpy array at %s: %s", imu_path, e)

        elif mode == "combined" and imu_path.is_file() and emg_path.is_file():
            try:
                emg = np.load(emg_path)
                imu = np.load(imu_path)
                if emg.shape[-1] == 0 or imu.shape[-1] == 0:
                    logger.warning(
                        "Skipping trial (subject=%s, label=%s, trial=%s): Empty signal",
                        row_dict.get("subject_id"),
                        row_dict.get("label_id"),
                        row_dict.get("trial_num"),
                    )
                    continue
                processed_imu_emg = combine_signals(emg, imu)
                rows.append(
                    {
                        "subject_id": subject_id,
                        "label_id": label_id,
                        "trial_num": trial_num,
                        "emg_imu_combined": processed_imu_emg,
                    }
                )
            except Exception as e:
                logger.error("Failed to load numpy array at %s: %s", imu_path, e)
        else:
            logger.warning("Expected file missing during iteration: %s")

    signal_features = pd.DataFrame(rows)
    if not signal_features.empty:
        signal_features = signal_features.sort_values(
            by=["subject_id", "trial_num"], ascending=True
        )
        signal_features = signal_features.reset_index(drop=True)

    logger.info("Pipeline completed. Returned %s features", signal_features.shape)
    return signal_features


# if __name__ == "__main__":
#     project_root = Path(__file__).resolve().parents[2]

#     raw_data_dir = project_root / "data" / "raw"
#     output_dir = project_root / "data" / "processed" / ""
#     output_dir.mkdir(parents=True, exist_ok=True)

#     df = fetch_signal_features(raw_data_dir, "combined")
#     df.to_pickle(output_dir / "processed_emg_imu.pkl")
