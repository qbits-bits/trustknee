"""Canonical Phase 3 experiment contract and subject split standards.

Defines the reproducible baseline contract used across all models (XGBoost,
Transformer, Minimal-Sensor studies) to guarantee identical data splits, window
parameters, label mappings, and evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src import config


@dataclass(frozen=True)
class ExperimentContract:
    """Immutable specification for Phase 3 benchmarking experiments."""

    name: str
    seed: int
    train_subjects: tuple[int, ...]
    val_subjects: tuple[int, ...]
    test_subjects: tuple[int, ...]
    held_out_subjects: tuple[int, ...]
    window_ms: float
    window_overlap: float
    expected_imu_samples: int
    expected_emg_samples: int
    tasks: tuple[str, ...]
    reporting_levels: tuple[str, ...]
    required_metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate subject isolation and coverage."""
        all_groups = [
            set(self.train_subjects),
            set(self.val_subjects),
            set(self.test_subjects),
            set(self.held_out_subjects),
        ]
        # Check pairwise disjointness
        for i in range(len(all_groups)):
            for j in range(i + 1, len(all_groups)):
                overlap = all_groups[i].intersection(all_groups[j])
                if overlap:
                    raise ValueError(
                        f"Subject leakage detected between split partitions: {overlap}"
                    )

    @property
    def all_benchmark_subjects(self) -> tuple[int, ...]:
        """All subjects used during training, validation, and testing."""
        return tuple(
            sorted(set(self.train_subjects) | set(self.val_subjects) | set(self.test_subjects))
        )

    @property
    def total_subject_count(self) -> int:
        """Total subjects including the held-out subject."""
        return len(self.all_benchmark_subjects) + len(self.held_out_subjects)

    def get_subject_split(self, subject_id: int) -> Literal["train", "val", "test", "held_out"]:
        """Return the partition name for a given subject ID."""
        if subject_id in self.train_subjects:
            return "train"
        if subject_id in self.val_subjects:
            return "val"
        if subject_id in self.test_subjects:
            return "test"
        if subject_id in self.held_out_subjects:
            return "held_out"
        raise ValueError(f"Subject ID {subject_id} is not recognized in experiment contract")

    def filter_manifest(
        self,
        manifest: pd.DataFrame,
        split: Literal["train", "val", "test", "held_out", "benchmark"],
    ) -> pd.DataFrame:
        """Filter a trial manifest DataFrame to only include subjects for a specific partition."""
        if "subject_id" not in manifest.columns:
            raise KeyError("Manifest must contain a 'subject_id' column")

        if split == "train":
            allowed = set(self.train_subjects)
        elif split == "val":
            allowed = set(self.val_subjects)
        elif split == "test":
            allowed = set(self.test_subjects)
        elif split == "held_out":
            allowed = set(self.held_out_subjects)
        elif split == "benchmark":
            allowed = set(self.all_benchmark_subjects)
        else:
            raise ValueError(
                f"Invalid split '{split}'. Expected train, val, test, held_out, or benchmark."
            )

        filtered = manifest[manifest["subject_id"].isin(allowed)].copy()
        return filtered.reset_index(drop=True)


CANONICAL_PHASE3_CONTRACT = ExperimentContract(
    name="Phase-3-Canonical",
    seed=config.PHASE3_RANDOM_SEED,
    train_subjects=config.PHASE3_TRAIN_SUBJECTS,
    val_subjects=config.PHASE3_VAL_SUBJECTS,
    test_subjects=config.PHASE3_TEST_SUBJECTS,
    held_out_subjects=config.PHASE3_HELD_OUT_SUBJECTS,
    window_ms=config.WINDOW_MS,
    window_overlap=config.WINDOW_OVERLAP,
    expected_imu_samples=30,
    expected_emg_samples=252,
    tasks=("binary", "nine_class"),
    reporting_levels=("window", "trial"),
    required_metrics=(
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "per_class_precision",
        "per_class_recall",
        "confusion_matrix",
        "ece",
        "brier_score",
        "inference_latency_ms",
    ),
)


def get_canonical_contract() -> ExperimentContract:
    """Return the frozen canonical Phase 3 experiment contract."""
    return CANONICAL_PHASE3_CONTRACT
