"""Leakage-safe Sequential Backward Elimination with Subject-Level Cross-Validation for TrustKnee sensors."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.contracts import get_canonical_contract  # noqa: E402
from src.ingestion import build_manifest  # noqa: E402
from src.models.evaluation import fixed_subject_split  # noqa: E402
from src.models.model_data import ModelDataset, build_model_inputs  # noqa: E402
from src.models.training import (  # noqa: E402
    QUICK_TRAINING,
    TrainingConfig,
    fit_xgboost,
    predict_xgboost,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trustknee.models.sbe")

AGGREGATION_METHODS = ("mean_probability", "majority_vote", "confidence_weighted")


def feature_sensor_ids(feature_name: str) -> set[int]:
    """Return all one-based physical sensor IDs used by a feature."""
    sensors = {int(value) for value in re.findall(r"s([1-8])_", feature_name)}
    pair = re.search(r"(?:sym_ratio_[a-z]+_|inter_sym_ctrl_)([1-8])([1-8])", feature_name)
    if pair:
        sensors.update((int(pair.group(1)), int(pair.group(2))))
    single = re.search(r"(?:ctrl_ratio_|loadshare_[a-z]+_)([1-8])(?:$|_)", feature_name)
    if single:
        sensors.add(int(single.group(1)))
    return sensors


def mask_tabular_features(
    features: pd.DataFrame,
    active_sensors: tuple[int, ...] | list[int],
) -> pd.DataFrame:
    """Keep only features whose complete sensor dependency is active."""
    active = {int(sensor) for sensor in active_sensors}
    if not active or any(sensor < 1 or sensor > config.N_SENSORS for sensor in active):
        raise ValueError("active_sensors must contain one-based IDs from 1 through 8")
    selected = []
    for column in features.columns:
        dependencies = feature_sensor_ids(str(column))
        if dependencies:
            if dependencies <= active:
                selected.append(column)
        elif str(column).startswith("lrshare_"):
            continue
        else:
            selected.append(column)
    if not selected:
        raise ValueError("Sensor mask removed every tabular feature")
    return features[selected].copy()


def aggregate_trial_probabilities(
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate aligned window probabilities using one selected rule."""
    if method not in AGGREGATION_METHODS:
        raise ValueError(f"Unknown aggregation method: {method}")
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    local = metadata.reset_index(drop=True)
    if values.ndim != 2 or len(values) != len(local) or len(targets) != len(local):
        raise ValueError("Probabilities, labels, and metadata must be aligned")
    trial_probabilities = []
    trial_labels = []
    for _, group in local.groupby("trial_id", sort=True):
        indices = group.index.to_numpy()
        group_values = values[indices]
        if method == "mean_probability":
            aggregated = group_values.mean(axis=0)
        elif method == "majority_vote":
            counts = np.bincount(group_values.argmax(axis=1), minlength=values.shape[1])
            aggregated = counts / counts.sum()
        else:
            confidence = group_values.max(axis=1)
            aggregated = np.average(group_values, axis=0, weights=confidence)
        trial_probabilities.append(aggregated)
        trial_labels.append(int(targets[indices[0]]))
    return np.asarray(trial_labels), np.asarray(trial_probabilities)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predictions = probabilities.argmax(axis=1)
    classes = list(range(probabilities.shape[1]))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "per_class_recall": recall_score(
            labels,
            predictions,
            labels=classes,
            average=None,
            zero_division=0,
        ).tolist(),
    }


def _evaluate_subset(
    dataset: ModelDataset,
    labels: np.ndarray,
    dev_indices: np.ndarray,  # Groups data strictly on subject-id for multiple subject-level fold validaton;
    sensors: tuple[int, ...],
    training_config: TrainingConfig,
    seed: int,
    n_splits: int = 5,
) -> tuple[dict[str, object], str]:
    """Evaluate a sensor subset across subject-level cross-validation folds (leakage-safe)."""
    features = mask_tabular_features(dataset.xgboost_features, sensors)
    dev_metadata = dataset.metadata.iloc[dev_indices].reset_index(drop=True)
    dev_labels = labels[dev_indices]

    # Subject-level folding to prevent data leakage across subjects;
    groups = (
        dev_metadata["subject_id"].to_numpy()
        if "subject_id" in dev_metadata.columns
        else dev_metadata.get("subject", np.zeros(len(dev_metadata)))
    )
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))

    fold_trial_f1s = []
    fold_trial_accs = []

    started = time.perf_counter()
    inference_elapsed_total = 0.0
    total_eval_samples = 0

    for train_fold_rel, val_fold_rel in gkf.split(dev_indices, dev_labels, groups=groups):
        train_idx = dev_indices[train_fold_rel]
        val_idx = dev_indices[val_fold_rel]

        model = fit_xgboost(
            features,
            labels,
            train_idx,
            num_classes=2,
            seed=seed,
            training_config=training_config,
        )
        pred_started = time.perf_counter()
        probabilities = predict_xgboost(model, features.iloc[val_idx], 2)
        inference_elapsed_total += time.perf_counter() - pred_started
        total_eval_samples += len(val_idx)

        # Aggregate per trial using default/mean probability for CV scoring;
        trial_labels, trial_probabilities = aggregate_trial_probabilities(
            probabilities,
            dataset.metadata.iloc[val_idx],
            labels[val_idx],
            "mean_probability",
        )
        fold_metrics = _metrics(trial_labels, trial_probabilities)
        fold_trial_f1s.append(fold_metrics["macro_f1"])
        fold_trial_accs.append(fold_metrics["accuracy"])

    elapsed = time.perf_counter() - started
    mean_f1 = float(np.mean(fold_trial_f1s))
    std_f1 = float(np.std(fold_trial_f1s))
    mean_acc = float(np.mean(fold_trial_accs))

    result: dict[str, object] = {
        "active_sensors": list(sensors),
        "sensor_count": len(sensors),
        "feature_count": features.shape[1],
        "aggregation_method": "mean_probability",
        "fit_predict_seconds": elapsed,
        "latency_ms_per_window": inference_elapsed_total * 1000.0 / max(1, total_eval_samples),
        "trial_macro_f1": mean_f1,
        "trial_macro_f1_std": std_f1,
        "trial_accuracy": mean_acc,
        "fold_macro_f1_scores": fold_trial_f1s,
    }
    return result, "mean_probability"


def run_sensor_selection(
    dataset: ModelDataset,
    output_dir: Path,
    *,
    seed: int = 42,
    min_sensors: int = 2,
    quick: bool = False,
    initial_sensors: tuple[int, ...] = tuple(range(1, 9)),
) -> dict[str, object]:
    """Run SBE across subject-level folds on development subjects and test only the recommendation."""
    if min_sensors < 1 or min_sensors >= len(initial_sensors):
        raise ValueError("min_sensors must be positive and smaller than initial sensor count")
    contract = get_canonical_contract()

    logger.info("Isolating development subjects from test partition...")
    # Strictly isolate development (train + val subjects) from test subjects;
    train_idx_fixed, val_idx_fixed, test_indices = fixed_subject_split(
        dataset.metadata,
        contract.train_subjects,
        contract.val_subjects,
        contract.test_subjects,
    )
    dev_indices = np.concatenate([train_idx_fixed, val_idx_fixed])
    labels = dataset.labels_binary
    training_config = QUICK_TRAINING if quick else TrainingConfig()

    current = tuple(sorted(initial_sensors))
    rows = []

    logger.info("Evaluating full baseline feature set with %d sensors...", len(current))
    full_result, _ = _evaluate_subset(dataset, labels, dev_indices, current, training_config, seed)
    full_result["retained_performance"] = 1.0
    rows.append(full_result)
    full_f1 = float(full_result["trial_macro_f1"])
    logger.info(
        "Full Baseline Evaluation Complete: Sensors: %d; Trial Macro-F1: %.4f (±%.4f); Trial Accuracy: %.4f",
        full_result["sensor_count"],
        full_result["trial_macro_f1"],
        full_result["trial_macro_f1_std"],
        full_result["trial_accuracy"],
    )

    logger.info("Starting Sequential Backward Elimination down to %d sensors:", min_sensors)
    selection_history = []
    while len(current) > min_sensors:
        candidates = []
        for removed_sensor in current:
            candidate = tuple(sensor for sensor in current if sensor != removed_sensor)
            result, _ = _evaluate_subset(
                dataset, labels, dev_indices, candidate, training_config, seed
            )
            result["removed_sensor"] = removed_sensor
            candidates.append(result)
        winner = max(
            candidates,
            key=lambda row: (
                float(row["trial_macro_f1"]),
                float(row["trial_accuracy"]),
                -int(row["removed_sensor"]),
            ),
        )
        winner["retained_performance"] = (
            float(winner["trial_macro_f1"]) / full_f1 if full_f1 > 0 else 0.0
        )
        logger.info(
            "Pruned Sensor %s; Active sensors: %s; Trial Macro-F1: %.4f; Retained: %.2f%%",
            winner.get("removed_sensor"),
            winner["active_sensors"],
            winner["trial_macro_f1"],
            winner["retained_performance"] * 100.0,
        )

        rows.append(winner)
        selection_history.append(winner["active_sensors"])
        current = tuple(int(sensor) for sensor in winner["active_sensors"])

    # These counts describe survival along this single greedy elimination path,
    # not selection stability across independent subject samples.
    sensor_survival_counts = {s: 0 for s in initial_sensors}
    for active_list in selection_history:
        for s in active_list:
            sensor_survival_counts[s] += 1

    reduced_candidates = [row for row in rows if int(row["sensor_count"]) <= 3]
    meeting_target = [
        row for row in reduced_candidates if float(row["retained_performance"]) >= 0.9
    ]
    recommendation = (
        min(meeting_target, key=lambda row: int(row["sensor_count"]))
        if meeting_target
        else max(reduced_candidates, key=lambda row: float(row["trial_macro_f1"]))
    )
    recommended_sensors = tuple(int(sensor) for sensor in recommendation["active_sensors"])
    logger.info("Selected optimal reduced sensor subset: %s", list(recommended_sensors))

    # Final evaluation on held-out test set using the recommendation;
    test_rows = []
    for subset_name, sensors in (
        ("all_sensors", tuple(sorted(initial_sensors))),
        ("recommended_reduced", recommended_sensors),
    ):
        logger.info("Running final evaluation on held-out test set for subset: %s", subset_name)
        # Train on all dev indices, test on test_indices;
        features = mask_tabular_features(dataset.xgboost_features, sensors)
        model = fit_xgboost(
            features,
            labels,
            dev_indices,
            num_classes=2,
            seed=seed,
            training_config=training_config,
        )
        probabilities = predict_xgboost(model, features.iloc[test_indices], 2)
        window_metrics = _metrics(labels[test_indices], probabilities)
        trial_labels, trial_probabilities = aggregate_trial_probabilities(
            probabilities,
            dataset.metadata.iloc[test_indices],
            labels[test_indices],
            "mean_probability",
        )
        test_metrics = _metrics(trial_labels, trial_probabilities)
        test_result = {
            "subset": subset_name,
            "active_sensors": list(sensors),
            "sensor_count": len(sensors),
            "test_window_macro_f1": window_metrics["macro_f1"],
            "test_trial_macro_f1": test_metrics["macro_f1"],
            "test_trial_accuracy": test_metrics["accuracy"],
        }
        test_rows.append(test_result)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / "sensor_subset_results.csv", index=False)
    pd.DataFrame(test_rows).to_csv(destination / "sensor_subset_test_results.csv", index=False)

    selection_summary = {
        "sensor_survival_counts_across_elimination_steps": sensor_survival_counts,
        "elimination_step_count": len(selection_history),
        "selection_stability_assessed": False,
        "sensor_survival_count_description": (
            "Number of steps in this single SBE path where each sensor remained active; "
            "not a cross-sample selection stability measure."
        ),
        "recommended_sensors": list(recommended_sensors),
        "validation_macro_f1_mean": float(recommendation["trial_macro_f1"]),
        "validation_macro_f1_std": float(recommendation["trial_macro_f1_std"]),
        "fold_scores": recommendation["fold_macro_f1_scores"],
    }
    (destination / "sensor_selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2), encoding="utf-8"
    )

    handoff = {
        "selected_sensor_ids": list(recommended_sensors),
        "sensor_names": [config.SENSOR_PLACEMENT[sensor] for sensor in recommended_sensors],
        "transformer_sensor_mask_boolean": [
            sensor in recommended_sensors for sensor in range(1, config.N_SENSORS + 1)
        ],
        "validation_retained_performance": float(recommendation["retained_performance"]),
        "validation_macro_f1_mean": float(recommendation["trial_macro_f1"]),
        "validation_macro_f1_std": float(recommendation["trial_macro_f1_std"]),
        "target_90pct_met": float(recommendation["retained_performance"]) >= 0.9,
        "selection_split": "subject_level_cv",
        "test_used_for_selection": False,
        "aggregation_method": "mean_probability",
        "seed": seed,
        "contract": asdict(contract),
        "quick": quick,
    }
    temporary = destination / "sensor_handoff.json.tmp"
    temporary.write_text(json.dumps(handoff, indent=2), encoding="utf-8")
    temporary.replace(destination / "sensor_handoff.json")

    logger.info(
        "Successfully exported sensor handoff configuration to %s",
        destination / "sensor_handoff.json",
    )
    return handoff


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-sensors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=config.PHASE3_RANDOM_SEED)
    parser.add_argument("--quick", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logger.info("Building model input dataset from manifest...")
    dataset = build_model_inputs(build_manifest(args.data_root), seed=args.seed)

    run_sensor_selection(
        dataset,
        args.output_dir,
        seed=args.seed,
        min_sensors=args.min_sensors,
        quick=args.quick,
    )
    # print(json.dumps(handoff, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
