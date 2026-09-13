"""Leakage-safe Sequential Backward Elimination for TrustKnee sensors."""

from __future__ import annotations

import argparse
import json
import re
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

from src import config
from src.contracts import get_canonical_contract
from src.ingestion import build_manifest
from src.models.evaluation import fixed_subject_split
from src.models.model_data import ModelDataset, build_model_inputs
from src.models.training import QUICK_TRAINING, TrainingConfig, fit_xgboost, predict_xgboost

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
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    sensors: tuple[int, ...],
    training_config: TrainingConfig,
    seed: int,
    aggregation_method: str | None = None,
) -> tuple[dict[str, object], str]:
    features = mask_tabular_features(dataset.xgboost_features, sensors)
    started = time.perf_counter()
    model = fit_xgboost(
        features,
        labels,
        train_indices,
        num_classes=2,
        seed=seed,
        training_config=training_config,
    )
    prediction_started = time.perf_counter()
    probabilities = predict_xgboost(model, features.iloc[evaluation_indices], 2)
    inference_elapsed = time.perf_counter() - prediction_started
    elapsed = time.perf_counter() - started
    window_metrics = _metrics(labels[evaluation_indices], probabilities)
    methods = (aggregation_method,) if aggregation_method else AGGREGATION_METHODS
    candidates = []
    for method in methods:
        trial_labels, trial_probabilities = aggregate_trial_probabilities(
            probabilities,
            dataset.metadata.iloc[evaluation_indices],
            labels[evaluation_indices],
            method,
        )
        candidates.append((method, _metrics(trial_labels, trial_probabilities)))
    selected_method, trial_metrics = max(
        candidates,
        key=lambda item: (
            float(item[1]["macro_f1"]),
            float(item[1]["accuracy"]),
            -AGGREGATION_METHODS.index(item[0]),
        ),
    )
    result: dict[str, object] = {
        "active_sensors": list(sensors),
        "sensor_count": len(sensors),
        "feature_count": features.shape[1],
        "aggregation_method": selected_method,
        "fit_predict_seconds": elapsed,
        "latency_ms_per_window": inference_elapsed * 1000.0 / len(evaluation_indices),
    }
    result.update({f"window_{key}": value for key, value in window_metrics.items()})
    result.update({f"trial_{key}": value for key, value in trial_metrics.items()})
    return result, selected_method


def run_sensor_selection(
    dataset: ModelDataset,
    output_dir: Path,
    *,
    seed: int = 42,
    min_sensors: int = 2,
    quick: bool = False,
    initial_sensors: tuple[int, ...] = tuple(range(1, 9)),
) -> dict[str, object]:
    """Run SBE on train/validation subjects and test only the recommendation."""
    if min_sensors < 1 or min_sensors >= len(initial_sensors):
        raise ValueError("min_sensors must be positive and smaller than the initial sensor count")
    contract = get_canonical_contract()
    train_indices, validation_indices, test_indices = fixed_subject_split(
        dataset.metadata,
        contract.train_subjects,
        contract.val_subjects,
        contract.test_subjects,
    )
    labels = dataset.labels_binary
    training_config = QUICK_TRAINING if quick else TrainingConfig()
    current = tuple(sorted(initial_sensors))
    rows = []
    full_result, _ = _evaluate_subset(
        dataset,
        labels,
        train_indices,
        validation_indices,
        current,
        training_config,
        seed,
    )
    full_result["retained_performance"] = 1.0
    rows.append(full_result)
    full_f1 = float(full_result["trial_macro_f1"])
    while len(current) > min_sensors:
        candidates = []
        for removed_sensor in current:
            candidate = tuple(sensor for sensor in current if sensor != removed_sensor)
            result, _ = _evaluate_subset(
                dataset,
                labels,
                train_indices,
                validation_indices,
                candidate,
                training_config,
                seed,
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
        rows.append(winner)
        current = tuple(int(sensor) for sensor in winner["active_sensors"])

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
    test_rows = []
    for subset_name, sensors, method in (
        ("all_sensors", tuple(sorted(initial_sensors)), str(full_result["aggregation_method"])),
        (
            "recommended_reduced",
            recommended_sensors,
            str(recommendation["aggregation_method"]),
        ),
    ):
        result, _ = _evaluate_subset(
            dataset,
            labels,
            train_indices,
            test_indices,
            sensors,
            training_config,
            seed,
            aggregation_method=method,
        )
        result["subset"] = subset_name
        test_rows.append(result)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    search_frame = pd.DataFrame(rows)
    search_frame.to_csv(destination / "sensor_subset_results.csv", index=False)
    pd.DataFrame(test_rows).to_csv(destination / "sensor_subset_test_results.csv", index=False)
    handoff = {
        "selected_sensor_ids": list(recommended_sensors),
        "sensor_names": [config.SENSOR_PLACEMENT[sensor] for sensor in recommended_sensors],
        "transformer_sensor_mask_boolean": [
            sensor in recommended_sensors for sensor in range(1, config.N_SENSORS + 1)
        ],
        "validation_retained_performance": float(recommendation["retained_performance"]),
        "target_90pct_met": float(recommendation["retained_performance"]) >= 0.9,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "aggregation_method": recommendation["aggregation_method"],
        "seed": seed,
        "contract": asdict(contract),
        "quick": quick,
    }
    temporary = destination / "sensor_handoff.json.tmp"
    temporary.write_text(json.dumps(handoff, indent=2), encoding="utf-8")
    temporary.replace(destination / "sensor_handoff.json")
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
    dataset = build_model_inputs(build_manifest(args.data_root), seed=args.seed)
    handoff = run_sensor_selection(
        dataset,
        args.output_dir,
        seed=args.seed,
        min_sensors=args.min_sensors,
        quick=args.quick,
    )
    print(json.dumps(handoff, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
