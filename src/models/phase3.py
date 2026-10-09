"""Phase 3 calibrated Transformer evaluation on the frozen subject split."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src import config
from src.contracts import get_canonical_contract
from src.ingestion import build_manifest
from src.models.artifacts import save_transformer_artifact
from src.models.calibration import (
    TemperatureScaler,
    apply_post_pool_temperature,
    compute_calibration_metrics,
    fit_post_pool_temperature,
    reliability_bins,
    risk_coverage_curve,
    select_binary_decision_threshold,
    select_uncertainty_threshold,
)
from src.models.evaluation import average_trial_probabilities, fixed_subject_split
from src.models.explainability import (
    apply_sensor_mask,
    integrated_gradients,
    sensor_feature_mask,
    summarize_sensor_attributions,
)
from src.models.model_data import ModelDataset, build_model_inputs
from src.models.normalization import fit_sequence_normalizer
from src.models.statistical_evaluation import (
    paired_subject_comparisons,
    per_subject_trial_metrics,
    validate_paired_trial_alignment,
)
from src.models.training import (
    QUICK_TRAINING,
    TrainingConfig,
    fit_transformer,
    fit_xgboost,
    predict_transformer_logits,
    predict_xgboost,
)

logger = logging.getLogger("trustknee.phase3")


@dataclass(frozen=True)
class EvaluationProtocol:
    name: str
    overlap: float
    trim_edge_windows: int
    tasks: tuple[str, ...]


CANONICAL_PROTOCOL = EvaluationProtocol("canonical-50pct-overlap", 0.5, 0, ("binary", "nine_class"))
MATCHED_PROTOCOL = EvaluationProtocol("matched-nonoverlap", 0.0, 2, ("binary",))
CALIBRATION_BIN_COUNT = 15
SUBJECT_BOOTSTRAP_SAMPLES = 10_000
TRANSFORMER_INPUT_REPRESENTATION = "synchronized_30x56_sequence"
XGBOOST_INPUT_REPRESENTATION = "handcrafted_features_from_aligned_windows"


def _input_representation(model: str) -> str:
    return (
        XGBOOST_INPUT_REPRESENTATION
        if model == "xgboost_aligned"
        else TRANSFORMER_INPUT_REPRESENTATION
    )


def _trim_trial_edges(dataset: ModelDataset, count: int) -> ModelDataset:
    if count == 0:
        return dataset
    if count < 0:
        raise ValueError("trim_edge_windows cannot be negative")
    keep: list[int] = []
    for _, group in dataset.metadata.groupby("trial_id", sort=False):
        indices = group.index.to_numpy()
        if len(indices) > 2 * count:
            keep.extend(indices[count:-count])
    if not keep:
        raise ValueError("Edge trimming removed every model window")
    return dataset.subset(np.asarray(keep, dtype=int))


def _dataset_fingerprint(dataset: ModelDataset, protocol: EvaluationProtocol) -> str:
    columns = ["subject_id", "label_id", "trial_num", "window_index", "start_time_s", "end_time_s"]
    digest = hashlib.sha256()
    digest.update(protocol.name.encode())
    digest.update(str(dataset.transformer_sequences.shape).encode())
    digest.update(
        pd.util.hash_pandas_object(dataset.metadata[columns], index=False).values.tobytes()
    )
    return digest.hexdigest()


def _task_labels(dataset: ModelDataset, task: str) -> tuple[np.ndarray, int]:
    if task == "binary":
        return dataset.labels_binary, 2
    if task == "nine_class":
        return dataset.labels_9, 9
    raise ValueError(f"Unknown Phase 3 task: {task}")


def _trial_values(
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    local_metadata = metadata.reset_index(drop=True)
    trials = average_trial_probabilities(probabilities, local_metadata)
    true_by_trial = (
        local_metadata.assign(_true_label=np.asarray(labels, dtype=int))
        .groupby("trial_id", sort=True)["_true_label"]
        .first()
    )
    true_labels = true_by_trial.loc[trials["trial_id"]].to_numpy(dtype=int)
    probability_columns = sorted(
        (column for column in trials if column.startswith("probability_")),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    return true_labels, trials[probability_columns].to_numpy(dtype=float), trials


def _classification_result(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    task: str,
    level: str,
    model: str,
    calibration: str,
    decision_threshold: float | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    values = np.asarray(probabilities, dtype=float)
    targets = np.asarray(labels, dtype=int)
    if decision_threshold is None:
        predictions = values.argmax(axis=1)
    else:
        if values.shape[1] != 2 or not 0 <= decision_threshold <= 1:
            raise ValueError("A binary decision threshold must be in [0, 1]")
        predictions = (values[:, 1] >= decision_threshold).astype(int)
    classes = list(range(values.shape[1]))
    precision, recall, class_f1, support = precision_recall_fscore_support(
        targets, predictions, labels=classes, zero_division=0
    )
    result = {
        "task": task,
        "level": level,
        "model": model,
        "calibration": calibration,
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, labels=classes, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "n_samples": len(targets),
        "decision_threshold": decision_threshold,
    }
    class_rows = [
        {
            "task": task,
            "level": level,
            "model": model,
            "calibration": calibration,
            "class_index": class_index,
            "precision": float(precision[class_index]),
            "recall": float(recall[class_index]),
            "f1": float(class_f1[class_index]),
            "support": int(support[class_index]),
            "decision_threshold": decision_threshold,
        }
        for class_index in classes
    ]
    matrix = confusion_matrix(targets, predictions, labels=classes)
    confusion_rows = [
        {
            "task": task,
            "level": level,
            "model": model,
            "calibration": calibration,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "count": int(matrix[true_class, predicted_class]),
            "decision_threshold": decision_threshold,
        }
        for true_class in classes
        for predicted_class in classes
    ]
    return result, class_rows, confusion_rows


def _trial_prediction_rows(
    probabilities: np.ndarray,
    labels: np.ndarray,
    trials: pd.DataFrame,
    *,
    protocol: str,
    task: str,
    model: str,
    calibration: str,
    decision_threshold: float | None = None,
) -> list[dict[str, object]]:
    values = np.asarray(probabilities, dtype=float)
    targets = np.asarray(labels, dtype=int)
    if values.ndim != 2 or len(values) != len(targets) or len(values) != len(trials):
        raise ValueError("Trial probabilities, labels, and metadata must align")
    if decision_threshold is None:
        predictions = values.argmax(axis=1)
    else:
        if values.shape[1] != 2 or not 0 <= decision_threshold <= 1:
            raise ValueError("A binary decision threshold must be in [0, 1]")
        predictions = (values[:, 1] >= decision_threshold).astype(int)

    rows = []
    local_trials = trials.reset_index(drop=True)
    for index, (target, prediction) in enumerate(zip(targets, predictions, strict=True)):
        row = {
            "protocol": protocol,
            "task": task,
            "model": model,
            "calibration": calibration,
            "subject_id": int(local_trials.iloc[index]["subject_id"]),
            "trial_id": str(local_trials.iloc[index]["trial_id"]),
            "true_class": int(target),
            "predicted_class": int(prediction),
            "num_classes": int(values.shape[1]),
            "decision_threshold": decision_threshold,
            "input_representation": _input_representation(model),
        }
        row.update(
            {
                f"probability_{class_index}": float(value)
                for class_index, value in enumerate(values[index])
            }
        )
        rows.append(row)
    return rows


def _aggregate_explanations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate representative-window attributions by exercise and class."""
    flattened = []
    for explanation in rows:
        for sensor in explanation["sensors"]:
            flattened.append(
                {
                    "protocol": explanation["protocol"],
                    "task": explanation["task"],
                    "input_representation": explanation["input_representation"],
                    "selection": explanation["selection"],
                    "exercise": explanation["exercise"],
                    "true_class": explanation["true_class"],
                    "predicted_class": explanation["predicted_class"],
                    "sensor_id": sensor["sensor_id"],
                    "sensor_name": sensor["sensor_name"],
                    "absolute_attribution": sensor["absolute_attribution"],
                    "signed_attribution": sensor["signed_attribution"],
                    "attribution_fraction": sensor["attribution_fraction"],
                    "top_modality": sensor["top_modality"],
                    "top_channel": sensor["top_channel"],
                }
            )
    if not flattened:
        return []

    frame = pd.DataFrame(flattened)
    groups = [
        "protocol",
        "task",
        "input_representation",
        "selection",
        "exercise",
        "true_class",
        "predicted_class",
        "sensor_id",
        "sensor_name",
    ]
    summaries = []
    for keys, group in frame.groupby(groups, sort=True):
        row = dict(zip(groups, keys, strict=True))
        mean_signed = float(group["signed_attribution"].mean())
        row.update(
            {
                "n_representative_windows": int(len(group)),
                "mean_absolute_attribution": float(group["absolute_attribution"].mean()),
                "mean_signed_attribution": mean_signed,
                "mean_attribution_fraction": float(group["attribution_fraction"].mean()),
                "direction": "supports" if mean_signed >= 0 else "opposes",
                "most_frequent_modality": str(group["top_modality"].mode().iloc[0]),
                "most_frequent_channel": str(group["top_channel"].mode().iloc[0]),
            }
        )
        summaries.append(row)
    return summaries


def _representative_indices(
    probabilities: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    per_class: int,
) -> list[tuple[int, str]]:
    """Choose correct representatives, with an explicitly labelled fallback."""
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    label_ids = metadata.reset_index(drop=True)["label_id"].to_numpy(dtype=int)
    chosen: list[tuple[int, str]] = []
    for label_id in sorted(np.unique(label_ids)):
        true_class = int(labels[np.flatnonzero(label_ids == label_id)[0]])
        candidates = np.flatnonzero(
            (label_ids == label_id) & (predictions == true_class) & (labels == true_class)
        )
        ranked = candidates[np.argsort(confidence[candidates])[::-1]]
        chosen.extend(
            (int(index), "correct_high_confidence_representative_by_label")
            for index in ranked[:per_class]
        )
    if not chosen:
        chosen.extend(
            (int(index), "fallback_high_confidence_any_prediction")
            for index in np.argsort(confidence)[::-1][:per_class]
        )
    return chosen


def _write_frame(rows: list[dict[str, object]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json(values: object, path: Path) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temporary.replace(path)


def _measure_latency(model, normalized: np.ndarray, batch_size: int) -> float:
    import torch

    device = next(model.parameters()).device
    sample = torch.from_numpy(normalized[: min(len(normalized), batch_size)]).to(device)
    if not len(sample):
        return 0.0
    with torch.no_grad():
        model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / len(sample)


def _explain_representatives(
    model,
    normalized_test: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    task: str,
    active_sensors: tuple[int, ...] | None,
    per_class: int,
    steps: int,
) -> list[dict[str, object]]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    local_metadata = metadata.reset_index(drop=True)
    selections = _representative_indices(probabilities, labels, local_metadata, per_class)
    chosen = [index for index, _ in selections]

    feature_mask = sensor_feature_mask(active_sensors) if active_sensors is not None else None
    attrs = integrated_gradients(
        model,
        normalized_test[chosen],
        predictions[chosen],
        steps=steps,
        feature_mask=feature_mask,
    )
    rows = []
    for (local_index, selection), attribution in zip(selections, attrs, strict=True):
        meta = local_metadata.iloc[local_index]
        label_id = int(meta["label_id"])
        rows.append(
            {
                "task": task,
                "input_representation": TRANSFORMER_INPUT_REPRESENTATION,
                "subject_id": int(meta["subject_id"]),
                "label_id": label_id,
                "exercise": config.LABELS[label_id].exercise,
                "label_description": config.LABELS[label_id].description,
                "trial_num": int(meta["trial_num"]),
                "window_index": int(meta["window_index"]),
                "predicted_class": int(predictions[local_index]),
                "true_class": int(labels[local_index]),
                "confidence": float(confidence[local_index]),
                "selection": selection,
                "time_axis": {
                    "window_ms": float(config.WINDOW_MS),
                    "positions": int(normalized_test.shape[1]),
                    "bin_width_ms": float(config.WINDOW_MS / normalized_test.shape[1]),
                    "interpretation": "model input bins, not isolated physiological events",
                },
                "sensors": summarize_sensor_attributions(attribution),
            }
        )
    return rows


def run_phase3_transformer(
    manifest: pd.DataFrame,
    output_dir: Path,
    *,
    protocol: EvaluationProtocol = CANONICAL_PROTOCOL,
    seed: int = 42,
    quick: bool = False,
    device: str = "cpu",
    batch_size: int | None = None,
    active_sensors: tuple[int, ...] | None = None,
    explanation_steps: int = 32,
    explanations_per_class: int = 2,
    prepared_dataset: ModelDataset | None = None,
) -> pd.DataFrame:
    """Train, calibrate, explain, and evaluate the Phase 3 Transformer."""
    if explanation_steps <= 0 or explanations_per_class <= 0:
        raise ValueError("explanation_steps and explanations_per_class must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    contract = get_canonical_contract()
    selected = set(contract.train_subjects + contract.val_subjects + contract.test_subjects)
    if prepared_dataset is None:
        evaluation_manifest = manifest[
            manifest["subject_id"].astype(int).isin(selected)
        ].reset_index(drop=True)
        dataset = build_model_inputs(evaluation_manifest, overlap=protocol.overlap, seed=seed)
    else:
        preparation = prepared_dataset.preparation
        if preparation is None:
            raise ValueError("prepared_dataset is missing preparation settings")
        mismatches = []
        if not np.isclose(preparation.window_ms, config.WINDOW_MS):
            mismatches.append(f"window_ms={preparation.window_ms}")
        if not np.isclose(preparation.overlap, protocol.overlap):
            mismatches.append(f"overlap={preparation.overlap}")
        if not preparation.preprocess:
            mismatches.append("preprocess=False")
        if preparation.augment_minority:
            mismatches.append("augment_minority=True")
        if preparation.seed != seed:
            mismatches.append(f"seed={preparation.seed}")
        if mismatches:
            raise ValueError(
                "prepared_dataset is incompatible with the requested protocol: "
                + ", ".join(mismatches)
            )
        prepared_subjects = prepared_dataset.metadata["subject_id"].astype(int)
        indices = np.flatnonzero(prepared_subjects.isin(selected).to_numpy())
        if not len(indices):
            raise ValueError("prepared_dataset contains no subjects from the evaluation contract")
        dataset = prepared_dataset.subset(indices)
    dataset = _trim_trial_edges(dataset, protocol.trim_edge_windows)
    train_indices, validation_indices, test_indices = fixed_subject_split(
        dataset.metadata,
        contract.train_subjects,
        contract.val_subjects,
        contract.test_subjects,
    )
    training_config = QUICK_TRAINING if quick else TrainingConfig()
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        training_config = replace(training_config, batch_size=batch_size)
    if active_sensors is not None:
        sensor_feature_mask(active_sensors)

    configuration = {
        "protocol": asdict(protocol),
        "contract": asdict(contract),
        "seed": seed,
        "quick": quick,
        "device": device,
        "active_sensors": list(active_sensors) if active_sensors is not None else None,
        "training": asdict(training_config),
        "explanation_steps": explanation_steps,
        "explanations_per_class": explanations_per_class,
        "calibration_bins": CALIBRATION_BIN_COUNT,
        "subject_bootstrap_samples": SUBJECT_BOOTSTRAP_SAMPLES,
        "dataset_fingerprint_sha256": _dataset_fingerprint(dataset, protocol),
        "n_windows": dataset.n_samples,
        "n_trials": int(dataset.metadata["trial_id"].nunique()),
        "dataset_preparation": (
            asdict(dataset.preparation) if dataset.preparation is not None else None
        ),
    }
    _write_json(configuration, destination / "config.json")

    metrics_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    explanation_rows: list[dict[str, object]] = []
    trial_prediction_rows: list[dict[str, object]] = []
    trial_calibrator_rows: list[dict[str, object]] = []
    decision_threshold_rows: list[dict[str, object]] = []

    normalizer = fit_sequence_normalizer(dataset.transformer_sequences, train_indices)
    normalized = normalizer.transform(dataset.transformer_sequences)
    if active_sensors is not None:
        normalized = apply_sensor_mask(normalized, active_sensors)

    for task in protocol.tasks:
        labels, num_classes = _task_labels(dataset, task)
        model, _, history = fit_transformer(
            dataset.transformer_sequences,
            labels,
            train_indices,
            validation_indices,
            num_classes=num_classes,
            seed=seed,
            training_config=training_config,
            device=device,
            normalizer=normalizer,
            normalized_sequences=normalized,
        )
        validation_logits = predict_transformer_logits(
            model,
            dataset.transformer_sequences[validation_indices],
            normalizer,
            active_sensors=active_sensors,
        )
        test_logits = predict_transformer_logits(
            model,
            dataset.transformer_sequences[test_indices],
            normalizer,
            active_sensors=active_sensors,
        )
        scaler = TemperatureScaler.fit(validation_logits, labels[validation_indices])
        calibrated_validation = scaler.predict_proba(validation_logits)
        raw_test = TemperatureScaler().predict_proba(test_logits)
        calibrated_test = scaler.predict_proba(test_logits)
        validation_trial_labels, validation_trial_probabilities, validation_trials = _trial_values(
            calibrated_validation,
            dataset.metadata.iloc[validation_indices],
            labels[validation_indices],
        )
        test_trial_labels, test_trial_probabilities, test_trials = _trial_values(
            calibrated_test,
            dataset.metadata.iloc[test_indices],
            labels[test_indices],
        )
        trial_scaler = fit_post_pool_temperature(
            validation_trial_probabilities,
            validation_trial_labels,
        )
        calibrated_validation_trials = apply_post_pool_temperature(
            trial_scaler,
            validation_trial_probabilities,
        )
        calibrated_test_trials = apply_post_pool_temperature(
            trial_scaler,
            test_trial_probabilities,
        )
        trial_calibrator_rows.append(
            {
                "task": task,
                "fit_split": "validation_trials",
                "pooling": "arithmetic_mean_window_probabilities",
                "method": "temperature_on_log_pooled_probabilities",
                "temperature": float(trial_scaler.temperature),
                "n_validation_trials": int(len(validation_trial_labels)),
                "n_test_trials": int(len(test_trial_labels)),
            }
        )
        threshold_selection = select_uncertainty_threshold(
            calibrated_validation, labels[validation_indices]
        )
        threshold = float(threshold_selection["threshold"])

        for calibration_name, probabilities in (
            ("uncalibrated", raw_test),
            ("temperature", calibrated_test),
        ):
            for level in ("window", "trial"):
                if level == "window":
                    level_labels = labels[test_indices]
                    level_probabilities = probabilities
                else:
                    level_labels, level_probabilities, _ = _trial_values(
                        probabilities,
                        dataset.metadata.iloc[test_indices],
                        labels[test_indices],
                    )
                result, per_class, confusion = _classification_result(
                    level_probabilities,
                    level_labels,
                    task=task,
                    level=level,
                    model="transformer",
                    calibration=calibration_name,
                )
                metrics_rows.append(result)
                class_rows.extend(per_class)
                confusion_rows.extend(confusion)
                bins = reliability_bins(
                    level_probabilities,
                    level_labels,
                    n_bins=CALIBRATION_BIN_COUNT,
                )
                calibration_metric = compute_calibration_metrics(
                    level_probabilities,
                    level_labels,
                    n_bins=CALIBRATION_BIN_COUNT,
                )
                calibration_rows.append(
                    {
                        "task": task,
                        "level": level,
                        "calibration": calibration_name,
                        "n_samples": int(len(level_labels)),
                        "n_bins": CALIBRATION_BIN_COUNT,
                        "nonempty_bins": int(sum(int(row["count"]) > 0 for row in bins)),
                        **calibration_metric,
                    }
                )
                for row in bins:
                    reliability_rows.append(
                        {
                            "task": task,
                            "level": level,
                            "calibration": calibration_name,
                            "split": "test",
                            "n_samples": int(len(level_labels)),
                            "n_bins": CALIBRATION_BIN_COUNT,
                            **row,
                        }
                    )

        trial_result, trial_classes, trial_confusion = _classification_result(
            calibrated_test_trials,
            test_trial_labels,
            task=task,
            level="trial",
            model="transformer",
            calibration="trial_temperature",
        )
        metrics_rows.append(trial_result)
        class_rows.extend(trial_classes)
        confusion_rows.extend(trial_confusion)
        trial_bins = reliability_bins(
            calibrated_test_trials,
            test_trial_labels,
            n_bins=CALIBRATION_BIN_COUNT,
        )
        calibration_rows.append(
            {
                "task": task,
                "level": "trial",
                "calibration": "trial_temperature",
                "n_samples": int(len(test_trial_labels)),
                "n_bins": CALIBRATION_BIN_COUNT,
                "nonempty_bins": int(sum(int(row["count"]) > 0 for row in trial_bins)),
                **compute_calibration_metrics(
                    calibrated_test_trials,
                    test_trial_labels,
                    n_bins=CALIBRATION_BIN_COUNT,
                ),
            }
        )
        for row in trial_bins:
            reliability_rows.append(
                {
                    "task": task,
                    "level": "trial",
                    "calibration": "trial_temperature",
                    "split": "test",
                    "n_samples": int(len(test_trial_labels)),
                    "n_bins": CALIBRATION_BIN_COUNT,
                    **row,
                }
            )

        trial_prediction_rows.extend(
            _trial_prediction_rows(
                test_trial_probabilities,
                test_trial_labels,
                test_trials,
                protocol=protocol.name,
                task=task,
                model="transformer",
                calibration="temperature",
            )
        )
        trial_prediction_rows.extend(
            _trial_prediction_rows(
                calibrated_test_trials,
                test_trial_labels,
                test_trials,
                protocol=protocol.name,
                task=task,
                model="transformer",
                calibration="trial_temperature",
            )
        )

        if task == "binary":
            decision_selection = select_binary_decision_threshold(
                calibrated_validation_trials,
                validation_trial_labels,
            )
            decision_threshold = float(decision_selection["threshold"])
            decision_threshold_rows.append(
                {
                    "task": task,
                    "fit_split": "validation_trials",
                    "calibration": "trial_temperature",
                    "n_validation_trials": int(len(validation_trials)),
                    **decision_selection,
                }
            )
            thresholded_result, thresholded_classes, thresholded_confusion = _classification_result(
                calibrated_test_trials,
                test_trial_labels,
                task=task,
                level="trial",
                model="transformer_thresholded",
                calibration="trial_temperature",
                decision_threshold=decision_threshold,
            )
            metrics_rows.append(thresholded_result)
            class_rows.extend(thresholded_classes)
            confusion_rows.extend(thresholded_confusion)
            trial_prediction_rows.extend(
                _trial_prediction_rows(
                    calibrated_test_trials,
                    test_trial_labels,
                    test_trials,
                    protocol=protocol.name,
                    task=task,
                    model="transformer_thresholded",
                    calibration="trial_temperature",
                    decision_threshold=decision_threshold,
                )
            )

        for split, probabilities, split_labels in (
            ("validation", calibrated_validation, labels[validation_indices]),
            ("test", calibrated_test, labels[test_indices]),
        ):
            for row in risk_coverage_curve(probabilities, split_labels):
                risk_rows.append({"task": task, "level": "window", "split": split, **row})
        test_confidence = calibrated_test.max(axis=1)
        accepted = test_confidence >= threshold
        accepted_predictions = calibrated_test.argmax(axis=1)[accepted]
        accepted_labels = labels[test_indices][accepted]
        metrics_rows.append(
            {
                "task": task,
                "level": "window",
                "model": "transformer_selective",
                "calibration": "temperature",
                "accuracy": float(accuracy_score(accepted_labels, accepted_predictions)),
                "macro_f1": float(
                    f1_score(
                        accepted_labels,
                        accepted_predictions,
                        labels=list(range(num_classes)),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(accepted_labels, accepted_predictions)
                ),
                "n_samples": int(accepted.sum()),
                "coverage": float(accepted.mean()),
                "uncertainty_threshold": threshold,
            }
        )

        latency = _measure_latency(model, normalized[test_indices], training_config.batch_size)
        for row in metrics_rows:
            if row["task"] == task and row["model"].startswith("transformer"):
                row["inference_latency_ms"] = latency

        label_mapping = (
            {0: "Correct", 1: "Wrong"}
            if task == "binary"
            else {index: config.LABELS[index].description for index in range(9)}
        )
        save_transformer_artifact(
            destination / f"transformer_{task}.pt",
            model,
            normalizer,
            scaler,
            threshold,
            label_mapping,
            sensor_mask=active_sensors,
            metadata={
                "protocol": protocol.name,
                "dataset_fingerprint_sha256": configuration["dataset_fingerprint_sha256"],
                "seed": seed,
            },
        )
        _write_json(history, destination / f"training_history_{task}.json")
        explanation_rows.extend(
            _explain_representatives(
                model,
                normalized[test_indices],
                calibrated_test,
                labels[test_indices],
                dataset.metadata.iloc[test_indices],
                task,
                active_sensors,
                explanations_per_class,
                explanation_steps,
            )
        )

        if active_sensors is not None:
            continue

        xgb = fit_xgboost(
            dataset.xgboost_features,
            labels,
            train_indices,
            num_classes,
            seed=seed,
            training_config=training_config,
        )
        xgb_probabilities = predict_xgboost(
            xgb, dataset.xgboost_features.iloc[test_indices], num_classes
        )
        xgb_trial_labels, xgb_trial_probabilities, xgb_trials = _trial_values(
            xgb_probabilities,
            dataset.metadata.iloc[test_indices],
            labels[test_indices],
        )
        for level in ("window", "trial"):
            if level == "window":
                level_labels = labels[test_indices]
                level_probabilities = xgb_probabilities
            else:
                level_labels = xgb_trial_labels
                level_probabilities = xgb_trial_probabilities
            result, per_class, confusion = _classification_result(
                level_probabilities,
                level_labels,
                task=task,
                level=level,
                model="xgboost_aligned",
                calibration="none",
            )
            metrics_rows.append(result)
            class_rows.extend(per_class)
            confusion_rows.extend(confusion)

        trial_prediction_rows.extend(
            _trial_prediction_rows(
                xgb_trial_probabilities,
                xgb_trial_labels,
                xgb_trials,
                protocol=protocol.name,
                task=task,
                model="xgboost_aligned",
                calibration="none",
            )
        )

    for rows in (
        metrics_rows,
        class_rows,
        confusion_rows,
        calibration_rows,
        reliability_rows,
        risk_rows,
        explanation_rows,
        trial_calibrator_rows,
        decision_threshold_rows,
        trial_prediction_rows,
    ):
        for row in rows:
            row.setdefault("protocol", protocol.name)
            if "model" in row:
                row.setdefault(
                    "input_representation",
                    _input_representation(str(row["model"])),
                )
            else:
                row.setdefault("input_representation", TRANSFORMER_INPUT_REPRESENTATION)

    trial_predictions = pd.DataFrame(trial_prediction_rows)
    subject_metrics = per_subject_trial_metrics(trial_predictions)
    primary_trial_predictions = trial_predictions[
        (
            (trial_predictions["model"] == "transformer")
            & (trial_predictions["calibration"] == "temperature")
        )
        | (trial_predictions["model"] == "xgboost_aligned")
    ].reset_index(drop=True)
    paired_input = subject_metrics[
        (
            (subject_metrics["model"] == "transformer")
            & (subject_metrics["calibration"] == "temperature")
        )
        | (subject_metrics["model"] == "xgboost_aligned")
    ].reset_index(drop=True)
    if {"transformer", "xgboost_aligned"}.issubset(primary_trial_predictions["model"].unique()):
        validate_paired_trial_alignment(primary_trial_predictions)
        paired_comparisons = paired_subject_comparisons(
            paired_input,
            seed=seed,
            bootstrap_samples=SUBJECT_BOOTSTRAP_SAMPLES,
        )
    else:
        paired_comparisons = pd.DataFrame()
    explanation_summaries = _aggregate_explanations(explanation_rows)

    _write_frame(metrics_rows, destination / "metrics.csv")
    _write_frame(class_rows, destination / "per_class_metrics.csv")
    _write_frame(confusion_rows, destination / "confusion_matrices.csv")
    _write_frame(calibration_rows, destination / "calibration_metrics.csv")
    _write_frame(reliability_rows, destination / "reliability_bins.csv")
    _write_frame(risk_rows, destination / "risk_coverage.csv")
    _write_frame(trial_prediction_rows, destination / "trial_predictions.csv")
    _write_frame(subject_metrics.to_dict("records"), destination / "per_subject_metrics.csv")
    _write_frame(
        paired_comparisons.to_dict("records"),
        destination / "paired_model_comparisons.csv",
    )
    _write_frame(trial_calibrator_rows, destination / "trial_calibration.csv")
    _write_frame(decision_threshold_rows, destination / "binary_decision_thresholds.csv")
    _write_frame(explanation_summaries, destination / "integrated_gradients_summary.csv")
    _write_json(explanation_rows, destination / "integrated_gradients.json")
    _write_json(
        {
            "protocol": protocol.name,
            "aligned_on": ["subject_split", "trial", "window", "target"],
            "transformer_input": "synchronized 30x56 IMU and sEMG-activity sequence",
            "xgboost_aligned_input": "handcrafted summaries from the aligned windows",
            "claim_boundary": (
                "Model differences apply only to this named protocol and representation; "
                "canonical, matched, and exact-engineered XGBoost results are not interchangeable."
            ),
        },
        destination / "comparison_scope.json",
    )
    return pd.DataFrame(metrics_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=["canonical", "matched", "both"], default="both")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=config.PHASE3_RANDOM_SEED)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--active-sensors", help="Comma-separated one-based sensor IDs")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--explanation-steps", type=int, default=32)
    parser.add_argument("--explanations-per-class", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    active_sensors = (
        tuple(int(value.strip()) for value in args.active_sensors.split(","))
        if args.active_sensors
        else None
    )
    manifest = build_manifest(args.data_root)
    protocols = {
        "canonical": (CANONICAL_PROTOCOL,),
        "matched": (MATCHED_PROTOCOL,),
        "both": (CANONICAL_PROTOCOL, MATCHED_PROTOCOL),
    }[args.protocol]
    for protocol in protocols:
        run_phase3_transformer(
            manifest,
            args.output_dir / protocol.name,
            protocol=protocol,
            seed=args.seed,
            quick=args.quick,
            device=args.device,
            batch_size=args.batch_size,
            active_sensors=active_sensors,
            explanation_steps=args.explanation_steps,
            explanations_per_class=args.explanations_per_class,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
