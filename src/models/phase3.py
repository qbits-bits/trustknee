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
    compute_calibration_metrics,
    reliability_bins,
    risk_coverage_curve,
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
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    values = np.asarray(probabilities, dtype=float)
    targets = np.asarray(labels, dtype=int)
    predictions = values.argmax(axis=1)
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
        }
        for true_class in classes
        for predicted_class in classes
    ]
    return result, class_rows, confusion_rows


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
    chosen: list[int] = []
    for class_index in range(probabilities.shape[1]):
        candidates = np.flatnonzero((predictions == class_index) & (labels == class_index))
        ranked = candidates[np.argsort(confidence[candidates])[::-1]]
        chosen.extend(ranked[:per_class].tolist())
    if not chosen:
        chosen = np.argsort(confidence)[::-1][:per_class].tolist()

    feature_mask = sensor_feature_mask(active_sensors) if active_sensors is not None else None
    attrs = integrated_gradients(
        model,
        normalized_test[chosen],
        predictions[chosen],
        steps=steps,
        feature_mask=feature_mask,
    )
    rows = []
    local_metadata = metadata.reset_index(drop=True)
    for local_index, attribution in zip(chosen, attrs, strict=True):
        meta = local_metadata.iloc[local_index]
        rows.append(
            {
                "task": task,
                "subject_id": int(meta["subject_id"]),
                "label_id": int(meta["label_id"]),
                "trial_num": int(meta["trial_num"]),
                "window_index": int(meta["window_index"]),
                "predicted_class": int(predictions[local_index]),
                "true_class": int(labels[local_index]),
                "confidence": float(confidence[local_index]),
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
) -> pd.DataFrame:
    """Train, calibrate, explain, and evaluate the Phase 3 Transformer."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    contract = get_canonical_contract()
    selected = set(contract.train_subjects + contract.val_subjects + contract.test_subjects)
    evaluation_manifest = manifest[manifest["subject_id"].astype(int).isin(selected)].reset_index(
        drop=True
    )
    dataset = build_model_inputs(evaluation_manifest, overlap=protocol.overlap, seed=seed)
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
        "dataset_fingerprint_sha256": _dataset_fingerprint(dataset, protocol),
        "n_windows": dataset.n_samples,
        "n_trials": int(dataset.metadata["trial_id"].nunique()),
    }
    _write_json(configuration, destination / "config.json")

    metrics_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    explanation_rows: list[dict[str, object]] = []

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
                calibration_metric = compute_calibration_metrics(level_probabilities, level_labels)
                calibration_rows.append(
                    {
                        "task": task,
                        "level": level,
                        "calibration": calibration_name,
                        **calibration_metric,
                    }
                )
                for row in reliability_bins(level_probabilities, level_labels):
                    reliability_rows.append(
                        {
                            "task": task,
                            "level": level,
                            "calibration": calibration_name,
                            **row,
                        }
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
        for level in ("window", "trial"):
            if level == "window":
                level_labels = labels[test_indices]
                level_probabilities = xgb_probabilities
            else:
                level_labels, level_probabilities, _ = _trial_values(
                    xgb_probabilities,
                    dataset.metadata.iloc[test_indices],
                    labels[test_indices],
                )
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

    _write_frame(metrics_rows, destination / "metrics.csv")
    _write_frame(class_rows, destination / "per_class_metrics.csv")
    _write_frame(confusion_rows, destination / "confusion_matrices.csv")
    _write_frame(calibration_rows, destination / "calibration_metrics.csv")
    _write_frame(reliability_rows, destination / "reliability_bins.csv")
    _write_frame(risk_rows, destination / "risk_coverage.csv")
    _write_json(explanation_rows, destination / "integrated_gradients.json")
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
