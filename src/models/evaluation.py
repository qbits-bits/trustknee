"""Leave-one-subject-out comparison of the Transformer and XGBoost baseline."""

from __future__ import annotations

import json
import logging
import platform
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from src import config as project_config
from src.models.model_data import ModelDataset, build_model_inputs
from src.models.normalization import normalize_sequences
from src.models.training import (
    QUICK_TRAINING,
    TrainingConfig,
    fit_transformer,
    fit_xgboost,
    predict_transformer,
    predict_xgboost,
)

logger = logging.getLogger("trustknee.models")

EXPECTED_FOLD_RESULT_ROWS = 3 * 2 * 2  # models × tasks × reporting levels

FIXED_TRAIN_SUBJECTS = (
    2,
    5,
    7,
    8,
    9,
    11,
    12,
    14,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    31,
)
FIXED_VALIDATION_SUBJECTS = (4, 13, 16, 30)
FIXED_TEST_SUBJECTS = (3, 6, 10, 15, 29)


def iter_loso_folds(metadata: pd.DataFrame) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(held_out_subject, train_rows, test_rows)`` without subject overlap."""
    subjects = np.sort(metadata["subject_id"].unique())
    is_synthetic = (
        metadata["synthetic"].to_numpy(dtype=bool)
        if "synthetic" in metadata.columns
        else np.zeros(len(metadata), dtype=bool)
    )
    for held_out in subjects:
        test_indices = np.flatnonzero(
            (metadata["subject_id"].to_numpy() == held_out) & (~is_synthetic)
        )
        train_indices = np.flatnonzero(metadata["subject_id"].to_numpy() != held_out)
        yield int(held_out), train_indices, test_indices


def fixed_subject_split(
    metadata: pd.DataFrame,
    train_subjects: Iterable[int] = FIXED_TRAIN_SUBJECTS,
    validation_subjects: Iterable[int] = FIXED_VALIDATION_SUBJECTS,
    test_subjects: Iterable[int] = FIXED_TEST_SUBJECTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row indices for one explicit, leakage-free subject partition."""
    partitions = {
        "train": {int(subject) for subject in train_subjects},
        "validation": {int(subject) for subject in validation_subjects},
        "test": {int(subject) for subject in test_subjects},
    }
    if any(not subjects for subjects in partitions.values()):
        raise ValueError("Fixed train, validation, and test subject sets must be non-empty")
    names = list(partitions)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = partitions[left_name] & partitions[right_name]
            if overlap:
                raise ValueError(
                    f"Fixed subject partitions overlap between {left_name} and {right_name}: "
                    f"{sorted(overlap)}"
                )

    available = {int(subject) for subject in metadata["subject_id"].unique()}
    requested = set().union(*partitions.values())
    missing = requested - available
    if missing:
        raise ValueError(f"Fixed subject partition is missing subjects: {sorted(missing)}")

    subject_ids = metadata["subject_id"].to_numpy()
    indices = tuple(
        np.flatnonzero(np.isin(subject_ids, sorted(partitions[name]))) for name in names
    )
    if any(not len(values) for values in indices):
        raise ValueError("Fixed subject partition produced an empty row set")
    return indices


def subject_validation_split(
    metadata: pd.DataFrame, train_indices: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Split training rows by subject for inner validation, never by window."""
    train_indices = np.asarray(train_indices, dtype=int)
    subjects = np.sort(metadata.iloc[train_indices]["subject_id"].unique())
    if len(subjects) < 2:
        return train_indices, np.empty(0, dtype=int)
    # A deterministic single-subject validation set keeps the small experiment
    # repeatable and avoids a random window split.
    validation_subject = subjects[np.random.default_rng(seed).integers(len(subjects))]
    is_synthetic = (
        metadata["synthetic"].to_numpy(dtype=bool)
        if "synthetic" in metadata.columns
        else np.zeros(len(metadata), dtype=bool)
    )
    validation_mask = metadata["subject_id"].to_numpy() == validation_subject
    validation_indices = train_indices[
        validation_mask[train_indices] & (~is_synthetic[train_indices])
    ]
    fit_indices = train_indices[~validation_mask[train_indices]]
    if not len(fit_indices):
        return train_indices, np.empty(0, dtype=int)
    return fit_indices, validation_indices


def average_trial_probabilities(probabilities: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    """Average window probabilities into one prediction per trial.

    Grouping uses subject and trial IDs, so overlapping windows from one trial
    cannot become independent trial-level observations.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(probabilities) != len(metadata):
        raise ValueError("probabilities and metadata must have the same number of rows")
    if not len(probabilities):
        return pd.DataFrame()
    metadata = metadata.reset_index(drop=True)
    group_columns = ["trial_id"] if "trial_id" in metadata else ["subject_id", "trial_num"]
    rows = []
    for _, group in metadata.groupby(group_columns, sort=True):
        indices = group.index.to_numpy()
        row = {
            "subject_id": group["subject_id"].iloc[0] if "subject_id" in group else None,
            "trial_num": group["trial_num"].iloc[0] if "trial_num" in group else None,
            "trial_id": group["trial_id"].iloc[0] if "trial_id" in group else None,
            "label_id": group["label_id"].iloc[0] if "label_id" in group else None,
            "execution": group["execution"].iloc[0] if "execution" in group else None,
            "n_windows": len(indices),
        }
        averaged = probabilities[indices].mean(axis=0)
        row.update(
            {f"probability_{class_index}": value for class_index, value in enumerate(averaged)}
        )
        row["predicted_label"] = int(np.argmax(averaged))
        rows.append(row)
    return pd.DataFrame(rows)


# Descriptive alias used in notebooks and reports.
trial_probability_average = average_trial_probabilities


def _metric_rows(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
    *,
    model: str,
    task: str,
    fold: int,
    level: str,
    held_out_subject: int | None,
    n_train_subjects: int,
    n_validation_subjects: int,
    n_test_subjects: int,
    n_train_trials: int,
    n_test_trials: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(y_true), num_classes):
        raise ValueError(
            f"Expected probability shape ({len(y_true)}, {num_classes}), got {probabilities.shape}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model produced non-finite probabilities")
    y_pred = probabilities.argmax(axis=1)
    class_indices = list(range(num_classes))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        precision, recall, class_f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=class_indices, zero_division=0
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        macro_f1 = f1_score(y_true, y_pred, labels=class_indices, average="macro", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    result = {
        "fold": fold,
        "held_out_subject": held_out_subject,
        "model": model,
        "task": task,
        "level": level,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy),
        "n_train_subjects": n_train_subjects,
        "n_validation_subjects": n_validation_subjects,
        "n_test_subjects": n_test_subjects,
        "n_train_trials": n_train_trials,
        "n_test_trials": n_test_trials,
        "n_test_windows": len(y_true) if level == "window" else None,
    }
    per_class = [
        {
            "fold": fold,
            "held_out_subject": held_out_subject,
            "model": model,
            "task": task,
            "level": level,
            "class_index": class_index,
            "precision": float(precision[class_index]),
            "recall": float(recall[class_index]),
            "f1": float(class_f1[class_index]),
            "support": int(support[class_index]),
        }
        for class_index in class_indices
    ]
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    confusion_rows = [
        {
            "fold": fold,
            "held_out_subject": held_out_subject,
            "model": model,
            "task": task,
            "level": level,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "count": int(matrix[true_class, predicted_class]),
        }
        for true_class in class_indices
        for predicted_class in class_indices
    ]
    return result, per_class, confusion_rows


def _evaluate_prediction(
    dataset: ModelDataset,
    test_indices: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    *,
    model: str,
    task: str,
    fold: int,
    held_out_subject: int | None,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    test_metadata = dataset.metadata.iloc[test_indices].reset_index(drop=True)
    train_metadata = dataset.metadata.iloc[train_indices]
    test_subjects = int(test_metadata["subject_id"].nunique())
    train_subjects = int(train_metadata["subject_id"].nunique())
    validation_subjects = int(dataset.metadata.iloc[validation_indices]["subject_id"].nunique())
    train_trials = int(train_metadata["trial_id"].nunique())
    test_trials = int(test_metadata["trial_id"].nunique())
    rows = []
    per_class_rows = []
    confusion_rows = []
    window_result, window_classes, window_confusion = _metric_rows(
        labels[test_indices],
        probabilities,
        num_classes,
        model=model,
        task=task,
        fold=fold,
        level="window",
        held_out_subject=held_out_subject,
        n_train_subjects=train_subjects,
        n_validation_subjects=validation_subjects,
        n_test_subjects=test_subjects,
        n_train_trials=train_trials,
        n_test_trials=test_trials,
    )
    rows.append(window_result)
    per_class_rows.extend(window_classes)
    confusion_rows.extend(window_confusion)

    trial_predictions = average_trial_probabilities(probabilities, test_metadata)
    trial_prob_columns = [f"probability_{index}" for index in range(num_classes)]
    trial_probabilities = trial_predictions[trial_prob_columns].to_numpy()
    true_trial_labels = (
        test_metadata.assign(_true_label=np.asarray(labels, dtype=int)[test_indices])
        .groupby(
            ["trial_id"] if "trial_id" in test_metadata else ["subject_id", "trial_num"],
            sort=True,
        )["_true_label"]
        .first()
        .to_numpy()
    )
    trial_result, trial_classes, trial_confusion = _metric_rows(
        true_trial_labels,
        trial_probabilities,
        num_classes,
        model=model,
        task=task,
        fold=fold,
        level="trial",
        held_out_subject=held_out_subject,
        n_train_subjects=train_subjects,
        n_validation_subjects=validation_subjects,
        n_test_subjects=test_subjects,
        n_train_trials=train_trials,
        n_test_trials=test_trials,
    )
    trial_result["n_test_windows"] = len(probabilities)
    trial_result["n_test_trials"] = len(trial_predictions)
    rows.append(trial_result)
    per_class_rows.extend(trial_classes)
    confusion_rows.extend(trial_confusion)
    return rows, per_class_rows, confusion_rows


def _majority_probabilities(
    labels: np.ndarray, train_indices: np.ndarray, num_classes: int
) -> np.ndarray:
    train_labels = np.asarray(labels, dtype=int)[train_indices]
    counts = np.bincount(train_labels, minlength=num_classes)
    majority = int(np.argmax(counts))
    probabilities = np.zeros((len(labels), num_classes), dtype=np.float32)
    probabilities[:, majority] = 1.0
    return probabilities


def _version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unavailable"


def _write_experiment_record(
    output_dir: Path,
    dataset: ModelDataset,
    manifest: pd.DataFrame,
    seed: int,
    quick: bool,
    training_config: TrainingConfig,
    device: str,
    evaluation_description: str,
    augment_minority: bool = False,
    aug_multiplier: int = 3,
    aug_methods: tuple[str, ...] = ("jitter", "magnitude_scale", "time_warp"),
    aug_target_labels: tuple[int, ...] = (6, 7, 8),
) -> None:
    counts = pd.Series(dataset.labels_9).value_counts().sort_index().to_dict()
    subject_count = int(dataset.metadata["subject_id"].nunique())
    trial_count = int(dataset.metadata["trial_id"].nunique())
    synthetic_count = (
        int(dataset.metadata["synthetic"].sum()) if "synthetic" in dataset.metadata.columns else 0
    )
    skipped = max(0, len(manifest) - trial_count)
    record = f"""# TrustKnee experiment record

This is a research-software record, not a clinical validation report.

- Dataset/version/source: KneE-PAD dataset, local copy of the source data described by Konstantoudakis et al. (Scientific Data, 2025) and `src/config.py`.
- Valid trials: {trial_count}; manifest rows skipped or without usable windows: {skipped}.
- Subjects: {subject_count}; windows: {dataset.n_samples} (synthetic: {synthetic_count}).
- Minority augmentation: {augment_minority} (multiplier={aug_multiplier}, methods={list(aug_methods)}, target_labels={list(aug_target_labels)}).
- Label counts by nine-class ID: {counts}.
- Window: {project_config.WINDOW_MS:g} ms with {project_config.WINDOW_OVERLAP:.0%} overlap.
- Transformer input: {dataset.transformer_sequences.shape[1]} time positions × {dataset.transformer_sequences.shape[2]} values (48 IMU + 8 rectified/binned sEMG activity channels).
- Primary task: binary Correct (0) versus Wrong (1). Secondary task: nine-class label prediction.
- Evaluation: {evaluation_description}.
- Random seed: {seed}; quick mode: {quick}.
- Transformer training device: {device}.
- Transformer: 2 encoder layers, 4 heads, hidden size 64, feed-forward size 128, dropout 0.1, AdamW lr {training_config.learning_rate}.
- XGBoost: {training_config.xgb_estimators} estimators, depth 4, learning rate 0.05, fixed seed.
- Hardware/software: {platform.machine()} ({platform.processor() or "CPU information unavailable"}); Python {platform.python_version()}, NumPy {_version("numpy")}, pandas {_version("pandas")}, PyTorch {_version("torch")}, XGBoost {_version("xgboost")}, scikit-learn {_version("scikit-learn")}.

Results are reported at both window and trial level. Trial probabilities are
the mean of that trial's window probabilities. Performance does not establish
clinical safety, diagnostic validity, or readiness for home use.
"""
    (output_dir / "experiment_record.md").write_text(record, encoding="utf-8")


def _summarize_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        fold_results.groupby(["model", "task", "level"])[
            ["accuracy", "macro_f1", "balanced_accuracy"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "model",
        "task",
        "level",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
    ]
    return summary.fillna(0.0)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _validate_or_write_run_config(
    output_dir: Path, config_record: dict[str, object], resume: bool
) -> None:
    config_path = output_dir / "config.json"
    checkpoint_paths = (
        output_dir / "fold_results.csv",
        output_dir / "per_class_results.csv",
        output_dir / "confusion_matrices.csv",
    )
    if resume and not config_path.exists() and any(path.exists() for path in checkpoint_paths):
        raise ValueError("Resume requires config.json to identify existing checkpoint files")
    if resume and config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != config_record:
            raise ValueError("Existing checkpoint configuration does not match this run")
        return
    config_path.write_text(json.dumps(config_record, indent=2), encoding="utf-8")


def _write_result_checkpoint(
    output_dir: Path,
    rows: list[dict[str, object]],
    per_class_rows: list[dict[str, object]],
    confusion_rows: list[dict[str, object]],
) -> pd.DataFrame:
    fold_results = pd.DataFrame(rows)
    per_class_results = pd.DataFrame(per_class_rows)
    confusion_results = pd.DataFrame(confusion_rows)
    _atomic_write_csv(per_class_results, output_dir / "per_class_results.csv")
    _atomic_write_csv(confusion_results, output_dir / "confusion_matrices.csv")
    for (model, task), group in confusion_results.groupby(["model", "task"]):
        _atomic_write_csv(group, output_dir / f"confusion_matrices_{model}_{task}.csv")
    _atomic_write_csv(_summarize_results(fold_results), output_dir / "summary_results.csv")
    # Write this completion marker last. Resume trusts only folds recorded here.
    _atomic_write_csv(fold_results, output_dir / "fold_results.csv")
    return fold_results


def _load_result_checkpoint(
    output_dir: Path, resume: bool
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    set[int],
]:
    if not resume:
        return [], [], [], set()
    paths = {
        "fold": output_dir / "fold_results.csv",
        "per_class": output_dir / "per_class_results.csv",
        "confusion": output_dir / "confusion_matrices.csv",
    }
    existing = {name: path.exists() for name, path in paths.items()}
    if not any(existing.values()):
        return [], [], [], set()
    if not all(existing.values()):
        raise ValueError("Resume requires a complete set of fold checkpoint files")

    fold_results = pd.read_csv(paths["fold"])
    counts = fold_results.groupby("fold").size()
    incomplete = counts[counts != EXPECTED_FOLD_RESULT_ROWS]
    if not incomplete.empty:
        raise ValueError(f"Incomplete fold checkpoint rows: {incomplete.to_dict()}")
    completed_folds = {int(fold) for fold in counts.index}
    per_class_results = pd.read_csv(paths["per_class"])
    confusion_results = pd.read_csv(paths["confusion"])
    per_class_results = per_class_results[per_class_results["fold"].isin(completed_folds)]
    confusion_results = confusion_results[confusion_results["fold"].isin(completed_folds)]
    return (
        fold_results.to_dict("records"),
        per_class_results.to_dict("records"),
        confusion_results.to_dict("records"),
        completed_folds,
    )


def run_loso_comparison(
    manifest: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    quick: bool = False,
    device: str = "cpu",
    resume: bool = False,
    batch_size: int | None = None,
    augment_minority: bool = False,
    aug_multiplier: int = 3,
    aug_methods: tuple[str, ...] = ("jitter", "magnitude_scale", "time_warp"),
    aug_target_labels: tuple[int, ...] = (6, 7, 8),
) -> pd.DataFrame:
    """Run binary and nine-class LOSO comparisons and write reproducible results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_model_inputs(
        manifest,
        augment_minority=augment_minority,
        target_labels=aug_target_labels,
        aug_methods=aug_methods,
        aug_multiplier=aug_multiplier,
        seed=seed,
    )
    if not len(dataset.metadata):
        raise ValueError("No usable windows were produced from the supplied manifest")
    if dataset.metadata["subject_id"].nunique() < 2:
        raise ValueError("LOSO evaluation requires at least two subjects")
    training_config = QUICK_TRAINING if quick else TrainingConfig()
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        training_config = replace(training_config, batch_size=batch_size)

    config_record = {
        "seed": seed,
        "quick": quick,
        "device": device,
        "window_ms": project_config.WINDOW_MS,
        "window_overlap": project_config.WINDOW_OVERLAP,
        "sequence_shape": list(dataset.transformer_sequences.shape[1:]),
        "emg_representation": "absolute value followed by equal-duration bin averages",
        "primary_task": "binary",
        "secondary_task": "nine_class",
        "evaluation": "leave_one_subject_out",
        "subjects": [int(value) for value in sorted(dataset.metadata["subject_id"].unique())],
        "n_windows": dataset.n_samples,
        "augmentation": {
            "enabled": augment_minority,
            "multiplier": aug_multiplier,
            "methods": list(aug_methods),
            "target_labels": list(aug_target_labels),
        },
        "transformer": {
            "input_dim": 56,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "max_seq_len": 30,
            "optimizer": "AdamW",
            "learning_rate": training_config.learning_rate,
            "batch_size": training_config.batch_size,
            "max_epochs": training_config.max_epochs,
            "patience": training_config.patience,
        },
        "xgboost": {
            "n_estimators": training_config.xgb_estimators,
            "max_depth": 4,
            "learning_rate": 0.05,
            "seed": seed,
            "n_jobs": training_config.xgb_n_jobs,
        },
    }
    _validate_or_write_run_config(output_dir, config_record, resume)
    _write_experiment_record(
        output_dir,
        dataset,
        manifest,
        seed,
        quick,
        training_config,
        device,
        "leave-one-subject-out; validation subjects are selected only from each training fold",
        augment_minority=augment_minority,
        aug_multiplier=aug_multiplier,
        aug_methods=aug_methods,
        aug_target_labels=aug_target_labels,
    )

    all_rows, all_per_class, all_confusion, completed_folds = _load_result_checkpoint(
        output_dir, resume
    )
    task_labels = {"binary": dataset.labels_binary, "nine_class": dataset.labels_9}
    task_classes = {"binary": 2, "nine_class": 9}

    for fold, (held_out, train_indices, test_indices) in enumerate(
        iter_loso_folds(dataset.metadata), start=1
    ):
        if fold in completed_folds:
            logger.info(
                "Fold %d already complete; resuming after held-out subject %s", fold, held_out
            )
            continue
        fit_indices, validation_indices = subject_validation_split(
            dataset.metadata, train_indices, seed=seed + fold
        )
        normalized_sequences, sequence_normalizer = normalize_sequences(
            dataset.transformer_sequences, fit_indices
        )
        logger.info(
            "Fold %d/%d: held-out subject %s (%d test windows)",
            fold,
            dataset.metadata["subject_id"].nunique(),
            held_out,
            len(test_indices),
        )
        for task, labels in task_labels.items():
            num_classes = task_classes[task]
            baseline_probabilities = _majority_probabilities(labels, fit_indices, num_classes)[
                test_indices
            ]
            baseline_rows, baseline_classes, baseline_confusion = _evaluate_prediction(
                dataset,
                test_indices,
                baseline_probabilities,
                labels,
                num_classes,
                model="majority_baseline",
                task=task,
                fold=fold,
                held_out_subject=held_out,
                train_indices=fit_indices,
                validation_indices=validation_indices,
            )
            all_rows.extend(baseline_rows)
            all_per_class.extend(baseline_classes)
            all_confusion.extend(baseline_confusion)

            transformer, normalizer, _ = fit_transformer(
                dataset.transformer_sequences,
                labels,
                fit_indices,
                validation_indices,
                num_classes=num_classes,
                seed=seed + fold,
                training_config=training_config,
                device=device,
                normalizer=sequence_normalizer,
                normalized_sequences=normalized_sequences,
            )
            transformer_probabilities = predict_transformer(
                transformer, dataset.transformer_sequences[test_indices], normalizer
            )
            transformer_rows, transformer_classes, transformer_confusion = _evaluate_prediction(
                dataset,
                test_indices,
                transformer_probabilities,
                labels,
                num_classes,
                model="transformer",
                task=task,
                fold=fold,
                held_out_subject=held_out,
                train_indices=fit_indices,
                validation_indices=validation_indices,
            )
            all_rows.extend(transformer_rows)
            all_per_class.extend(transformer_classes)
            all_confusion.extend(transformer_confusion)

            xgb_model = fit_xgboost(
                dataset.xgboost_features,
                labels,
                fit_indices,
                num_classes=num_classes,
                seed=seed + fold,
                training_config=training_config,
            )
            xgb_probabilities = predict_xgboost(
                xgb_model, dataset.xgboost_features.iloc[test_indices], num_classes
            )
            xgb_rows, xgb_classes, xgb_confusion = _evaluate_prediction(
                dataset,
                test_indices,
                xgb_probabilities,
                labels,
                num_classes,
                model="xgboost",
                task=task,
                fold=fold,
                held_out_subject=held_out,
                train_indices=fit_indices,
                validation_indices=validation_indices,
            )
            all_rows.extend(xgb_rows)
            all_per_class.extend(xgb_classes)
            all_confusion.extend(xgb_confusion)

        _write_result_checkpoint(output_dir, all_rows, all_per_class, all_confusion)

    _write_result_checkpoint(output_dir, all_rows, all_per_class, all_confusion)
    return pd.read_csv(output_dir / "fold_results.csv")


def run_fixed_split_comparison(
    manifest: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    quick: bool = False,
    device: str = "cpu",
    resume: bool = False,
    batch_size: int | None = None,
    train_subjects: Iterable[int] = FIXED_TRAIN_SUBJECTS,
    validation_subjects: Iterable[int] = FIXED_VALIDATION_SUBJECTS,
    test_subjects: Iterable[int] = FIXED_TEST_SUBJECTS,
) -> pd.DataFrame:
    """Run one controlled subject-wise train/validation/test comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_subject_list = sorted({int(value) for value in train_subjects})
    validation_subject_list = sorted({int(value) for value in validation_subjects})
    test_subject_list = sorted({int(value) for value in test_subjects})
    selected_subjects = set(train_subject_list + validation_subject_list + test_subject_list)
    evaluation_manifest = manifest.loc[
        manifest["subject_id"].astype(int).isin(selected_subjects)
    ].reset_index(drop=True)
    dataset = build_model_inputs(evaluation_manifest)
    if not len(dataset.metadata):
        raise ValueError("No usable windows were produced from the supplied manifest")

    train_indices, validation_indices, test_indices = fixed_subject_split(
        dataset.metadata,
        train_subject_list,
        validation_subject_list,
        test_subject_list,
    )
    training_config = QUICK_TRAINING if quick else TrainingConfig()
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        training_config = replace(training_config, batch_size=batch_size)

    config_record = {
        "seed": seed,
        "quick": quick,
        "device": device,
        "window_ms": project_config.WINDOW_MS,
        "window_overlap": project_config.WINDOW_OVERLAP,
        "sequence_shape": list(dataset.transformer_sequences.shape[1:]),
        "emg_representation": "absolute value followed by equal-duration bin averages",
        "primary_task": "binary",
        "secondary_task": "nine_class",
        "evaluation": "fixed_subject_split",
        "train_subjects": train_subject_list,
        "validation_subjects": validation_subject_list,
        "test_subjects": test_subject_list,
        "excluded_subjects": sorted(
            {int(value) for value in manifest["subject_id"].unique()} - selected_subjects
        ),
        "n_windows": len(train_indices) + len(validation_indices) + len(test_indices),
        "transformer": {
            "input_dim": 56,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "max_seq_len": 30,
            "optimizer": "AdamW",
            "learning_rate": training_config.learning_rate,
            "batch_size": training_config.batch_size,
            "max_epochs": training_config.max_epochs,
            "patience": training_config.patience,
        },
        "xgboost": {
            "n_estimators": training_config.xgb_estimators,
            "max_depth": 4,
            "learning_rate": 0.05,
            "seed": seed,
            "n_jobs": training_config.xgb_n_jobs,
        },
    }
    _validate_or_write_run_config(output_dir, config_record, resume)
    _write_experiment_record(
        output_dir,
        dataset,
        evaluation_manifest,
        seed,
        quick,
        training_config,
        device,
        f"fixed subject split with {len(train_subject_list)} training, "
        f"{len(validation_subject_list)} validation, and "
        f"{len(test_subject_list)} untouched test subjects",
    )

    all_rows, all_per_class, all_confusion, completed_folds = _load_result_checkpoint(
        output_dir, resume
    )
    if 1 in completed_folds:
        return pd.DataFrame(all_rows)

    normalized_sequences, sequence_normalizer = normalize_sequences(
        dataset.transformer_sequences, train_indices
    )
    task_labels = {"binary": dataset.labels_binary, "nine_class": dataset.labels_9}
    task_classes = {"binary": 2, "nine_class": 9}
    for task, labels in task_labels.items():
        num_classes = task_classes[task]
        baseline_probabilities = _majority_probabilities(labels, train_indices, num_classes)[
            test_indices
        ]
        baseline_rows, baseline_classes, baseline_confusion = _evaluate_prediction(
            dataset,
            test_indices,
            baseline_probabilities,
            labels,
            num_classes,
            model="majority_baseline",
            task=task,
            fold=1,
            held_out_subject=None,
            train_indices=train_indices,
            validation_indices=validation_indices,
        )
        all_rows.extend(baseline_rows)
        all_per_class.extend(baseline_classes)
        all_confusion.extend(baseline_confusion)

        transformer, normalizer, _ = fit_transformer(
            dataset.transformer_sequences,
            labels,
            train_indices,
            validation_indices,
            num_classes=num_classes,
            seed=seed,
            training_config=training_config,
            device=device,
            normalizer=sequence_normalizer,
            normalized_sequences=normalized_sequences,
        )
        transformer_probabilities = predict_transformer(
            transformer, dataset.transformer_sequences[test_indices], normalizer
        )
        transformer_rows, transformer_classes, transformer_confusion = _evaluate_prediction(
            dataset,
            test_indices,
            transformer_probabilities,
            labels,
            num_classes,
            model="transformer",
            task=task,
            fold=1,
            held_out_subject=None,
            train_indices=train_indices,
            validation_indices=validation_indices,
        )
        all_rows.extend(transformer_rows)
        all_per_class.extend(transformer_classes)
        all_confusion.extend(transformer_confusion)

        xgb_model = fit_xgboost(
            dataset.xgboost_features,
            labels,
            train_indices,
            num_classes=num_classes,
            seed=seed,
            training_config=training_config,
        )
        xgb_probabilities = predict_xgboost(
            xgb_model, dataset.xgboost_features.iloc[test_indices], num_classes
        )
        xgb_rows, xgb_classes, xgb_confusion = _evaluate_prediction(
            dataset,
            test_indices,
            xgb_probabilities,
            labels,
            num_classes,
            model="xgboost",
            task=task,
            fold=1,
            held_out_subject=None,
            train_indices=train_indices,
            validation_indices=validation_indices,
        )
        all_rows.extend(xgb_rows)
        all_per_class.extend(xgb_classes)
        all_confusion.extend(xgb_confusion)

    _write_result_checkpoint(output_dir, all_rows, all_per_class, all_confusion)
    return pd.read_csv(output_dir / "fold_results.csv")
