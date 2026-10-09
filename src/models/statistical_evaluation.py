"""Subject-clustered evidence for paired model comparisons."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def validate_paired_trial_alignment(
    trial_predictions: pd.DataFrame,
    *,
    reference_model: str = "transformer",
    baseline_model: str = "xgboost_aligned",
) -> None:
    """Reject a paired comparison unless both models cover identical labelled trials."""
    required = {
        "protocol",
        "task",
        "model",
        "subject_id",
        "trial_id",
        "true_class",
    }
    missing = sorted(required.difference(trial_predictions.columns))
    if missing:
        raise ValueError(f"Trial predictions are missing columns: {missing}")

    keys = ["protocol", "task", "subject_id", "trial_id"]
    reference = trial_predictions[trial_predictions["model"] == reference_model][
        keys + ["true_class"]
    ]
    baseline = trial_predictions[trial_predictions["model"] == baseline_model][
        keys + ["true_class"]
    ]
    if reference.empty or baseline.empty:
        raise ValueError("Both paired comparison models must have trial predictions")
    if reference.duplicated(keys).any() or baseline.duplicated(keys).any():
        raise ValueError("Each model must contain exactly one prediction per trial")

    aligned = reference.merge(
        baseline,
        on=keys,
        how="outer",
        suffixes=("_reference", "_baseline"),
        indicator=True,
    )
    if not (aligned["_merge"] == "both").all():
        raise ValueError("Paired comparison models do not cover identical trials")
    if not (
        aligned["true_class_reference"].to_numpy() == aligned["true_class_baseline"].to_numpy()
    ).all():
        raise ValueError("Paired comparison models disagree on trial labels")


def per_subject_trial_metrics(trial_predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize trial predictions without treating one subject's trials as independent."""
    required = {
        "protocol",
        "task",
        "model",
        "calibration",
        "subject_id",
        "trial_id",
        "true_class",
        "predicted_class",
        "num_classes",
        "input_representation",
    }
    missing = sorted(required.difference(trial_predictions.columns))
    if missing:
        raise ValueError(f"Trial predictions are missing columns: {missing}")
    if trial_predictions.empty:
        raise ValueError("Trial predictions must not be empty")

    rows: list[dict[str, object]] = []
    group_columns = [
        "protocol",
        "task",
        "model",
        "calibration",
        "input_representation",
        "subject_id",
    ]
    for keys, group in trial_predictions.groupby(group_columns, sort=True):
        protocol, task, model, calibration, input_representation, subject_id = keys
        num_classes_values = group["num_classes"].unique()
        if len(num_classes_values) != 1:
            raise ValueError("num_classes must be constant within a subject/model/task group")
        classes = list(range(int(num_classes_values[0])))
        targets = group["true_class"].to_numpy(dtype=int)
        predictions = group["predicted_class"].to_numpy(dtype=int)
        rows.append(
            {
                "protocol": protocol,
                "task": task,
                "level": "trial",
                "model": model,
                "calibration": calibration,
                "input_representation": input_representation,
                "subject_id": int(subject_id),
                "accuracy": float(accuracy_score(targets, predictions)),
                "macro_f1": float(
                    f1_score(
                        targets,
                        predictions,
                        labels=classes,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
                "n_trials": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _paired_sign_flip_p_value(differences: np.ndarray) -> float:
    """Return an exact two-sided paired randomization p-value over subjects."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("differences must be a non-empty finite vector")
    if len(values) > 20:
        raise ValueError("Exact sign-flip testing is limited to 20 paired subjects")
    observed = abs(float(values.mean()))
    permuted = [
        abs(float(np.mean(values * signs))) for signs in product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-12))


def paired_subject_comparisons(
    subject_metrics: pd.DataFrame,
    *,
    reference_model: str = "transformer",
    baseline_model: str = "xgboost_aligned",
    seed: int = 42,
    bootstrap_samples: int = 10_000,
) -> pd.DataFrame:
    """Compare models using paired subject metrics and subject-bootstrap CIs."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    required = {
        "protocol",
        "task",
        "model",
        "subject_id",
        "accuracy",
        "macro_f1",
        "input_representation",
    }
    missing = sorted(required.difference(subject_metrics.columns))
    if missing:
        raise ValueError(f"Subject metrics are missing columns: {missing}")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (protocol, task), group in subject_metrics.groupby(["protocol", "task"], sort=True):
        reference = group[group["model"] == reference_model].set_index("subject_id")
        baseline = group[group["model"] == baseline_model].set_index("subject_id")
        subjects = reference.index.intersection(baseline.index).sort_values()
        if subjects.empty:
            continue
        if len(subjects) != len(reference) or len(subjects) != len(baseline):
            raise ValueError("Paired model comparison requires the same subjects for both models")

        for metric in ("accuracy", "macro_f1"):
            reference_values = reference.loc[subjects, metric].to_numpy(dtype=float)
            baseline_values = baseline.loc[subjects, metric].to_numpy(dtype=float)
            differences = reference_values - baseline_values
            reference_representations = reference.loc[subjects, "input_representation"].unique()
            baseline_representations = baseline.loc[subjects, "input_representation"].unique()
            if len(reference_representations) != 1 or len(baseline_representations) != 1:
                raise ValueError("Each paired model must use one declared input representation")
            sampled_subjects = rng.integers(
                0, len(subjects), size=(bootstrap_samples, len(subjects))
            )
            bootstrap_means = differences[sampled_subjects].mean(axis=1)
            rows.append(
                {
                    "protocol": protocol,
                    "task": task,
                    "level": "trial",
                    "metric": metric,
                    "reference_model": reference_model,
                    "baseline_model": baseline_model,
                    "reference_input_representation": str(reference_representations[0]),
                    "baseline_input_representation": str(baseline_representations[0]),
                    "n_subjects": int(len(subjects)),
                    "reference_subject_mean": float(reference_values.mean()),
                    "reference_subject_sd": float(reference_values.std(ddof=1))
                    if len(subjects) > 1
                    else 0.0,
                    "baseline_subject_mean": float(baseline_values.mean()),
                    "baseline_subject_sd": float(baseline_values.std(ddof=1))
                    if len(subjects) > 1
                    else 0.0,
                    "paired_mean_difference": float(differences.mean()),
                    "subject_bootstrap_ci_lower": float(np.quantile(bootstrap_means, 0.025)),
                    "subject_bootstrap_ci_upper": float(np.quantile(bootstrap_means, 0.975)),
                    "paired_sign_flip_p_value": _paired_sign_flip_p_value(differences),
                    "bootstrap_samples": int(bootstrap_samples),
                    "bootstrap_unit": "subject",
                    "inference_scope": "exploratory_five_subject_test",
                }
            )
    return pd.DataFrame(rows)
