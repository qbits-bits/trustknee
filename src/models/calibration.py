"""Validation-only probability calibration and uncertainty selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_probabilities(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(targets) or not len(values):
        raise ValueError("Probabilities must be a non-empty 2D array aligned with labels")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    if np.any(targets < 0) or np.any(targets >= values.shape[1]):
        raise ValueError("Labels fall outside the probability class range")
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("Probability rows must sum to one")
    return values, targets


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Logits must be a non-empty finite 2D array")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Return multiclass mean negative log likelihood."""
    values, targets = _validate_probabilities(probabilities, labels)
    selected = np.clip(values[np.arange(len(targets)), targets], 1e-12, 1.0)
    return float(-np.log(selected).mean())


@dataclass(frozen=True)
class TemperatureScaler:
    """A positive scalar temperature fitted on validation logits only."""

    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")

    @classmethod
    def fit(
        cls,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 50,
    ) -> TemperatureScaler:
        """Fit temperature by minimizing validation-set cross entropy."""
        import torch
        from torch.nn import functional as torch_functional

        values = np.asarray(logits, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64)
        if values.ndim != 2 or len(values) != len(targets) or not len(values):
            raise ValueError("Logits must be a non-empty 2D array aligned with labels")
        if not np.all(np.isfinite(values)):
            raise ValueError("Logits must be finite")
        if np.any(targets < 0) or np.any(targets >= values.shape[1]):
            raise ValueError("Labels fall outside the logit class range")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        logits_tensor = torch.as_tensor(values, dtype=torch.float64)
        labels_tensor = torch.as_tensor(targets, dtype=torch.long)
        log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [log_temperature],
            lr=0.1,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            temperature = log_temperature.exp().clamp(0.05, 10.0)
            loss = torch_functional.cross_entropy(logits_tensor / temperature, labels_tensor)
            loss.backward()
            return loss

        initial_nll = negative_log_likelihood(_softmax(values), targets)
        optimizer.step(closure)
        fitted = float(log_temperature.detach().exp().clamp(0.05, 10.0).item())
        scaler = cls(fitted)
        fitted_nll = negative_log_likelihood(scaler.predict_proba(values), targets)
        return scaler if fitted_nll <= initial_nll + 1e-10 else cls()

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=np.float64)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("Logits must be a finite 2D array")
        return values / self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return _softmax(self.transform_logits(logits)).astype(np.float32)

    def to_dict(self) -> dict[str, float]:
        return {"temperature": float(self.temperature)}

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> TemperatureScaler:
        return cls(float(values["temperature"]))


def reliability_bins(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> list[dict[str, float | int]]:
    """Create equal-width confidence bins for a reliability diagram."""
    values, targets = _validate_probabilities(probabilities, labels)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    confidence = values.max(axis=1)
    correct = values.argmax(axis=1) == targets
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for index in range(n_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": float(confidence[mask].mean()) if count else 0.0,
                "accuracy": float(correct[mask].mean()) if count else 0.0,
            }
        )
    return rows


def compute_calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict[str, float]:
    """Compute ECE, generalized multiclass Brier score, and NLL."""
    values, targets = _validate_probabilities(probabilities, labels)
    bins = reliability_bins(values, targets, n_bins=n_bins)
    ece = sum(
        row["count"] / len(values) * abs(row["accuracy"] - row["mean_confidence"]) for row in bins
    )
    one_hot = np.eye(values.shape[1], dtype=np.float64)[targets]
    brier = np.square(values - one_hot).sum(axis=1).mean()
    return {
        "ece": float(ece),
        "brier_score": float(brier),
        "nll": negative_log_likelihood(values, targets),
    }


def risk_coverage_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, float]]:
    """Return selective performance for every attainable confidence threshold."""
    values, targets = _validate_probabilities(probabilities, labels)
    confidence = values.max(axis=1)
    predictions = values.argmax(axis=1)
    order = np.argsort(-confidence, kind="stable")
    sorted_confidence = confidence[order]
    sorted_targets = targets[order]
    sorted_predictions = predictions[order]
    matrix = np.zeros((values.shape[1], values.shape[1]), dtype=np.int64)
    descending_rows: list[dict[str, float]] = []
    accepted_count = 0
    cursor = 0
    while cursor < len(values):
        threshold = sorted_confidence[cursor]
        group_end = cursor + 1
        while group_end < len(values) and sorted_confidence[group_end] == threshold:
            group_end += 1
        np.add.at(
            matrix,
            (sorted_targets[cursor:group_end], sorted_predictions[cursor:group_end]),
            1,
        )
        accepted_count = group_end
        true_positive = np.diag(matrix).astype(np.float64)
        false_positive = matrix.sum(axis=0) - true_positive
        false_negative = matrix.sum(axis=1) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1 = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=denominator != 0,
        )
        accuracy = float(true_positive.sum() / accepted_count)
        descending_rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(accepted_count / len(values)),
                "accuracy": accuracy,
                "selective_risk": 1.0 - accuracy,
                "macro_f1": float(class_f1.mean()),
            }
        )
        cursor = group_end

    rows = list(reversed(descending_rows))
    if rows[0]["threshold"] > 0.0:
        rows.insert(0, {**rows[0], "threshold": 0.0})
    return rows


def select_uncertainty_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    min_coverage: float = 0.80,
) -> dict[str, float]:
    """Select the validation threshold with best Macro-F1 above a coverage floor."""
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    eligible = [
        row for row in risk_coverage_curve(probabilities, labels) if row["coverage"] >= min_coverage
    ]
    if not eligible:
        raise ValueError("No uncertainty threshold satisfies the minimum coverage")
    return max(
        eligible,
        key=lambda row: (
            row["macro_f1"],
            row["accuracy"],
            row["coverage"],
            -row["threshold"],
        ),
    )
