import warnings

import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score


def aggregate_trial_predictions(
    window_probs: np.ndarray,
    trial_ids: np.ndarray,
    method: str = "mean_prob"
) -> dict:
    """
    Aggregates window-level probabilities to trial-level predictions.
    window_probs: shape (N_windows, N_classes)
    """
    unique_trials = np.unique(trial_ids)
    trial_preds = []

    for t_id in unique_trials:
        mask = (trial_ids == t_id)
        t_probs = window_probs[mask]

        if method == "mean_prob":
            avg_prob = np.mean(t_probs, axis=0)
            pred = np.argmax(avg_prob)

        elif method == "majority_vote":
            window_preds = np.argmax(t_probs, axis=1)
            pred = mode(window_preds, keepdims=False).mode

        elif method == "confidence_weighted":
            # Confidence metric: distance from uniform/uncertainty or max probability
            confidences = np.max(t_probs, axis=1, keepdims=True)
            weighted_probs = np.sum(t_probs * confidences, axis=0) / (np.sum(confidences) + 1e-8)
            pred = np.argmax(weighted_probs)

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        trial_preds.append((t_id, pred))

    return dict(trial_preds)

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    all_classes: list | None = None,
) -> dict:
    labels = all_classes if all_classes is not None else np.unique(y_true)

    with warnings.catch_warnings():
        # Temporarily suppress scikit-learn warnings about missing classes in y_true
        warnings.simplefilter("ignore", category=UserWarning)

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "per_class_recall": np.atleast_1d(
                recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
            ).tolist()
        }
