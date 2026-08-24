"""Repeatable CPU training helpers for the two model families."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from src.models.normalization import SequenceNormalizer, fit_sequence_normalizer


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5
    validation_batch_size: int = 256
    xgb_estimators: int = 120


QUICK_TRAINING = TrainingConfig(max_epochs=2, patience=1, xgb_estimators=20)


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when it is available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CPU CI is expected
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - only relevant for XGBoost-only use
        pass


def _class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    present = counts > 0
    weights = np.ones(num_classes, dtype=np.float32)
    if present.any():
        weights[present] = len(labels) / (present.sum() * counts[present])
    return weights


def fit_transformer(
    sequences: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    num_classes: int,
    seed: int = 42,
    training_config: TrainingConfig | None = None,
) -> tuple[Any, SequenceNormalizer, list[dict[str, float]]]:
    """Train the Transformer and select its state using validation macro-F1."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.transformer import TransformerEncoderClassifier

    cfg = training_config or TrainingConfig()
    set_random_seed(seed)
    train_indices = np.asarray(train_indices, dtype=int)
    validation_indices = np.asarray(validation_indices, dtype=int)
    normalizer = fit_sequence_normalizer(sequences, train_indices)
    normalized = normalizer.transform(sequences)
    x_train = torch.from_numpy(normalized[train_indices])
    y_train = torch.from_numpy(np.asarray(labels, dtype=np.int64)[train_indices])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = TransformerEncoderClassifier(num_classes=num_classes)
    class_weights = torch.from_numpy(_class_weights(y_train.numpy(), num_classes))
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    validation_x = torch.from_numpy(normalized[validation_indices])
    validation_y = np.asarray(labels, dtype=np.int64)[validation_indices]

    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(cfg.max_epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            if len(validation_indices):
                validation_logits = model(validation_x)
                validation_predictions = validation_logits.argmax(dim=1).cpu().numpy()
                score = float(
                    f1_score(
                        validation_y,
                        validation_predictions,
                        labels=list(range(num_classes)),
                        average="macro",
                        zero_division=0,
                    )
                )
            else:
                # With a single training participant there is no independent
                # subject available for inner validation; retain the first state.
                score = -float(np.mean(losses)) if losses else -np.inf
        history.append(
            {"epoch": float(epoch + 1), "loss": float(np.mean(losses)), "val_macro_f1": score}
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, normalizer, history


def predict_transformer(
    model: Any, sequences: np.ndarray, normalizer: SequenceNormalizer, batch_size: int = 256
) -> np.ndarray:
    """Return finite softmax probabilities for all supplied sequences."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    values = normalizer.transform(sequences)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values)), batch_size=batch_size, shuffle=False
    )
    probabilities = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            probabilities.append(torch.softmax(model(batch_x), dim=1).cpu().numpy())
    if not probabilities:
        return np.empty((0, model.classifier.out_features), dtype=np.float32)
    result = np.concatenate(probabilities, axis=0).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("Transformer produced non-finite probabilities")
    return result


class ConstantClassifier:
    """Probability-producing fallback for a fold whose training data has one class."""

    def __init__(self, class_index: int, num_classes: int) -> None:
        self.class_index = int(class_index)
        self.num_classes = int(num_classes)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        result = np.zeros((len(values), self.num_classes), dtype=np.float32)
        result[:, self.class_index] = 1.0
        return result


def fit_xgboost(
    features,
    labels: np.ndarray,
    train_indices: np.ndarray,
    num_classes: int,
    seed: int = 42,
    training_config: TrainingConfig | None = None,
):
    """Fit the fixed-configuration XGBoost baseline on training rows only."""
    cfg = training_config or TrainingConfig()
    x_train = features.iloc[np.asarray(train_indices, dtype=int)].to_numpy(dtype=np.float32)
    y_train = np.asarray(labels, dtype=np.int64)[np.asarray(train_indices, dtype=int)]
    unique = np.unique(y_train)
    if len(unique) < 2:
        return ConstantClassifier(int(unique[0]), num_classes)

    from xgboost import XGBClassifier

    weights_by_class = _class_weights(y_train, num_classes)
    sample_weights = weights_by_class[y_train]
    params = {
        "n_estimators": cfg.xgb_estimators,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob" if num_classes > 2 else "binary:logistic",
        "eval_metric": "mlogloss" if num_classes > 2 else "logloss",
        "random_state": seed,
        "n_jobs": 1,
        "tree_method": "hist",
    }
    if num_classes > 2:
        params["num_class"] = num_classes
    model = XGBClassifier(**params)
    model.fit(x_train, y_train, sample_weight=sample_weights)
    return model


def predict_xgboost(model, features, num_classes: int) -> np.ndarray:
    """Expand estimator probabilities to the requested global class order."""
    probabilities = model.predict_proba(features.to_numpy(dtype=np.float32))
    if probabilities.shape[1] == num_classes:
        result = np.asarray(probabilities, dtype=np.float32)
    else:
        result = np.zeros((len(features), num_classes), dtype=np.float32)
        for column, class_index in enumerate(model.classes_):
            result[:, int(class_index)] = probabilities[:, column]
    if not np.all(np.isfinite(result)):
        raise ValueError("XGBoost produced non-finite probabilities")
    return result
