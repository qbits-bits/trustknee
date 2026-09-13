"""Variable-length full-trial TCN experiment using the shared model dataset."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn

from src import config
from src.contracts import get_canonical_contract
from src.ingestion import build_manifest
from src.models.evaluation import fixed_subject_split
from src.models.model_data import ModelDataset, build_model_inputs
from src.models.training import set_random_seed
from src.wrapper.dataset_seq import SequenceDataset, dynamic_padding_collate


@dataclass(frozen=True)
class TCNConfig:
    input_dim: int = 56
    hidden_dim: int = 64
    kernel_size: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 16
    max_epochs: int = 30
    patience: int = 5


class TemporalConvNetClassifier(nn.Module):
    """Small temporal convolutional classifier with padding-aware pooling."""

    def __init__(
        self,
        input_dim: int = 56,
        num_classes: int = 2,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes < 2 or hidden_dim <= 0 or kernel_size <= 0:
            raise ValueError("TCN dimensions must be positive and num_classes must be at least two")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so temporal length is preserved")
        self.model_config = {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dim": hidden_dim,
            "kernel_size": kernel_size,
            "dropout": dropout,
        }
        self.first_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.second_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            padding=kernel_size - 1,
            dilation=2,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, values, mask=None):
        if mask is None:
            valid = None
        else:
            if mask.shape != (values.shape[0], values.shape[-1]):
                raise ValueError("mask must have shape (batch_size, sequence_length)")
            valid = mask.to(values.device, dtype=values.dtype).unsqueeze(1)
            values = values * valid
        encoded = self.dropout(self.activation(self.first_conv(values)))
        if valid is not None:
            encoded = encoded * valid
        encoded = self.dropout(self.activation(self.second_conv(encoded)))
        if valid is None:
            pooled = encoded.mean(dim=-1)
        else:
            encoded = encoded * valid
            pooled = (encoded * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        return self.classifier(pooled)


def build_full_trial_sequences(
    dataset: ModelDataset,
    labels: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, pd.DataFrame]:
    """Convert synchronized windows into variable-length 56-channel trial sequences."""
    sequences = []
    trial_labels = []
    metadata_rows = []
    local = dataset.metadata.reset_index(drop=True)
    targets = np.asarray(labels, dtype=np.int64)
    for trial_id, group in local.groupby("trial_id", sort=True):
        indices = group.index.to_numpy()
        # One token per 200 ms window. Averaging inside each synchronized window
        # avoids concatenating duplicated samples when windows overlap.
        tokens = dataset.transformer_sequences[indices].mean(axis=1)
        sequences.append(tokens.T.astype(np.float32))
        trial_labels.append(int(targets[indices[0]]))
        metadata_rows.append(
            {
                "trial_id": trial_id,
                "subject_id": int(group["subject_id"].iloc[0]),
                "label_id": int(group["label_id"].iloc[0]),
                "trial_num": int(group["trial_num"].iloc[0]),
                "n_windows": len(indices),
            }
        )
    return sequences, np.asarray(trial_labels), pd.DataFrame(metadata_rows)


def _fit_trial_normalizer(
    sequences: list[np.ndarray],
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    training_values = np.concatenate([sequences[index] for index in train_indices], axis=1)
    mean = training_values.mean(axis=1, keepdims=True).astype(np.float32)
    scale = training_values.std(axis=1, keepdims=True).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _loader(sequences, labels, indices, batch_size, shuffle):
    from torch.utils.data import DataLoader

    selected_sequences = [(sequences[index]).astype(np.float32) for index in indices]
    selected_labels = np.asarray(labels, dtype=np.int64)[indices]
    return DataLoader(
        SequenceDataset(selected_sequences, selected_labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=dynamic_padding_collate,
    )


def _predict(model, loader, device):
    import torch

    probabilities = []
    labels = []
    model.eval()
    with torch.no_grad():
        for values, target, mask in loader:
            logits = model(values.to(device), mask.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def _result(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    classes = list(range(probabilities.shape[1]))
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=classes,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": class_f1.tolist(),
        "support": support.tolist(),
    }


def run_tcn_experiment(
    dataset: ModelDataset,
    output_dir: Path,
    *,
    task: str = "binary",
    seed: int = 42,
    device: str = "cpu",
    quick: bool = False,
) -> dict[str, object]:
    """Train and evaluate a reproducible variable-length full-trial TCN."""
    import torch
    from torch import nn

    if task == "binary":
        labels, num_classes = dataset.labels_binary, 2
    elif task == "nine_class":
        labels, num_classes = dataset.labels_9, 9
    else:
        raise ValueError("task must be binary or nine_class")
    set_random_seed(seed)
    sequences, trial_labels, trial_metadata = build_full_trial_sequences(dataset, labels)
    contract = get_canonical_contract()
    train_indices, validation_indices, test_indices = fixed_subject_split(
        trial_metadata,
        contract.train_subjects,
        contract.val_subjects,
        contract.test_subjects,
    )
    mean, scale = _fit_trial_normalizer(sequences, train_indices)
    normalized = [(sequence - mean) / scale for sequence in sequences]
    cfg = TCNConfig(max_epochs=2, patience=1) if quick else TCNConfig()
    train_loader = _loader(normalized, trial_labels, train_indices, cfg.batch_size, True)
    validation_loader = _loader(
        normalized,
        trial_labels,
        validation_indices,
        cfg.batch_size,
        False,
    )
    test_loader = _loader(normalized, trial_labels, test_indices, cfg.batch_size, False)
    model = TemporalConvNetClassifier(
        input_dim=cfg.input_dim,
        num_classes=num_classes,
        hidden_dim=cfg.hidden_dim,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    ).to(device)
    counts = np.bincount(trial_labels[train_indices], minlength=num_classes).astype(float)
    weights = np.ones(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = len(train_indices) / (present.sum() * counts[present])
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    stale = 0
    history = []
    for epoch in range(cfg.max_epochs):
        model.train()
        losses = []
        for values, target, mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values.to(device), mask.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_labels, validation_probabilities = _predict(model, validation_loader, device)
        validation_f1 = float(
            f1_score(
                validation_labels,
                validation_probabilities.argmax(axis=1),
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_macro_f1": validation_f1,
            }
        )
        if validation_f1 > best_f1 + 1e-8:
            best_f1 = validation_f1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    started = time.perf_counter()
    test_labels, test_probabilities = _predict(model, test_loader, device)
    latency_ms = (time.perf_counter() - started) * 1000.0 / len(test_indices)
    result: dict[str, object] = {
        "task": task,
        "model": "full_trial_tcn",
        **_result(test_labels, test_probabilities),
        "latency_ms_per_trial": latency_ms,
        "n_train_trials": len(train_indices),
        "n_validation_trials": len(validation_indices),
        "n_test_trials": len(test_indices),
        "seed": seed,
        "quick": quick,
        "config": asdict(cfg),
        "contract": asdict(contract),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_config": model.model_config,
            "model_state_dict": model.state_dict(),
            "normalizer_mean": torch.from_numpy(mean),
            "normalizer_scale": torch.from_numpy(scale),
            "task": task,
            "metadata": result,
        },
        destination / f"tcn_{task}.pt",
    )
    (destination / f"tcn_{task}_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (destination / f"tcn_{task}_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("binary", "nine_class"), default="binary")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=config.PHASE3_RANDOM_SEED)
    parser.add_argument("--quick", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = build_model_inputs(build_manifest(args.data_root), seed=args.seed)
    result = run_tcn_experiment(
        dataset,
        args.output_dir,
        task=args.task,
        seed=args.seed,
        device=args.device,
        quick=args.quick,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
