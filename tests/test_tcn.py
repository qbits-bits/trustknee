from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.tcn import (
    TemporalConvNetClassifier,
    build_full_trial_sequences,
    run_tcn_experiment,
)
from tests.test_sensor_selection import _synthetic_dataset


def test_tcn_accepts_dynamic_padding_and_returns_trial_logits():
    import torch
    from torch.nn.utils.rnn import pad_sequence

    model = TemporalConvNetClassifier(num_classes=2, dropout=0.0)
    sequences = [torch.randn(56, 5), torch.randn(56, 8)]
    values = pad_sequence([sequence.T for sequence in sequences], batch_first=True).transpose(1, 2)
    mask = torch.arange(8)[None, :] < torch.tensor([5, 8])[:, None]

    logits = model(values, mask)

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_tcn_rejects_even_kernel_that_changes_temporal_length():
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        TemporalConvNetClassifier(kernel_size=2)


def test_tcn_prediction_is_invariant_to_other_trials_padding_length():
    import torch

    torch.manual_seed(3)
    model = TemporalConvNetClassifier(num_classes=2, dropout=0.0).eval()
    short = torch.randn(1, 56, 5)
    short_mask = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        alone = model(short, short_mask)
        mixed_values = torch.zeros(2, 56, 9)
        mixed_values[0, :, :5] = short[0]
        mixed_values[1] = torch.randn(56, 9)
        mixed_mask = torch.arange(9)[None, :] < torch.tensor([5, 9])[:, None]
        mixed = model(mixed_values, mixed_mask)[0:1]

    assert torch.allclose(alone, mixed, atol=1e-6)


def test_full_trial_conversion_preserves_variable_window_counts():
    dataset = _synthetic_dataset()
    extra_index = 0
    dataset.transformer_sequences = np.concatenate(
        [dataset.transformer_sequences, dataset.transformer_sequences[[extra_index]]],
        axis=0,
    )
    dataset.xgboost_features.loc[len(dataset.xgboost_features)] = dataset.xgboost_features.iloc[
        extra_index
    ]
    dataset.labels_9 = np.append(dataset.labels_9, dataset.labels_9[extra_index])
    dataset.labels_binary = np.append(dataset.labels_binary, dataset.labels_binary[extra_index])
    extra_metadata = dataset.metadata.iloc[[extra_index]].copy()
    extra_metadata["window_index"] = 1
    dataset.metadata = pd.concat([dataset.metadata, extra_metadata], ignore_index=True)

    sequences, labels, metadata = build_full_trial_sequences(dataset, dataset.labels_binary)

    assert len(sequences) == len(labels) == len(metadata)
    assert {sequence.shape[1] for sequence in sequences} == {1, 2}


def test_quick_tcn_run_writes_reproducible_result(tmp_path):
    result = run_tcn_experiment(
        _synthetic_dataset(),
        tmp_path,
        quick=True,
    )

    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert (tmp_path / "tcn_binary.pt").exists()
    assert (tmp_path / "tcn_binary_result.json").exists()
