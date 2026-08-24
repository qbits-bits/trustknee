"""Training-only sequence normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceNormalizer:
    """Per-channel standardizer with statistics shaped for sequence broadcasting."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        values = np.asarray(sequences, dtype=np.float32)
        if values.ndim != 3 or values.shape[-1] != self.mean.shape[-1]:
            raise ValueError("Sequences do not match the fitted normalizer")
        return ((values - self.mean) / self.scale).astype(np.float32)


def fit_sequence_normalizer(sequences: np.ndarray, train_indices: np.ndarray) -> SequenceNormalizer:
    """Fit statistics using only rows in ``train_indices``.

    Means and scales are per sensor channel and are calculated over both batch
    and time dimensions.  A small scale floor handles constant synthetic data.
    """
    values = np.asarray(sequences, dtype=np.float32)
    indices = np.asarray(train_indices, dtype=int)
    if values.ndim != 3 or not len(indices):
        raise ValueError("Need non-empty 3D sequences and training indices")
    training_values = values[indices]
    mean = training_values.mean(axis=(0, 1), keepdims=True)
    scale = training_values.std(axis=(0, 1), keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return SequenceNormalizer(mean=mean.astype(np.float32), scale=scale.astype(np.float32))


def normalize_sequences(
    sequences: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, SequenceNormalizer]:
    """Convenience function returning normalized all-row sequences and the fitted object."""
    normalizer = fit_sequence_normalizer(sequences, train_indices)
    return normalizer.transform(sequences), normalizer
