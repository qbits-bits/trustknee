import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Expose normalized channel-first sequences and integer class labels to PyTorch.

    Args:
        X: NumPy array shaped ``(samples, channels, time)``.
        y: NumPy array of one integer label per sample.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Convert pre-processed 3D signal matrix (N, Channels, Time) to Tensor
        self.signals = torch.tensor(X, dtype=torch.float32)

        # Convert target labels (N,) to long Tensor
        self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of labeled sequences."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one sequence and its class label as tensors."""
        return {
            "signal": self.signals[idx],  # Shape: (Channels, Time)
            "label": self.labels[idx],  # Scalar label ID
        }
