import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Expose normalized channel-first sequences and integer class labels to PyTorch.

    Args:
        X: NumPy array shaped ``(samples, channels, time)``.
        y: NumPy array of one integer label per sample.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.signals = [torch.tensor(signal, dtype=torch.float32).T for signal in X]
        self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of labeled sequences."""
        return len(self.labels)

    def __getitem__(self, idx: int):
        """Return one sequence and its class label as tensors."""
        return self.signals[idx], self.labels[idx]


def dynamic_padding_collate(batch):
    # Unpack tuple list: batch = [(signal, label), (signal, label), ...];
    signals, labels = zip(*batch, strict=False)
    signals_list = list(signals)
    # Pad sequences along time (dim 0);
    padded_signals = pad_sequence(
        signals_list, batch_first=True, padding_value=0.0
    )  # shape: (batch_size, max_time_steps, channels);

    # Transpose to standard 1D Convolution/TCN shape;
    padded_signals = padded_signals.transpose(
        1, 2
    )  # shape: (batch_size, channels, max_time_steps);

    # Create attention/padding mask; shape: (batch_size, max_time_steps);
    # Fetch original sequence lengths for each sample;
    lengths = torch.tensor([s.shape[0] for s in signals])
    max_len = int(padded_signals.shape[-1])
    mask = torch.arange(max_len)[None, :] < lengths[:, None]

    # Stack labels;
    labels = torch.stack(labels)

    return padded_signals, labels, mask
