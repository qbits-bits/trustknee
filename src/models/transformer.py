"""Small, explainable Transformer encoder for synchronized sensor sequences."""

from __future__ import annotations

import math

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised only without optional deps
    raise ImportError(
        "Transformer support requires PyTorch. Install the project requirements first."
    ) from exc


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding that marks the temporal position of each step."""

    def __init__(self, d_model: int, max_seq_len: int = 30) -> None:
        super().__init__()
        if d_model <= 0 or d_model % 2:
            raise ValueError("d_model must be a positive even number")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_seq_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3:
            raise ValueError("Positional encoding expects (batch, sequence, features)")
        if values.shape[1] > self.encoding.shape[1]:
            raise ValueError("Sequence is longer than the configured positional encoding")
        return values + self.encoding[:, : values.shape[1], :]


class TransformerEncoderClassifier(nn.Module):
    """Transformer encoder classifier for input shaped ``(batch, 30, 56)``."""

    def __init__(
        self,
        input_dim: int = 56,
        num_classes: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 30,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes <= 1:
            raise ValueError("input_dim must be positive and num_classes must be at least 2")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.model_config = {
            "input_dim": int(input_dim),
            "num_classes": int(num_classes),
            "d_model": int(d_model),
            "nhead": int(nhead),
            "num_layers": int(num_layers),
            "dim_feedforward": int(dim_feedforward),
            "dropout": float(dropout),
            "max_seq_len": int(max_seq_len),
        }
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3:
            raise ValueError("Transformer input must have shape (batch, sequence, features)")
        projected = self.input_projection(values)
        encoded = self.encoder(self.positional_encoding(projected))
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)
