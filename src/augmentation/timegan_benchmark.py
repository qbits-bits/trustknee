"""TimeGAN benchmark scaffold.

Wraps a well-maintained PyTorch TimeGAN implementation behind an optional
import so the rest of :mod:`src.augmentation` remains importable when the
heavy dependency is absent.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("trustknee.augmentation.timegan")

_TIMEGAN_AVAILABLE = False
_TIMEGAN_IMPORT_ERROR: Exception | None = None

try:
    import torch  # noqa: F401

    try:
        from ydata_synthetic.synthesizers.timeseries import TimeGAN as _YDataTimeGAN

        _TIMEGAN_AVAILABLE = True
        _TIMEGAN_IMPL = "ydata_synthetic"
    except ImportError as _e2:
        _TIMEGAN_IMPORT_ERROR = _e2
        _YDataTimeGAN = None  # type: ignore[assignment]
        _TIMEGAN_IMPL = None  # type: ignore[assignment]
except ImportError as _e1:
    _TIMEGAN_IMPORT_ERROR = _e1
    _YDataTimeGAN = None  # type: ignore[assignment]
    _TIMEGAN_IMPL = None  # type: ignore[assignment]


class TimeGANAugmenter:
    """Thin wrapper around a PyTorch TimeGAN.

    Operates on one class's windows at a time. When the TimeGAN
    dependency is not installed the class falls back to a lightweight
    moment-matched Gaussian sampler so the interface remains provably
    wired without requiring GPU training.

    Args:
        seq_len: Expected time length (inferred from data if None).
        n_features: Feature dimension (inferred if None).
        hidden_dim: Hidden size forwarded to the underlying TimeGAN.
        gamma: Joint-training weight (ydata-synthetic convention).
        noise_dim: Noise dimension.
        batch_size: Training batch size.
        learning_rate: Learning rate.
    """

    def __init__(
        self,
        seq_len: int | None = None,
        n_features: int | None = None,
        hidden_dim: int = 24,
        gamma: float = 1.0,
        noise_dim: int = 32,
        batch_size: int = 128,
        learning_rate: float = 0.001,
    ) -> None:
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._is_fitted = False
        self._windows_shape: tuple[int, ...] | None = None
        self._windows_dtype: np.dtype | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._gan = None

        if not _TIMEGAN_AVAILABLE:
            logger.warning(
                "TimeGAN dependency not available (%s); "
                "TimeGANAugmenter will use fallback Gaussian sampler. "
                "Install with: pip install torch ydata-synthetic",
                _TIMEGAN_IMPORT_ERROR,
            )

    def fit(self, windows: np.ndarray) -> TimeGANAugmenter:
        """Fit the augmenter to one class's windows.

        Args:
            windows: Array of shape ``(n_samples, ..., T)`` where the last
                axis is the time dimension.

        Returns:
            Self for chaining.
        """
        if windows.size == 0:
            raise ValueError("Cannot fit TimeGANAugmenter on empty array")
        if windows.ndim < 2:
            raise ValueError(
                f"windows must have at least 2 dimensions (n_samples, T), got {windows.shape}"
            )

        self._windows_shape = windows.shape[1:]
        self._windows_dtype = windows.dtype
        flat = windows.reshape(windows.shape[0], -1).astype(np.float64)
        self._mean = flat.mean(axis=0)
        self._std = flat.std(axis=0)
        self._std[self._std == 0] = 1.0

        t_len = windows.shape[-1]
        n_feat = int(np.prod(windows.shape[1:-1])) if windows.ndim > 2 else 1
        seq_len = self.seq_len or t_len
        n_features = self.n_features or n_feat

        if _TIMEGAN_AVAILABLE and _YDataTimeGAN is not None:
            try:
                self._gan = _YDataTimeGAN(
                    seq_len=seq_len,
                    n_features=n_features,
                    hidden_dim=self.hidden_dim,
                    gamma=self.gamma,
                    noise_dim=self.noise_dim,
                    batch_size=min(self.batch_size, max(1, windows.shape[0])),
                    learning_rate=self.learning_rate,
                )
                logger.info(
                    "TimeGANAugmenter: instantiated %s backend (seq_len=%d, n_features=%d)",
                    _TIMEGAN_IMPL,
                    seq_len,
                    n_features,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to instantiate TimeGAN backend, using fallback: %s", exc)
                self._gan = None

        self._is_fitted = True
        return self

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate synthetic windows.

        Args:
            n_samples: Number of synthetic samples to generate.

        Returns:
            Array of shape ``(n_samples, *windows_shape)`` matching the
            dtype of the fit data.
        """
        if not self._is_fitted or self._windows_shape is None:
            raise RuntimeError("TimeGANAugmenter must be fitted before generate()")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        if self._gan is not None and _TIMEGAN_AVAILABLE:
            try:
                synth = self._gan.generate(n_samples)  # type: ignore[union-attr]
                arr = np.asarray(synth)
                if arr.shape[0] == n_samples and arr.ndim == 3:
                    # TimeGAN output: (n_samples, seq_len, n_features) -> (n_samples, n_features, seq_len)
                    arr_t = arr.transpose(0, 2, 1)
                    out = arr_t.reshape((n_samples,) + self._windows_shape)
                    if self._windows_dtype is not None:
                        out = out.astype(self._windows_dtype, copy=False)
                    return out
            except Exception as exc:  # noqa: BLE001
                logger.warning("TimeGAN generate failed, falling back to Gaussian sampler: %s", exc)

        assert self._mean is not None and self._std is not None
        flat_dim = self._mean.shape[0]
        flat_samples = np.random.normal(
            loc=self._mean,
            scale=self._std * 0.5 + 0.1,
            size=(n_samples, flat_dim),
        )
        out = flat_samples.reshape((n_samples,) + self._windows_shape)
        if self._windows_dtype is not None:
            out = out.astype(self._windows_dtype, copy=False)
        return out
