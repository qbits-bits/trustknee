"""Simple time-series augmentations operating along the last (time) axis.

All functions are shape-agnostic: they accept any ``(..., T)`` array and
transform along ``axis=-1`` so IMU ``(N_SENSORS, C, T)``, EMG
``(N_SENSORS, T)``, or future joined-tensor formats work without
special-casing. Input arrays are never mutated.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import CubicSpline

logger = logging.getLogger("trustknee.augmentation.warping")


def jitter(signal: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """Add channel-wise scaled Gaussian noise.

    Noise std is ``sigma * per_channel_std`` where ``per_channel_std`` is
    computed along the time axis for each channel independently, so the
    perturbation is relative to the channel's own amplitude.

    Args:
        signal: Array of shape ``(..., T)``.
        sigma: Relative noise scale (fraction of per-channel std).

    Returns:
        Augmented copy with same shape and dtype as input.
    """
    if signal.size == 0:
        return signal.copy()
    orig_dtype = signal.dtype
    x = signal.astype(np.float64, copy=False)
    ch_std = np.std(x, axis=-1, keepdims=True)
    ch_std = np.where(ch_std == 0, 1.0, ch_std)
    noise = np.random.normal(0.0, 1.0, size=x.shape) * (sigma * ch_std)
    out = x + noise
    return out.astype(orig_dtype, copy=False)


def magnitude_scale(
    signal: np.ndarray,
    scale_range: tuple[float, float] = (0.9, 1.1),
    per_channel: bool = False,
) -> np.ndarray:
    """Random magnitude scaling.

    Multiplies the signal by a scalar drawn uniformly from ``scale_range``.
    When ``per_channel`` is False a single scalar is applied to the whole
    array; when True a distinct scalar is drawn for each ``(..., 1)``
    channel slice along the time axis.

    Args:
        signal: Array of shape ``(..., T)``.
        scale_range: ``(low, high)`` interval for the uniform draw.
        per_channel: If True draw one scale per channel.

    Returns:
        Augmented copy with same shape and dtype as input.
    """
    if signal.size == 0:
        return signal.copy()
    orig_dtype = signal.dtype
    x = signal.astype(np.float64, copy=False)
    low, high = scale_range
    if per_channel:
        scale_shape = x.shape[:-1] + (1,)
        scales = np.random.uniform(low, high, size=scale_shape)
        scales = np.broadcast_to(scales, x.shape)
    else:
        scalar = float(np.random.uniform(low, high))
        scales = scalar
    out = x * scales
    return out.astype(orig_dtype, copy=False)


def time_warp(
    signal: np.ndarray,
    sigma: float = 0.2,
    n_knots: int = 4,
    warp_factors: np.ndarray | None = None,
) -> np.ndarray:
    """Smooth random time warping via cubic-spline warp path.

    Following Um et al. 2017 conventions for wearable-sensor augmentation,
    a smooth warp factor curve is defined on continuous normalized time
    ``[0, 1]`` using ``n_knots + 2`` knots, integrated continuously via
    analytic cubic-spline antiderivative, and normalised to ``[0, 1]`` to
    obtain a monotonic continuous time mapping. The signal is then
    resampled at the warped coordinates via linear interpolation.

    When ``warp_factors`` is pre-sampled and shared across paired modalities
    with different sampling rates (e.g. 30-sample IMU and 252-sample EMG),
    the underlying continuous time deformation is mathematically identical
    and perfectly synchronized in physical time.

    Args:
        signal: Array of shape ``(..., T)``.
        sigma: Std of the random warp factors centred at 1.0.
        n_knots: Number of interior knots (total knots = n_knots + 2).
        warp_factors: Optional pre-sampled knot warp factors. If provided,
            these are used directly to ensure synchronized warping across
            paired modalities (e.g. IMU and EMG).

    Returns:
        Augmented copy with same shape and dtype as input.
    """
    if signal.size == 0:
        return signal.copy()
    orig_dtype = signal.dtype
    x = signal.astype(np.float64, copy=False)
    t_len = x.shape[-1]
    if t_len <= 1:
        return signal.copy()

    if warp_factors is None:
        n_total_knots = n_knots + 2
        wf = np.random.normal(loc=1.0, scale=sigma, size=n_total_knots)
    else:
        wf = np.asarray(warp_factors, dtype=np.float64)
        n_total_knots = len(wf)

    wf = np.clip(wf, 0.1, 3.0)

    # Formulate spline in continuous normalized time [0.0, 1.0]
    knot_t = np.linspace(0.0, 1.0, n_total_knots)
    cs = CubicSpline(knot_t, wf, bc_type="natural")
    cs_anti = cs.antiderivative()

    t_query = np.linspace(0.0, 1.0, t_len)
    integrated = cs_anti(t_query)
    total_area = float(cs_anti(1.0) - cs_anti(0.0))

    if abs(total_area) < 1e-9:
        warp_steps = np.arange(t_len, dtype=np.float64)
    else:
        norm_warped_t = (integrated - float(cs_anti(0.0))) / total_area
        warp_steps = np.clip(norm_warped_t * (t_len - 1), 0.0, t_len - 1)

    flat = x.reshape(-1, t_len)
    out_flat = np.empty_like(flat)
    xp = np.arange(t_len, dtype=np.float64)
    for i in range(flat.shape[0]):
        out_flat[i] = np.interp(warp_steps, xp, flat[i])

    out = out_flat.reshape(x.shape)
    return out.astype(orig_dtype, copy=False)


def permute_segments(
    signal: np.ndarray,
    n_segments: int = 4,
    permutation: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """Randomly permute time segments.

    Warning:
        Permutation breaks temporal continuity. Models that rely on
        ordered dynamics (TCN, Transformer, LSTM) may degrade when
        trained on permuted data. This augmentation is disabled by
        default in :func:`augment_minority_classes` and should only be
        enabled explicitly for bag-of-segments or permutation-invariant
        architectures.

    Args:
        signal: Array of shape ``(..., T)``.
        n_segments: Number of equal-length segments to permute.
        permutation: Optional pre-sampled permutation array. If provided,
            ensures synchronized segment rearrangement across paired
            modalities (e.g. IMU and EMG).

    Returns:
        Augmented copy with same shape and dtype as input.
    """
    if signal.size == 0:
        return signal.copy()
    orig_dtype = signal.dtype
    x = signal.astype(np.float64, copy=False)
    t_len = x.shape[-1]
    if n_segments <= 1 or t_len < n_segments:
        return signal.copy()

    perm = (
        np.random.permutation(n_segments)
        if permutation is None
        else np.asarray(permutation, dtype=int)
    )
    if len(perm) != n_segments:
        raise ValueError(f"permutation length ({len(perm)}) must equal n_segments ({n_segments})")

    # Continuous normalized time cut points rounded to nearest sample index
    cut_points = [int(round(i * t_len / n_segments)) for i in range(n_segments + 1)]
    segs = [(cut_points[i], cut_points[i + 1]) for i in range(n_segments)]

    flat = x.reshape(-1, t_len)
    out_flat = np.empty_like(flat)
    for i in range(flat.shape[0]):
        parts = [flat[i][s:e] for s, e in segs]
        permuted = [parts[p] for p in perm]
        out_flat[i] = np.concatenate(permuted)

    out = out_flat.reshape(x.shape)
    return out.astype(orig_dtype, copy=False)
