"""Pipeline for augmenting minority classes (e.g. Walking labels 6/7/8)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.augmentation.warping import jitter, magnitude_scale, permute_segments, time_warp
from src.preprocessing.windowing import WindowedTrial, generate_windows_from_manifest

logger = logging.getLogger("trustknee.augmentation.pipeline")

_METHOD_MAP = {
    "jitter": jitter,
    "magnitude_scale": magnitude_scale,
    "time_warp": time_warp,
    "permute_segments": permute_segments,
}


def _apply_methods(
    signal: np.ndarray,
    methods: tuple[str, ...],
) -> np.ndarray:
    out = signal
    for name in methods:
        fn = _METHOD_MAP.get(name)
        if fn is None:
            raise ValueError(
                f"Unknown augmentation method: {name!r}. Available: {sorted(_METHOD_MAP)}"
            )
        out = fn(out)
    return out


def _apply_multimodal_methods(
    imu: np.ndarray,
    emg: np.ndarray,
    methods: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    out_imu = imu
    out_emg = emg
    for name in methods:
        if name == "jitter":
            out_imu = jitter(out_imu)
            out_emg = jitter(out_emg)
        elif name == "magnitude_scale":
            out_imu = magnitude_scale(out_imu)
            out_emg = magnitude_scale(out_emg)
        elif name == "time_warp":
            n_total_knots = 4 + 2
            wf = np.random.normal(loc=1.0, scale=0.2, size=n_total_knots)
            out_imu = time_warp(out_imu, warp_factors=wf)
            out_emg = time_warp(out_emg, warp_factors=wf)
        elif name == "permute_segments":
            n_segments = 4
            perm = np.random.permutation(n_segments)
            out_imu = permute_segments(out_imu, n_segments=n_segments, permutation=perm)
            out_emg = permute_segments(out_emg, n_segments=n_segments, permutation=perm)
        else:
            raise ValueError(
                f"Unknown augmentation method: {name!r}. Available: {sorted(_METHOD_MAP)}"
            )
    return out_imu, out_emg


def augment_minority_classes(
    manifest: pd.DataFrame,
    target_labels: tuple[int, ...] = (6, 7, 8),
    methods: tuple[str, ...] = ("jitter", "magnitude_scale", "time_warp"),
    multiplier: int = 3,
    window_ms: float = config.WINDOW_MS,
    overlap: float = config.WINDOW_OVERLAP,
    preprocess: bool = True,
) -> list[WindowedTrial]:
    """Augment windows for minority classes.

    For each :class:`WindowedTrial` whose ``label_id`` is in
    ``target_labels`` and whose ``subject_id`` is not in
    ``config.HELD_OUT_LOSO_SUBJECTS``, generate ``multiplier`` synthetic
    copies by composing the requested augmentation methods in order.
    Multimodal time-domain transformations (e.g. time warping and
    permutation) are synchronized across IMU and EMG channels.

    Original trials are preserved with ``metadata["synthetic"] == False``;
    synthetic copies carry ``metadata["synthetic"] == True`` so LOSO
    evaluation can exclude them from held-out splits.

    Permutation (``permute_segments``) is included only when explicitly
    listed in ``methods``.

    Args:
        manifest: DataFrame from :func:`src.ingestion.build_manifest`.
        target_labels: Labels to augment.
        methods: Augmentation method names to compose in order.
        multiplier: Number of synthetic copies per eligible trial.
        window_ms: Windowing parameter forwarded to
            :func:`generate_windows_from_manifest`.
        overlap: Windowing parameter forwarded.
        preprocess: Whether to filter before windowing.

    Returns:
        List of :class:`WindowedTrial` containing originals followed by
        synthetic copies. The return is intentionally a ``list`` rather
        than a lazy iterator so the caller can inspect ``synthetic``
        counts without consuming an iterator.
    """
    if multiplier < 0:
        raise ValueError(f"multiplier must be non-negative, got {multiplier}")

    for m in methods:
        if m not in _METHOD_MAP:
            raise ValueError(f"Unknown method {m!r}. Choose from {sorted(_METHOD_MAP)}")

    target_set = set(target_labels)
    held_out = set(config.HELD_OUT_LOSO_SUBJECTS)

    base_trials: list[WindowedTrial] = list(
        generate_windows_from_manifest(
            manifest,
            window_ms=window_ms,
            overlap=overlap,
            preprocess=preprocess,
        )
    )

    for wt in base_trials:
        if "synthetic" not in wt.metadata.columns:
            wt.metadata["synthetic"] = False
        else:
            wt.metadata["synthetic"] = wt.metadata["synthetic"].astype(bool)

    augmented: list[WindowedTrial] = list(base_trials)

    for wt in base_trials:
        if wt.label_id not in target_set:
            continue
        if wt.subject_id in held_out:
            logger.info(
                "Skipping LOSO held-out subject %d (label %d) for augmentation",
                wt.subject_id,
                wt.label_id,
            )
            continue
        if wt.n_windows == 0:
            continue
        for _ in range(multiplier):
            aug_imu, aug_emg = _apply_multimodal_methods(wt.imu_windows, wt.emg_windows, methods)

            new_meta = wt.metadata.copy()
            new_meta["synthetic"] = True

            synthetic_trial = WindowedTrial(
                subject_id=wt.subject_id,
                label_id=wt.label_id,
                trial_num=wt.trial_num,
                execution=wt.execution,
                exercise=wt.exercise,
                imu_windows=aug_imu,
                emg_windows=aug_emg,
                metadata=new_meta,
            )
            augmented.append(synthetic_trial)

    n_synth = sum(1 for t in augmented if bool(t.metadata["synthetic"].iloc[0])) if augmented else 0
    logger.info(
        "Augmentation complete: %d base trials, %d synthetic trials (target_labels=%s, methods=%s, multiplier=%d)",
        len(base_trials),
        n_synth,
        sorted(target_set),
        methods,
        multiplier,
    )
    return augmented
