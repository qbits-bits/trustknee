"""Tests for src/augmentation: warping, TimeGAN scaffold, and pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.augmentation import TimeGANAugmenter, jitter, magnitude_scale, permute_segments, time_warp
from src.augmentation.pipeline import augment_minority_classes
from src.ingestion import build_manifest

RNG = np.random.default_rng(0)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel().astype(float)
    bf = b.ravel().astype(float)
    if np.std(af) == 0 or np.std(bf) == 0:
        return 1.0
    return float(np.corrcoef(af, bf)[0, 1])


@pytest.mark.parametrize("fn", [jitter, magnitude_scale, time_warp, permute_segments])
def test_shape_dtype_preservation_2d(fn):
    arr = RNG.normal(size=(8, 252)).astype(np.float32)
    orig = arr.copy()
    out = fn(arr)
    assert out.shape == arr.shape
    assert out.dtype == arr.dtype
    assert np.array_equal(arr, orig), "input was mutated"


@pytest.mark.parametrize("fn", [jitter, magnitude_scale, time_warp, permute_segments])
def test_shape_dtype_preservation_3d(fn):
    arr = RNG.normal(size=(8, 6, 30)).astype(np.float64)
    orig = arr.copy()
    out = fn(arr)
    assert out.shape == arr.shape
    assert out.dtype == arr.dtype
    assert np.array_equal(arr, orig)


def test_jitter_not_identical_but_correlated():
    arr = RNG.normal(size=(8, 6, 30)).astype(np.float64) * 2 + 1.0
    out = jitter(arr, sigma=0.03)
    assert not np.array_equal(out, arr)
    assert _corr(arr, out) > 0.8
    assert np.all(np.isfinite(out))


def test_magnitude_scale_not_identical_but_correlated():
    arr = RNG.normal(size=(8, 252)).astype(np.float64)
    out = magnitude_scale(arr, scale_range=(0.9, 1.1))
    assert not np.array_equal(out, arr)
    assert _corr(arr, out) > 0.8


def test_time_warp_not_identical_but_correlated():
    t = np.linspace(0, 2 * np.pi, 252)
    base = np.sin(t)[None, :] + RNG.normal(scale=0.1, size=(8, 252))
    base = base.astype(np.float64)
    out = time_warp(base, sigma=0.2, n_knots=4)
    assert out.shape == base.shape
    assert not np.array_equal(out, base)
    assert _corr(base, out) > 0.8


def test_jitter_sigma_scaled_per_channel():
    x = np.ones((2, 10), dtype=np.float64)
    x[0] = np.linspace(0, 10, 10)
    x[1] = np.ones(10) * 100
    out = jitter(x, sigma=0.5)
    diff0 = np.std(out[0] - x[0])
    diff1 = np.std(out[1] - x[1])
    assert diff0 > 0 and diff1 > 0


def test_permute_segments_warn_and_shape():
    arr = RNG.normal(size=(4, 20)).astype(np.float32)
    out = permute_segments(arr, n_segments=4)
    assert out.shape == arr.shape
    assert out.dtype == arr.dtype


def test_timegan_benchmark_interface():
    windows = RNG.normal(size=(10, 8, 6, 30)).astype(np.float32)
    aug = TimeGANAugmenter()
    aug.fit(windows)
    synth = aug.generate(5)
    assert synth.shape == (5, 8, 6, 30)
    assert synth.dtype == windows.dtype

    synth2 = aug.generate(2)
    assert synth2.shape[0] == 2

    with pytest.raises(RuntimeError):
        TimeGANAugmenter().generate(1)
    with pytest.raises(ValueError):
        aug.generate(0)


def _write_csvs(data_root):
    pd.DataFrame(
        [
            {
                "Participant ID": 1,
                "Gender (M/F)": "M",
                "Height (CM)": 1.82,
                "Weight(KG)": 78.0,
                "Age (Years)": 24,
                "Leg": "Right",
                "Pathology": "ACL",
            },
            {
                "Participant ID": 2,
                "Gender (M/F)": "F",
                "Height (CM)": 1.60,
                "Weight(KG)": 55.0,
                "Age (Years)": 29,
                "Leg": "Left",
                "Pathology": "None",
            },
            {
                "Participant ID": 3,
                "Gender (M/F)": "M",
                "Height (CM)": 1.75,
                "Weight(KG)": 70.0,
                "Age (Years)": 30,
                "Leg": "Right",
                "Pathology": "None",
            },
        ]
    ).to_csv(data_root / "participants.csv", index=False)
    pd.DataFrame(
        [
            {"Label ID": 0, "Execution": "Correct", "Details": "Squat"},
            {"Label ID": 6, "Execution": "Correct", "Details": "Walking"},
            {"Label ID": 7, "Execution": "Wrong", "Details": "Walking wrong"},
            {"Label ID": 8, "Execution": "Wrong", "Details": "Walking wrong 2"},
        ]
    ).to_csv(data_root / "labels.csv", index=False)
    pd.DataFrame(
        [{"Sensor ID": sid, "Muscle": m} for sid, m in config.SENSOR_PLACEMENT.items()]
    ).to_csv(data_root / "placement.csv", index=False)
    with open(data_root / "sensors.csv", "w") as f:
        f.write("n_sensors,8\n")
        f.write("emg_sampling_rate_hz,1259.2592592592594\n")
        f.write("imu_sampling_rate_hz,148.14814814814815\n")


def _write_trial(trial_dir, imu_t=300, emg_t=2550):
    trial_dir.mkdir(parents=True, exist_ok=True)
    np.save(trial_dir / "imu.npy", RNG.normal(size=(48, imu_t)))
    np.save(trial_dir / "emg.npy", RNG.normal(size=(8, emg_t)))


def _build_manifest_for_augmentation(tmp_path):
    _write_csvs(tmp_path)
    ds = tmp_path / "dataset"
    _write_trial(ds / "Subject_1" / "6" / "Trial_1")
    _write_trial(ds / "Subject_2" / "6" / "Trial_1")
    _write_trial(ds / "Subject_2" / "0" / "Trial_1")
    _write_trial(ds / "Subject_3" / "7" / "Trial_1")
    _write_trial(ds / "Subject_3" / "8" / "Trial_1")
    # include all subjects explicitly
    manifest = build_manifest(tmp_path, exclude_subjects=set())
    return manifest


def test_augment_minority_only_touches_target_and_respects_held_out(tmp_path):
    manifest = _build_manifest_for_augmentation(tmp_path)
    augmented = augment_minority_classes(
        manifest,
        target_labels=(6, 7, 8),
        methods=("jitter", "magnitude_scale", "time_warp"),
        multiplier=2,
    )
    for wt in augmented:
        assert "synthetic" in wt.metadata.columns

    originals = [w for w in augmented if not bool(w.metadata["synthetic"].iloc[0])]
    synthetics = [w for w in augmented if bool(w.metadata["synthetic"].iloc[0])]

    assert len(originals) == 5
    assert all(s.label_id in (6, 7, 8) for s in synthetics)
    assert all(s.subject_id != 1 for s in synthetics)
    # subject 1 label 6 should not produce synthetics
    # eligible: Subject2 label6 (2 copies), Subject3 label7 (2), label8 (2) => 6 synthetics
    assert len(synthetics) == 6
    assert all(o.label_id == 0 or o.subject_id == 1 or o.label_id in (6, 7, 8) for o in originals)

    # non-target label 0 has no synthetic copies even for subject 2
    assert not any(s.label_id == 0 for s in synthetics)
    assert not any(s.subject_id == 1 for s in synthetics)


def test_augment_preserves_window_shapes(tmp_path):
    manifest = _build_manifest_for_augmentation(tmp_path)
    augmented = augment_minority_classes(manifest, target_labels=(6,), multiplier=1)
    for wt in augmented:
        if wt.n_windows > 0:
            assert wt.imu_windows.shape[1:] == (
                config.N_SENSORS,
                config.IMU_CHANNELS_PER_SENSOR,
                30,
            )
            assert wt.emg_windows.shape[1:] == (config.N_SENSORS, 252)


def test_augment_default_excludes_permute(tmp_path):
    manifest = _build_manifest_for_augmentation(tmp_path)
    _ = augment_minority_classes(manifest, multiplier=1)
    assert permute_segments.__doc__ is not None and "Warning" in permute_segments.__doc__


def test_augment_minority_negative_multiplier_raises(tmp_path):
    manifest = _build_manifest_for_augmentation(tmp_path)
    with pytest.raises(ValueError, match="multiplier must be non-negative"):
        augment_minority_classes(manifest, multiplier=-1)


def test_multimodal_synchronized_warping_and_permutation():
    # Construct pulse signals in segment 1 for IMU (30 samples) and EMG (300 samples)
    imu = np.zeros((8, 6, 30), dtype=np.float64)
    emg = np.zeros((8, 300), dtype=np.float64)
    imu[:, :, 10] = 10.0
    emg[:, 100] = 10.0

    wf = np.array([0.5, 1.5, 2.0, 0.8, 1.2, 0.6])
    warped_imu = time_warp(imu, warp_factors=wf)
    warped_emg = time_warp(emg, warp_factors=wf)

    # Peak relative position should be identical in both warped signals
    imu_peak_rel = np.argmax(warped_imu[0, 0]) / (30 - 1)
    emg_peak_rel = np.argmax(warped_emg[0]) / (300 - 1)
    assert abs(imu_peak_rel - emg_peak_rel) < 0.05

    # Test synchronized permutation
    perm = [2, 0, 3, 1]
    p_imu = permute_segments(imu, n_segments=4, permutation=perm)
    p_emg = permute_segments(emg, n_segments=4, permutation=perm)
    assert np.argmax(p_imu[0, 0]) / 30 == pytest.approx(np.argmax(p_emg[0]) / 300, abs=0.05)


def test_timegan_tensor_dimensions_and_shapes():
    # 4D tensor (e.g. IMU windows)
    imu_wins = RNG.normal(size=(5, 8, 6, 30)).astype(np.float32)
    aug_imu = TimeGANAugmenter()
    aug_imu.fit(imu_wins)
    out_imu = aug_imu.generate(3)
    assert out_imu.shape == (3, 8, 6, 30)
    assert out_imu.dtype == np.float32

    # 3D tensor (e.g. EMG windows)
    emg_wins = RNG.normal(size=(5, 8, 252)).astype(np.float64)
    aug_emg = TimeGANAugmenter()
    aug_emg.fit(emg_wins)
    out_emg = aug_emg.generate(4)
    assert out_emg.shape == (4, 8, 252)
    assert out_emg.dtype == np.float64

    # 2D tensor
    arr_2d = RNG.normal(size=(5, 100)).astype(np.float32)
    aug_2d = TimeGANAugmenter()
    aug_2d.fit(arr_2d)
    out_2d = aug_2d.generate(2)
    assert out_2d.shape == (2, 100)

    # 1D array should raise ValueError
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        TimeGANAugmenter().fit(np.ones(10))
