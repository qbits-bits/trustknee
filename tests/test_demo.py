from __future__ import annotations

import json

import numpy as np

from src import config
from src.demo import DISCLAIMER, run_demo
from src.models.artifacts import save_transformer_artifact
from src.models.calibration import TemperatureScaler
from src.models.normalization import SequenceNormalizer
from src.models.transformer import TransformerEncoderClassifier


def test_recorded_trial_demo_runs_real_pipeline_and_writes_safe_output(tmp_path):
    import torch

    torch.manual_seed(42)
    artifact = save_transformer_artifact(
        tmp_path / "model.pt",
        TransformerEncoderClassifier(num_classes=2, dropout=0.0),
        SequenceNormalizer(
            mean=np.zeros((1, 1, 56), dtype=np.float32),
            scale=np.ones((1, 1, 56), dtype=np.float32),
        ),
        TemperatureScaler(1.0),
        0.6,
        {0: "Correct", 1: "Wrong"},
    )
    duration_s = 0.6
    imu = np.random.default_rng(42).normal(
        size=(48, round(config.IMU_SAMPLING_RATE_HZ * duration_s))
    )
    emg = np.random.default_rng(43).normal(
        size=(8, round(config.EMG_SAMPLING_RATE_HZ * duration_s))
    )
    imu_path = tmp_path / "imu.npy"
    emg_path = tmp_path / "emg.npy"
    np.save(imu_path, imu.astype(np.float32))
    np.save(emg_path, emg.astype(np.float32))

    result = run_demo(
        artifact_path=artifact,
        imu_path=imu_path,
        emg_path=emg_path,
        output_dir=tmp_path / "demo",
        subject_id=3,
        label_id=0,
        trial_num=1,
        explain=False,
    )

    saved = json.loads((tmp_path / "demo" / "demo_result.json").read_text())
    html = (tmp_path / "demo" / "demo_result.html").read_text()
    assert result["n_windows"] > 0
    assert result["persisted_trial_id"] > 0
    assert saved["disclaimer"] == DISCLAIMER
    assert "not a diagnosis" in html
    assert (tmp_path / "demo" / "trustknee_demo.sqlite").exists()
