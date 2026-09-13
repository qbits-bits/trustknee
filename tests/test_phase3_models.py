"""Tests for Phase 3 calibration, explanations, and Transformer artifacts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.artifacts import load_transformer_artifact, save_transformer_artifact
from src.models.calibration import (
    TemperatureScaler,
    compute_calibration_metrics,
    negative_log_likelihood,
    reliability_bins,
    risk_coverage_curve,
    select_uncertainty_threshold,
)
from src.models.explainability import (
    integrated_gradients,
    sensor_feature_mask,
    summarize_sensor_attributions,
)
from src.models.normalization import SequenceNormalizer
from src.models.phase3 import CANONICAL_PROTOCOL, run_phase3_transformer
from src.models.transformer import TransformerEncoderClassifier


def test_temperature_scaling_is_positive_and_does_not_increase_validation_nll():
    logits = np.array([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 0.0]])
    labels = np.array([0, 1, 1, 1])
    raw = TemperatureScaler().predict_proba(logits)
    scaler = TemperatureScaler.fit(logits, labels)
    calibrated = scaler.predict_proba(logits)

    assert scaler.temperature > 0
    assert negative_log_likelihood(calibrated, labels) <= negative_log_likelihood(raw, labels)
    assert np.allclose(calibrated.sum(axis=1), 1.0)


def test_calibration_metrics_and_reliability_bins_are_finite():
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6], [0.9, 0.1]])
    labels = np.array([0, 1, 1])
    metrics = compute_calibration_metrics(probabilities, labels, n_bins=5)
    bins = reliability_bins(probabilities, labels, n_bins=5)

    assert set(metrics) == {"ece", "brier_score", "nll"}
    assert np.isfinite(list(metrics.values())).all()
    assert len(bins) == 5
    assert sum(row["count"] for row in bins) == 3


def test_uncertainty_threshold_is_deterministic_and_respects_coverage():
    probabilities = np.array([[0.99, 0.01], [0.85, 0.15], [0.55, 0.45], [0.52, 0.48], [0.40, 0.60]])
    labels = np.array([0, 0, 1, 1, 1])
    first = select_uncertainty_threshold(probabilities, labels, min_coverage=0.8)
    second = select_uncertainty_threshold(probabilities, labels, min_coverage=0.8)

    assert first == second
    assert first["coverage"] >= 0.8


def test_risk_coverage_curve_matches_direct_metrics_with_tied_confidence():
    from sklearn.metrics import accuracy_score, f1_score

    probabilities = np.array([[0.8, 0.2], [0.8, 0.2], [0.3, 0.7], [0.55, 0.45], [0.4, 0.6]])
    labels = np.array([0, 1, 1, 1, 0])

    for row in risk_coverage_curve(probabilities, labels):
        accepted = probabilities.max(axis=1) >= row["threshold"]
        predictions = probabilities.argmax(axis=1)[accepted]
        assert row["coverage"] == pytest.approx(accepted.mean())
        assert row["accuracy"] == pytest.approx(accuracy_score(labels[accepted], predictions))
        assert row["macro_f1"] == pytest.approx(
            f1_score(
                labels[accepted],
                predictions,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )


def test_integrated_gradients_matches_linear_logit_difference_and_mask():
    import torch
    from torch import nn

    model = nn.Sequential(nn.Flatten(), nn.Linear(30 * 56, 2, bias=False))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].weight[0].fill_(0.01)
        model[1].weight[1].fill_(-0.01)
    values = np.ones((1, 30, 56), dtype=np.float32)
    mask = sensor_feature_mask((1, 3))
    attrs = integrated_gradients(model, values, np.array([0]), steps=8, feature_mask=mask)
    masked_values = values.copy()
    masked_values[..., ~mask] = 0.0
    with torch.no_grad():
        expected = float(model(torch.from_numpy(masked_values))[0, 0])

    assert attrs.shape == values.shape
    assert np.all(attrs[..., ~mask] == 0.0)
    assert float(attrs.sum()) == pytest.approx(expected, abs=1e-5)


def test_sensor_attribution_summary_uses_physical_sensor_names():
    attribution = np.zeros((30, 56), dtype=np.float32)
    attribution[:, :6] = 2.0
    summary = summarize_sensor_attributions(attribution)

    assert summary[0]["sensor_id"] == 1
    assert summary[0]["sensor_name"] == config.SENSOR_PLACEMENT[1]
    assert summary[0]["top_modality"] == "IMU"
    assert summary[0]["top_channel"] == "acc_x"
    assert summary[0]["peak_time_ms"] > 0
    assert sum(float(row["attribution_fraction"]) for row in summary) == pytest.approx(1.0)


def test_transformer_artifact_round_trip_preserves_predictions(tmp_path):
    import torch

    torch.manual_seed(42)
    model = TransformerEncoderClassifier(num_classes=2, dropout=0.0)
    normalizer = SequenceNormalizer(
        mean=np.zeros((1, 1, 56), dtype=np.float32),
        scale=np.ones((1, 1, 56), dtype=np.float32),
    )
    values = torch.randn(2, 30, 56)
    with torch.no_grad():
        expected = model(values).numpy()
    path = save_transformer_artifact(
        tmp_path / "model.pt",
        model,
        normalizer,
        TemperatureScaler(1.7),
        0.73,
        {0: "Correct", 1: "Wrong"},
        sensor_mask=(1, 3, 5),
        metadata={"contract": "test"},
    )
    loaded = load_transformer_artifact(path)
    with torch.no_grad():
        actual = loaded.model(values).numpy()

    assert np.allclose(actual, expected)
    assert loaded.temperature_scaler.temperature == pytest.approx(1.7)
    assert loaded.uncertainty_threshold == pytest.approx(0.73)
    assert loaded.sensor_mask == (1, 3, 5)


def _write_phase3_dataset(root):
    subjects = range(1, 32)
    pd.DataFrame(
        [
            {
                "Participant ID": subject,
                "Gender (M/F)": "M",
                "Height (CM)": 1.8,
                "Weight(KG)": 80,
                "Age (Years)": 30,
                "Leg": "Right",
                "Pathology": "None",
            }
            for subject in subjects
        ]
    ).to_csv(root / "participants.csv", index=False)
    pd.DataFrame(
        [
            {
                "Label ID": label,
                "Execution": config.LABELS[label].execution,
                "Details": config.LABELS[label].description,
            }
            for label in range(9)
        ]
    ).to_csv(root / "labels.csv", index=False)
    pd.DataFrame(
        [
            {"Sensor ID": sensor, "Muscle": placement}
            for sensor, placement in config.SENSOR_PLACEMENT.items()
        ]
    ).to_csv(root / "placement.csv", index=False)
    (root / "sensors.csv").write_text("n_sensors,8\n", encoding="utf-8")
    rng = np.random.default_rng(42)
    for subject in subjects:
        label = (subject - 1) % 9
        trial = root / "dataset" / f"Subject_{subject}" / str(label) / "Trial_1"
        trial.mkdir(parents=True)
        np.save(trial / "imu.npy", rng.normal(size=(48, 45)).astype(np.float32))
        np.save(trial / "emg.npy", rng.normal(size=(8, 383)).astype(np.float32))


def test_quick_phase3_run_writes_calibration_explanations_and_artifacts(tmp_path):
    from src.ingestion import build_manifest

    _write_phase3_dataset(tmp_path)
    output = tmp_path / "results"
    results = run_phase3_transformer(
        build_manifest(tmp_path, exclude_subjects=set()),
        output,
        protocol=CANONICAL_PROTOCOL,
        quick=True,
        explanation_steps=2,
        explanations_per_class=1,
    )

    assert {"binary", "nine_class"} == set(results["task"])
    assert {"transformer", "transformer_selective", "xgboost_aligned"} <= set(results["model"])
    for filename in (
        "metrics.csv",
        "calibration_metrics.csv",
        "reliability_bins.csv",
        "risk_coverage.csv",
        "integrated_gradients.json",
        "transformer_binary.pt",
        "transformer_nine_class.pt",
    ):
        assert (output / filename).exists()
    explanations = json.loads((output / "integrated_gradients.json").read_text())
    assert explanations
