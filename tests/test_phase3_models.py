"""Tests for Phase 3 calibration, explanations, and Transformer artifacts."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src import config
from src.inference.mock_engine import MockInferenceEngine
from src.models.artifacts import load_transformer_artifact, save_transformer_artifact
from src.models.calibration import (
    TemperatureScaler,
    apply_post_pool_temperature,
    compute_calibration_metrics,
    fit_post_pool_temperature,
    negative_log_likelihood,
    reliability_bins,
    risk_coverage_curve,
    select_binary_decision_threshold,
    select_uncertainty_threshold,
)
from src.models.explainability import (
    integrated_gradients,
    sensor_feature_mask,
    summarize_sensor_attributions,
)
from src.models.normalization import SequenceNormalizer
from src.models.phase3 import CANONICAL_PROTOCOL, run_phase3_transformer
from src.models.statistical_evaluation import (
    paired_subject_comparisons,
    per_subject_trial_metrics,
    validate_paired_trial_alignment,
)
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


def test_trial_temperature_is_fitted_and_applied_without_validation_leakage():
    probabilities = np.array([[0.99, 0.01], [0.95, 0.05], [0.10, 0.90], [0.85, 0.15]], dtype=float)
    labels = np.array([0, 1, 1, 1])

    scaler = fit_post_pool_temperature(probabilities, labels)
    calibrated = apply_post_pool_temperature(scaler, probabilities)

    assert calibrated.shape == probabilities.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert negative_log_likelihood(calibrated, labels) <= negative_log_likelihood(
        probabilities, labels
    )


def test_binary_decision_threshold_is_selected_on_supplied_validation_trials():
    probabilities = np.array([[0.70, 0.30], [0.60, 0.40], [0.55, 0.45], [0.40, 0.60], [0.35, 0.65]])
    labels = np.array([0, 0, 1, 1, 1])

    selected = select_binary_decision_threshold(probabilities, labels)

    assert 0.0 <= selected["threshold"] <= 1.0
    assert selected["macro_f1"] >= 0.8
    assert {"correct_recall", "wrong_recall"} <= set(selected)


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
    assert summary[0]["peak_time_bin"] == 0
    assert summary[0]["peak_bin_start_ms"] == 0
    assert summary[0]["peak_bin_end_ms"] == pytest.approx(config.WINDOW_MS / 30)
    assert summary[0]["peak_bin_center_ms"] == pytest.approx(config.WINDOW_MS / 60)
    assert summary[0]["signal_interpretation"] == "segment kinematics from the IMU"
    assert sum(float(row["attribution_fraction"]) for row in summary) == pytest.approx(1.0)


def test_subject_clustered_comparison_reports_variability_ci_and_paired_test():
    predictions = pd.DataFrame(
        [
            {
                "protocol": "matched",
                "task": "binary",
                "model": model,
                "calibration": calibration,
                "subject_id": subject,
                "trial_id": f"s{subject}-t{trial}",
                "true_class": true,
                "predicted_class": predicted,
                "num_classes": 2,
            }
            for model, calibration, predicted_by_subject in (
                ("transformer", "temperature", {1: [0, 1], 2: [0, 1], 3: [0, 1]}),
                ("xgboost_aligned", "none", {1: [1, 1], 2: [0, 0], 3: [1, 1]}),
            )
            for subject, predictions_for_subject in predicted_by_subject.items()
            for trial, (true, predicted) in enumerate(
                zip([0, 1], predictions_for_subject, strict=True), start=1
            )
        ]
    )

    subject_metrics = per_subject_trial_metrics(predictions)
    validate_paired_trial_alignment(predictions)
    comparison = paired_subject_comparisons(
        subject_metrics,
        seed=7,
        bootstrap_samples=500,
    )

    assert len(subject_metrics) == 6
    assert set(comparison["metric"]) == {"accuracy", "macro_f1"}
    assert (comparison["n_subjects"] == 3).all()
    assert (comparison["bootstrap_unit"] == "subject").all()
    assert comparison["paired_sign_flip_p_value"].between(0, 1).all()
    assert (
        comparison["subject_bootstrap_ci_lower"] <= comparison["subject_bootstrap_ci_upper"]
    ).all()

    incomplete = predictions.drop(predictions.index[-1])
    with pytest.raises(ValueError, match="identical trials"):
        validate_paired_trial_alignment(incomplete)


def test_trial_explanation_excludes_zero_attribution_masked_sensors():
    engine = MockInferenceEngine()
    base = engine.predict_window(
        np.zeros((8, 6, 30)),
        np.zeros((8, 252)),
    )
    explained = replace(
        base,
        feature_attributions={"active sensor": 1.0, "masked sensor": 0.0},
    )
    all_zero = replace(
        base,
        feature_attributions={"masked sensor": 0.0},
    )

    assert engine.aggregate_trial([explained]).top_contributing_sensors == ["active sensor"]
    assert engine.aggregate_trial([all_zero]).top_contributing_sensors is None


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
        "trial_predictions.csv",
        "per_subject_metrics.csv",
        "paired_model_comparisons.csv",
        "trial_calibration.csv",
        "binary_decision_thresholds.csv",
        "integrated_gradients.json",
        "integrated_gradients_summary.csv",
        "comparison_scope.json",
        "transformer_binary.pt",
        "transformer_nine_class.pt",
    ):
        assert (output / filename).exists()
    explanations = json.loads((output / "integrated_gradients.json").read_text())
    assert explanations
    assert all(item["protocol"] == CANONICAL_PROTOCOL.name for item in explanations)
    summary = pd.read_csv(output / "integrated_gradients_summary.csv")
    assert {"exercise", "true_class", "sensor_name", "n_representative_windows"} <= set(summary)
    calibration = pd.read_csv(output / "calibration_metrics.csv")
    assert "trial_temperature" in set(calibration["calibration"])
    assert (calibration["n_bins"] == 15).all()
    reliability = pd.read_csv(output / "reliability_bins.csv")
    assert reliability.groupby(["task", "level", "calibration"])["count"].sum().gt(0).all()
    per_subject = pd.read_csv(output / "per_subject_metrics.csv")
    assert set(per_subject["subject_id"]) == set(config.PHASE3_TEST_SUBJECTS)
    comparisons = pd.read_csv(output / "paired_model_comparisons.csv")
    assert set(comparisons["metric"]) == {"accuracy", "macro_f1"}
    assert (comparisons["n_subjects"] == len(config.PHASE3_TEST_SUBJECTS)).all()


def test_phase3_can_reuse_a_prepared_dataset(tmp_path):
    from src.ingestion import build_manifest
    from src.models.model_data import build_model_inputs

    _write_phase3_dataset(tmp_path)
    manifest = build_manifest(tmp_path, exclude_subjects=set())
    prepared = build_model_inputs(manifest, preprocess=True)

    results = run_phase3_transformer(
        manifest,
        tmp_path / "prepared-results",
        protocol=CANONICAL_PROTOCOL,
        quick=True,
        explanation_steps=2,
        explanations_per_class=1,
        prepared_dataset=prepared,
    )

    assert not results.empty


def test_phase3_rejects_prepared_dataset_with_incompatible_protocol(tmp_path):
    from src.ingestion import build_manifest
    from src.models.model_data import build_model_inputs

    _write_phase3_dataset(tmp_path)
    manifest = build_manifest(tmp_path, exclude_subjects=set())
    prepared = build_model_inputs(manifest, preprocess=False)

    with pytest.raises(ValueError, match="incompatible.*preprocess=False"):
        run_phase3_transformer(
            manifest,
            tmp_path / "invalid-prepared-results",
            protocol=CANONICAL_PROTOCOL,
            quick=True,
            explanation_steps=2,
            explanations_per_class=1,
            prepared_dataset=prepared,
        )
