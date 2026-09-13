"""Unit tests for the XGBoost feature-bridge adapter.

Verifies artifact integrity, rejection of missing features, and prediction
parity using a representative sample of the dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.xgboost_adapter import XGBoostArtifactAdapter, describe_engineered_feature
from src.models.xgboost_pipeline import (
    _classification_metrics,
    _trial_probabilities,
    build_features,
    load_and_trim,
    save_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "models" / "xgboost_canonical" / "xgboost_artifact.joblib"
SAMPLE_DATA_PATH = REPO_ROOT / "tests" / "data" / "sample_dataset.csv"

needs_artifact = pytest.mark.skipif(
    not ARTIFACT_PATH.exists(), reason="trained artifact not available"
)
needs_sample_data = pytest.mark.skipif(
    not SAMPLE_DATA_PATH.exists(), reason="sample dataset not available"
)


@pytest.fixture(scope="module")
def adapter():
    return XGBoostArtifactAdapter(str(ARTIFACT_PATH))


def test_artifact_loads_with_required_structure(adapter):
    """The artifact must expose model, feature order, clip, and threshold."""
    assert adapter.model is not None
    assert len(adapter.feature_order) > 0
    assert len(adapter.clip) == len(adapter.feature_order)
    assert 0.0 <= adapter.threshold <= 1.0


def test_rejects_missing_features(adapter, monkeypatch):
    """Adapter must raise when engineered features are absent from the input."""

    def fake_build_features(df):
        return df, []

    monkeypatch.setattr("src.models.xgboost_adapter.build_features", fake_build_features)
    dummy = pd.DataFrame({"subject_id": [1], "trial_num": [1], "window_index": [0]})
    with pytest.raises(ValueError, match="missing required features"):
        adapter.predict_trial_features(dummy)


@needs_artifact
@needs_sample_data
def test_prediction_parity_on_sample(adapter):
    """Adapter predictions must match the pipeline logic exactly."""
    df = load_and_trim(str(SAMPLE_DATA_PATH))

    # Build features exactly as the pipeline does
    df_feat, _ = build_features(df)

    # Prepare features for the raw model (select + clip)
    F = adapter.feature_order
    X = df_feat[F].copy()
    for c in F:
        lo, hi = adapter.clip[c]
        X[c] = X[c].clip(lo, hi)

    # Get the raw model's probabilities and apply the CALIBRATED threshold
    # (Using probabilities avoids the mismatch with model.predict's default 0.5)
    raw_proba = adapter.model.predict_proba(X)[:, 1]
    raw_preds = (raw_proba >= adapter.threshold).astype(int)

    # Get the adapter's predictions
    adapter_preds = np.array([p.predicted_label_id for p in adapter.predict_trial_features(df)])

    # They must match exactly
    assert (raw_preds == adapter_preds).all()


@needs_artifact
@needs_sample_data
def test_predictions_satisfy_contract(adapter):
    """Each prediction must be a well-formed WindowPrediction."""
    df = load_and_trim(str(SAMPLE_DATA_PATH))
    predictions = adapter.predict_trial_features(df)
    for p in predictions:
        assert p.execution in ("Correct", "Wrong", "Uncertain")
        assert 0.0 <= p.confidence_score <= 1.0
        assert p.predicted_label_id in (0, 1)
        assert len(p.probabilities) == 2
        assert p.feature_attributions


def test_engineered_feature_description_names_sensor_and_modality():
    description = describe_engineered_feature("sym_ratio_emg_15_rm5")

    assert "muscle activity" in description
    assert config.SENSOR_PLACEMENT[1] in description
    assert config.SENSOR_PLACEMENT[5] in description


def _temporal_feature_rows():
    rows = []
    for label_id in (0, 1):
        for window_index in range(6):
            value = float(10 * label_id + window_index + 1)
            rows.append(
                {
                    "subject_id": 2,
                    "label_id": label_id,
                    "trial_num": 1,
                    "window_index": window_index,
                    "start_time_s": window_index * 0.2,
                    "end_time_s": (window_index + 1) * 0.2,
                    "exercise": "Squat",
                    "execution": "Correct" if label_id == 0 else "Wrong",
                    "s1_emg_mav": value,
                    "s1_gyro_peak_angular_vel": value + 1,
                    "s1_acc_z_rms": value + 2,
                }
            )
    return pd.DataFrame(rows)


def test_temporal_features_are_causal_and_isolated_by_full_trial_key():
    original = _temporal_feature_rows()
    changed_future = original.copy()
    changed_future.loc[
        (changed_future["label_id"] == 0) & (changed_future["window_index"] == 5),
        "s1_acc_z_rms",
    ] = 1_000_000.0

    first, _ = build_features(original)
    second, _ = build_features(changed_future)
    earlier = (first["label_id"] == 0) & (first["window_index"] < 5)
    engineered = [
        column
        for column in first
        if column not in original.columns and pd.api.types.is_numeric_dtype(first[column])
    ]

    assert np.allclose(first.loc[earlier, engineered], second.loc[earlier, engineered])
    lag_column = "s1_acc_z_rms_norm_lag1"
    first_of_each_label = first.groupby(["subject_id", "label_id", "trial_num"]).head(1)
    assert np.all(first_of_each_label[lag_column] == 0.0)
    assert first.iloc[0]["s1_acc_z_rms_norm"] == pytest.approx(first.iloc[0]["s1_acc_z_rms"])


def test_feature_construction_accepts_one_unlabelled_trial_without_target_dependency():
    unlabelled = _temporal_feature_rows().query("label_id == 0").drop(columns="label_id")

    features, columns = build_features(unlabelled)

    assert columns
    assert features["trial_id"].nunique() == 1
    assert "label_id" not in features


def test_edge_trimming_does_not_merge_equal_trial_numbers_across_labels(tmp_path):
    path = tmp_path / "features.csv"
    _temporal_feature_rows().to_csv(path, index=False)

    trimmed = load_and_trim(path)

    assert len(trimmed) == 4
    assert trimmed.groupby(["subject_id", "label_id", "trial_num"]).size().to_dict() == {
        (2, 0, 1): 2,
        (2, 1, 1): 2,
    }


def test_xgboost_trial_metrics_use_complete_physical_trial_key():
    frame = pd.DataFrame(
        {
            "subject_id": [2, 2, 2, 2],
            "label_id": [0, 0, 1, 1],
            "trial_num": [1, 1, 1, 1],
            "label": [0, 0, 1, 1],
        }
    )

    labels, probabilities = _trial_probabilities(frame, [0.1, 0.3, 0.7, 0.9])
    metrics = _classification_metrics(labels, probabilities, 0.5)

    assert labels.tolist() == [0, 1]
    assert probabilities.tolist() == pytest.approx([0.2, 0.8])
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_xgboost_result_writer_has_no_authorship_or_generated_watermark(tmp_path):
    level_metrics = {
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "balanced_accuracy": 1.0,
        "per_class_precision": [1.0, 1.0],
        "per_class_recall": [1.0, 1.0],
        "per_class_f1": [1.0, 1.0],
        "support": [1, 1],
        "confusion_matrix": [[1, 0], [0, 1]],
        "n_samples": 2,
        "inference_latency_ms": 0.1,
    }
    bundle = {
        "model": {"test": True},
        "feature_order": ["feature"],
        "clip": {"feature": (0.0, 1.0)},
        "threshold": 0.5,
        "selected_K": 1,
        "config_name": "test",
        "config": {},
        "metrics": {"window": level_metrics, "trial": level_metrics},
    }

    save_artifacts(bundle, tmp_path, {"split": "test"})
    config_data = json.loads((tmp_path / "xgboost_config.json").read_text())

    assert "generated_at" not in config_data
    assert "author" not in config_data
    assert (tmp_path / "xgboost_metrics.csv").exists()
    assert (tmp_path / "xgboost_per_class_metrics.csv").exists()
    assert (tmp_path / "xgboost_confusion_matrices.csv").exists()
