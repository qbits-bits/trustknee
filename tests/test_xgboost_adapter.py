"""Unit tests for the XGBoost feature-bridge adapter.

Verifies artifact integrity, rejection of missing features, prediction parity
on the canonical test split, and WindowPrediction contract compliance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.xgboost_adapter import XGBoostArtifactAdapter
from src.models.xgboost_pipeline import load_and_trim

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "models" / "xgboost_canonical" / "xgboost_artifact.joblib"
DATA_PATH = REPO_ROOT / "data" / "processed" / "kneepad_features.csv"

needs_data = pytest.mark.skipif(not DATA_PATH.exists(), reason="canonical dataset not available")


@pytest.fixture(scope="module")
def adapter():
    return XGBoostArtifactAdapter(str(ARTIFACT_PATH))


@pytest.fixture(scope="module")
def canonical_test_result(adapter):
    df = load_and_trim(str(DATA_PATH))
    test_ids = list(config.PHASE3_TEST_SUBJECTS)
    test_df = df[df["subject_id"].isin(test_ids)].copy()
    test_df = test_df.sort_values(["subject_id", "trial_num", "window_index"])
    predictions = adapter.predict_trial_features(test_df)
    return predictions, test_df


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


@needs_data
def test_prediction_parity_on_canonical_test(adapter, canonical_test_result):
    """Reloading the artifact must reproduce the reported canonical accuracy."""
    predictions, test_df = canonical_test_result
    preds = np.array([p.predicted_label_id for p in predictions])
    truth = test_df["label"].values
    acc = (preds == truth).mean()
    assert acc == pytest.approx(0.915, abs=0.005)


@needs_data
def test_predictions_satisfy_contract(adapter, canonical_test_result):
    """Each prediction must be a well-formed WindowPrediction."""
    predictions, _ = canonical_test_result
    for p in predictions:
        assert p.execution in ("Correct", "Wrong", "Uncertain")
        assert 0.0 <= p.confidence_score <= 1.0
        assert p.predicted_label_id in (0, 1)
        assert len(p.probabilities) == 2
