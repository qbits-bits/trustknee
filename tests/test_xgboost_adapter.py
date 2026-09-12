"""Unit tests for the XGBoost feature-bridge adapter.

Verifies artifact integrity, rejection of missing features, and prediction
parity using a representative sample of the dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_adapter import XGBoostArtifactAdapter
from src.models.xgboost_pipeline import build_features, load_and_trim

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
