import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from src.features.masking import apply_sensor_mask_tabular, apply_sensor_mask_sequence # noqa: E402
from src.models.trial_aggregation import aggregate_trial_predictions, compute_metrics # noqa: E402

def test_tabular_sensor_mask():
    """Verify tabular mask keeps metadata columns and only active sensor features."""
    df = pd.DataFrame({
        "s0_mean": [1, 2],
        "s1_mean": [3, 4],
        "s2_mean": [5, 6],
        "label": [0, 1],
        "subject_id": ["sub1", "sub2"]
    })
    masked_df = apply_sensor_mask_tabular(df, active_sensors=[0, 2], sensor_column_prefix="s")
    
    assert "s0_mean" in masked_df.columns
    assert "s2_mean" in masked_df.columns
    assert "s1_mean" not in masked_df.columns
    assert "label" in masked_df.columns
    assert "subject_id" in masked_df.columns

def test_sequence_sensor_mask_channel_zeroing():
    """Verify 3D array channel masking zeroes out inactive channels."""
    # Batch=2, Channels=8, Sequence Length=100
    x_dummy = np.ones((2, 8, 100))
    active_sensors = [0, 3, 7]
    masked = apply_sensor_mask_sequence(x_dummy, active_sensors=active_sensors)
    
    # Active channels retain values (1.0)
    assert np.all(masked[:, 0, :] == 1.0)
    assert np.all(masked[:, 3, :] == 1.0)
    assert np.all(masked[:, 7, :] == 1.0)
    
    # Inactive channels zeroed out (0.0)
    assert np.all(masked[:, 1, :] == 0.0)
    assert np.all(masked[:, 2, :] == 0.0)
    assert np.all(masked[:, 4, :] == 0.0)

def test_aggregation_methods():
    """Verify trial-level probability aggregation logic across window samples."""
    probs = np.array([
        [0.8, 0.2],
        [0.4, 0.6],
        [0.9, 0.1]
    ])
    trial_ids = np.array(["sub1_t1", "sub1_t1", "sub1_t1"])
    
    mean_res = aggregate_trial_predictions(probs, trial_ids, method="mean_prob")
    vote_res = aggregate_trial_predictions(probs, trial_ids, method="majority_vote")
    conf_res = aggregate_trial_predictions(probs, trial_ids, method="confidence_weighted")
    
    assert mean_res["sub1_t1"] == 0  
    assert vote_res["sub1_t1"] == 0 
    assert conf_res["sub1_t1"] in [0, 1]

def test_compute_metrics_safe_zero_division():
    """Verify compute_metrics returns clean dictionary without throwing zero-division warnings."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 0]) 
    
    metrics = compute_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["accuracy"] == 0.5