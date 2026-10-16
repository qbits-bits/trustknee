from __future__ import annotations

import json

import numpy as np
import pandas as pd

import src.models.sensor_selection as sensor_selection
from src.contracts import get_canonical_contract
from src.models.evaluation import fixed_subject_split
from src.models.model_data import ModelDataset
from src.models.sensor_selection import (
    aggregate_trial_probabilities,
    feature_sensor_ids,
    mask_tabular_features,
    run_sensor_selection,
)


def test_tabular_mask_removes_direct_and_derived_inactive_sensor_features():
    features = pd.DataFrame(
        {
            "s1_emg_rms": [1.0],
            "s2_emg_rms": [2.0],
            "sym_ratio_emg_15_rm5": [3.0],
            "ctrl_ratio_1": [4.0],
            "ctx_Squat": [1.0],
            "lrshare_emg": [0.5],
        }
    )

    masked = mask_tabular_features(features, (1,))

    assert list(masked) == ["s1_emg_rms", "ctrl_ratio_1", "ctx_Squat"]
    assert feature_sensor_ids("sym_ratio_emg_15_rm5") == {1, 5}


def test_trial_aggregation_methods_are_aligned_and_finite():
    probabilities = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
    metadata = pd.DataFrame({"trial_id": ["a", "a", "b"]})
    labels = np.array([0, 0, 1])

    for method in ("mean_probability", "majority_vote", "confidence_weighted"):
        trial_labels, trial_probabilities = aggregate_trial_probabilities(
            probabilities,
            metadata,
            labels,
            method,
        )
        assert trial_labels.tolist() == [0, 1]
        assert trial_probabilities.shape == (2, 2)
        assert np.isfinite(trial_probabilities).all()
        assert np.allclose(trial_probabilities.sum(axis=1), 1.0)


def _synthetic_dataset() -> ModelDataset:
    rng = np.random.default_rng(42)
    subjects = list(range(2, 32))
    metadata_rows = []
    feature_rows = []
    labels = []
    for subject in subjects:
        for trial in (1, 2):
            label = (subject + trial) % 2
            metadata_rows.append(
                {
                    "subject_id": subject,
                    "label_id": label,
                    "trial_num": trial,
                    "trial_id": f"{subject}-{label}-{trial}",
                    "window_index": 0,
                }
            )
            feature_rows.append(
                {
                    "s1_signal": label + rng.normal(scale=0.05),
                    "s2_signal": rng.normal(),
                    "s3_signal": rng.normal(),
                }
            )
            labels.append(label)
    count = len(labels)
    return ModelDataset(
        transformer_sequences=np.zeros((count, 30, 56), dtype=np.float32),
        xgboost_features=pd.DataFrame(feature_rows, dtype=np.float32),
        labels_9=np.asarray(labels),
        labels_binary=np.asarray(labels),
        metadata=pd.DataFrame(metadata_rows),
    )


def test_quick_sbe_writes_validation_only_handoff_and_test_comparison(tmp_path):
    handoff = run_sensor_selection(
        _synthetic_dataset(),
        tmp_path,
        quick=True,
        min_sensors=2,
        initial_sensors=(1, 2, 3),
    )

    assert len(handoff["selected_sensor_ids"]) == 2
    assert handoff["test_used_for_selection"] is False
    assert len(handoff["transformer_sensor_mask_boolean"]) == 8
    assert (tmp_path / "sensor_subset_results.csv").exists()
    test_results = pd.read_csv(tmp_path / "sensor_subset_test_results.csv")
    assert set(test_results["subset"]) == {"all_sensors", "recommended_reduced"}
    summary = json.loads((tmp_path / "sensor_selection_summary.json").read_text())
    survival_counts = summary["sensor_survival_counts_across_elimination_steps"]
    assert sum(survival_counts.values()) == 2
    assert summary["elimination_step_count"] == 1
    assert summary["selection_stability_assessed"] is False
    assert (
        "not a cross-sample selection stability measure"
        in summary["sensor_survival_count_description"]
    )


def test_sensor_selection_keeps_subject_folds_disjoint_and_defers_test_predictions(
    tmp_path, monkeypatch
):
    dataset = _synthetic_dataset()
    contract = get_canonical_contract()
    train_indices, val_indices, test_indices = fixed_subject_split(
        dataset.metadata,
        contract.train_subjects,
        contract.val_subjects,
        contract.test_subjects,
    )
    dev_indices = np.concatenate([train_indices, val_indices])
    fit_rows = []
    prediction_rows = []
    completed_subset_evaluations = 0

    def fake_fit_xgboost(features, labels, train_idx, **kwargs):
        rows = np.asarray(train_idx)
        fit_rows.append(rows)
        return rows

    def fake_predict_xgboost(model, features, num_classes):
        prediction_rows.append(
            {
                "train_rows": np.asarray(model),
                "predicted_rows": features.index.to_numpy(),
                "completed_subset_evaluations": completed_subset_evaluations,
            }
        )
        return np.full((len(features), num_classes), 1 / num_classes)

    original_evaluate_subset = sensor_selection._evaluate_subset

    def tracked_evaluate_subset(*args, **kwargs):
        nonlocal completed_subset_evaluations
        result = original_evaluate_subset(*args, **kwargs)
        completed_subset_evaluations += 1
        return result

    monkeypatch.setattr(sensor_selection, "fit_xgboost", fake_fit_xgboost)
    monkeypatch.setattr(sensor_selection, "predict_xgboost", fake_predict_xgboost)
    monkeypatch.setattr(sensor_selection, "_evaluate_subset", tracked_evaluate_subset)

    run_sensor_selection(
        dataset,
        tmp_path,
        quick=True,
        min_sensors=2,
        initial_sensors=(1, 2, 3),
    )

    subject_ids = dataset.metadata["subject_id"].to_numpy()
    expected_evaluations = 1 + len((1, 2, 3))
    assert completed_subset_evaluations == expected_evaluations

    for prediction in prediction_rows:
        train_subjects = set(subject_ids[prediction["train_rows"]])
        predicted_subjects = set(subject_ids[prediction["predicted_rows"]])
        assert train_subjects.isdisjoint(predicted_subjects)

    test_rows = set(test_indices)
    test_predictions = [
        prediction
        for prediction in prediction_rows
        if test_rows.intersection(prediction["predicted_rows"])
    ]
    assert len(test_predictions) == 2
    assert all(set(prediction["predicted_rows"]) == test_rows for prediction in test_predictions)
    assert all(set(prediction["train_rows"]) == set(dev_indices) for prediction in test_predictions)
    assert all(
        prediction["completed_subset_evaluations"] == expected_evaluations
        for prediction in test_predictions
    )
    assert all(not test_rows.intersection(rows) for rows in fit_rows)
