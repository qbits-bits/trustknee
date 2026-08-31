"""Focused tests for TrustKnee model inputs and evaluation safeguards."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.evaluation import (
    FIXED_TEST_SUBJECTS,
    FIXED_TRAIN_SUBJECTS,
    FIXED_VALIDATION_SUBJECTS,
    average_trial_probabilities,
    fixed_subject_split,
    iter_loso_folds,
    run_fixed_split_comparison,
    run_loso_comparison,
)
from src.models.labels import binary_label_from_execution, labels_to_binary
from src.models.model_data import (
    build_model_inputs,
    create_emg_activity_envelope,
    make_transformer_sequences,
)
from src.models.normalization import fit_sequence_normalizer
from src.models.training import QUICK_TRAINING, fit_xgboost, predict_xgboost
from src.models.transformer import SinusoidalPositionalEncoding, TransformerEncoderClassifier


def test_emg_activity_envelope_uses_rectification_and_30_equal_bins():
    emg = np.zeros((8, 252), dtype=np.float32)
    emg[0, :126] = -2.0
    emg[0, 126:] = 4.0
    envelope = create_emg_activity_envelope(emg)

    assert envelope.shape == (8, 30)
    assert np.allclose(envelope[0, :15], 2.0)
    assert np.allclose(envelope[0, 15:], 4.0)


def test_transformer_input_has_30_positions_and_56_values():
    imu = np.zeros((2, config.N_SENSORS, config.IMU_CHANNELS_PER_SENSOR, 30), dtype=np.float32)
    emg = np.ones((2, config.N_SENSORS, 252), dtype=np.float32)
    sequences = make_transformer_sequences(imu, emg)

    assert sequences.shape == (2, 30, 56)
    assert np.all(sequences[:, :, 48:] == 1.0)


def test_transformer_outputs_binary_and_nine_class_logits():
    import torch

    values = torch.zeros(3, 30, 56)
    assert TransformerEncoderClassifier(num_classes=2)(values).shape == (3, 2)
    assert TransformerEncoderClassifier(num_classes=9)(values).shape == (3, 9)


def test_positional_encoding_changes_time_positions():
    import torch

    encoding = SinusoidalPositionalEncoding(d_model=64, max_seq_len=30)
    output = encoding(torch.zeros(1, 30, 64))
    assert not torch.equal(output[:, 0], output[:, 1])


def test_binary_label_mapping_is_explicit():
    assert binary_label_from_execution("Correct") == 0
    assert binary_label_from_execution("wrong") == 1
    assert labels_to_binary([0, 1, 3, 8]).tolist() == [0, 1, 0, 1]
    with pytest.raises(ValueError, match="conflicts"):
        labels_to_binary([0], ["Wrong"])


def test_normalization_uses_training_rows_only():
    values = np.zeros((3, 30, 56), dtype=np.float32)
    values[0] = 1.0
    values[1] = 3.0
    values[2] = 100.0  # held out and must not affect the fitted mean
    normalizer = fit_sequence_normalizer(values, np.array([0, 1]))

    assert float(normalizer.mean[0, 0, 0]) == pytest.approx(2.0)
    assert float(normalizer.transform(values)[2, 0, 0]) == pytest.approx(98.0)


def test_loso_folds_keep_subjects_separate():
    metadata = pd.DataFrame({"subject_id": [1, 1, 2, 2, 3], "trial_num": [1, 1, 1, 2, 1]})
    for held_out, train_indices, test_indices in iter_loso_folds(metadata):
        assert held_out not in set(metadata.iloc[train_indices].subject_id)
        assert set(metadata.iloc[test_indices].subject_id) == {held_out}


def test_fixed_subject_split_is_deterministic_and_disjoint():
    metadata = pd.DataFrame({"subject_id": np.repeat(np.arange(1, 32), 2)})
    train_indices, validation_indices, test_indices = fixed_subject_split(metadata)
    partitions = [
        set(metadata.iloc[indices].subject_id)
        for indices in (train_indices, validation_indices, test_indices)
    ]

    assert partitions == [
        set(FIXED_TRAIN_SUBJECTS),
        set(FIXED_VALIDATION_SUBJECTS),
        set(FIXED_TEST_SUBJECTS),
    ]
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]
    assert 1 not in set().union(*partitions)


def test_fixed_subject_split_rejects_overlap_and_missing_subjects():
    metadata = pd.DataFrame({"subject_id": [1, 2, 3]})
    with pytest.raises(ValueError, match="overlap"):
        fixed_subject_split(metadata, [1], [1], [3])
    with pytest.raises(ValueError, match="missing subjects"):
        fixed_subject_split(metadata, [1], [2], [4])


def test_trial_probability_averaging_groups_overlapping_windows():
    metadata = pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "trial_num": [1, 1, 2],
            "trial_id": ["subject_1_trial_1", "subject_1_trial_1", "subject_1_trial_2"],
            "label_id": [0, 0, 1],
            "execution": ["Correct", "Correct", "Wrong"],
        }
    )
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]])
    trials = average_trial_probabilities(probabilities, metadata)

    assert len(trials) == 2
    first = trials[trials["trial_num"] == 1].iloc[0]
    assert first["probability_0"] == pytest.approx(0.6)
    assert first["n_windows"] == 2


def test_model_dataset_keeps_metadata_out_of_model_features(tmp_path):
    _write_model_dataset(tmp_path)
    from src.ingestion import build_manifest

    dataset = build_model_inputs(build_manifest(tmp_path, exclude_subjects=set()), preprocess=False)

    assert dataset.transformer_sequences.shape[1:] == (30, 56)
    assert not set(dataset.metadata.columns) & set(dataset.feature_names)
    assert not {"subject_id", "trial_id", "window_index", "start_time_s"} & set(
        dataset.feature_names
    )


def test_xgboost_maps_sparse_training_classes_to_global_probabilities():
    features = pd.DataFrame(
        {
            "feature_a": np.arange(12, dtype=np.float32),
            "feature_b": np.tile([0.0, 1.0, 2.0], 4),
        }
    )
    labels = np.array([0, 2, 8, 0, 2, 8, 0, 2, 8, 0, 2, 8])
    indices = np.arange(len(labels))

    model = fit_xgboost(
        features,
        labels,
        indices,
        num_classes=9,
        training_config=QUICK_TRAINING,
    )
    probabilities = predict_xgboost(model, features, num_classes=9)

    assert probabilities.shape == (len(features), 9)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities[:, [1, 3, 4, 5, 6, 7]] == 0.0)
    assert model.estimator.get_params()["n_jobs"] == -1


def _write_model_dataset(data_root, subjects=(1, 2)):
    participants = pd.DataFrame(
        [
            {
                "Participant ID": subject,
                "Gender (M/F)": "M",
                "Height (CM)": 1.80,
                "Weight(KG)": 80.0,
                "Age (Years)": 30,
                "Leg": "Right",
                "Pathology": "None",
            }
            for subject in subjects
        ]
    )
    participants.to_csv(data_root / "participants.csv", index=False)
    pd.DataFrame(
        [
            {"Label ID": 0, "Execution": "Correct", "Details": "Squat"},
            {"Label ID": 1, "Execution": "Wrong", "Details": "Wrong squat"},
        ]
    ).to_csv(data_root / "labels.csv", index=False)
    pd.DataFrame(
        [
            {"Sensor ID": sensor, "Muscle": muscle}
            for sensor, muscle in config.SENSOR_PLACEMENT.items()
        ]
    ).to_csv(data_root / "placement.csv", index=False)
    (data_root / "sensors.csv").write_text("n_sensors,8\n", encoding="utf-8")
    rng = np.random.default_rng(42)
    for subject in subjects:
        for label in [0, 1]:
            trial_dir = data_root / "dataset" / f"Subject_{subject}" / str(label) / "Trial_1"
            trial_dir.mkdir(parents=True)
            np.save(trial_dir / "imu.npy", rng.normal(size=(48, 300)))
            np.save(trial_dir / "emg.npy", rng.normal(size=(8, 2550)))


def test_quick_end_to_end_writes_loso_results(tmp_path):
    _write_model_dataset(tmp_path)
    from src.ingestion import build_manifest

    results = run_loso_comparison(
        build_manifest(tmp_path, exclude_subjects=set()), tmp_path / "reports", quick=True
    )

    assert {"transformer", "xgboost", "majority_baseline"} <= set(results["model"])
    assert {"binary", "nine_class"} <= set(results["task"])
    assert {"window", "trial"} <= set(results["level"])
    assert np.isfinite(results[["accuracy", "macro_f1", "balanced_accuracy"]]).all().all()
    for filename in ["fold_results.csv", "summary_results.csv", "config.json"]:
        assert (tmp_path / "reports" / filename).exists()

    resumed = run_loso_comparison(
        build_manifest(tmp_path, exclude_subjects=set()),
        tmp_path / "reports",
        quick=True,
        resume=True,
    )
    pd.testing.assert_frame_equal(
        results.sort_values(["fold", "model", "task", "level"]).reset_index(drop=True),
        resumed.sort_values(["fold", "model", "task", "level"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_quick_end_to_end_writes_fixed_split_results(tmp_path):
    _write_model_dataset(tmp_path, subjects=(1, 2, 3))
    from src.ingestion import build_manifest

    manifest = build_manifest(tmp_path, exclude_subjects=set())
    results = run_fixed_split_comparison(
        manifest,
        tmp_path / "fixed-reports",
        quick=True,
        train_subjects=[1],
        validation_subjects=[2],
        test_subjects=[3],
    )

    assert len(results) == 12
    assert set(results["n_train_subjects"]) == {1}
    assert set(results["n_validation_subjects"]) == {1}
    assert set(results["n_test_subjects"]) == {1}
    assert np.isfinite(results[["accuracy", "macro_f1", "balanced_accuracy"]]).all().all()
    config = (tmp_path / "fixed-reports" / "config.json").read_text(encoding="utf-8")
    assert '"evaluation": "fixed_subject_split"' in config

    resumed = run_fixed_split_comparison(
        manifest,
        tmp_path / "fixed-reports",
        quick=True,
        resume=True,
        train_subjects=[1],
        validation_subjects=[2],
        test_subjects=[3],
    )
    pd.testing.assert_frame_equal(results, resumed, check_dtype=False)

    imu_path = tmp_path / "dataset" / "Subject_1" / "0" / "Trial_1" / "imu.npy"
    changed_imu = np.load(imu_path)
    changed_imu[0, 0] += 1.0
    np.save(imu_path, changed_imu)
    with pytest.raises(ValueError, match="configuration does not match"):
        run_fixed_split_comparison(
            manifest,
            tmp_path / "fixed-reports",
            quick=True,
            resume=True,
            train_subjects=[1],
            validation_subjects=[2],
            test_subjects=[3],
        )

    (tmp_path / "fixed-reports" / "config.json").unlink()
    with pytest.raises(ValueError, match="requires config.json"):
        run_fixed_split_comparison(
            manifest,
            tmp_path / "fixed-reports",
            quick=True,
            resume=True,
            train_subjects=[1],
            validation_subjects=[2],
            test_subjects=[3],
        )


def test_evaluate_cli_parser_defaults():
    from src.models.evaluate import build_parser

    parser = build_parser()
    args = parser.parse_args(["--data-root", "data", "--output-dir", "out"])
    assert not args.include_held_out
    assert not args.quick
    assert not args.resume
    assert args.batch_size is None
    assert args.evaluation == "loso"
    assert args.seed == 42

    args_held = parser.parse_args(
        [
            "--data-root",
            "data",
            "--output-dir",
            "out",
            "--include-held-out",
            "--quick",
            "--augment-minority",
            "--aug-multiplier",
            "5",
            "--aug-methods",
            "jitter,time_warp",
            "--aug-targets",
            "6,7",
        ]
    )
    assert args_held.include_held_out
    assert args_held.quick
    assert args_held.augment_minority
    assert args_held.aug_multiplier == 5
    assert args_held.aug_methods == "jitter,time_warp"
    assert args_held.aug_targets == "6,7"


def test_iter_loso_folds_strictly_excludes_synthetic_from_test():
    from src.models.evaluation import iter_loso_folds

    metadata = pd.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2, 2],
            "synthetic": [False, False, True, False, True, True],
        }
    )
    folds = list(iter_loso_folds(metadata))
    assert len(folds) == 2

    # Fold 1: held_out = 1
    held_out_1, train_1, test_1 = folds[0]
    assert held_out_1 == 1
    # test_1 must only contain real subject 1 indices (0, 1) and never synthetic (2)
    assert list(test_1) == [0, 1]
    # train_1 contains all subject 2 indices (3, 4, 5)
    assert list(train_1) == [3, 4, 5]

    # Fold 2: held_out = 2
    held_out_2, train_2, test_2 = folds[1]
    assert held_out_2 == 2
    # test_2 must only contain real subject 2 indices (3) and never synthetic (4, 5)
    assert list(test_2) == [3]
    # train_2 contains all subject 1 indices (0, 1, 2)
    assert list(train_2) == [0, 1, 2]


def test_subject_validation_split_strictly_excludes_synthetic_from_validation():
    from src.models.evaluation import subject_validation_split

    metadata = pd.DataFrame(
        {
            "subject_id": [2, 2, 2, 3, 3, 3],
            "synthetic": [False, False, True, False, False, True],
        }
    )
    train_indices = np.array([0, 1, 2, 3, 4, 5])
    fit_indices, val_indices = subject_validation_split(metadata, train_indices, seed=42)

    # val_indices must only contain non-synthetic rows
    assert not metadata.iloc[val_indices]["synthetic"].any()
    # fit_indices + val_indices must cover the training subjects
    assert set(fit_indices).isdisjoint(set(val_indices))


def test_quick_end_to_end_with_augmentation_writes_loso_results(tmp_path):
    _write_model_dataset(tmp_path)
    from src.ingestion import build_manifest

    reports_dir = tmp_path / "aug_reports"
    results = run_loso_comparison(
        build_manifest(tmp_path, exclude_subjects=set()),
        reports_dir,
        quick=True,
        augment_minority=True,
        aug_multiplier=2,
        aug_target_labels=(0,),  # label 0 exists in synthetic test dataset
    )

    assert {"transformer", "xgboost", "majority_baseline"} <= set(results["model"])
    assert (reports_dir / "config.json").exists()
    config_data = json.loads((reports_dir / "config.json").read_text(encoding="utf-8"))
    assert config_data["augmentation"]["enabled"] is True
    assert config_data["augmentation"]["multiplier"] == 2


def test_build_model_inputs_deterministic_with_seed(tmp_path):
    _write_model_dataset(tmp_path)
    from src.ingestion import build_manifest

    manifest = build_manifest(tmp_path, exclude_subjects=set())
    ds1 = build_model_inputs(
        manifest,
        augment_minority=True,
        aug_multiplier=2,
        target_labels=(0,),
        seed=42,
    )
    ds2 = build_model_inputs(
        manifest,
        augment_minority=True,
        aug_multiplier=2,
        target_labels=(0,),
        seed=42,
    )
    np.testing.assert_array_equal(ds1.transformer_sequences, ds2.transformer_sequences)
    pd.testing.assert_frame_equal(ds1.xgboost_features, ds2.xgboost_features)
    pd.testing.assert_frame_equal(ds1.metadata, ds2.metadata)
