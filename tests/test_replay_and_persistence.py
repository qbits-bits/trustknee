"""Integration tests for streaming replay, common inference engines, and database persistence."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from src import config
from src.inference import (
    MockInferenceEngine,
    TransformerInferenceEngine,
    WindowPrediction,
    XGBoostInferenceEngine,
)
from src.ingestion.ingest import Trial
from src.models.normalization import fit_sequence_normalizer
from src.persistence.db import DatabaseManager
from src.preprocessing.windowing import slice_trial_windows
from src.streaming import replay_trial_pipeline


class DummyPyTorchModel:
    """Minimal PyTorch-like mock model for Transformer testing."""

    def __init__(self, num_classes: int = 2) -> None:
        self.num_classes = num_classes

    def eval(self) -> None:
        pass

    def __call__(self, x):
        import torch

        # Return static unnormalized logits
        batch_size = x.shape[0]
        logits = torch.zeros((batch_size, self.num_classes), dtype=torch.float32)
        logits[:, 0] = 2.0  # favor class 0
        return logits


class DummyXGBoostModel:
    """Minimal XGBoost-like mock model for tabular testing."""

    def __init__(self, num_classes: int = 2) -> None:
        self.num_classes = num_classes

    def predict_proba(self, X):
        n_samples = len(X)
        probs = np.zeros((n_samples, self.num_classes), dtype=np.float32)
        probs[:, 0] = 0.88
        probs[:, 1] = 0.12
        return probs


@pytest.fixture
def sample_trial() -> Trial:
    """Generate a valid synthetic 1.0 second exercise trial."""
    n_imu = int(round(1.0 * config.IMU_SAMPLING_RATE_HZ))
    n_emg = int(round(1.0 * config.EMG_SAMPLING_RATE_HZ))
    imu = np.random.randn(8, 6, n_imu).astype(np.float32)
    emg = np.random.randn(8, n_emg).astype(np.float32)

    return Trial(
        subject_id=2,
        label_id=0,
        trial_num=1,
        imu=imu,
        emg=emg,
        duration_s=1.0,
    )


class TestDatabaseManager:
    """Test SQLite database operations and schema integrity."""

    def test_schema_init_and_seeding(self) -> None:
        db = DatabaseManager(db_path=":memory:")
        conn = db.get_connection()
        cur = conn.cursor()

        # Check Sensors table seeded with 8 hardware sensors
        cur.execute("SELECT COUNT(*) FROM Sensors;")
        assert cur.fetchone()[0] == 8

        # Check Labels table seeded with 9 classes
        cur.execute("SELECT COUNT(*) FROM Labels;")
        assert cur.fetchone()[0] == 9

        # Check Sessions table created
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Sessions';")
        assert cur.fetchone() is not None

    def test_persistence_lifecycle(self) -> None:
        db = DatabaseManager(db_path=":memory:")

        # 1. Subject and Session
        db.insert_subject(subject_id=2, gender="Male", height_cm=175.0, weight_kg=72, age_years=28)
        session_id = db.create_session(subject_id=2, notes="Baseline rehab test")
        assert session_id > 0

        # 2. Trial
        trial_id = db.insert_trial(
            subject_id=2,
            label_id=0,
            trial_num=1,
            imu_path="test_imu.npy",
            emg_path="test_emg.npy",
            duration_s=1.0,
            n_imu_samples=148,
            n_emg_samples=1259,
        )
        assert trial_id > 0

        # 3. Window Features and Prediction
        win_id = db.insert_window_features(
            trial_id=trial_id, window_index=0, start_time_s=0.0, end_time_s=0.2
        )
        assert win_id > 0

        pred_id = db.insert_prediction(
            window_id=win_id,
            model_name="transformer",
            predicted_label_id=0,
            confidence_score=0.92,
            is_flagged_uncertain=False,
        )
        assert pred_id > 0

        # 4. Attributions and Feedback
        db.insert_attributions(
            prediction_id=pred_id, attributions={"s1_acc_x_rms": 0.45, "s2_emg_mav": -0.12}
        )
        fb_id = db.insert_feedback(
            trial_id=trial_id, source_prediction_id=pred_id, feedback_text="Good form"
        )
        assert fb_id > 0

        # 5. Query and verify
        trial_data = db.get_trial(trial_id)
        assert trial_data is not None
        assert trial_data["subject_id"] == 2

        preds = db.get_trial_predictions(trial_id)
        assert len(preds) == 1
        assert preds[0]["confidence_score"] == 0.92
        assert preds[0]["execution"] == "Correct"


class TestInferenceEngines:
    """Test common inference contract across mock, transformer, and xgboost implementations."""

    def test_mock_engine(self) -> None:
        engine = MockInferenceEngine(predicted_class=0, confidence=0.75, uncertainty_threshold=0.60)
        imu = np.zeros((8, 6, 30), dtype=np.float32)
        emg = np.zeros((8, 252), dtype=np.float32)

        pred = engine.predict_window(imu, emg)
        assert isinstance(pred, WindowPrediction)
        assert pred.predicted_label_id == 0
        assert pred.confidence_score == 0.75
        assert not pred.is_flagged_uncertain
        assert pred.execution == "Correct"

        # Test uncertainty threshold triggering
        uncertain_engine = MockInferenceEngine(
            predicted_class=0, confidence=0.55, uncertainty_threshold=0.60
        )
        pred_u = uncertain_engine.predict_window(imu, emg)
        assert pred_u.is_flagged_uncertain
        assert pred_u.execution == "Uncertain"

    def test_transformer_engine(self) -> None:
        dummy_seqs = np.zeros((10, 30, 56), dtype=np.float32)
        normalizer = fit_sequence_normalizer(dummy_seqs, np.arange(10))
        model = DummyPyTorchModel(num_classes=2)

        engine = TransformerInferenceEngine(
            model=model,
            normalizer=normalizer,
            num_classes=2,
            temperature=1.0,
            uncertainty_threshold=0.60,
        )

        imu = np.zeros((8, 6, 30), dtype=np.float32)
        emg = np.zeros((8, 252), dtype=np.float32)
        pred = engine.predict_window(imu, emg)
        assert pred.predicted_label_id == 0
        assert pred.confidence_score > 0.80

    def test_xgboost_engine(self) -> None:
        model = DummyXGBoostModel(num_classes=2)
        engine = XGBoostInferenceEngine(model=model, num_classes=2, uncertainty_threshold=0.60)

        imu = np.zeros((8, 6, 30), dtype=np.float32)
        emg = np.zeros((8, 252), dtype=np.float32)
        pred = engine.predict_window(imu, emg)
        assert pred.predicted_label_id == 0
        assert pred.confidence_score == pytest.approx(0.88, abs=1e-4)


class TestReplayPipelineIntegration:
    """Test end-to-end replay pipeline and window parity against offline slicing."""

    def test_replay_pipeline_end_to_end(self, sample_trial: Trial) -> None:
        db = DatabaseManager(db_path=":memory:")
        engine = MockInferenceEngine(predicted_class=0, confidence=0.85)

        # 1. Run replay pipeline
        result = replay_trial_pipeline(
            trial=sample_trial,
            inference_engine=engine,
            database_manager=db,
            chunk_duration_ms=50.0,
            aggregation_method="mean_probability",
            generate_feedback=True,
        )

        assert result.database_persisted
        assert result.trial_id > 0
        assert result.n_windows > 0
        assert result.trial_prediction.predicted_label_id == 0

        # 2. Verify window parity against offline slice_trial_windows
        offline_windowed = slice_trial_windows(sample_trial, window_ms=200.0, overlap=0.5)
        assert result.n_windows == offline_windowed.n_windows

        # 3. Verify SQLite records
        persisted_preds = db.get_trial_predictions(result.trial_id)
        assert len(persisted_preds) == result.n_windows
        assert all(p["model_name"] == "mock_model" for p in persisted_preds)

    def test_replay_pipeline_requires_inference_engine(self, sample_trial: Trial) -> None:
        with pytest.raises(ValueError, match="inference_engine is required"):
            replay_trial_pipeline(trial=sample_trial, inference_engine=None)

    def test_subject_reuse_preserves_demographics(self) -> None:
        db = DatabaseManager(db_path=":memory:")
        # 1. Insert subject with custom demographics
        db.insert_subject(
            subject_id=3,
            gender="Female",
            height_cm=165.0,
            weight_kg=60,
            age_years=45,
            injured_leg="left",
            pathology="ACL rupture",
        )

        # 2. Insert trial referencing subject
        trial_id = db.insert_trial(
            subject_id=3,
            label_id=0,
            trial_num=1,
            imu_path="test_imu.npy",
            emg_path="test_emg.npy",
            duration_s=1.0,
            n_imu_samples=148,
            n_emg_samples=1259,
        )
        assert trial_id > 0

        # 3. Re-inserting or calling create_session should not overwrite demographics or cause FK violation
        db.create_session(subject_id=3, notes="Follow-up")

        subject = db.get_subject(3)
        assert subject is not None
        assert subject["gender"] == "Female"
        assert subject["pathology"] == "ACL rupture"

    def test_atomic_persistence_rollback_on_failure(self, sample_trial: Trial) -> None:
        db = DatabaseManager(db_path=":memory:")

        # Attempt atomic persist with a window frame that has invalid types or throws
        class BrokenPrediction:
            model_name = "test"
            predicted_label_id = "not_an_int"  # will trigger schema constraint/failure
            confidence_score = "not_a_float"
            is_flagged_uncertain = False
            feature_attributions = None

        class MockWindow:
            window_index = 0
            start_time_s = 0.0
            end_time_s = 0.2

        with pytest.raises(sqlite3.IntegrityError), db.transaction() as cur:
            cur.execute(
                "INSERT INTO Trials (subject_id, label_id, trial_num, imu_path, emg_path, duration_s, n_imu_samples, n_emg_samples) VALUES (1, 0, 1, 'a', 'b', 1.0, 10, 10);"
            )
            # Force a constraint violation
            cur.execute(
                "INSERT INTO WindowFeatures (trial_id, window_index, start_time_s, end_time_s) VALUES (-9999, 'bad', 'bad', 'bad');"
            )

        # Verify Trials table is still empty because transaction rolled back
        conn = db.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Trials;")
        assert cur.fetchone()[0] == 0
