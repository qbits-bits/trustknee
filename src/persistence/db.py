"""SQLite persistence layer implementing the TrustKnee database schema.

Covers Subjects, Sensors, Labels, Trials, WindowFeatures, Predictions,
ShapAttributions, Feedback, and Sessions matching db/schema.sql.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config

logger = logging.getLogger("trustknee.persistence")

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "db" / "schema.sql"


class DatabaseManager:
    """Manages SQLite storage, migrations, and queries according to db/schema.sql."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        schema_path: str | Path | None = None,
        auto_init: bool = True,
    ) -> None:
        self.db_path = str(db_path)
        self.schema_path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
        self._conn: sqlite3.Connection | None = None

        if auto_init:
            self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Return the persistent connection or create a new one."""
        if self._conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            self._conn = conn
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager providing transactional execution."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def initialize_schema(self) -> None:
        """Create tables from schema.sql and seed static Sensors and Labels."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found at {self.schema_path}")

        schema_sql = self.schema_path.read_text(encoding="utf-8")

        with self.transaction() as cur:
            cur.executescript(schema_sql)

            # Ensure Sessions table exists for session tracking
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS Sessions (
                    session_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_id    INTEGER NOT NULL,
                    session_time  TEXT NOT NULL,
                    notes         TEXT,
                    FOREIGN KEY (subject_id) REFERENCES Subjects (subject_id)
                );
                """
            )

            # Seed static hardware sensors (1..8)
            for sensor_id, placement in config.SENSOR_PLACEMENT.items():
                cur.execute(
                    """
                    INSERT OR IGNORE INTO Sensors (sensor_id, muscle_placement, emg_channels, imu_channels)
                    VALUES (?, ?, ?, ?);
                    """,
                    (
                        sensor_id,
                        placement,
                        config.EMG_CHANNELS_PER_SENSOR,
                        config.IMU_CHANNELS_PER_SENSOR,
                    ),
                )

            # Seed static exercise labels (0..8)
            for label_id, info in config.LABELS.items():
                cur.execute(
                    """
                    INSERT OR IGNORE INTO Labels (label_id, execution, exercise, description)
                    VALUES (?, ?, ?, ?);
                    """,
                    (
                        label_id,
                        info.execution,
                        info.exercise,
                        info.description,
                    ),
                )

    PLACEHOLDER_SUBJECT: dict[str, Any] = {
        "gender": "Unknown",
        "height_cm": 170.0,
        "weight_kg": 70,
        "age_years": 30,
        "injured_leg": "right",
        "pathology": None,
    }

    def insert_subject(
        self,
        subject_id: int,
        gender: str = "Unknown",
        height_cm: float = 170.0,
        weight_kg: int = 70,
        age_years: int = 30,
        injured_leg: str = "right",
        pathology: str | None = None,
        update_existing: bool = False,
    ) -> None:
        """Insert a subject, upgrading placeholder rows with authoritative values."""
        with self.transaction() as cur:
            cur.execute(
                "SELECT gender, height_cm, weight_kg, age_years, injured_leg, pathology FROM Subjects WHERE subject_id = ?;",
                (subject_id,),
            )
            existing = cur.fetchone()
            if existing is None:
                cur.execute(
                    """
                    INSERT INTO Subjects (subject_id, gender, height_cm, weight_kg, age_years, injured_leg, pathology)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (subject_id, gender, height_cm, weight_kg, age_years, injured_leg, pathology),
                )
                return
            is_placeholder = all(existing[k] == v for k, v in self.PLACEHOLDER_SUBJECT.items())
            has_authoritative = (
                gender != "Unknown"
                or height_cm != 170.0
                or weight_kg != 70
                or age_years != 30
                or injured_leg != "right"
                or pathology is not None
            )
            if update_existing or (is_placeholder and has_authoritative):
                cur.execute(
                    """
                    UPDATE Subjects SET gender=?, height_cm=?, weight_kg=?, age_years=?, injured_leg=?, pathology=?
                    WHERE subject_id=?;
                    """,
                    (gender, height_cm, weight_kg, age_years, injured_leg, pathology, subject_id),
                )

    def create_session(
        self,
        subject_id: int,
        session_time: str | None = None,
        notes: str | None = None,
    ) -> int:
        """Create a new session record for a subject and return session_id."""
        if session_time is None:
            session_time = datetime.now(timezone.utc).isoformat()

        # Ensure subject exists
        self.insert_subject(subject_id=subject_id)

        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO Sessions (subject_id, session_time, notes)
                VALUES (?, ?, ?);
                """,
                (subject_id, session_time, notes),
            )
            return int(cur.lastrowid)

    def insert_trial(
        self,
        subject_id: int,
        label_id: int,
        trial_num: int,
        imu_path: str,
        emg_path: str,
        duration_s: float,
        n_imu_samples: int,
        n_emg_samples: int,
    ) -> int:
        """Record a completed trial and return its trial_id."""
        self.insert_subject(subject_id=subject_id)

        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO Trials (subject_id, label_id, trial_num, imu_path, emg_path, duration_s, n_imu_samples, n_emg_samples)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    subject_id,
                    label_id,
                    trial_num,
                    str(imu_path),
                    str(emg_path),
                    float(duration_s),
                    int(n_imu_samples),
                    int(n_emg_samples),
                ),
            )
            return int(cur.lastrowid)

    def insert_window_features(
        self,
        trial_id: int,
        window_index: int,
        start_time_s: float,
        end_time_s: float,
    ) -> int:
        """Insert a window record and return window_id."""
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO WindowFeatures (trial_id, window_index, start_time_s, end_time_s)
                VALUES (?, ?, ?, ?);
                """,
                (trial_id, window_index, float(start_time_s), float(end_time_s)),
            )
            return int(cur.lastrowid)

    def insert_prediction(
        self,
        window_id: int,
        model_name: str,
        predicted_label_id: int,
        confidence_score: float,
        is_flagged_uncertain: bool = False,
    ) -> int:
        """Insert a window prediction and return prediction_id."""
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO Predictions (window_id, model_name, predicted_label_id, confidence_score, is_flagged_uncertain)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    window_id,
                    model_name,
                    predicted_label_id,
                    float(confidence_score),
                    bool(is_flagged_uncertain),
                ),
            )
            return int(cur.lastrowid)

    def insert_attributions(
        self,
        prediction_id: int,
        attributions: dict[str, float],
    ) -> None:
        """Insert feature or sensor attributions for a prediction."""
        if not attributions:
            return
        with self.transaction() as cur:
            for feature_name, value in attributions.items():
                cur.execute(
                    """
                    INSERT INTO ShapAttributions (prediction_id, feature_name, attribution_value)
                    VALUES (?, ?, ?);
                    """,
                    (prediction_id, str(feature_name), float(value)),
                )

    def insert_feedback(
        self,
        trial_id: int,
        source_prediction_id: int,
        feedback_text: str,
    ) -> int:
        """Insert patient/clinician feedback for a trial."""
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO Feedback (trial_id, source_prediction_id, feedback_text)
                VALUES (?, ?, ?);
                """,
                (trial_id, source_prediction_id, str(feedback_text)),
            )
            return int(cur.lastrowid)

    def get_subject(self, subject_id: int) -> dict[str, Any] | None:
        """Query subject demographics by ID."""
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM Subjects WHERE subject_id = ?;", (subject_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_trial(self, trial_id: int) -> dict[str, Any] | None:
        """Query trial metadata by ID."""
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM Trials WHERE trial_id = ?;", (trial_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_trial_predictions(self, trial_id: int) -> list[dict[str, Any]]:
        """Query all window predictions for a trial."""
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT p.prediction_id, w.window_index, w.start_time_s, w.end_time_s,
                   p.model_name, p.predicted_label_id, p.confidence_score,
                   p.is_flagged_uncertain, l.execution, l.exercise
            FROM Predictions p
            JOIN WindowFeatures w ON p.window_id = w.window_id
            JOIN Labels l ON p.predicted_label_id = l.label_id
            WHERE w.trial_id = ?
            ORDER BY w.window_index ASC;
            """,
            (trial_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def persist_replay_trial(
        self,
        subject_id: int,
        label_id: int,
        trial_num: int,
        imu_path: str,
        emg_path: str,
        duration_s: float,
        n_imu_samples: int,
        n_emg_samples: int,
        windows: list[Any],
        predictions: list[Any],
        feedback_text: str | None = None,
    ) -> int:
        """Atomically persist a replayed trial, its windows, predictions, attributions, and feedback in one transaction."""
        with self.transaction() as cur:
            # 1. Ensure subject exists without overwriting demographics
            cur.execute(
                """
                INSERT INTO Subjects (subject_id, gender, height_cm, weight_kg, age_years, injured_leg, pathology)
                VALUES (?, 'Unknown', 170.0, 70, 30, 'right', NULL)
                ON CONFLICT(subject_id) DO NOTHING;
                """,
                (subject_id,),
            )

            # 2. Insert Trial record
            cur.execute(
                """
                INSERT INTO Trials (subject_id, label_id, trial_num, imu_path, emg_path, duration_s, n_imu_samples, n_emg_samples)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    subject_id,
                    label_id,
                    trial_num,
                    str(imu_path),
                    str(emg_path),
                    float(duration_s),
                    int(n_imu_samples),
                    int(n_emg_samples),
                ),
            )
            trial_id = int(cur.lastrowid)

            # 3. Insert Windows, Predictions, and Attributions
            source_pid: int | None = None
            for win, pred in zip(windows, predictions, strict=True):
                cur.execute(
                    """
                    INSERT INTO WindowFeatures (trial_id, window_index, start_time_s, end_time_s)
                    VALUES (?, ?, ?, ?);
                    """,
                    (trial_id, win.window_index, float(win.start_time_s), float(win.end_time_s)),
                )
                win_id = int(cur.lastrowid)

                cur.execute(
                    """
                    INSERT INTO Predictions (window_id, model_name, predicted_label_id, confidence_score, is_flagged_uncertain)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (
                        win_id,
                        pred.model_name,
                        int(pred.predicted_label_id),
                        float(pred.confidence_score),
                        bool(pred.is_flagged_uncertain),
                    ),
                )
                pred_id = int(cur.lastrowid)
                if source_pid is None:
                    source_pid = pred_id

                if getattr(pred, "feature_attributions", None):
                    for feat_name, feat_val in pred.feature_attributions.items():
                        cur.execute(
                            """
                            INSERT INTO ShapAttributions (prediction_id, feature_name, attribution_value)
                            VALUES (?, ?, ?);
                            """,
                            (pred_id, str(feat_name), float(feat_val)),
                        )

            # 4. Insert Feedback (linked to first prediction)
            if feedback_text and source_pid is not None:
                cur.execute(
                    """
                    INSERT INTO Feedback (trial_id, source_prediction_id, feedback_text)
                    VALUES (?, ?, ?);
                    """,
                    (trial_id, source_pid, str(feedback_text)),
                )

            return trial_id

    def close(self) -> None:
        """Close connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
