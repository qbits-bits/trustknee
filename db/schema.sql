-- TrustKnee Phase 2, Section 5.4 database schema.
--
-- Covers the full pipeline for the KneE-PAD dataset: raw trial ingestion,
-- windowed features, model predictions, SHAP attributions, patient feedback.
-- Mirrors the constants in src/config.py (N_SENSORS, LABELS, SENSOR_PLACEMENT).

PRAGMA foreign_keys = ON;

-- One row per participant (participants.csv).
CREATE TABLE Subjects (
    subject_id   INTEGER PRIMARY KEY,
    gender       VARCHAR(16) NOT NULL,
    height_cm    REAL NOT NULL,
    weight_kg    INTEGER NOT NULL,
    age_years    INTEGER NOT NULL,
    injured_leg  VARCHAR(8) NOT NULL,   -- 'left' | 'right'
    pathology    VARCHAR(64)            -- e.g. 'meniscus', 'osteoarthritis'
);

-- Fixed hardware description of the 8-sensor rig (placement.csv). No FK from
-- WindowFeatures since every trial uses all 8 sensors the same way, so it's
-- a fixed 1:8 reference, not a per-row foreign key.
CREATE TABLE Sensors (
    sensor_id         INTEGER PRIMARY KEY,   -- 1-8
    muscle_placement  VARCHAR(64) NOT NULL,  -- e.g. 'right rectus femoris'
    emg_channels      INTEGER NOT NULL DEFAULT 1,
    imu_channels      INTEGER NOT NULL DEFAULT 6
);

-- The 9 exercise/execution labels (labels.csv).
CREATE TABLE Labels (
    label_id   INTEGER PRIMARY KEY,   -- 0-8
    execution  VARCHAR(8) NOT NULL,   -- 'Correct' | 'Wrong'
    exercise   VARCHAR(32) NOT NULL,  -- 'Squat' | 'Seated leg extension' | 'Walking'
    description TEXT NOT NULL
);

-- One row per recorded trial (subject x label x trial_num), matching
-- ingestion's manifest of raw IMU/EMG file pairs.
CREATE TABLE Trials (
    trial_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id     INTEGER NOT NULL,
    label_id       INTEGER NOT NULL,
    trial_num      INTEGER NOT NULL,
    imu_path       VARCHAR(255) NOT NULL,
    emg_path       VARCHAR(255) NOT NULL,
    duration_s     REAL NOT NULL,
    n_imu_samples  INTEGER NOT NULL,
    n_emg_samples  INTEGER NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES Subjects (subject_id),
    FOREIGN KEY (label_id)   REFERENCES Labels (label_id)
);

-- Sliding-window segments of a trial (WINDOW_MS / WINDOW_OVERLAP in
-- config.py), one feature vector per window. The real table will have many
-- more float feature columns than shown here, named per sensor/channel/stat
-- (e.g. s1_acc_x_rms, s1_emg_mean); feature extraction owns the exact set.
CREATE TABLE WindowFeatures (
    window_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id      INTEGER NOT NULL,
    window_index  INTEGER NOT NULL,
    start_time_s  REAL NOT NULL,
    end_time_s    REAL NOT NULL,
    -- feature_* REAL columns go here (peak, mean, RMS, std, jerk, ROM, etc)
    FOREIGN KEY (trial_id) REFERENCES Trials (trial_id)
);

-- Model output for a single window.
CREATE TABLE Predictions (
    prediction_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id            INTEGER NOT NULL,
    model_name            VARCHAR(64) NOT NULL,
    predicted_label_id    INTEGER NOT NULL,
    confidence_score       REAL NOT NULL,   -- calibrated probability
    is_flagged_uncertain   BOOLEAN NOT NULL DEFAULT 0,
    FOREIGN KEY (window_id) REFERENCES WindowFeatures (window_id),
    FOREIGN KEY (predicted_label_id) REFERENCES Labels (label_id)
);

-- Per-feature SHAP attribution for a single prediction.
CREATE TABLE ShapAttributions (
    attribution_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id      INTEGER NOT NULL,
    feature_name       VARCHAR(128) NOT NULL,  -- matches a WindowFeatures column
    attribution_value  REAL NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES Predictions (prediction_id)
);

-- Patient-facing feedback, generated per trial (per repetition, not per
-- window), traceable back to the prediction that triggered it.
CREATE TABLE Feedback (
    feedback_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id              INTEGER NOT NULL,
    source_prediction_id  INTEGER NOT NULL,
    feedback_text         TEXT NOT NULL,
    FOREIGN KEY (trial_id) REFERENCES Trials (trial_id),
    FOREIGN KEY (source_prediction_id) REFERENCES Predictions (prediction_id)
);
