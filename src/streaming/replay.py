"""End-to-end recorded trial replay pipeline connecting streaming, inference, and persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.inference.base import InferenceEngine, TrialPrediction
from src.ingestion.ingest import Trial, load_trial
from src.persistence.db import DatabaseManager
from src.preprocessing.filters import preprocess_emg, preprocess_imu
from src.streaming.buffer import StreamingWindowBuffer
from src.streaming.source import RecordedTrialSource

logger = logging.getLogger("trustknee.streaming")


@dataclass(frozen=True)
class ReplayResult:
    """The outcome of replaying a trial through streaming, inference, and persistence."""

    trial_id: int
    subject_id: int
    label_id: int
    trial_num: int
    n_windows: int
    trial_prediction: TrialPrediction
    database_persisted: bool


def replay_trial_pipeline(
    trial: Trial | None = None,
    imu_path: str | Path | None = None,
    emg_path: str | Path | None = None,
    subject_id: int = 1,
    label_id: int = 0,
    trial_num: int = 1,
    inference_engine: InferenceEngine | None = None,
    database_manager: DatabaseManager | None = None,
    chunk_duration_ms: float = 50.0,
    pacing_realtime: bool = False,
    aggregation_method: str = "mean_probability",
    generate_feedback: bool = True,
) -> ReplayResult:
    """Replay a recorded trial through ingestion, filtering, streaming windowing, inference, and persistence.

    Guarantees zero cumulative drift and identical windowing to offline batch processing.
    """
    if trial is None:
        if imu_path is None or emg_path is None:
            raise ValueError("Must provide either a Trial instance or both imu_path and emg_path")
        trial = load_trial(
            imu_path=imu_path,
            emg_path=emg_path,
            subject_id=subject_id,
            label_id=label_id,
            trial_num=trial_num,
        )

    # 1. Biomedical Filtering
    filtered_imu = preprocess_imu(trial.imu)
    filtered_emg = preprocess_emg(trial.emg)

    filtered_trial = Trial(
        subject_id=trial.subject_id,
        label_id=trial.label_id,
        trial_num=trial.trial_num,
        imu=filtered_imu,
        emg=filtered_emg,
        duration_s=trial.duration_s,
    )

    # 2. Streaming source and buffer
    source = RecordedTrialSource(
        trial=filtered_trial,
        chunk_duration_ms=chunk_duration_ms,
        pacing_realtime=pacing_realtime,
    )
    buffer = StreamingWindowBuffer()

    emitted_windows = []
    for frame in source.stream_frames():
        new_windows = buffer.push(
            imu_chunk=frame.imu_data,
            emg_chunk=frame.emg_data,
            subject_id=frame.subject_id,
            label_id=frame.label_id,
            trial_num=frame.trial_num,
        )
        emitted_windows.extend(new_windows)

    if not emitted_windows:
        raise RuntimeError(
            "No windows were emitted during trial replay (trial duration may be < 200 ms)"
        )

    # 3. Model Inference
    if inference_engine is None:
        raise ValueError(
            "inference_engine is required for replay_trial_pipeline. "
            "Pass an InferenceEngine instance (e.g. TransformerInferenceEngine, XGBoostInferenceEngine, or MockInferenceEngine)."
        )

    window_preds = []
    for win in emitted_windows:
        pred = inference_engine.predict_window(
            imu_window=win.imu,
            emg_window=win.emg,
            window_index=win.window_index,
            start_time_s=win.start_time_s,
            end_time_s=win.end_time_s,
            subject_id=win.subject_id,
            trial_num=win.trial_num,
        )
        window_preds.append(pred)

    trial_pred = inference_engine.aggregate_trial(
        window_predictions=window_preds,
        method=aggregation_method,
    )

    # 4. Atomic Database Persistence
    db_persisted = False
    persisted_trial_id = -1
    if database_manager is not None:
        feedback_text = None
        if generate_feedback and window_preds:
            if trial_pred.is_flagged_uncertain:
                feedback_text = (
                    "The model is not confident enough to classify this movement. Check sensor "
                    "placement and repeat only if that agrees with the clinician-provided plan."
                )
            elif trial_pred.execution == "Correct":
                feedback_text = (
                    "The model classified this recorded movement as correct. Treat this as a "
                    "software result, not clinical confirmation; continue to follow clinician guidance."
                )
            else:
                feedback_text = (
                    "The model detected a pattern associated with an incorrect execution label. "
                    "Do not change the exercise independently; review it with a clinician."
                )

        persisted_trial_id = database_manager.persist_replay_trial(
            subject_id=trial.subject_id,
            label_id=trial.label_id,
            trial_num=trial.trial_num,
            imu_path=str(imu_path or "replayed_in_memory"),
            emg_path=str(emg_path or "replayed_in_memory"),
            duration_s=trial.duration_s,
            n_imu_samples=trial.imu.shape[-1],
            n_emg_samples=trial.emg.shape[-1],
            windows=emitted_windows,
            predictions=window_preds,
            feedback_text=feedback_text,
        )
        db_persisted = True

    return ReplayResult(
        trial_id=persisted_trial_id,
        subject_id=trial.subject_id,
        label_id=trial.label_id,
        trial_num=trial.trial_num,
        n_windows=len(emitted_windows),
        trial_prediction=trial_pred,
        database_persisted=db_persisted,
    )
