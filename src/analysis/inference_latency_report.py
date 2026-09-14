"""Inference latency benchmark for the canonical XGBoost artifact.

Exercises the repository's reachable streaming inference path: the
feature-bridge adapter's ``predict_trial_features`` call (raw sensor history
in -> feature engineering + clipping + prediction + WindowPrediction
construction out).  The adapter contract is subject-session granularity with
full window history, because single-exercise trial slices do not contain the
complete exercise one context columns the model requires.  Reports
per-window mean/median/p95 against the 200 ms streaming  and persists
the end-to-end metric and MEETS/EXCEEDS verdict inside the CSV artifact.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.xgboost_adapter import XGBoostArtifactAdapter
from src.models.xgboost_pipeline import build_features, load_and_trim

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "models" / "xgboost_canonical" / "xgboost_artifact.joblib"
CONFIG_PATH = REPO_ROOT / "models" / "xgboost_canonical" / "xgboost_config.json"
DATA_PATH = REPO_ROOT / "data" / "processed" / "kneepad_features.csv"
OUTPUT_DIR = REPO_ROOT / "models" / "xgboost_canonical" / "latency"

WINDOW_STRIDE_MS = 200.0  # a new window arrives every 0.2 s in streaming
FULL_SPLIT_REPS = 4


def main() -> None:
    print("[1/4] Loading feature-bridge adapter + config...")
    adapter = XGBoostArtifactAdapter(str(ARTIFACT_PATH))
    with open(CONFIG_PATH, encoding="utf-8") as f:
        test_subjects = json.load(f)["subject_ids"]["test"]

    print("[2/4] Preparing canonical test split...")
    df = load_and_trim(str(DATA_PATH))
    df_test = df[df["subject_id"].isin(test_subjects)].copy()
    n = len(df_test)
    print(f"  -> {len(test_subjects)} subjects, {n} windows")

    print("[3/4] Benchmarking reachable streaming path (predict_trial_features)...")
    adapter.predict_trial_features(df_test.copy())  # warmup (not counted)

    per_window_ms: list[float] = []
    session_ms: list[float] = []

    for _ in range(FULL_SPLIT_REPS):
        t0 = time.perf_counter()
        adapter.predict_trial_features(df_test.copy())
        dt_ms = (time.perf_counter() - t0) * 1000.0
        session_ms.append(dt_ms)
        per_window_ms.append(dt_ms / n)

    skipped = []
    for sid in test_subjects:
        sub_df = df_test[df_test["subject_id"] == sid]
        try:
            t0 = time.perf_counter()
            adapter.predict_trial_features(sub_df.copy())
            dt_ms = (time.perf_counter() - t0) * 1000.0
        except ValueError:
            skipped.append(sid)
            continue
        session_ms.append(dt_ms)
        per_window_ms.append(dt_ms / len(sub_df))
    if skipped:
        print(f"  -> skipped subjects (incomplete exercise context): {skipped}")

    per_window = np.asarray(per_window_ms)
    session = np.asarray(session_ms)

    print("[4/4] Model-only component + writing report...")
    df_feat, _ = build_features(df_test)
    F = adapter.feature_order
    X = df_feat[F].copy()
    for c in F:
        lo, hi = adapter.clip[c]
        X[c] = X[c].clip(lo, hi)
    adapter.model.predict_proba(X[:100])  # warmup
    batch_ms: list[float] = []
    for _ in range(10):
        t0 = time.perf_counter()
        adapter.model.predict_proba(X)
        batch_ms.append((time.perf_counter() - t0) * 1000.0)
    model_per_window = np.asarray(batch_ms) / len(X)

    end_to_end = float(per_window.mean())
    verdict = "MEETS" if end_to_end < WINDOW_STRIDE_MS else "EXCEEDS"
    rows = [
        ("streaming_end_to_end_ms_per_window_mean", end_to_end),
        ("streaming_end_to_end_ms_per_window_median", float(np.percentile(per_window, 50))),
        ("streaming_end_to_end_ms_per_window_p95", float(np.percentile(per_window, 95))),
        ("streaming_call_ms_mean", float(session.mean())),
        ("model_only_batch_ms_per_window_mean", float(model_per_window.mean())),
        ("throughput_windows_per_sec", 1000.0 / float(model_per_window.mean())),
        ("streaming_budget_ms_per_window", WINDOW_STRIDE_MS),
        ("end_to_end_ms_per_window", end_to_end),
        ("verdict", verdict),
    ]
    report = pd.DataFrame(rows, columns=["metric", "value"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_DIR / "latency_report.csv", index=False)

    print(report.to_string(index=False))
    print(
        f"\nEnd-to-end per window: {end_to_end:.2f} ms "
        f"vs {WINDOW_STRIDE_MS:.0f} ms budget -> {verdict} real-time requirement"
    )
    print(f"Saved -> {OUTPUT_DIR / 'latency_report.csv'}")


if __name__ == "__main__":
    main()
