"""Inference latency benchmark for the canonical XGBoost artifact.

Measures feature-engineering and prediction latency on the canonical test
split and compares end-to-end per-window cost against the 200 ms streaming
(one new window every 0.2 s).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.xgboost_pipeline import build_features, load_and_trim

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "models" / "xgboost_canonical" / "xgboost_artifact.joblib"
CONFIG_PATH = REPO_ROOT / "models" / "xgboost_canonical" / "xgboost_config.json"
DATA_PATH = REPO_ROOT / "data" / "processed" / "kneepad_features.csv"
OUTPUT_DIR = REPO_ROOT / "models" / "xgboost_canonical" / "latency"

WINDOW_STRIDE_MS = 200.0  # a new window arrives every 0.2 s in streaming


def main() -> None:
    print("[1/4] Loading artifact + config...")
    artifact = joblib.load(ARTIFACT_PATH)
    model = artifact["model"]
    feature_order = list(artifact["feature_order"])
    clip = artifact["clip"]
    with open(CONFIG_PATH, encoding="utf-8") as f:
        test_subjects = json.load(f)["subject_ids"]["test"]

    print("[2/4] Preparing canonical test features...")
    df = load_and_trim(str(DATA_PATH))
    df_test = df[df["subject_id"].isin(test_subjects)].copy()
    t0 = time.perf_counter()
    df_feat, _ = build_features(df_test)
    feat_seconds = time.perf_counter() - t0
    X = df_feat[feature_order].copy()
    for c in feature_order:
        lo, hi = clip[c]
        X[c] = X[c].clip(lo, hi)
    n = len(X)

    print("[3/4] Benchmarking prediction latency...")
    model.predict_proba(X[:100])  # warmup: thread pools / init outside timing

    reps = 10
    batch_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model.predict_proba(X)
        batch_ms.append((time.perf_counter() - t0) * 1000.0)
    per_window_batch = np.array(batch_ms) / n

    single_ms = []
    for i in range(min(n, 500)):
        t0 = time.perf_counter()
        model.predict_proba(X.iloc[i : i + 1])
        single_ms.append((time.perf_counter() - t0) * 1000.0)
    single_ms = np.array(single_ms)

    feat_per_window_ms = feat_seconds * 1000.0 / n
    rows = [
        ("feature_engineering_ms_per_window", feat_per_window_ms),
        ("batch_predict_ms_per_window_mean", float(per_window_batch.mean())),
        ("batch_predict_ms_per_window_median", float(np.percentile(per_window_batch, 50))),
        ("single_window_predict_ms_mean", float(single_ms.mean())),
        ("single_window_predict_ms_median", float(np.percentile(single_ms, 50))),
        ("single_window_predict_ms_p95", float(np.percentile(single_ms, 95))),
        ("throughput_windows_per_sec", 1000.0 / float(per_window_batch.mean())),
        ("streaming_budget_ms_per_window", WINDOW_STRIDE_MS),
    ]
    report = pd.DataFrame(rows, columns=["metric", "value"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_DIR / "latency_report.csv", index=False)

    print("[4/4] Latency report:")
    print(report.to_string(index=False))
    end_to_end = feat_per_window_ms + float(single_ms.mean())
    verdict = "MEETS" if end_to_end < WINDOW_STRIDE_MS else "EXCEEDS"
    print(
        f"\nEnd-to-end per window: {end_to_end:.2f} ms "
        f"vs {WINDOW_STRIDE_MS:.0f} ms budget -> {verdict} real-time requirement"
    )
    print(f"Saved -> {OUTPUT_DIR / 'latency_report.csv'}")


if __name__ == "__main__":
    main()
