"""Tests for exercise-context manifest and context-column handling (reviewer R6)."""

from __future__ import annotations

import json
from pathlib import Path

from src.models.xgboost_pipeline import (
    build_feature_manifest,
    drop_context_columns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = REPO_ROOT / "models" / "xgboost_canonical" / "xgboost_config.json"


def test_manifest_identifies_context_columns():
    cols = ["s1_acc_x_rms", "ctx_Walking", "ctx_Squat", "sym_ratio_emg_15"]
    manifest = build_feature_manifest(cols)
    assert manifest["context_columns"] == ["ctx_Walking", "ctx_Squat"]
    assert manifest["forbidden_columns"] == []
    assert manifest["n_features"] == 4


def test_manifest_flags_identity_timestamp_and_label_columns():
    cols = ["subject_id", "label", "start_time_s", "window_index", "trial_num", "ctx_Walking"]
    manifest = build_feature_manifest(cols)
    assert set(manifest["forbidden_columns"]) == {
        "subject_id",
        "label",
        "start_time_s",
        "window_index",
        "trial_num",
    }
    assert manifest["context_columns"] == ["ctx_Walking"]


def test_drop_context_columns_removes_only_context():
    cols = ["ctx_Walking", "ctx_Squat", "s1_acc_x_rms", "sym_ratio_emg_15"]
    assert drop_context_columns(cols) == ["s1_acc_x_rms", "sym_ratio_emg_15"]


def test_canonical_feature_order_is_manifest_clean():
    config = json.loads(CANONICAL_CONFIG.read_text(encoding="utf-8"))
    manifest = build_feature_manifest(config["feature_order"])
    assert manifest["context_columns"], "canonical model must include exercise context"
    assert manifest["forbidden_columns"] == []