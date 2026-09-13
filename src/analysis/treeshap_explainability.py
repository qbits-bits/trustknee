"""TreeSHAP explainability analysis for the canonical XGBoost artifact.

Generates global and per-class feature importance plots using SHAP's
TreeExplainer. Uses the canonical subject-split test set (loaded from
the saved config) to ensure explanations reflect real deployment conditions.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.xgboost_pipeline import build_features, load_and_trim

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "models" / "xgboost_canonical" / "xgboost_artifact.joblib"
CONFIG_PATH = REPO_ROOT / "models" / "xgboost_canonical" / "xgboost_config.json"
DATA_PATH = REPO_ROOT / "data" / "processed" / "kneepad_features.csv"
OUTPUT_DIR = REPO_ROOT / "models" / "xgboost_canonical" / "shap"


def main() -> None:
    print("[1/6] Loading artifact and config...")
    artifact = joblib.load(ARTIFACT_PATH)
    model = artifact["model"]
    feature_order = list(artifact["feature_order"])
    clip = artifact["clip"]

    # Compatibility patches (same as adapter)
    if not hasattr(model, "use_label_encoder"):
        model.use_label_encoder = False
    if not hasattr(model, "gpu_id"):
        model.gpu_id = -1
    if not hasattr(model, "predictor"):
        model.predictor = "cpu_predictor"

    # Load test subjects from the saved config (proper way, not hardcoded)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    test_subjects = config["subject_ids"]["test"]

    print("[2/6] Loading canonical test split...")
    df_full = load_and_trim(str(DATA_PATH))
    df_test = df_full[df_full["subject_id"].isin(test_subjects)].copy()
    print(f"  -> Test subjects: {test_subjects}")
    print(f"  -> Test windows: {len(df_test)}")

    print("[3/6] Building features...")
    df_feat, _ = build_features(df_test)
    X = df_feat[feature_order].copy()
    for c in feature_order:
        lo, hi = clip[c]
        X[c] = X[c].clip(lo, hi)

    # Subsample for speed (TreeSHAP is O(n_trees * n_features))
    MAX_SAMPLES = 2000
    if len(X) > MAX_SAMPLES:
        print(f"  -> Subsampling to {MAX_SAMPLES} windows for SHAP speed...")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=MAX_SAMPLES, replace=False)
        X = X.iloc[idx].reset_index(drop=True)

    print("[4/6] Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[5/6] Generating plots...")

    # Plot 1: Global summary (beeswarm)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=20)
    plt.title("Global Feature Importance (All Test Windows)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Bar plot of mean |SHAP|
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
    plt.title("Mean |SHAP| per Feature (Top 20)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Per-class (Correct=0, Wrong=1) summary
    for cls_id, cls_name in [(0, "Correct"), (1, "Wrong")]:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values[..., cls_id] if shap_values.values.ndim == 3 else shap_values,
            X,
            show=False,
            max_display=15,
        )
        plt.title(f"Feature Importance for class={cls_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"summary_class_{cls_name.lower()}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # Save ranked feature importance CSV
    print("[6/6] Saving feature importance ranking...")
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    if mean_abs.ndim == 2:
        mean_abs = mean_abs[:, 1]  # Use class=1 (Wrong) for ranking
    ranking = (
        pd.DataFrame({"feature": feature_order, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    print(f"\nDone! Outputs saved to {OUTPUT_DIR}/")
    print("Top 10 features:")
    print(ranking.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
