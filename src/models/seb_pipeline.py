
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.contracts import get_canonical_contract  # noqa: E402
from src.features.masking import apply_sensor_mask_tabular  # noqa: E402
from src.models.trial_aggregation import aggregate_trial_predictions, compute_metrics  # noqa: E402
from src.models.xgboost_pipeline import build_features, load_and_trim, make_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trustknee.models")

AGGREGATION_METHODS = ["mean_prob", "majority_vote", "confidence_weighted"]

def evaluate_subset(tr_df, va_df, active_sensors, is_candidate_search=False):
    """Evaluates a sensor subset using masking.py and trial_aggregation.py."""

    canonical = get_canonical_contract()
    SEED = canonical.seed

    # Apply tabular channel masking from masking.py;
    train_masked = apply_sensor_mask_tabular(tr_df, active_sensors, sensor_column_prefix="s")
    val_masked = apply_sensor_mask_tabular(va_df, active_sensors, sensor_column_prefix="s")

    meta_cols = {
        "window_index",
        "start_time_s",
        "end_time_s",
        "subject_id",
        "label_id",
        "trial_num",
        "execution",
        "exercise",
        "label"
    }

    feature_cols = [col for col in train_masked.select_dtypes(include=[np.number]).columns if col not in meta_cols]

    # Train-only Quantile Clipping;
    lo = train_masked[feature_cols].quantile(0.01)
    hi = train_masked[feature_cols].quantile(0.99)

    X_train = train_masked[feature_cols].clip(lo, hi, axis=1)
    y_train = train_masked["label"].values

    X_val = val_masked[feature_cols].clip(lo, hi, axis=1)
    y_val = val_masked["label"].values

    # Train-only SMOTE Oversampling;
    sm = SMOTE(random_state=SEED)
    resample = sm.fit_resample(X_train, y_train)
    X_train_resampled, y_train_resampled = resample[0], resample[1]

    # Fit Canonical XGBoost (Shallow Config);
    # Fast parameters during candidate search; deeper parameters for full evaluation
    n_trees = 150 if is_candidate_search else 1500
    params = {
        "n_estimators": n_trees,
        "max_depth": 5,
        "learning_rate": 0.02,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 2,
        "reg_lambda": 1.5,
        "reg_alpha": 0.05,

        # Execution options;
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
        "eval_metric": "logloss"
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)

    # Measure Latency & Window-Level Predictions;
    start_t = time.time()
    val_probs = model.predict_proba(X_val)  # shape: (N_windows, 2)
    latency_per_sample = (time.time() - start_t) / len(X_val)

    val_probs_pos = val_probs[:, 1]

    # Tune Probability Threshold on Validation Window Predictions;
    best_t, best_win_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.61, 0.02):
        win_preds = (val_probs_pos >= t).astype(int)
        win_f1 = compute_metrics(np.asarray(y_val), win_preds)["macro_f1"]
        if win_f1 > best_win_f1:
            best_win_f1, best_t = win_f1, t

    opt_win_preds = (val_probs_pos >= best_t).astype(int)
    win_metrics = compute_metrics(np.asarray(y_val), np.asarray(opt_win_preds))

    # Trial numbers restart per subject and label directory, so label_id is part
    # of the trial key to prevent aggregating distinct physical trials together.
    va_trial_ids = (
        val_masked["subject_id"].astype(str)
        + "_"
        + val_masked["label_id"].astype(str)
        + "_"
        + val_masked["trial_num"].astype(str)
    ).values
    # Build ground truth map per unique trial;
    unique_trials = np.unique(np.asarray(va_trial_ids, dtype=str))
    y_true_trial = np.array([val_masked[va_trial_ids == tid]["label"].iloc[0] for tid in unique_trials])

    # Keep a valid metrics mapping even if no aggregation method produces an improvement, so the return block cannot subscript None;
    best_agg_method = AGGREGATION_METHODS[0]
    best_trial_metrics = compute_metrics(y_true_trial, y_true_trial)
    best_trial_f1 = -1.0

    for method in AGGREGATION_METHODS:
        pred_dict = aggregate_trial_predictions(val_probs, np.asarray(va_trial_ids, dtype=str), method=method)
        y_pred_trial = np.array([pred_dict[tid] for tid in unique_trials])

        t_metrics = compute_metrics(y_true_trial, y_pred_trial)
        if t_metrics["macro_f1"] > best_trial_f1:
            best_trial_f1 = t_metrics["macro_f1"]
            best_agg_method = method
            best_trial_metrics = t_metrics

    return {
        "feature_count": len(feature_cols),
        "latency_sec": latency_per_sample,
        "best_threshold": best_t,

        # Window Metrics;
        "win_acc": win_metrics["accuracy"],
        "win_macro_f1": win_metrics["macro_f1"],
        "win_balanced_acc": win_metrics["balanced_accuracy"],
        "win_per_class_recall": win_metrics["per_class_recall"],

        # Trial Metrics;
        "best_agg_method": best_agg_method,
        "trial_acc": best_trial_metrics["accuracy"],
        "trial_macro_f1": best_trial_metrics["macro_f1"],
        "trial_balanced_acc": best_trial_metrics["balanced_accuracy"],
        "trial_per_class_recall": best_trial_metrics["per_class_recall"]
    }

def _eval_candidate_worker(sensor_to_drop, current_sensors, train_df, val_df):
    """Worker function for parallel candidate evaluation."""
    candidate = [x for x in current_sensors if x != sensor_to_drop]
    cand_metrics = evaluate_subset(train_df, val_df, candidate, is_candidate_search=True)
    if cand_metrics is None or "win_macro_f1" not in cand_metrics:
        logger.warning("Candidate evaluation returned no metrics for Sensor %s", sensor_to_drop)
        return None
    return sensor_to_drop, cand_metrics["win_macro_f1"]


def run_sbe_experiment(data_path, min_sensors=2, n_jobs = -1):
    if not 1 <= min_sensors <= 8:
        raise ValueError("min_sensors must be between 1 and 8")

    logger.info("Loading Data & Generating Engineered Features...")

    import warnings  # noqa: E402

    from pandas.errors import PerformanceWarning  # noqa: E402
    warnings.filterwarnings("ignore", category=PerformanceWarning) # noqa: E402

    df = load_and_trim(data_path)
    df, _ = build_features(df)

    canonical = {
        "train": list(config.PHASE3_TRAIN_SUBJECTS),
        "val": list(config.PHASE3_VAL_SUBJECTS),
        "test": list(config.PHASE3_TEST_SUBJECTS),
    }
    train_df, val_df, test_df, _ = make_split(df, canonical=canonical)

    current_sensors = list(range(1, 9))
    full_baseline_f1 = None
    results = []

    logger.info(f"Starting Sequential Backward Elimination from {len(current_sensors)} down to {min_sensors} sensors...")

    while len(current_sensors) >= min_sensors:
        logger.info(f"Evaluating Sensor Set: {current_sensors}")
        m = evaluate_subset(train_df, val_df, current_sensors)

        if full_baseline_f1 is None:
            full_baseline_f1 = m["win_macro_f1"]

        retained_pct = (m["win_macro_f1"] / full_baseline_f1) * 100.0

        logger.info(
            f"Active: {len(current_sensors)} sensors; Win Macro-F1: {m['win_macro_f1']:.4f}; "
            f"Trial Macro-F1: {m['trial_macro_f1']:.4f} ({m['best_agg_method']}); "
            f"Retained: {retained_pct:.2f}%"
        )

        results.append({
            "num_sensors": len(current_sensors),
            "active_sensors": str(current_sensors),
            "feature_count": m["feature_count"],
            "latency_sec": m["latency_sec"],
            "best_threshold": m["best_threshold"],
            "win_accuracy": m["win_acc"],
            "win_macro_f1": m["win_macro_f1"],
            "win_balanced_accuracy": m["win_balanced_acc"],
            "win_per_class_recall": str(m["win_per_class_recall"]),
            "best_agg_method": m["best_agg_method"],
            "trial_accuracy": m["trial_acc"],
            "trial_macro_f1": m["trial_macro_f1"],
            "trial_balanced_accuracy": m["trial_balanced_acc"],
            "trial_per_class_recall": str(m["trial_per_class_recall"]),
            "retained_performance_pct": retained_pct
        })

        if len(current_sensors) == min_sensors:
            break

        # Parallel Candidate Evaluation;
        logger.debug(f"Evaluating candidate removals across parallel workers (n_jobs={n_jobs})")
        candidate_evals = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_eval_candidate_worker)(sensor, current_sensors, train_df, val_df)
            for sensor in current_sensors
        )

        # Ignore failed workers before indexing candidate results;
        valid_candidate_evals = [
            result for result in candidate_evals
            if result is not None
            and len(result) == 2
            and result[0] is not None
            and result[1] is not None
        ]
        if not valid_candidate_evals:
            logger.warning("No valid candidate evaluations remain; stopping sensor pruning.")
            break

        # Select candidate whose removal drops performance the least
        worst_sensor, best_cand_f1 = max(valid_candidate_evals, key=lambda x: x[1])
        logger.info(f"Pruning Sensor {worst_sensor} (Retained Candidate Val F1: {best_cand_f1:.4f})")
        current_sensors.remove(worst_sensor)


    # Export SBE Experiment CSV
    project_src = Path(__file__).resolve().parents[1]
    config_dir = project_src / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    sbe_results_csv = config_dir / "sbe_sensor_subset_results.csv"
    res_df = pd.DataFrame(results)
    res_df.to_csv(sbe_results_csv, index=False)
    logger.info("Saved SBE Experiment log to sbe_sensor_subset_results.csv")

    # Dynamically select the target row: try 4 (or 3), fallback to minimum evaluated;
    target_count = min_sensors
    matching_subset = res_df[res_df["num_sensors"] == target_count]

    if not matching_subset.empty:
        target_row = matching_subset.iloc[0]
    else:
        min_count = res_df["num_sensors"].min()
        logger.warning(
            f"{target_count}-sensor subset not found. Falling back to lowest evaluated subset ({min_count} sensors)."
        )
        target_row = res_df[res_df["num_sensors"] == min_count].iloc[0]

    # Safely parse active sensors list;
    raw_sensors = target_row["active_sensors"]
    selected_sensors = (
        eval(raw_sensors) if isinstance(raw_sensors, str) else list(raw_sensors)
    )
    selected_sensors = [int(s) for s in selected_sensors]

    # Generate 8-channel boolean mask (handles 0-based [0..7] or 1-based [1..8] indexing);
    is_one_based = any(s == 8 for s in selected_sensors) or not any(s == 0 for s in selected_sensors)
    sensor_range = range(1, 9) if is_one_based else range(0, 8)
    boolean_mask = [bool(i in selected_sensors) for i in sensor_range]

    # Performance Retention Check;
    retained_pct = float(target_row["retained_performance_pct"])
    # If retained_pct is logged as a percentage (e.g., 92.5), check >= 90.0;
    # If logged as a decimal fraction (e.g., 0.925), check >= 0.90;
    target_met = retained_pct >= 90.0 if retained_pct > 1.0 else retained_pct >= 0.90

    handoff = {
        "num_sensors_selected": int(target_row["num_sensors"]),
        "selected_sensor_ids": selected_sensors,
        "transformer_sensor_mask_boolean": boolean_mask,
        "retained_performance_pct": retained_pct,
        "target_90pct_met": bool(target_met),
        "validation_best_aggregation_method": str(target_row["best_agg_method"]),
        "val_window_macro_f1": float(target_row["win_macro_f1"]),
        "val_trial_macro_f1": float(target_row["trial_macro_f1"])
    }

    handoff_json = config_dir / "sensor_selection_handoff.json"
    with open(handoff_json, "w") as f:
        json.dump(handoff, f, indent=4)

    logger.info("Exported Transformer Handoff Mask to sensor_selection_handoff.json")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed"
    data_csv = data_dir / "kneepad_features.csv"

    parser = argparse.ArgumentParser(description="SBE Sensor Selection")
    parser.add_argument(
        "--min-sensors",
        type=int,
        default=2,
        choices=range(1, 9),
        help="Minimum number of sensors to eliminate down to (default: 2)."
    )
    parser.add_argument(
            "--n-jobs",
            type=int,
            default=-1,
            help="Number of CPU cores for parallel evaluation (-1 uses all cores)."
        )
    args = parser.parse_args()
    run_sbe_experiment(
        data_path=data_csv,
        min_sensors=args.min_sensors,
        n_jobs=args.n_jobs
    )
