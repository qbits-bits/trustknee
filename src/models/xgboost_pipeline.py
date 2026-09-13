"""Reproducible Phase-2 XGBoost pipeline (Pakki's contribution).

Moves the exact Phase-2 experiment into a repository module:
  - identical feature engineering, SMOTE, clipping, selection, calibration
  - leakage-safe: clip / SMOTE / feature-selection / threshold fit on TRAIN only
  - causal subject calibration (running median, no future windows)
  - canonical Phase-3 subject split imported from src.config (Nihal's freeze)
  - saves one loadable artifact (model + ordered features + clip + threshold)
    plus a full JSON config for reproducibility

Run from the repo root:
    python -m src.models.xgboost_pipeline --data-root data/raw
    python -m src.models.xgboost_pipeline --legacy   # Phase-2 split
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src import config

EPS = 1e-9
SEED = 42
ID_COLS = [
    "window_index",
    "start_time_s",
    "end_time_s",
    "subject_id",
    "trial_num",
    "trial_id",
    "label_id",
    "exercise",
    "execution",
    "label",
]
TRIAL_GROUP_COLUMNS = ["subject_id", "label_id", "trial_num"]
ENG_PATTERNS = [
    "sym_ratio",
    "ctrl_ratio",
    "loadshare",
    "lrshare",
    "ctx_",
    "_norm",
    "_delta",
    "_absd",
    "_lag",
    "_ema",
    "_rmin",
    "_rmax",
    "_rrange",
    "_rm",
    "_rs",
    "inter_",
    "_symdelta",
]
MODEL_CONFIGS = {
    "deep_slow": dict(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=2.0,
        reg_alpha=0.1,
    ),
    "wide_reg": dict(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=3.0,
        reg_alpha=0.2,
    ),
    "shallow": dict(
        n_estimators=1500,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=1.5,
        reg_alpha=0.05,
    ),
}


# ---------------------------------------------------------------- load / trim
def trim_trial_edges(df):
    """Label and remove two boundary windows from each physical trial."""
    df = df.copy()
    df["label"] = df["label_id"].apply(lambda x: 0 if x in [0, 3, 6] else 1)
    df = df.sort_values([*TRIAL_GROUP_COLUMNS, "window_index"])
    pos = df.groupby(TRIAL_GROUP_COLUMNS).cumcount()
    tot = df.groupby(TRIAL_GROUP_COLUMNS)["window_index"].transform("count")
    return df[(pos >= 2) & (pos < tot - 2)].copy()


def load_and_trim(path):
    return trim_trial_edges(pd.read_csv(path))


def load_raw_features(data_root):
    """Build the existing window-feature table directly from prepared raw data."""
    from src.ingestion import build_manifest
    from src.models.model_data import build_model_inputs

    dataset = build_model_inputs(build_manifest(data_root), seed=SEED)
    return pd.concat(
        [
            dataset.metadata.reset_index(drop=True),
            dataset.xgboost_features.reset_index(drop=True),
        ],
        axis=1,
    )


# ------------------------------------------------------------- feature build
def _sensor_col(df, s, sig_type):
    cand = [c for c in df.columns if c.startswith(f"s{s}_") and sig_type in c]
    if not cand:
        return None
    pri = {
        "emg": ["emg_mav", "emg_rms", "emg_std", "emg_power"],
        "gyro": ["gyro_peak_angular_vel", "gyro_angular_vel_mean"],
        "acc": ["acc_z_rms", "acc_z_mean", "acc_y_rms", "acc_mag_mean", "acc_mag_rms"],
    }
    for p in pri.get(sig_type, []):
        m = [c for c in cand if p in c]
        if m:
            return m[0]
    return cand[0]


def build_features(df):
    """Exact Phase-2 feature engineering with review fixes. Returns (df, feat_cols)."""
    required = {*TRIAL_GROUP_COLUMNS, "window_index", "exercise"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Feature rows are missing required grouping columns: {missing}")
    df = df.sort_values([*TRIAL_GROUP_COLUMNS, "window_index"]).copy()
    anchors = {s: {t: _sensor_col(df, s, t) for t in ["emg", "gyro", "acc"]} for s in range(1, 9)}

    # Causal subject calibration: a window can use prior windows from the same
    # subject, but neither the current row nor any future row contributes to
    # its normalization denominator. Edge windows are trimmed by the caller.
    anchor_cols = [
        anchors[s][t] for s in range(1, 9) for t in ["emg", "gyro", "acc"] if anchors[s][t]
    ]
    subj_med_including_current = (
        df.groupby("subject_id")[anchor_cols].expanding().median().reset_index(level=0, drop=True)
    )
    subj_med = subj_med_including_current.groupby(df["subject_id"], sort=False).shift(1)
    for c in anchor_cols:
        denominator = subj_med[c].where(subj_med[c].abs() > EPS).fillna(1.0)
        df[c + "_norm"] = df[c] / denominator

    # base invariants
    for L, R in [(1, 5), (2, 6), (3, 7), (4, 8)]:
        for t in ["emg", "gyro", "acc"]:
            cL, cR = anchors[L][t], anchors[R][t]
            if cL and cR:
                df[f"sym_ratio_{t}_{L}{R}"] = df[cL] / (df[cR] + EPS)
    for s in range(1, 9):
        e, g = anchors[s]["emg"], anchors[s]["gyro"]
        if e and g:
            df[f"ctrl_ratio_{s}"] = df[e] / (df[g] + EPS)
    for t in ["emg", "gyro", "acc"]:
        cols = [anchors[s][t] for s in range(1, 9) if anchors[s][t]]
        tt = df[cols].sum(axis=1) + EPS
        for c in cols:
            df[f"loadshare_{t}_{c[1]}"] = df[c] / tt
        left = [anchors[s][t] for s in range(1, 5) if anchors[s][t]]
        df[f"lrshare_{t}"] = df[left].sum(axis=1) / tt
    ctx = pd.get_dummies(df["exercise"], prefix="ctx", dtype=int)
    df = pd.concat([df, ctx], axis=1)
    grp = df.groupby(TRIAL_GROUP_COLUMNS)

    # kinematics (delta2 now grouped: no cross-trial contamination)
    for s in range(1, 9):
        for t in ["emg", "gyro", "acc"]:
            c = f"{anchors[s][t]}_norm" if anchors[s][t] else None
            if c and c in df.columns:
                d1 = grp[c].diff().fillna(0)
                df[c + "_delta1"] = d1
                df[c + "_delta2"] = (
                    d1.groupby([df[column] for column in TRIAL_GROUP_COLUMNS]).diff().fillna(0)
                )
                df[c + "_absd1"] = d1.abs()

    # trajectory lags
    for s in range(1, 9):
        for t in ["gyro", "acc"]:
            c = f"{anchors[s][t]}_norm" if anchors[s][t] else None
            if c and c in df.columns:
                for k in (1, 2, 3, 4):
                    df[c + f"_lag{k}"] = grp[c].shift(k).fillna(0)

    # EMA trends
    def ema(c, span):
        return grp[c].transform(lambda x: x.ewm(span=span, adjust=False).mean())

    for s in range(1, 9):
        for t in ["emg", "gyro", "acc"]:
            c = f"{anchors[s][t]}_norm" if anchors[s][t] else None
            if c and c in df.columns:
                df[c + "_ema5"] = ema(c, 5)

    # rolling + interactions
    def rmean(c, w):
        return grp[c].transform(lambda x: x.rolling(w, min_periods=1).mean())

    def rstd(c, w):
        return grp[c].transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))

    def rmin(c, w):
        return grp[c].transform(lambda x: x.rolling(w, min_periods=1).min())

    def rmax(c, w):
        return grp[c].transform(lambda x: x.rolling(w, min_periods=1).max())

    for L, R in [(1, 5), (2, 6), (3, 7), (4, 8)]:
        c = f"sym_ratio_emg_{L}{R}"
        if c in df.columns:
            df[c + "_rm3"] = rmean(c, 3)
            df[c + "_rm5"] = rmean(c, 5)
            df[c + "_rm7"] = rmean(c, 7)
            df[c + "_rs5"] = rstd(c, 5)
            df[c + "_rrange5"] = rmax(c, 5) - rmin(c, 5)
            df[c + "_symdelta"] = grp[c].diff().fillna(0)
    for s in range(1, 9):
        c = f"{anchors[s]['emg']}_norm" if anchors[s]["emg"] else None
        if c and c in df.columns:
            df[c + "_rm5"] = rmean(c, 5)
            df[c + "_rs5"] = rstd(c, 5)
    for L, R in [(1, 5), (2, 6), (3, 7), (4, 8)]:
        sym, cl = f"sym_ratio_emg_{L}{R}", f"ctrl_ratio_{L}"
        if sym in df.columns and cl in df.columns:
            df[f"inter_sym_ctrl_{L}{R}"] = df[sym] * df[cl]

    feat_cols = [c for c in df.columns if c not in ID_COLS and any(p in c for p in ENG_PATTERNS)]
    return df, feat_cols


# ------------------------------------------------------------------- splitting
def make_split(df, canonical=None):
    """canonical = {'train':[...],'val':[...],'test':[...]} (Nihal's freeze).
    If None, reproduces the original Phase-2 seeded 70/15/15 subject split."""
    subs = df["subject_id"].unique()
    if canonical is None:
        np.random.seed(SEED)
        np.random.shuffle(subs)
        a, b = int(0.7 * len(subs)), int(0.85 * len(subs))
        tr_ids, va_ids, te_ids = subs[:a], subs[a:b], subs[b:]
    else:
        tr_ids = np.array(canonical["train"])
        va_ids = np.array(canonical["val"])
        te_ids = np.array(canonical["test"])
    tr = df[df["subject_id"].isin(tr_ids)].copy()
    va = df[df["subject_id"].isin(va_ids)].copy()
    te = df[df["subject_id"].isin(te_ids)].copy()
    ids = {
        "train": [int(x) for x in tr_ids],
        "val": [int(x) for x in va_ids],
        "test": [int(x) for x in te_ids],
    }
    return tr, va, te, ids


# -------------------------------------------------------------------- fitting
def _classification_metrics(labels, probabilities, threshold):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    p_c, r_c, f_c, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "per_class_precision": p_c.tolist(),
        "per_class_recall": r_c.tolist(),
        "per_class_f1": f_c.tolist(),
        "support": support.tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def _trial_probabilities(frame, probabilities):
    local = frame[[*TRIAL_GROUP_COLUMNS, "label"]].copy()
    local["probability_wrong"] = np.asarray(probabilities, dtype=float)
    grouped = local.groupby(TRIAL_GROUP_COLUMNS, sort=True, as_index=False).agg(
        label=("label", "first"),
        probability_wrong=("probability_wrong", "mean"),
    )
    return grouped["label"].to_numpy(dtype=int), grouped["probability_wrong"].to_numpy()


def fit_pipeline(tr, va, te, feat_cols):
    """All leakage-prone steps (clip / SMOTE / selection / calibration) on TRAIN only."""
    lo = tr[feat_cols].quantile(0.01)
    hi = tr[feat_cols].quantile(0.99)
    for d in (tr, va, te):
        d[feat_cols] = d[feat_cols].clip(lo, hi, axis=1)

    sm = SMOTE(random_state=SEED)
    X_tr, y_tr = sm.fit_resample(tr[feat_cols], tr["label"])

    # feature selection (train-only)
    rk = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
        verbosity=0,
    )
    rk.fit(X_tr, y_tr)
    imp = pd.Series(rk.feature_importances_, index=feat_cols)
    imp = imp.sort_values(ascending=False)
    best = None
    for K in [k for k in [220, 270, 320, len(feat_cols)] if k <= len(feat_cols)]:
        F = imp.head(K).index.tolist()
        q = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED,
            verbosity=0,
            eval_metric="logloss",
        )
        q.fit(X_tr[F], y_tr)
        va_pred = (q.predict_proba(va[F])[:, 1] >= 0.5).astype(int)
        f1 = f1_score(va["label"], va_pred, average="macro")
        if best is None or f1 > best[0]:
            best = (f1, K)
    F = imp.head(best[1]).index.tolist()

    # model config search (train-only fit, val for selection)
    best_cfg = None
    for name, cfg in MODEL_CONFIGS.items():
        m = xgb.XGBClassifier(
            tree_method="hist",
            n_jobs=-1,
            random_state=SEED,
            verbosity=0,
            eval_metric="logloss",
            **cfg,
        )
        m.fit(X_tr[F], y_tr)
        vp = m.predict_proba(va[F])[:, 1]
        f1 = f1_score(va["label"], (vp >= 0.5).astype(int), average="macro")
        if best_cfg is None or f1 > best_cfg[0]:
            best_cfg = (f1, name, cfg)

    model = xgb.XGBClassifier(
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
        verbosity=0,
        eval_metric="logloss",
        **best_cfg[2],
    )
    model.fit(X_tr[F], y_tr)

    # threshold calibration on VALIDATION only
    vp = model.predict_proba(va[F])[:, 1]
    best_t, bf1 = 0.5, 0.0
    for t in np.arange(0.30, 0.61, 0.01):
        f = f1_score(va["label"], (vp >= t).astype(int), average="macro")
        if f > bf1:
            bf1, best_t = f, t

    started = time.perf_counter()
    tp = model.predict_proba(te[F])[:, 1]
    latency_ms = (time.perf_counter() - started) * 1000.0 / len(te)
    trial_labels, trial_probabilities = _trial_probabilities(te, tp)
    window_metrics = _classification_metrics(te["label"].to_numpy(), tp, best_t)
    trial_metrics = _classification_metrics(trial_labels, trial_probabilities, best_t)
    window_metrics["n_samples"] = int(len(te))
    window_metrics["inference_latency_ms"] = float(latency_ms)
    trial_metrics["n_samples"] = int(len(trial_labels))
    trial_metrics["inference_latency_ms"] = float(latency_ms)

    return {
        "model": model,
        "feature_order": F,
        "selected_K": best[1],
        "clip": {c: (float(lo[c]), float(hi[c])) for c in F},
        "threshold": float(best_t),
        "config_name": best_cfg[1],
        "config": best_cfg[2],
        "metrics": {"window": window_metrics, "trial": trial_metrics},
        "subject_ids": None,  # filled by caller
    }


# ------------------------------------------------------------------- artifacts
def save_artifacts(bundle, out_dir, extra_config):
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    art_path = destination / "xgboost_artifact.joblib"
    joblib.dump(
        {k: bundle[k] for k in ["model", "feature_order", "clip", "threshold"]},
        art_path,
    )
    cfg = dict(extra_config)
    cfg.update(
        {
            "feature_order": bundle["feature_order"],
            "selected_K": bundle["selected_K"],
            "model_config_name": bundle["config_name"],
            "model_params": bundle["config"],
            "threshold": bundle["threshold"],
            "threshold_rule": "argmax macro-F1 over [0.30,0.60] on validation",
            "metrics": bundle["metrics"],
            "seed": SEED,
            "eps": EPS,
            "label_map": {"0,3,6": 0, "else": 1},
            "smote": {"method": "SMOTE", "random_state": SEED, "train_only": True},
        }
    )
    (destination / "xgboost_config.json").write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8",
    )
    metric_rows = []
    class_rows = []
    confusion_rows = []
    for level, metrics in bundle["metrics"].items():
        metric_rows.append(
            {
                "level": level,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "n_samples": metrics["n_samples"],
                "inference_latency_ms": metrics["inference_latency_ms"],
            }
        )
        for class_id in (0, 1):
            class_rows.append(
                {
                    "level": level,
                    "class_id": class_id,
                    "precision": metrics["per_class_precision"][class_id],
                    "recall": metrics["per_class_recall"][class_id],
                    "f1": metrics["per_class_f1"][class_id],
                    "support": metrics["support"][class_id],
                }
            )
            for predicted_class in (0, 1):
                confusion_rows.append(
                    {
                        "level": level,
                        "true_class": class_id,
                        "predicted_class": predicted_class,
                        "count": metrics["confusion_matrix"][class_id][predicted_class],
                    }
                )
    pd.DataFrame(metric_rows).to_csv(destination / "xgboost_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(destination / "xgboost_per_class_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(destination / "xgboost_confusion_matrices.csv", index=False)
    return art_path


def load_artifact(path):
    art = joblib.load(path)
    for k in ["model", "feature_order", "clip", "threshold"]:
        if k not in art:
            raise ValueError(f"artifact missing key: {k}")
    return art


def predict(artifact, df_features):
    """Inference: clip + ordered select + predict. NO SMOTE, NO leakage."""
    F = artifact["feature_order"]
    missing = [c for c in F if c not in df_features.columns]
    if missing:
        raise ValueError(f"missing features: {missing[:5]}...")
    X = df_features[F].copy()
    for c in F:
        lo, hi = artifact["clip"][c]
        X[c] = X[c].clip(lo, hi)
    proba = artifact["model"].predict_proba(X)[:, 1]
    return proba, (proba >= artifact["threshold"]).astype(int)


# ------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(
        description="Reproducible Phase-2/3 XGBoost pipeline",
    )
    ap.add_argument("--data", default="data/processed/kneepad_features.csv")
    ap.add_argument(
        "--data-root",
        type=Path,
        help="build features directly from the prepared raw dataset instead of --data",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--legacy", action="store_true", help="use original Phase-2 seeded 70/15/15 split"
    )
    args = ap.parse_args()
    out_dir = args.out or (
        "models/xgboost_phase2split" if args.legacy else "models/xgboost_canonical"
    )

    t0 = time.time()
    df = (
        trim_trial_edges(load_raw_features(args.data_root))
        if args.data_root is not None
        else load_and_trim(args.data)
    )
    df, feat_cols = build_features(df)

    if args.legacy:
        canonical, split_name = None, "phase2-legacy-70/15/15"
    else:
        canonical = {
            "train": list(config.PHASE3_TRAIN_SUBJECTS),
            "val": list(config.PHASE3_VAL_SUBJECTS),
            "test": list(config.PHASE3_TEST_SUBJECTS),
        }
        split_name = "phase3-canonical"

    tr, va, te, ids = make_split(df, canonical=canonical)
    bundle = fit_pipeline(tr, va, te, feat_cols)
    bundle["subject_ids"] = ids

    m = bundle["metrics"]["window"]
    print("=" * 70)
    print(f" SPLIT: {split_name}")
    print(f" ACCURACY : {m['accuracy'] * 100:.1f}%   MACRO-F1 : {m['macro_f1'] * 100:.1f}%")
    print(
        f" threshold: {bundle['threshold']:.2f}  K={bundle['selected_K']}  cfg={bundle['config_name']}"
    )
    print(f" subjects  train={ids['train']}")
    print(f"           val={ids['val']}")
    print(f"           test={ids['test']}")
    print("=" * 70)

    extra = {
        "subject_ids": ids,
        "n_features_all": len(feat_cols),
        "split": split_name,
    }
    if not args.legacy:
        extra["held_out_subjects"] = list(config.PHASE3_HELD_OUT_SUBJECTS)
    save_artifacts(bundle, out_dir, extra)
    print(f"saved -> {out_dir}/xgboost_artifact.joblib + xgboost_config.json")
    print(f"time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
