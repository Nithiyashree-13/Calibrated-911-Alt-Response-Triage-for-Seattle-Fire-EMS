#!/usr/bin/env python3
"""
baseline_911_quick_train.py

Train quick baselines on the dataset produced by prepare_911_dataset.py
and print PR-AUC, Brier score, and Recall @ 0.90 precision on a time-based split.

Usage:
  python baseline_911_quick_train.py --data path/to/care_dataset.csv --outdir out_metrics

Optional args:
  --test_months 12   # how many most-recent months to hold out for testing (default 12)
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def pick_threshold_for_precision(y_true, y_prob, target_precision=0.90):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    idx = np.where(precision >= target_precision)[0]
    if len(idx) == 0:
        # return best precision you can get
        best_i = int(np.argmax(precision))
        thr = thresholds[best_i-1] if best_i > 0 and best_i-1 < len(thresholds) else 0.5
        return float(thr), float(precision[best_i]), float(recall[best_i])
    best_i = idx[np.argmax(recall[idx])]
    thr = thresholds[best_i-1] if best_i > 0 and best_i-1 < len(thresholds) else 0.5
    return float(thr), float(precision[best_i]), float(recall[best_i])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--outdir", required=False, type=Path, default=Path("./out_metrics"))
    ap.add_argument("--test_months", type=int, default=12)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["Datetime"])
    # Basic cleaning
    df = df.dropna(subset=["Datetime", "Latitude", "Longitude", "label"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Time split: last N months as test
    max_dt = df["Datetime"].max()
    cutoff = max_dt - pd.DateOffset(months=args.test_months)
    train = df[df["Datetime"] < cutoff].copy()
    test  = df[df["Datetime"] >= cutoff].copy()

    if len(test) < 1000:  # fallback to last 20% if test too small
        n = len(df)
        split = int(n * 0.8)
        train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    # Features: all numeric except target; drop identifiers/text
    drop_cols = {"label", "Incident Number", "Datetime", "Address", "units_raw"}
    num_cols = [c for c in train.columns if c not in drop_cols and np.issubdtype(train[c].dtype, np.number)]
    Xtr, ytr = train[num_cols].values, train["label"].values
    Xte, yte = test[num_cols].values, test["label"].values

    results = {}

    # Common imputer for NaNs -> 0
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)

    # 1) Logistic Regression (balanced)
    logit = Pipeline([
        ("impute", imputer),
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None))
    ])
    logit.fit(Xtr, ytr)
    prob_lr = logit.predict_proba(Xte)[:,1]
    prauc_lr = average_precision_score(yte, prob_lr)
    brier_lr = brier_score_loss(yte, prob_lr)
    thr_lr, p_lr, r_lr = pick_threshold_for_precision(yte, prob_lr, 0.90)
    results["logistic_regression"] = {
        "prauc": prauc_lr, "brier": brier_lr,
        "thr_for_p>=0.90": thr_lr, "precision_at_thr": p_lr, "recall_at_thr": r_lr
    }

    # 2) Random Forest (balanced_subsample)
    rf = Pipeline([
        ("impute", imputer),
        ("clf", RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        ))
    ])
    rf.fit(Xtr, ytr)
    prob_rf = rf.predict_proba(Xte)[:,1]
    prauc_rf = average_precision_score(yte, prob_rf)
    brier_rf = brier_score_loss(yte, prob_rf)
    thr_rf, p_rf, r_rf = pick_threshold_for_precision(yte, prob_rf, 0.90)
    results["random_forest"] = {
        "prauc": prauc_rf, "brier": brier_rf,
        "thr_for_p>=0.90": thr_rf, "precision_at_thr": p_rf, "recall_at_thr": r_rf
    }

    # Save metrics
    with open(args.outdir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save small predictions sample
    out_pred = test[["Incident Number", "Datetime"]].copy()
    out_pred["y_true"] = yte
    out_pred["p_lr"] = prob_lr
    out_pred["p_rf"] = prob_rf
    out_pred.to_csv(args.outdir / "pred_sample.csv", index=False)

    print(json.dumps(results, indent=2))
    print("\nSaved metrics to:", args.outdir / "metrics.json")
    print("Saved sample predictions to:", args.outdir / "pred_sample.csv")
    print("\nFeature columns used ({}): {}".format(len(num_cols), num_cols))

if __name__ == "__main__":
    main()
