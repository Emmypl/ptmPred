#!/usr/bin/env python3
"""
Headless Baseline Runner for PTM Site Prediction.
Executes models across leakage conditions and logs results to output/experiment_summary.csv.

Usage:
    python run_baselines.py --encoding onehot --split group
    python run_baselines.py --all
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import clean_tabular_features, get_feature_and_label_matrices, LABEL_COLS
from src.splits import get_split, audit_leakage
from src.models import train_multi_label_models, predict_multi_label_probabilities
from src.metrics import evaluate_predictions, log_experiment

SUMMARY_PATH = PROJECT_ROOT / "output" / "experiment_summary.csv"


def run_experiment_suite(encoding: str = "onehot", split_strategy: str = "group", models: list = None):
    if models is None:
        models = ["xgboost", "random_forest", "logistic_regression", "svm", "knn"]

    data_file = PROJECT_ROOT / f"data_engineered/{encoding}/train_with_features_{encoding}.csv"
    if not data_file.exists():
        data_file = PROJECT_ROOT / "data_engineered/train_with_features.csv"

    print(f"\n{'='*75}\nSTARTING SUITE: {encoding.upper()} | SPLIT: {split_strategy.upper()}\n{'='*75}")
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)

    # 1. Split data
    train_df, val_df = get_split(df, strategy=split_strategy, test_size=0.20, seed=42, group_col="ID")
    diagnostics = audit_leakage(train_df, val_df, group_col="ID")
    print(f"Leakage Overlap: {diagnostics['leakage_overlap_pct']}% of validation proteins seen in train.")

    # 2. Clean features
    clean_tr = clean_tabular_features(train_df, drop_constants=True, encode_categoricals=False)
    clean_va = clean_tabular_features(val_df, drop_constants=True, encode_categoricals=False)

    X_train, y_train, feat_cols = get_feature_and_label_matrices(clean_tr)
    X_val, y_val, _ = get_feature_and_label_matrices(clean_va)

    common_cols = [c for c in feat_cols if c in X_val.columns]
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]

    # 3. Train & Evaluate each model
    for model_name in models:
        print(f"\n--- Training {model_name.upper()} ---")
        fitted = train_multi_label_models(model_name, X_train, y_train, LABEL_COLS)
        probs_df = predict_multi_label_probabilities(fitted, X_val, LABEL_COLS)

        for label in LABEL_COLS:
            metrics = evaluate_predictions(y_val[label].values, probs_df[label].values)
            record = {
                "run_id": f"{model_name}_{encoding}_{split_strategy}",
                "split_condition": split_strategy,
                "model": model_name,
                "encoding": encoding,
                "label": label,
                "leakage_overlap_pct": diagnostics['leakage_overlap_pct'],
                **metrics
            }
            log_experiment(str(SUMMARY_PATH), record)
            print(f"  {label:<22}: PR-AUC={metrics['pr_auc']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | F1={metrics['f1_optimal']:.4f}")

    print(f"\n✓ Completed suite. Results logged to {SUMMARY_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PTM baseline model experiments.")
    parser.add_argument("--encoding", type=str, default="onehot", choices=["onehot", "blosum", "aapc", "hybrid"])
    parser.add_argument("--split", type=str, default="group", choices=["random", "group", "cluster"])
    parser.add_argument("--models", nargs="+", default=["xgboost", "logistic_regression", "random_forest"])
    parser.add_argument("--all", action="store_true", help="Run full 2x3 baseline matrix")

    args = parser.parse_args()

    if args.all:
        for split in ["random", "group"]:
            for enc in ["onehot", "blosum"]:
                run_experiment_suite(encoding=enc, split_strategy=split, models=args.models)
    else:
        run_experiment_suite(encoding=args.encoding, split_strategy=args.split, models=args.models)
