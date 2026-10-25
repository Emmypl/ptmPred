#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
from src.metrics import evaluate_predictions, log_experiment
from src.models import train_multi_label_models, predict_multi_label_probabilities
from src.splits import get_split, audit_leakage
from src.data import LABEL_COLS

CACHE_DIR = Path("output/99_legacy_runs/transformer_gru/run01_frozen_esm2/transformer_cache")
SUMMARY_PATH = Path("output/experiment_summary.csv")

print("Loading cached ESM-2 embeddings...")
E_tr = np.load(CACHE_DIR / "train_embeddings_esm2_t12_35M_UR50D.npy")
E_va = np.load(CACHE_DIR / "val_embeddings_esm2_t12_35M_UR50D.npy")
E_full = np.concatenate([E_tr, E_va], axis=0)

X_esm = pd.DataFrame(E_full.mean(axis=1), columns=[f"esm_{i}" for i in range(480)])

d_tr = pd.read_csv("data_engineered_legacy/train_with_features_split.csv", usecols=['ID', 'Sequence'] + LABEL_COLS)
d_va = pd.read_csv("data_engineered_legacy/val_with_features_split.csv", usecols=['ID', 'Sequence'] + LABEL_COLS)
df_meta = pd.concat([d_tr, d_va], ignore_index=True)

df_all = pd.concat([df_meta[['ID', 'Sequence'] + LABEL_COLS], X_esm], axis=1)

split_strategy = "random"
train_df, val_df = get_split(df_all, strategy=split_strategy, test_size=0.20, seed=42, group_col="ID")
diagnostics = audit_leakage(train_df, val_df, group_col="ID")

feat_cols = [f"esm_{i}" for i in range(480)]
X_train, y_train = train_df[feat_cols], train_df[LABEL_COLS]
X_val, y_val = val_df[feat_cols], val_df[LABEL_COLS]

model_name = "xgboost"
print(f"Training {model_name} on ESM-2 ({split_strategy})...")
fitted = train_multi_label_models(model_name, X_train, y_train, LABEL_COLS)
probs_df = predict_multi_label_probabilities(fitted, X_val, LABEL_COLS)

for label in LABEL_COLS:
    metrics = evaluate_predictions(y_val[label].values, probs_df[label].values)
    record = {
        "run_id": f"{model_name}_esm2_mean_{split_strategy}",
        "split_condition": split_strategy,
        "model": model_name,
        "encoding": "esm2_mean",
        "label": label,
        "leakage_overlap_pct": diagnostics['leakage_overlap_pct'],
        **metrics
    }
    log_experiment(str(SUMMARY_PATH), record)
    print(f"    {label:<22}: PR-AUC={metrics['pr_auc']:.4f} | ROC-AUC={metrics['roc_auc']:.4f} | F1={metrics['f1_optimal']:.4f}")

print("✓ Done!")
