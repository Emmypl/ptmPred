import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import average_precision_score

print("Loading augmented data...")
df = pd.read_csv("data_engineered/train_augmented_v2_with_features.csv")

# Identify feature columns
labels = ['S-glutathionylation', 'S-nitrosylation', 'S-palmitoylation']
metadata_cols = ['ID', 'Sequence'] + labels

# One-hot encode the categorical context columns
cat_cols = ['cys_ctx_-2', 'cys_ctx_-1', 'cys_ctx_1', 'cys_ctx_2', 'middle_aa']
# only encode if they exist
cat_cols = [c for c in cat_cols if c in df.columns]
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols)

X_cols = [c for c in df.columns if c not in metadata_cols]

X = df[X_cols].copy()
for col in X.columns:
    if X[col].dtype == object:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

print(f"Data loaded: {len(df)} rows. Features: {len(X_cols)}")

print("\n--- Condition 1: Random Split ---")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, df[labels], test_size=0.2, random_state=42
)

print("\n--- Condition 2: Group Split (by ID) ---")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, df[labels], groups=df['ID']))
X_train_g, X_test_g = X.iloc[train_idx], X.iloc[test_idx]
y_train_g, y_test_g = df[labels].iloc[train_idx], df[labels].iloc[test_idx]

# Check leakage
train_ids_r = set(df.iloc[y_train_r.index]['ID'])
test_ids_r = set(df.iloc[y_test_r.index]['ID'])
leak_r = len(train_ids_r.intersection(test_ids_r)) / len(test_ids_r)
print(f"Random split: {leak_r*100:.1f}% of test proteins are in training set.")

train_ids_g = set(df.iloc[train_idx]['ID'])
test_ids_g = set(df.iloc[test_idx]['ID'])
leak_g = len(train_ids_g.intersection(test_ids_g)) / len(test_ids_g)
print(f"Group split: {leak_g*100:.1f}% of test proteins are in training set.")

def evaluate_split(X_tr, X_te, y_tr, y_te, split_name):
    print(f"\nEvaluating {split_name}...")
    results = {}
    for label in labels:
        pos_count = y_tr[label].sum()
        neg_count = len(y_tr) - pos_count
        if pos_count == 0:
            results[label] = 0.0
            continue
        scale_pos_weight = neg_count / pos_count
        
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42,
            eval_metric='aucpr'
        )
        
        clf.fit(X_tr, y_tr[label])
        preds = clf.predict_proba(X_te)[:, 1]
        pr_auc = average_precision_score(y_te[label], preds)
        results[label] = pr_auc
        print(f"  {label}: PR-AUC = {pr_auc:.4f}")
    return results

res_random = evaluate_split(X_train_r, X_test_r, y_train_r, y_test_r, "Random Split")
res_group = evaluate_split(X_train_g, X_test_g, y_train_g, y_test_g, "Group Split")

print("\n--- Summary: PR-AUC Comparison ---")
for label in labels:
    diff = res_random[label] - res_group[label]
    print(f"{label}:")
    print(f"  Random: {res_random[label]:.4f}")
    print(f"  Group : {res_group[label]:.4f}")
    print(f"  Drop  : {diff:.4f}")
