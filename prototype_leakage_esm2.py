import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression

print("Loading data labels...")
df_tr = pd.read_csv("data_engineered/train_with_features_split.csv")
df_val = pd.read_csv("data_engineered/val_with_features_split.csv")
df = pd.concat([df_tr, df_val], ignore_index=True)
labels = ['S-glutathionylation', 'S-nitrosylation', 'S-palmitoylation']

print("Loading embeddings and applying mean pooling...")
emb_tr = np.load("output/transformer_gru/run01_frozen_esm2/transformer_cache/train_embeddings_esm2_t12_35M_UR50D.npy")
emb_val = np.load("output/transformer_gru/run01_frozen_esm2/transformer_cache/val_embeddings_esm2_t12_35M_UR50D.npy")
emb_full = np.concatenate([emb_tr, emb_val], axis=0)

# Mean pooling over the sequence length (axis 1)
# Shape goes from (89010, 31, 480) -> (89010, 480)
X = emb_full.mean(axis=1)

print(f"Data shape: {X.shape}")

print("\n--- Condition 1: Random Split ---")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, df[labels], test_size=0.2, random_state=42
)

print("\n--- Condition 2: Group Split (by ID) ---")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, df[labels], groups=df['ID']))
X_train_g, X_test_g = X[train_idx], X[test_idx]
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
        class_weight = {0: 1.0, 1: neg_count / pos_count}
        
        clf = LogisticRegression(max_iter=1000, class_weight=class_weight, n_jobs=-1, random_state=42)
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
