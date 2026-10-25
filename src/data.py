"""
Data loading, cleaning, and matrix preparation utilities.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

LABEL_COLS = ['S-glutathionylation', 'S-nitrosylation', 'S-palmitoylation']
METADATA_COLS = ['ID', 'Sequence']
CONSTANT_COLS = ['cys_present', 'cys_middle_dist', 'middle_aa']
CATEGORICAL_COLS = ['cys_ctx_-2', 'cys_ctx_-1', 'cys_ctx_1', 'cys_ctx_2']


def load_dataset(filepath: str) -> pd.DataFrame:
	"""Load CSV dataset with basic sanity check that all sequences are 31-mers."""
	df = pd.read_csv(filepath)
	if 'Sequence' in df.columns:
		# Verify 31-mer length and center cysteine
		assert (df['Sequence'].str.len() == 31).all(), "Sequences must be 31-mers"
	return df


def clean_tabular_features(
	df: pd.DataFrame,
	drop_constants: bool = True,
	encode_categoricals: bool = True
) -> pd.DataFrame:
	"""
	Cleans engineered tabular features:
	1. Drops zero-variance columns (cys_present, cys_middle_dist, middle_aa).
	2. One-hot encodes or coerces categorical context columns to numeric.
	3. Converts boolean masks to 0/1 integers.
	"""
	df_clean = df.copy()

	# 1. Drop constant zero-information columns
	if drop_constants:
		to_drop = [c for c in CONSTANT_COLS if c in df_clean.columns]
		if to_drop:
			df_clean = df_clean.drop(columns=to_drop)

	# 2. Encode categorical context features
	if encode_categoricals:
		cat_present = [c for c in CATEGORICAL_COLS if c in df_clean.columns]
		if cat_present:
			df_clean = pd.get_dummies(df_clean, columns=cat_present, dtype=int)

	# 3. Coerce any remaining object or boolean columns
	for col in df_clean.columns:
		if col not in METADATA_COLS + LABEL_COLS:
			if df_clean[col].dtype == object:
				df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
			elif df_clean[col].dtype == bool:
				df_clean[col] = df_clean[col].astype(int)

	return df_clean


def get_feature_and_label_matrices(
	df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
	"""
	Extracts feature matrix X and label matrix y from a cleaned dataset.
	Returns: (X, y, feature_names)
	"""
	target_labels = [c for c in LABEL_COLS if c in df.columns]
	feature_cols = [c for c in df.columns if c not in METADATA_COLS + LABEL_COLS]

	X = df[feature_cols].copy()
	y = df[target_labels].copy() if target_labels else pd.DataFrame()

	return X, y, feature_cols
