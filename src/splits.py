"""
Dataset partitioning and homology leakage audit utilities.
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from src.data import LABEL_COLS


def get_split(
	df: pd.DataFrame,
	strategy: str = "group",
	test_size: float = 0.20,
	seed: int = 42,
	group_col: str = "ID"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Partitions dataset under specified leakage control strategy.

	Parameters:
		df: Input DataFrame containing features, labels, and metadata.
		strategy:
			- 'random': Site-level split (~95% protein overlap).
			- 'group': Group split by protein ID (0% protein overlap).
			- 'cluster': Homology/family cluster split.
		test_size: Proportion of dataset to hold out for evaluation.
		seed: Random seed for reproducibility.
		group_col: Column name identifying protein groups.

	Returns:
		(train_df, val_df)
	"""
	labels = [c for c in LABEL_COLS if c in df.columns]

	if strategy == "random":
		# Stratify by total modification count if labels exist
		stratify_key = df[labels].sum(axis=1) if labels else None
		train_df, val_df = train_test_split(
			df,
			test_size=test_size,
			random_state=seed,
			stratify=stratify_key
		)
	elif strategy == "group":
		assert group_col in df.columns, f"Group column '{group_col}' not found"
		gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
		split_target = df[labels] if labels else df.index
		train_idx, val_idx = next(gss.split(df, split_target, groups=df[group_col]))
		train_df = df.iloc[train_idx]
		val_df = df.iloc[val_idx]
	elif strategy == "cluster":
		# Group by Cluster column if present, otherwise group by protein prefix
		group_key = df['Cluster'] if 'Cluster' in df.columns else df[group_col].str[:3]
		gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
		split_target = df[labels] if labels else df.index
		train_idx, val_idx = next(gss.split(df, split_target, groups=group_key))
		train_df = df.iloc[train_idx]
		val_df = df.iloc[val_idx]
	else:
		raise ValueError(f"Unknown split strategy: '{strategy}'. Choose 'random', 'group', or 'cluster'.")

	return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def audit_leakage(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	group_col: str = "ID"
) -> Dict[str, Any]:
	"""
	Computes exact protein-level overlap diagnostics between train and validation partitions.
	"""
	train_groups = set(train_df[group_col].dropna())
	val_groups = set(val_df[group_col].dropna())
	overlap = train_groups.intersection(val_groups)

	leak_pct = len(overlap) / max(len(val_groups), 1) * 100.0

	return {
		"train_samples": len(train_df),
		"val_samples": len(val_df),
		"train_unique_proteins": len(train_groups),
		"val_unique_proteins": len(val_groups),
		"overlapping_proteins": len(overlap),
		"leakage_overlap_pct": round(leak_pct, 2)
	}
