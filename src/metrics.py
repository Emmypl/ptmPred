"""
Evaluation metrics, threshold optimization, and centralized experiment logger.
"""

from typing import Dict, Any, Union
import os
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
	average_precision_score,
	roc_auc_score,
	f1_score,
	precision_score,
	recall_score,
	matthews_corrcoef
)

SUMMARY_COLUMNS = [
	'timestamp',
	'run_id',
	'split_condition',
	'model',
	'encoding',
	'label',
	'pr_auc',
	'roc_auc',
	'optimal_threshold',
	'f1_optimal',
	'precision_optimal',
	'recall_optimal',
	'mcc_optimal',
	'leakage_overlap_pct'
]


def evaluate_predictions(
	y_true: Union[np.ndarray, pd.Series],
	y_probs: Union[np.ndarray, pd.Series],
	threshold_step: float = 0.02
) -> Dict[str, float]:
	"""
	Evaluates multi-metric performance on imbalanced binary predictions.
	Sweeps thresholds in [0.05, 0.95] to find the F1-maximizing cutoff.
	"""
	y_t = np.asarray(y_true).astype(int)
	y_p = np.asarray(y_probs).astype(float)

	# Calculate threshold-independent rank metrics
	pr_auc = float(average_precision_score(y_t, y_p))
	roc_auc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.5

	# Sweep thresholds to find optimal F1 cutoff
	best_thresh = 0.5
	best_f1 = -1.0

	for t in np.arange(0.05, 0.95, threshold_step):
		preds = (y_p >= t).astype(int)
		score = f1_score(y_t, preds, zero_division=0)
		if score > best_f1:
			best_f1 = score
			best_thresh = t

	final_preds = (y_p >= best_thresh).astype(int)

	return {
		"pr_auc": round(pr_auc, 4),
		"roc_auc": round(roc_auc, 4),
		"optimal_threshold": round(float(best_thresh), 3),
		"f1_optimal": round(float(f1_score(y_t, final_preds, zero_division=0)), 4),
		"precision_optimal": round(float(precision_score(y_t, final_preds, zero_division=0)), 4),
		"recall_optimal": round(float(recall_score(y_t, final_preds, zero_division=0)), 4),
		"mcc_optimal": round(float(matthews_corrcoef(y_t, final_preds)), 4)
	}


def log_experiment(
	summary_path: str,
	record: Dict[str, Any]
) -> None:
	"""
	Appends a single standardized experiment record to the centralized summary CSV.
	Creates file and parent directories with header if it does not yet exist.
	"""
	os.makedirs(os.path.dirname(summary_path), exist_ok=True)

	if 'timestamp' not in record:
		record['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

	row_df = pd.DataFrame([record])

	# Reorder columns matching standard schema, leaving missing as NaN
	for col in SUMMARY_COLUMNS:
		if col not in row_df.columns:
			row_df[col] = np.nan
	row_df = row_df[SUMMARY_COLUMNS]

	write_header = not os.path.exists(summary_path)
	row_df.to_csv(summary_path, mode='a', header=write_header, index=False)
