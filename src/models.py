"""
Model initialization, training wrappers, and multi-label prediction utilities.
Supported classical models: XGBoost, Random Forest, Logistic Regression, Linear SVM, KNN.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

DEFAULT_PARAMS = {
    'xgboost': {
        'n_estimators': 150,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_jobs': -1,
        'random_state': 42,
        'eval_metric': 'aucpr'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'n_jobs': -1,
        'random_state': 42,
        'class_weight': 'balanced'
    },
    'logistic_regression': {
        'max_iter': 500,
        'class_weight': 'balanced',
        'random_state': 42
    },
    'svm': {
        'max_iter': 1000,
        'class_weight': 'balanced',
        'random_state': 42,
        'dual': 'auto'
    },
    'knn': {
        'n_neighbors': 7,
        'weights': 'distance',
        'n_jobs': -1
    }
}


def get_binary_model(model_name: str, pos_ratio: Optional[float] = None, **custom_params) -> Any:
    """
    Initializes a classifier configured for severe class imbalance.

    Parameters:
        model_name: One of 'xgboost', 'random_forest', 'logistic_regression', 'svm', 'knn'.
        pos_ratio: Negative-to-positive ratio for XGBoost scale_pos_weight.
        custom_params: Overrides for default hyperparameters.
    """
    name = model_name.lower()
    params = DEFAULT_PARAMS.get(name, {}).copy()
    params.update(custom_params)

    if name == 'xgboost':
        if pos_ratio is not None and 'scale_pos_weight' not in custom_params:
            params['scale_pos_weight'] = max(pos_ratio, 1.0)
        return xgb.XGBClassifier(**params)

    elif name == 'random_forest':
        return RandomForestClassifier(**params)

    elif name == 'logistic_regression':
        return LogisticRegression(**params)

    elif name == 'svm':
        base_svc = LinearSVC(**params)
        return CalibratedClassifierCV(base_svc, cv=3, n_jobs=1)

    elif name == 'knn':
        return KNeighborsClassifier(**params)

    else:
        raise ValueError(f"Unknown model: '{model_name}'. Choose from: {list(DEFAULT_PARAMS.keys())}")


def train_multi_label_models(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    label_cols: List[str],
    **custom_params
) -> Dict[str, Any]:
    """
    Trains independent binary classifiers for each PTM target label.

    Returns:
        Dictionary of {label: fitted_model or None if single class}
    """
    fitted_models = {}

    for label in label_cols:
        y_label = y_train[label].values
        unique_classes = np.unique(y_label)

        if len(unique_classes) < 2:
            fitted_models[label] = None
            continue

        pos_count = int(y_label.sum())
        neg_count = len(y_label) - pos_count
        spw = neg_count / max(pos_count, 1)

        clf = get_binary_model(model_name, pos_ratio=spw, **custom_params)
        clf.fit(X_train, y_label)
        fitted_models[label] = clf

    return fitted_models


def predict_multi_label_probabilities(
    fitted_models: Dict[str, Any],
    X: pd.DataFrame,
    label_cols: List[str]
) -> pd.DataFrame:
    """
    Generates predicted probabilities for each PTM target label.
    Robust against edge-case single-class partitions.

    Returns:
        DataFrame with columns matching label_cols.
    """
    probs_dict = {}
    for label in label_cols:
        clf = fitted_models.get(label)
        if clf is None:
            probs_dict[label] = np.zeros(len(X), dtype=float)
            continue

        if hasattr(clf, "classes_"):
            classes = list(clf.classes_)
            if 1 in classes:
                col_idx = classes.index(1)
                probs = clf.predict_proba(X)[:, col_idx]
            else:
                probs = np.zeros(len(X), dtype=float)
        else:
            probs = clf.predict_proba(X)[:, 1]

        probs_dict[label] = probs

    return pd.DataFrame(probs_dict, index=X.index)
