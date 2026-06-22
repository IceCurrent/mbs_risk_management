"""
src/default_model.py

Logistic regression default model with Platt scaling calibration (spec §5.2).
Temporal train/valid/test split to prevent data leakage.
SHAP feature importance via LinearExplainer.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# shap is an optional dependency — only needed for compute_shap_values()
try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

# Features used in the logistic model — ordered list for reproducible column selection
FEATURE_COLS = [
    "credit_score",
    "ltv",
    "dti",
    "orig_interest_rate",
    "months_remaining_ratio",
    "loan_age_yrs",
    "high_dti",
    "high_ltv",
    "low_fico",
    "in_negative_equity",
    "mortgage_rate_orig",
    "unemp_rate_orig",
    "hpi_yoy_orig",
    "fedfunds_orig",
    "in_recession_orig",
    "high_dti_x_high_rate",
    "neg_equity_x_hpi_drop",
    "occ_S",
    "occ_I",
    "is_judicial",
]

TARGET_COL = "default"


def temporal_split(
    loan_df: pd.DataFrame,
    train_cutoff: str = "2024-01-01",
    valid_cutoff: str = "2024-07-01",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split loan-level data by last observed reporting period (spec §5.2).

    Train:      last_period < train_cutoff
    Validation: train_cutoff <= last_period < valid_cutoff
    Test:       last_period >= valid_cutoff

    Parameters
    ----------
    loan_df : pd.DataFrame  Loan-level features from build_logistic_features().
    train_cutoff, valid_cutoff : str  ISO date strings.

    Returns
    -------
    Tuple of (train_df, valid_df, test_df).
    """
    # last_period may arrive as a YYYYMM integer (build_model_aggregates.py),
    # a pandas/np datetime (reporting_period max), or a string. Normalize all
    # three to a datetime before comparing against the cutoffs. Parsing a raw
    # Timestamp with format="%Y%m" silently coerces every row to NaT, which
    # would empty all three splits — so branch on dtype.
    last_period_raw = loan_df["last_period"]
    if pd.api.types.is_datetime64_any_dtype(last_period_raw):
        last_period = pd.to_datetime(last_period_raw, errors="coerce")
    else:
        last_period = pd.to_datetime(
            last_period_raw.astype("Int64").astype(str), format="%Y%m", errors="coerce"
        )
    train_mask = last_period < train_cutoff
    valid_mask = (last_period >= train_cutoff) & (last_period < valid_cutoff)
    test_mask  = last_period >= valid_cutoff

    return (
        loan_df[train_mask].copy(),
        loan_df[valid_mask].copy(),
        loan_df[test_mask].copy(),
    )


def _prepare_xy(
    loan_df: pd.DataFrame,
    feature_cols: list,
    target_col: str = TARGET_COL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix and target vector, dropping rows with any NaN in features.

    Parameters
    ----------
    loan_df : pd.DataFrame
    feature_cols : list
    target_col : str

    Returns
    -------
    Tuple (X, y) as numpy arrays.
    """
    subset = loan_df[feature_cols + [target_col]].dropna()
    return subset[feature_cols].values, subset[target_col].values


def train_logistic(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    scale_features: bool = True,
) -> Tuple[CalibratedClassifierCV, StandardScaler, Dict[str, float]]:
    """
    Train a logistic regression default model with Platt scaling calibration.

    Fits on train_df, calibrates with Platt scaling on valid_df.
    Uses saga solver with balanced class weights for class imbalance (spec §5.2).

    Parameters
    ----------
    train_df : pd.DataFrame   Training set from temporal_split().
    valid_df : pd.DataFrame   Validation set (used for calibration).
    feature_cols : list       Feature column names (default FEATURE_COLS).
    scale_features : bool     Standardize features before fitting.

    Returns
    -------
    calibrated_model : CalibratedClassifierCV  Fitted and calibrated model.
    scaler : StandardScaler                    Fitted scaler (or identity).
    metrics : dict                             Validation ROC-AUC and PR-AUC.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    x_train, y_train = _prepare_xy(train_df, feature_cols)
    x_valid, y_valid = _prepare_xy(valid_df, feature_cols)

    scaler = StandardScaler()
    if scale_features:
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)

    base_model = LogisticRegression(
        class_weight="balanced",
        solver="saga",
        max_iter=1000,
        random_state=42,
    )
    base_model.fit(x_train, y_train)

    # Platt scaling calibration on a held-out validation set (spec §5.2).
    # sklearn >=1.6 replaced cv="prefit" with FrozenEstimator; cv="prefit" was
    # removed entirely in 1.9. Support both so the model trains across versions.
    try:
        from sklearn.frozen import FrozenEstimator  # pylint: disable=import-outside-toplevel
        calibrated_model = CalibratedClassifierCV(
            FrozenEstimator(base_model), method="sigmoid"
        )
    except ImportError:  # sklearn < 1.6
        calibrated_model = CalibratedClassifierCV(
            base_model, method="sigmoid", cv="prefit"
        )
    calibrated_model.fit(x_valid, y_valid)

    # Evaluate on validation set
    y_prob = calibrated_model.predict_proba(x_valid)[:, 1]
    metrics = {
        "val_roc_auc": roc_auc_score(y_valid, y_prob),
        "val_pr_auc":  average_precision_score(y_valid, y_prob),
        "val_default_rate": float(y_valid.mean()),
        "n_train": int(len(y_train)),
        "n_valid": int(len(y_valid)),
    }

    return calibrated_model, scaler, metrics


def evaluate_on_test(
    model: CalibratedClassifierCV,
    scaler: StandardScaler,
    test_df: pd.DataFrame,
    feature_cols: Optional[list] = None,
) -> Dict[str, float]:
    """
    Evaluate the calibrated model on the held-out test set.

    Parameters
    ----------
    model : CalibratedClassifierCV
    scaler : StandardScaler
    test_df : pd.DataFrame   Test set from temporal_split().
    feature_cols : list

    Returns
    -------
    dict  ROC-AUC, PR-AUC, and default rate on test set.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    x_test, y_test = _prepare_xy(test_df, feature_cols)
    x_test_scaled = scaler.transform(x_test)
    y_prob = model.predict_proba(x_test_scaled)[:, 1]

    return {
        "test_roc_auc":      roc_auc_score(y_test, y_prob),
        "test_pr_auc":       average_precision_score(y_test, y_prob),
        "test_default_rate": float(y_test.mean()),
        "n_test":            int(len(y_test)),
    }


def compute_shap_values(
    model: CalibratedClassifierCV,
    scaler: StandardScaler,
    x_background: np.ndarray,
    x_explain: np.ndarray,
    feature_cols: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values via LinearExplainer on the calibrated logistic model.

    Uses the inner LogisticRegression from the CalibratedClassifierCV wrapper.

    Parameters
    ----------
    model : CalibratedClassifierCV
    scaler : StandardScaler
    x_background : np.ndarray  Reference dataset (pre-scaled raw features).
    x_explain : np.ndarray     Samples to explain (pre-scaled raw features).
    feature_cols : list

    Returns
    -------
    Tuple (shap_values, expected_value) as numpy arrays.
    """
    if not _HAS_SHAP:
        raise ImportError("shap not installed. Run: pip install shap")

    if feature_cols is None:
        feature_cols = FEATURE_COLS

    x_bg_scaled  = scaler.transform(x_background)
    x_exp_scaled = scaler.transform(x_explain)

    # Extract the underlying LogisticRegression from the calibrated wrapper.
    # On sklearn >=1.6 the prefit estimator is wrapped in a FrozenEstimator, so
    # unwrap one level (.estimator) to reach the LogisticRegression with coef_.
    inner = model.calibrated_classifiers_[0].estimator
    inner_lr = getattr(inner, "estimator", inner)

    explainer = _shap.LinearExplainer(inner_lr, x_bg_scaled)
    shap_values = explainer.shap_values(x_exp_scaled)

    return shap_values, explainer.expected_value


def save_model(
    model: CalibratedClassifierCV,
    scaler: StandardScaler,
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Serialize the calibrated model and scaler to artifacts/models/.

    Parameters
    ----------
    model : CalibratedClassifierCV
    scaler : StandardScaler
    out_dir : Path  Default artifacts/models/ relative to project root.

    Returns
    -------
    Path  Path to the saved model bundle.
    """
    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "artifacts" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {"model": model, "scaler": scaler, "feature_cols": FEATURE_COLS}
    out_path = out_dir / "logistic_model.pkl"
    joblib.dump(bundle, out_path)
    return out_path


def load_model(model_path: Optional[Path] = None) -> Dict:
    """
    Load a serialized model bundle from disk.

    Parameters
    ----------
    model_path : Path  Default artifacts/models/logistic_model.pkl.

    Returns
    -------
    dict  {'model': CalibratedClassifierCV, 'scaler': StandardScaler,
           'feature_cols': list}
    """
    if model_path is None:
        model_path = (
            Path(__file__).parent.parent / "artifacts" / "models" / "logistic_model.pkl"
        )
    try:
        return joblib.load(model_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run the training notebook (03_default_logistic.ipynb) first."
        ) from exc
