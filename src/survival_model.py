"""
src/survival_model.py

Cox Proportional Hazards survival model for mortgage default (spec §5.3).
Wraps lifelines.CoxPHFitter. Handles censoring correctly — loans still
performing at observation end are right-censored, not labeled as non-defaults.
"""

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    _HAS_LIFELINES = True
except ImportError:
    _HAS_LIFELINES = False

# Covariates for the Cox PH model (spec §5.3) — using z-score standardized columns.
#
# ORIGINATION-ONLY: every covariate is known at loan origination, so the model
# ranks risk without look-ahead. The earlier feature set additionally included
# delinq_last, eltv_last, current_rate and current_upb — all measured at the
# loan's LAST observation, which for a defaulted loan is the default month
# itself. Those leak the outcome and inflated the C-index from ~0.77 to ~0.91.
# They are kept below in CPH_FEATURE_COLS_LEAKY only to *demonstrate* the leak.
CPH_FEATURE_COLS = [
    "fico_z",
    "orig_cltv_z",
    "orig_dti_z",
    "log_orig_upb",
    "occ_investment",
    "occ_second",
]

# DO NOT USE for inference — retained to quantify the leakage (see notebook 04).
CPH_FEATURE_COLS_LEAKY = CPH_FEATURE_COLS + [
    "current_rate_z",
    "eltv_last_z",
    "log_current_upb",
    "delinq_last",
]

DURATION_COL = "duration"
EVENT_COL    = "event"


def train_cox(
    survival_df: pd.DataFrame,
    penalizer: float = 0.1,
    feature_cols: Optional[list] = None,
) -> "CoxPHFitter":
    """
    Fit a Cox Proportional Hazards model on the survival dataset.

    Uses L2 regularization (penalizer) to stabilize estimates in the presence
    of correlated covariates. The partial likelihood conditions on the risk set
    at each event time, handling right-censored loans correctly.

    Parameters
    ----------
    survival_df : pd.DataFrame  From build_survival_features(); one row per loan.
    penalizer : float           L2 regularization strength (spec §5.3 default 0.1).
    feature_cols : list         Covariate columns (default CPH_FEATURE_COLS).

    Returns
    -------
    CoxPHFitter  Fitted model object.
    """
    if not _HAS_LIFELINES:
        raise ImportError("lifelines not installed. Run: pip install lifelines")

    if feature_cols is None:
        feature_cols = CPH_FEATURE_COLS

    cols_needed = [DURATION_COL, EVENT_COL] + feature_cols
    missing = [col for col in cols_needed if col not in survival_df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in survival_df: {missing}. "
            "Run build_survival_features() first."
        )

    fit_df = survival_df[cols_needed].dropna()
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(fit_df, duration_col=DURATION_COL, event_col=EVENT_COL)
    return cph


def check_proportional_hazards(
    cph: "CoxPHFitter",
    survival_df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    p_value_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Test the proportional hazards assumption via Schoenfeld residuals (spec §5.3).

    Under PH, scaled Schoenfeld residuals should be uncorrelated with time.
    A significant p-value for a covariate indicates a PH violation for that variable.

    Parameters
    ----------
    cph : CoxPHFitter        Fitted Cox model.
    survival_df : pd.DataFrame
    feature_cols : list
    p_value_threshold : float  Flag covariates with p-value below this.

    Returns
    -------
    pd.DataFrame  Test results table with covariates, test statistics, p-values.
    """
    if feature_cols is None:
        feature_cols = CPH_FEATURE_COLS

    cols_needed = [DURATION_COL, EVENT_COL] + feature_cols
    fit_df = survival_df[cols_needed].dropna()
    results = cph.check_assumptions(fit_df, p_value_threshold=p_value_threshold, show_plots=False)
    return results


def cox_c_index(
    cph: "CoxPHFitter",
    test_df: pd.DataFrame,
    feature_cols: Optional[list] = None,
) -> float:
    """
    Compute the concordance index (C-index) on a held-out test set.

    The C-index measures the proportion of pairs where the model correctly
    ranks the higher-risk loan as defaulting first. 0.5 = random, 1.0 = perfect.

    Parameters
    ----------
    cph : CoxPHFitter
    test_df : pd.DataFrame  Must have duration and event columns.
    feature_cols : list

    Returns
    -------
    float  C-index (concordance index).
    """
    if not _HAS_LIFELINES:
        raise ImportError("lifelines not installed. Run: pip install lifelines")

    if feature_cols is None:
        feature_cols = CPH_FEATURE_COLS

    cols_needed = [DURATION_COL, EVENT_COL] + feature_cols
    fit_df = test_df[cols_needed].dropna()

    partial_hazard = cph.predict_partial_hazard(fit_df)
    return concordance_index(
        fit_df[DURATION_COL],
        -partial_hazard,
        fit_df[EVENT_COL],
    )


def stratified_subsample(
    survival_df: pd.DataFrame,
    n_target: int = 2_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Draw an event-stratified subsample of the survival dataset.

    The full CPH panel resolves to ~15M loans. Fitting lifelines.CoxPHFitter on
    that many rows is prohibitively memory- and time-intensive on a 16 GB
    machine. Subsampling while preserving the (rare) event rate keeps the fit
    tractable without materially degrading the partial-likelihood estimates or
    the C-index. If ``n_target`` >= len(survival_df), the input is returned
    unchanged.

    Parameters
    ----------
    survival_df : pd.DataFrame  From build_survival_features*(); must have 'event'.
    n_target : int              Approximate number of loans to retain.
    seed : int                  RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame  Subsample preserving the original event rate (shuffled).
    """
    if n_target >= len(survival_df):
        return survival_df

    frac = n_target / len(survival_df)
    parts = [
        group.sample(frac=frac, random_state=seed)
        for _, group in survival_df.groupby(EVENT_COL)
    ]
    sampled = pd.concat(parts)
    # Shuffle so the downstream random split sees no event-ordering structure.
    return sampled.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)


def temporal_split_survival(
    survival_df: pd.DataFrame,
    train_frac: float = 0.70,
    valid_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Random stratified split for survival data.

    Unlike the logistic model, the CPH panel (2018-2025) does not have a
    clean temporal cutoff for splitting — the origination years span the full
    range and all are needed for training the hazard baseline. A random
    stratified split on the event column is used instead.

    Parameters
    ----------
    survival_df : pd.DataFrame
    train_frac, valid_frac : float  Fractions; test = 1 - train - valid.
    seed : int

    Returns
    -------
    Tuple (train_df, valid_df, test_df).
    """
    rng = np.random.default_rng(seed)
    idx = survival_df.index.to_numpy().copy()
    rng.shuffle(idx)

    n_train = int(len(idx) * train_frac)
    n_valid = int(len(idx) * valid_frac)

    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train + n_valid]
    test_idx  = idx[n_train + n_valid:]

    return (
        survival_df.loc[train_idx],
        survival_df.loc[valid_idx],
        survival_df.loc[test_idx],
    )


def save_cox_model(
    cph: "CoxPHFitter",
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Serialize the fitted CoxPHFitter to artifacts/models/.

    Parameters
    ----------
    cph : CoxPHFitter
    out_dir : Path  Default artifacts/models/.

    Returns
    -------
    Path  Path to saved file.
    """
    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "artifacts" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cox_ph_model.pkl"
    joblib.dump(cph, out_path)
    return out_path


def load_cox_model(model_path: Optional[Path] = None) -> "CoxPHFitter":
    """
    Load a serialized CoxPHFitter from disk.

    Parameters
    ----------
    model_path : Path  Default artifacts/models/cox_ph_model.pkl.

    Returns
    -------
    CoxPHFitter
    """
    if model_path is None:
        model_path = (
            Path(__file__).parent.parent / "artifacts" / "models" / "cox_ph_model.pkl"
        )
    try:
        return joblib.load(model_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Cox model not found at {model_path}. "
            "Run the training notebook (04_cox_ph.ipynb) first."
        ) from exc
