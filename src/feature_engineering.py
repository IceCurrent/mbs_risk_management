"""
src/feature_engineering.py

Build modeling features for the default models (spec §5.2).
Joins FRED macro series at the origination quarter.
Outputs a loan-level DataFrame ready for logistic regression
and a panel DataFrame for Cox PH survival analysis.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# fredapi is an optional dependency — only needed for fetch_macro_quarterly()
try:
    from fredapi import Fred as _Fred
    _HAS_FREDAPI = True
except ImportError:
    _HAS_FREDAPI = False

# ── Judicial foreclosure states (spec §5.2) ───────────────────────────────────
# Require court approval → longer timelines → affects observed default rates.
JUDICIAL_STATES = {
    "CT", "DE", "FL", "HI", "IL", "IN", "KS", "KY", "LA", "ME",
    "MD", "MA", "NE", "NJ", "NM", "NY", "ND", "OH", "OK", "PA",
    "SC", "SD", "VT", "WI",
}

# FRED series to fetch for macro features
FRED_SERIES = {
    "UNRATE":       "unemp_rate",
    "MORTGAGE30US": "mortgage_rate",
    "FEDFUNDS":     "fedfunds",
    "SPCS20RSA":    "hpi",
    "USREC":        "in_recession",
}


# ── Macro data helpers ────────────────────────────────────────────────────────

def fetch_macro_quarterly(
    start: str = "2000-01-01",
    end: str = "2025-12-31",
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fetch FRED macro series and resample to quarter-end.

    Requires FRED_API_KEY in environment (loaded from .env via dotenv).
    Caches result to CSV to avoid repeated API calls.

    Parameters
    ----------
    start, end : str  Date range for FRED queries.
    cache_path : Path  If provided and exists, load from cache instead of FRED.

    Returns
    -------
    pd.DataFrame  Index = PeriodIndex (quarterly), columns = macro features.
    """
    if cache_path is not None and Path(cache_path).exists():
        macro = pd.read_csv(cache_path, index_col=0)
        macro.index = pd.PeriodIndex(macro.index, freq="Q")
        return macro

    if not _HAS_FREDAPI:
        raise ImportError("fredapi not installed. Run: pip install fredapi")

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Add it to .env and call load_dotenv() first."
        )

    fred = _Fred(api_key=api_key)
    raw = {}
    for series_id in FRED_SERIES:
        try:
            raw[series_id] = fred.get_series(series_id, start=start, end=end)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch FRED series '{series_id}': {exc}"
            ) from exc

    # Build DataFrame from dict of Series to avoid pylint subscript-assignment warnings
    aggregated = {
        sid: (sdata.resample("QE").max() if sid == "USREC"
              else sdata.resample("QE").last())
        for sid, sdata in raw.items()
    }
    macro = pd.DataFrame(aggregated).ffill()
    macro["HPI_YoY"] = macro["SPCS20RSA"].pct_change(4)
    macro.index = macro.index.to_period("Q")

    # Rename to friendlier column names used in the model
    macro = macro.rename(columns={
        "UNRATE":       "unemp_rate_orig",
        "MORTGAGE30US": "mortgage_rate_orig",
        "FEDFUNDS":     "fedfunds_orig",
        "SPCS20RSA":    "hpi_level_orig",
        "USREC":        "in_recession_orig",
    })
    macro = macro.rename(columns={"HPI_YoY": "hpi_yoy_orig"})

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        macro.to_csv(cache_path)

    return macro


# ── Default target ────────────────────────────────────────────────────────────

def add_default_flag(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add strict default binary target to a panel (spec §4.3).

    Default = 90+ DPD OR terminal zero-balance code (short sale,
    repurchase, REO/foreclosure). Excludes simple prepayment ('01').

    Parameters
    ----------
    panel : pd.DataFrame  Must have delinquency_status and zero_balance_code.

    Returns
    -------
    pd.DataFrame  Input with 'default' column added (0 or 1).
    """
    terminal = panel["zero_balance_code"].isin(["03", "06", "09"])
    dpd90    = pd.to_numeric(
        panel["delinquency_status"], errors="coerce"
    ).fillna(0) >= 3
    panel["default"] = (terminal | dpd90).astype(int)
    return panel


# ── Logistic regression feature engineering ───────────────────────────────────

def build_logistic_features(
    panel: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build loan-level feature matrix for logistic regression (spec §5.2).

    Collapses the monthly panel to one row per loan. The target is whether
    the loan ever reached 90+ DPD or a terminal zero-balance code.

    Parameters
    ----------
    panel : pd.DataFrame  Monthly panel (2021-2025 per spec §4.6).
    macro : pd.DataFrame  Quarterly macro features from fetch_macro_quarterly().

    Returns
    -------
    pd.DataFrame  One row per loan with all features and 'default' target.
    """
    panel = add_default_flag(panel.copy())

    # Derive origination quarter join key
    panel["orig_quarter"] = pd.to_datetime(
        panel["first_payment_date"].astype(str), format="%Y%m", errors="coerce"
    ).dt.to_period("Q")

    # Collapse to loan level — take first observation for static fields
    loan = panel.groupby("loan_id").agg(
        credit_score       = ("credit_score",       "first"),
        ltv                = ("ltv",                "first"),
        cltv               = ("cltv",               "first"),
        dti                = ("dti",                "first"),
        orig_interest_rate = ("orig_interest_rate", "first"),
        orig_loan_term     = ("orig_loan_term",     "first"),
        property_state     = ("property_state",     "first"),
        occupancy_status   = ("occupancy_status",   "first"),
        n_borrowers        = ("n_borrowers",        "first"),
        orig_quarter       = ("orig_quarter",       "first"),
        max_loan_age       = ("loan_age",           "max"),
        last_period        = ("reporting_period",   "max"),
        default            = ("default",            "max"),
    ).reset_index()

    # ── Loan-level features (spec §5.2) ──────────────────────────────────────

    # Clip FICO to valid range
    loan["credit_score"] = loan["credit_score"].clip(300, 850)
    # Clip LTV at 150 (extreme values are data errors)
    loan["ltv"] = loan["ltv"].clip(upper=150)
    # Impute median DTI for missing
    dti_median = loan["dti"].median()
    loan["dti"] = loan["dti"].fillna(dti_median)

    # Seasoning ratio: fraction of term remaining at last observation
    loan["months_remaining_ratio"] = (
        (loan["orig_loan_term"] - loan["max_loan_age"]) / loan["orig_loan_term"]
    ).clip(0, 1)
    loan["loan_age_yrs"] = loan["max_loan_age"] / 12

    # Binary flags
    loan["high_dti"]          = (loan["dti"] > 43).astype(int)
    loan["high_ltv"]          = (loan["ltv"] > 80).astype(int)
    loan["low_fico"]          = (loan["credit_score"] < 660).astype(int)
    loan["in_negative_equity"] = (loan["ltv"] > 100).astype(int)
    loan["is_judicial"]       = loan["property_state"].isin(JUDICIAL_STATES).astype(int)

    # Borrower count categorical (merge sparse categories)
    loan["n_borrowers_cat"] = loan["n_borrowers"].astype(str).where(
        loan["n_borrowers"].astype(str).isin(["1", "2"]), other="3+"
    )

    # ── Macro join (at origination quarter, not current — prevents look-ahead) ─
    macro_indexed = macro.copy()
    macro_indexed.index = macro_indexed.index.astype(str)
    loan["orig_quarter_str"] = loan["orig_quarter"].astype(str)
    loan = loan.merge(
        macro_indexed.reset_index().rename(columns={"index": "orig_quarter_str"}),
        on="orig_quarter_str",
        how="left",
    )

    # ── Interaction features (spec §5.2) ─────────────────────────────────────
    rate_median = loan["mortgage_rate_orig"].median()
    loan["high_dti_x_high_rate"] = (
        loan["high_dti"] * (loan["mortgage_rate_orig"] > rate_median).astype(int)
    )
    loan["neg_equity_x_hpi_drop"] = (
        loan["in_negative_equity"] * (loan["hpi_yoy_orig"] < 0).astype(int)
    )

    # ── One-hot encode occupancy status ──────────────────────────────────────
    occ_dummies = pd.get_dummies(
        loan["occupancy_status"], prefix="occ", drop_first=False
    ).astype(int)
    for col in ["occ_O", "occ_S", "occ_I"]:
        if col not in occ_dummies.columns:
            occ_dummies[col] = 0
    loan = pd.concat([loan, occ_dummies[["occ_S", "occ_I"]]], axis=1)

    # ── One-hot encode n_borrowers_cat ────────────────────────────────────────
    bor_dummies = pd.get_dummies(
        loan["n_borrowers_cat"], prefix="n_bor", drop_first=True
    ).astype(int)
    loan = pd.concat([loan, bor_dummies], axis=1)

    return loan


# ── Cox PH feature engineering ────────────────────────────────────────────────

def delinquency_ordinal(status: str) -> int:
    """
    Map delinquency_status string to ordinal severity scale (spec §5.3).

    Returns
    -------
    int  0=current, 1=30dpd, 2=60dpd, 3=90dpd/REO/foreclosure.
    """
    status_str = str(status).strip()
    if status_str in ("RA", "RF", "RE", "999"):
        return 3
    try:
        return min(int(status_str), 3)
    except (ValueError, TypeError):
        return 0


def build_survival_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly panel to loan-level survival tuples for Cox PH (spec §5.3).

    Applies LOCF imputation for eltv (populated only when delinquent),
    encodes delinquency ordinal, and standardizes continuous covariates.

    Parameters
    ----------
    panel : pd.DataFrame  Monthly panel (2018-2025 per spec §4.6).

    Returns
    -------
    pd.DataFrame  One row per loan with duration, event, and covariates.
    """
    panel = panel.copy()
    panel["delinquency_ordinal"] = panel["delinquency_status"].map(delinquency_ordinal)

    # LOCF for eltv: forward-fill within loan, then fill remaining NaN with cltv
    panel = panel.sort_values(["loan_id", "loan_age"])
    panel["eltv"] = panel.groupby("loan_id")["eltv"].ffill()
    panel["eltv"] = panel["eltv"].fillna(panel["cltv"])

    # Binary occupancy dummies
    panel["occ_investment"] = (panel["occupancy_status"] == "I").astype(int)
    panel["occ_second"]     = (panel["occupancy_status"] == "S").astype(int)

    survival = panel.groupby("loan_id").agg(
        duration       = ("loan_age",            "last"),
        event          = ("delinquency_ordinal", lambda x: int(x.max() >= 3)),
        fico           = ("credit_score",        "first"),
        orig_cltv      = ("cltv",               "first"),
        orig_dti       = ("dti",                "first"),
        orig_upb       = ("orig_upb",           "first"),
        orig_term      = ("orig_loan_term",      "first"),
        occ_investment = ("occ_investment",      "first"),
        occ_second     = ("occ_second",          "first"),
        delinq_last    = ("delinquency_ordinal", "last"),
        eltv_last      = ("eltv",               "last"),
        current_rate   = ("current_interest_rate", "last"),
        current_upb    = ("current_upb",        "last"),
    ).reset_index()

    # Impute missing values
    for col in ["orig_cltv", "orig_dti", "fico", "eltv_last"]:
        col_median = survival[col].median()
        survival[col] = survival[col].fillna(col_median)

    # Standardize continuous covariates (z-score)
    cont_cols = [
        "fico", "orig_cltv", "orig_dti", "current_rate",
        "eltv_last", "current_upb", "orig_upb",
    ]
    for col in cont_cols:
        col_mean = survival[col].mean()
        col_std  = survival[col].std()
        if col_std > 0:
            survival[f"{col}_z"] = (survival[col] - col_mean) / col_std

    # log-transform UPB before standardizing (heavy right skew)
    survival["log_orig_upb"]    = np.log1p(survival["orig_upb"])
    survival["log_current_upb"] = np.log1p(survival["current_upb"])

    return survival
