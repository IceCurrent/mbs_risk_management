"""Hard data-quality gates for actual Agency MBS security snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReconciliationResult:
    cusip: str
    as_of_date: str
    loan_count_disclosed: Optional[int]
    loan_count_calculated: int
    current_upb_disclosed: float
    current_upb_calculated: float
    upb_difference: float
    factor_disclosed: float
    factor_calculated: float
    factor_difference: float
    wac_disclosed: Optional[float]
    wac_calculated: float
    wac_difference_bps: Optional[float]
    wam_disclosed: Optional[float]
    wam_calculated: float
    wam_difference_months: Optional[float]
    wala_disclosed: Optional[float]
    wala_calculated: float
    wala_difference_months: Optional[float]
    available_after_as_of: bool
    passed: bool


def _weighted_average(frame: pd.DataFrame, value: str, weight: str = "current_upb") -> float:
    valid = frame[[value, weight]].dropna()
    total = valid[weight].sum()
    if total <= 0:
        raise ValueError(f"Cannot calculate weighted {value}: non-positive balance.")
    return float(np.average(valid[value], weights=valid[weight]))


def reconcile_security_snapshot(
    security: pd.Series | Dict,
    loans: pd.DataFrame,
    upb_tolerance: float = 1.00,
    factor_tolerance: float = 1e-6,
    wac_tolerance_bps: float = 1.0,
    term_tolerance_months: float = 1.0,
    raise_on_failure: bool = True,
) -> ReconciliationResult:
    """Reconcile one CUSIP/month before any model cash flow is generated."""
    security = pd.Series(security)
    cusip = str(security["cusip"]).upper()
    selected = loans.loc[loans["cusip"].astype(str).str.upper() == cusip].copy()
    if selected.empty:
        raise ValueError(f"No disclosed loans for CUSIP {cusip}.")
    if selected.duplicated(["cusip", "loan_id"]).any():
        raise ValueError(f"Duplicate loans in {cusip} snapshot.")
    if (selected["current_upb"].fillna(0) < 0).any():
        raise ValueError(f"Negative current UPB in {cusip}.")

    current_disclosed = float(security["current_upb"])
    current_calculated = float(selected["current_upb"].sum())
    upb_difference = current_calculated - current_disclosed
    original_upb = float(security["original_upb"])
    factor_calculated = current_calculated / original_upb
    factor_disclosed = float(security.get("factor", current_disclosed / original_upb))
    factor_difference = factor_calculated - factor_disclosed

    rate_column = (
        "current_interest_rate"
        if "current_interest_rate" in selected
        else "orig_interest_rate"
    )
    wac_calculated = _weighted_average(selected, rate_column)
    wam_calculated = _weighted_average(selected, "months_remaining")
    wala_calculated = _weighted_average(selected, "loan_age")

    wac_disclosed = security.get("wac")
    wam_disclosed = security.get("wam")
    wala_disclosed = security.get("wala")
    loan_count_disclosed = security.get("loan_count")
    wac_diff = (
        (wac_calculated - float(wac_disclosed)) * 10_000
        if pd.notna(wac_disclosed)
        else None
    )
    wam_diff = (
        wam_calculated - float(wam_disclosed) if pd.notna(wam_disclosed) else None
    )
    wala_diff = (
        wala_calculated - float(wala_disclosed) if pd.notna(wala_disclosed) else None
    )
    count_pass = (
        True
        if pd.isna(loan_count_disclosed)
        else int(loan_count_disclosed) == len(selected)
    )
    available_date = pd.Timestamp(security["available_date"])
    as_of_date = pd.Timestamp(security["as_of_date"])
    temporal_pass = available_date >= as_of_date
    checks = [
        abs(upb_difference) <= upb_tolerance,
        abs(factor_difference) <= factor_tolerance,
        count_pass,
        temporal_pass,
        wac_diff is None or abs(wac_diff) <= wac_tolerance_bps,
        wam_diff is None or abs(wam_diff) <= term_tolerance_months,
        wala_diff is None or abs(wala_diff) <= term_tolerance_months,
    ]
    result = ReconciliationResult(
        cusip=cusip,
        as_of_date=as_of_date.date().isoformat(),
        loan_count_disclosed=(
            int(loan_count_disclosed) if pd.notna(loan_count_disclosed) else None
        ),
        loan_count_calculated=len(selected),
        current_upb_disclosed=current_disclosed,
        current_upb_calculated=current_calculated,
        upb_difference=upb_difference,
        factor_disclosed=factor_disclosed,
        factor_calculated=factor_calculated,
        factor_difference=factor_difference,
        wac_disclosed=float(wac_disclosed) if pd.notna(wac_disclosed) else None,
        wac_calculated=wac_calculated,
        wac_difference_bps=wac_diff,
        wam_disclosed=float(wam_disclosed) if pd.notna(wam_disclosed) else None,
        wam_calculated=wam_calculated,
        wam_difference_months=wam_diff,
        wala_disclosed=float(wala_disclosed) if pd.notna(wala_disclosed) else None,
        wala_calculated=wala_calculated,
        wala_difference_months=wala_diff,
        available_after_as_of=temporal_pass,
        passed=all(checks),
    )
    if raise_on_failure and not result.passed:
        failures = {
            key: value
            for key, value in asdict(result).items()
            if key.endswith(("difference", "difference_bps", "difference_months"))
        }
        raise ValueError(f"Disclosure reconciliation failed for {cusip}: {failures}")
    return result


def reconcile_factor_runoff(
    prior_loans: pd.DataFrame,
    current_loans: pd.DataFrame,
    prior_security: pd.Series | Dict,
    current_security: pd.Series | Dict,
    tolerance: float = 1.00,
) -> Dict[str, float | bool]:
    """Reconcile disclosed factor runoff to loan-level principal change."""
    prior_security = pd.Series(prior_security)
    current_security = pd.Series(current_security)
    prior_upb = float(prior_security["current_upb"])
    current_upb = float(current_security["current_upb"])
    security_runoff = prior_upb - current_upb
    loan_runoff = float(prior_loans["current_upb"].sum() - current_loans["current_upb"].sum())
    difference = loan_runoff - security_runoff
    return {
        "security_runoff": security_runoff,
        "loan_runoff": loan_runoff,
        "difference": difference,
        "passed": abs(difference) <= tolerance,
    }
