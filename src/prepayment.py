"""
src/prepayment.py

PSA prepayment benchmark model and rate-dependent CPR extension (spec §6).

Implements:
- PSA 100% CPR schedule
- SMM ↔ CPR conversion
- Scheduled principal computation from amortization formula
- Pool-level CPR time series from Freddie Mac performance data
- PSA scalar fitting (least-squares projection)
- Rate-dependent PSA speed multiplier (logistic in refinancing incentive)
- Vectorized CPR path generation for Monte Carlo simulation
"""

import numpy as np
import pandas as pd


# ── SMM / CPR conversion ──────────────────────────────────────────────────────

def cpr_to_smm(cpr: np.ndarray) -> np.ndarray:
    """
    Convert annual Conditional Prepayment Rate to Single Monthly Mortality.

    SMM = 1 - (1 - CPR)^(1/12)

    Parameters
    ----------
    cpr : np.ndarray  CPR in decimal (e.g. 0.06 for 6%).

    Returns
    -------
    np.ndarray  SMM in decimal.
    """
    return 1.0 - (1.0 - np.asarray(cpr, dtype=float)) ** (1.0 / 12)


def smm_to_cpr(smm: np.ndarray) -> np.ndarray:
    """
    Convert Single Monthly Mortality to annual Conditional Prepayment Rate.

    CPR = 1 - (1 - SMM)^12

    Parameters
    ----------
    smm : np.ndarray  SMM in decimal.

    Returns
    -------
    np.ndarray  CPR in decimal.
    """
    return 1.0 - (1.0 - np.asarray(smm, dtype=float)) ** 12


# ── PSA benchmark ─────────────────────────────────────────────────────────────

def psa_cpr_schedule(months: np.ndarray, psa_speed: float = 1.0) -> np.ndarray:
    """
    Compute the PSA CPR schedule at a given speed multiple (spec §6.1).

    100% PSA: CPR ramps linearly at 0.2% * t for t < 30, then flat at 6%.
    k% PSA = k/100 times this schedule at each month.

    Parameters
    ----------
    months : np.ndarray  Loan age in months (1-indexed).
    psa_speed : float    PSA multiple (1.0 = 100% PSA, 1.5 = 150% PSA).

    Returns
    -------
    np.ndarray  CPR in decimal at each loan age.
    """
    months_arr = np.asarray(months, dtype=float)
    base_cpr = np.where(months_arr < 30, 0.002 * months_arr, 0.06)
    return psa_speed * base_cpr


# ── Scheduled principal computation ──────────────────────────────────────────

def compute_scheduled_principal(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute scheduled monthly principal and prepayment for each FRM observation.

    Implements the annuity formula on the previous period balance (spec §6.2).
    Only valid for fixed-rate mortgages (amortization_type == 'FRM').

    Parameters
    ----------
    panel : pd.DataFrame  Must have columns:
        loan_id, loan_age, orig_upb, orig_interest_rate, orig_loan_term,
        current_interest_rate, interest_bearing_upb, zero_balance_code.

    Returns
    -------
    pd.DataFrame  Input with added columns:
        bal_prev, scheduled_principal, prepayment.
    """
    result = panel.sort_values(["loan_id", "loan_age"]).copy()

    monthly_rate = (result["current_interest_rate"] / 100.0) / 12.0
    remaining    = (result["orig_loan_term"] - result["loan_age"]).clip(lower=1)

    # Previous period balance: shift within loan; month 1 uses orig_upb
    result["bal_prev"] = result.groupby("loan_id")["interest_bearing_upb"].shift(1)
    first_obs_mask = (
        result["loan_age"]
        == result.groupby("loan_id")["loan_age"].transform("min")
    )
    result.loc[first_obs_mask, "bal_prev"] = result.loc[first_obs_mask, "orig_upb"]

    bal = result["bal_prev"].values
    rate = monthly_rate.values
    rem  = remaining.values

    # Vectorized annuity payment
    with np.errstate(divide="ignore", invalid="ignore"):
        pmt = np.where(
            rate == 0.0,
            bal / rem,
            bal * (rate * (1.0 + rate) ** rem) / ((1.0 + rate) ** rem - 1.0),
        )

    interest   = bal * rate
    sched_prin = np.minimum(pmt - interest, bal)
    bal_after  = bal - sched_prin
    prepayment = np.maximum(
        bal_after - result["interest_bearing_upb"].values, 0.0
    )

    # Exclude terminal months (zero_balance_code set) from prepayment
    terminal_mask = result["zero_balance_code"].notna().values
    prepayment[terminal_mask] = 0.0

    result["scheduled_principal"] = sched_prin
    result["prepayment"]          = prepayment
    return result


# ── Pool-level CPR ────────────────────────────────────────────────────────────

def pool_cpr(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly pool-level CPR from loan-level prepayment data (spec §6.2).

    Must call compute_scheduled_principal() first.

    Parameters
    ----------
    panel : pd.DataFrame  Must have bal_prev, scheduled_principal, prepayment,
                          reporting_period.

    Returns
    -------
    pd.DataFrame  Monthly series with columns:
        reporting_period, beg_bal, sched_p, prepay, smm, cpr.
    """
    monthly = panel.groupby("reporting_period").agg(
        beg_bal=("bal_prev",             "sum"),
        sched_p=("scheduled_principal",  "sum"),
        prepay =("prepayment",           "sum"),
    ).reset_index()

    denom = (monthly["beg_bal"] - monthly["sched_p"]).clip(lower=1.0)
    monthly["smm"] = monthly["prepay"] / denom
    monthly["cpr"] = smm_to_cpr(monthly["smm"].values)
    return monthly


# ── PSA speed fitting ─────────────────────────────────────────────────────────

def fit_psa_speed(monthly_cpr: pd.DataFrame) -> float:
    """
    Fit a scalar PSA speed multiplier to observed monthly CPR (spec §6.3).

    Least-squares projection: k = (CPR · CPR_100psa) / ||CPR_100psa||²

    Parameters
    ----------
    monthly_cpr : pd.DataFrame  Must have columns 'cpr' and 'loan_age' or 't'.
                                Use pool_cpr() output indexed from month 1.

    Returns
    -------
    float  PSA speed multiplier (e.g. 1.42 means 142% PSA).
    """
    monthly = monthly_cpr.copy().reset_index(drop=True)
    monthly["t"] = np.arange(1, len(monthly) + 1)
    monthly["cpr_100psa"] = np.where(monthly["t"] < 30, 0.002 * monthly["t"], 0.06)

    numerator   = (monthly["cpr"] * monthly["cpr_100psa"]).sum()
    denominator = (monthly["cpr_100psa"] ** 2).sum()

    if denominator == 0:
        raise ValueError("PSA denominator is zero — check that monthly CPR data is non-empty.")
    return float(numerator / denominator)


# ── Rate-dependent CPR ────────────────────────────────────────────────────────

def rate_dependent_psa_k(
    refinancing_incentive: np.ndarray,
    k_min: float = 0.5,
    k_max: float = 3.0,
    alpha: float = -0.04,
    beta:  float = 8.0,
) -> np.ndarray:
    """
    Logistic PSA speed multiplier as a function of refinancing incentive (spec §6.4).

    k(RI) = k_min + (k_max - k_min) * sigmoid(alpha + beta * RI)

    At RI = 0, k(0) ≈ 1.0 (pool prepays at ~100% PSA when neutral).
    As RI rises (rates fall), k increases toward k_max (fast prepayment).
    As RI falls (rates rise), k decreases toward k_min (slow prepayment).

    Parameters
    ----------
    refinancing_incentive : np.ndarray  coupon - current_mortgage_rate (decimal).
    k_min, k_max : float  Bounds on PSA multiplier.
    alpha : float         Logistic intercept (shifts inflection point).
    beta : float          Steepness of S-curve.

    Returns
    -------
    np.ndarray  PSA speed multiplier, same shape as refinancing_incentive.
    """
    refi_inc = np.asarray(refinancing_incentive, dtype=float)
    return k_min + (k_max - k_min) / (1.0 + np.exp(-(alpha + beta * refi_inc)))


def compute_cpr_paths(  # pylint: disable=too-many-arguments
    mortgage_rate_paths: np.ndarray,
    coupon: float,
    k_min: float = 0.5,
    k_max: float = 3.0,
    alpha: float = -0.04,
    beta:  float = 8.0,
) -> np.ndarray:
    """
    Generate rate-dependent CPR paths for Monte Carlo simulation (spec §6.4).

    Parameters
    ----------
    mortgage_rate_paths : np.ndarray  Shape (n_paths, n_steps). 30Y mortgage rate
                                      on each simulated path (decimal).
    coupon : float  Weighted-average pool coupon (decimal).
    k_min, k_max, alpha, beta : float  Rate-dependent PSA parameters.

    Returns
    -------
    np.ndarray  CPR paths of shape (n_paths, n_steps), clipped to [0, 0.99].
    """
    _, n_steps = mortgage_rate_paths.shape
    months   = np.arange(1, n_steps + 1, dtype=float)
    psa_base = np.where(months < 30, 0.002 * months, 0.06)  # shape (n_steps,)

    refi_inc = coupon - mortgage_rate_paths                   # (n_paths, n_steps)
    psa_k = rate_dependent_psa_k(refi_inc, k_min, k_max, alpha, beta)
    cpr   = psa_k * psa_base[np.newaxis, :]                  # (n_paths, n_steps)
    return np.clip(cpr, 0.0, 0.99)


# ── Burnout dampening (optional enhancement) ──────────────────────────────────

def apply_burnout(
    cpr_paths: np.ndarray,
    balances: np.ndarray,
    orig_upb: float,
    gamma: float = 0.3,
) -> np.ndarray:
    """
    Apply burnout dampening to CPR paths (spec §6.4 optional enhancement).

    Remaining-balance fraction (B_t/B_0)^gamma attenuates PSA speed as the
    most rate-sensitive borrowers refinance first.

    Parameters
    ----------
    cpr_paths : np.ndarray  Shape (n_paths, n_steps) from compute_cpr_paths().
    balances : np.ndarray   Shape (n_paths, n_steps+1) from generate_cashflows().
    orig_upb : float        Original pool balance.
    gamma : float           Burnout exponent (typically 0.3–0.5).

    Returns
    -------
    np.ndarray  Burnout-adjusted CPR paths, same shape as cpr_paths.
    """
    bal_ratio = (balances[:, :-1] / orig_upb).clip(0.0, 1.0)
    return cpr_paths * bal_ratio ** gamma
