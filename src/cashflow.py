"""
src/cashflow.py

Pool-level MBS cash flow generator (spec §8.2).

Implements:
- generate_cashflows: month-by-month annuity simulation across all Monte Carlo paths
- compute_discount_factors: stochastic path-wise discount factors B(0,t,ω)

Both functions are fully vectorised over paths; the inner loop over months is the
only Python loop (unavoidable due to balance carry-forward).
"""

import numpy as np


def generate_cashflows(
    smm_paths: np.ndarray,
    coupon: float,
    orig_upb: float,
    term: int,
) -> tuple:
    """
    Simulate monthly MBS pool cash flows across Monte Carlo paths (spec §8.2).

    Each month: annuity payment on surviving balance → split into interest and
    scheduled principal → remaining balance prepaid at the path SMM.

    Parameters
    ----------
    smm_paths : np.ndarray
        Shape (n_paths, n_steps).  Single Monthly Mortality on each path.
        Columns index months 1…n_steps.
    coupon : float
        Pool weighted-average coupon (WAC) as decimal (e.g. 0.065).
    orig_upb : float
        Original pool unpaid principal balance in dollars.
    term : int
        Total loan term in months (e.g. 360 for 30-year).

    Returns
    -------
    cashflows : np.ndarray  Shape (n_paths, n_steps).
        Total cash flow (interest + principal + prepayment) in each month.
    balances : np.ndarray  Shape (n_paths, n_steps + 1).
        Pool UPB at start of each period; balances[:, 0] = orig_upb.
    """
    n_steps      = smm_paths.shape[1]
    monthly_rate = coupon / 12.0

    balances  = np.zeros((smm_paths.shape[0], n_steps + 1))
    cashflows = np.zeros((smm_paths.shape[0], n_steps))
    balances[:, 0] = orig_upb

    for step in range(n_steps):
        current_bal = balances[:, step]
        remaining   = term - step

        if remaining <= 0:
            cashflows[:, step]     = current_bal
            balances[:, step + 1]  = 0.0
            continue

        if monthly_rate == 0.0:
            pmt = current_bal / remaining
        else:
            pmt = current_bal * (
                monthly_rate * (1.0 + monthly_rate) ** remaining
            ) / ((1.0 + monthly_rate) ** remaining - 1.0)

        sched_prin   = np.minimum(pmt - current_bal * monthly_rate, current_bal)
        bal_after_sp = current_bal - sched_prin
        prepay       = bal_after_sp * smm_paths[:, step]

        cashflows[:, step]     = current_bal * monthly_rate + sched_prin + prepay
        balances[:, step + 1]  = np.maximum(bal_after_sp - prepay, 0.0)

    return cashflows, balances


def compute_discount_factors(
    rate_paths: np.ndarray,
    time_step: float = 1.0 / 12,
) -> np.ndarray:
    """
    Compute path-wise stochastic discount factors (spec §8.3).

    B(0, t, ω) = exp(-∫₀ᵗ r(s,ω) ds) ≈ exp(-Σₛ₌₀ᵗ⁻¹ r(s,ω) · Δt)

    The discount factor in column t represents discounting cash flows received
    at the *end* of period t+1 back to time zero.

    Parameters
    ----------
    rate_paths : np.ndarray
        Shape (n_paths, n_steps + 1).  Short rate paths from simulate_vasicek();
        column 0 = initial rate, column t = rate at month t.
    time_step : float
        Discretisation step in years (1/12 for monthly).

    Returns
    -------
    np.ndarray  Shape (n_paths, n_steps).
        disc[:, t] = B(0, t+1, ω) — discount factor from 0 to end of period t.
    """
    # Use rate_paths[:, :-1]: rates at start of each period (exclude terminal column)
    cum = np.cumsum(rate_paths[:, :-1] * time_step, axis=1)
    return np.exp(-cum)
