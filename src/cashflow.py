"""
src/cashflow.py

Pool-level MBS cash flow generator (spec §8.2).

Implements:
- generate_cashflows: month-by-month annuity simulation across all Monte Carlo paths,
  with an optional servicing/guarantee-fee net-coupon strip and an optional
  default (MDR) buyout-at-par leg that wires the credit model into the price
- compute_discount_factors: stochastic path-wise discount factors B(0,t,ω)

Both functions are fully vectorised over paths; the inner loop over months is the
only Python loop (unavoidable due to balance carry-forward).

Pass-through structure
----------------------
Borrowers amortise at the note rate (WAC).  The investor in an agency pass-through
receives a *net* coupon = WAC − servicing − guarantee fee, plus 100% of principal
(scheduled + voluntary prepayment + involuntary default buyout).  For agency
collateral a default is a buyout at par by the guarantor, so it behaves exactly
like an involuntary prepayment — there is no loss to the investor, only an
acceleration of principal.  Severity/loss would only matter for non-agency paper.
"""

from typing import Optional

import numpy as np


def generate_cashflows(  # pylint: disable=too-many-arguments,too-many-locals
    smm_paths: np.ndarray,
    coupon: float,
    orig_upb: float,
    term: int,
    net_coupon: Optional[float] = None,
    mdr: Optional[np.ndarray] = None,
) -> tuple:
    """
    Simulate monthly MBS pool cash flows across Monte Carlo paths (spec §8.2).

    Each month: annuity payment on surviving balance → split into investor
    interest (net coupon) and scheduled principal → remaining balance reduced by
    voluntary prepayment (path SMM) and then by an involuntary default buyout
    (MDR), both returned to the investor at par.

    Parameters
    ----------
    smm_paths : np.ndarray
        Shape (n_paths, n_steps).  Voluntary Single Monthly Mortality per path.
        Columns index months 1…n_steps.
    coupon : float
        Pool weighted-average *note* rate (WAC) as decimal (e.g. 0.065).  Drives
        amortisation and the borrower payment.
    orig_upb : float
        Original pool unpaid principal balance in dollars.
    term : int
        Total loan term in months (e.g. 360 for 30-year).
    net_coupon : float, optional
        Investor pass-through coupon = WAC − servicing − g-fee (e.g. 0.060).
        If None, defaults to ``coupon`` (no strip; backward compatible).
    mdr : np.ndarray, optional
        Monthly default rate applied as an involuntary buyout at par.  Either a
        1-D schedule of shape (n_steps,) broadcast across paths, or a 2-D array
        of shape (n_paths, n_steps).  If None, no default leg (backward compatible).

    Returns
    -------
    cashflows : np.ndarray  Shape (n_paths, n_steps).
        Investor cash flow (net interest + scheduled principal + voluntary
        prepayment + default buyout) in each month.
    balances : np.ndarray  Shape (n_paths, n_steps + 1).
        Pool UPB at start of each period; balances[:, 0] = orig_upb.
    """
    n_paths, n_steps = smm_paths.shape
    monthly_rate     = coupon / 12.0
    net_rate         = (coupon if net_coupon is None else net_coupon) / 12.0

    if mdr is None:
        mdr_paths = np.zeros((1, n_steps))
    else:
        mdr_arr   = np.asarray(mdr, dtype=float)
        mdr_paths = mdr_arr[np.newaxis, :] if mdr_arr.ndim == 1 else mdr_arr

    balances  = np.zeros((n_paths, n_steps + 1))
    cashflows = np.zeros((n_paths, n_steps))
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

        # Voluntary prepayment first, then involuntary default buyout on the remainder.
        vol_prepay   = bal_after_sp * smm_paths[:, step]
        bal_after_vp = bal_after_sp - vol_prepay
        default_bo   = bal_after_vp * mdr_paths[:, step]

        cashflows[:, step]    = (
            current_bal * net_rate + sched_prin + vol_prepay + default_bo
        )
        balances[:, step + 1] = np.maximum(bal_after_vp - default_bo, 0.0)

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
