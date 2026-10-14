"""Pool cash flows from disclosed loans and pathwise SMM.

Every loan amortises on its own note rate and remaining term. Prepayment is a
pool-level SMM (the S-curve is estimated on pool data), applied pro rata, so
loan i's balance is sched_i(t) * S(t) with S(t) = prod_{j<=t} (1 - SMM_j).
The pool therefore needs only the deterministic scheduled-balance curve
SB(t) = sum_i sched_i(t):

    opening_t   = S_{t-1} SB_{t-1}
    interest_t  = (coupon / 12) opening_t          (pass-through coupon)
    scheduled_t = S_{t-1} (SB_{t-1} - SB_t)
    prepaid_t   = SMM_t S_{t-1} SB_t
    closing_t   = S_t SB_t

Month t's interest and principal are paid after the security's delay,
counted from the first of the accrual month (UMBS: 55 days, the 25th of the
following month), rolled to the next business day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dates import UMBS_DELAY_DAYS, month_start, payment_date, year_fraction


def scheduled_balance_curve(loans: pd.DataFrame, n_months: int | None = None) -> np.ndarray:
    """SB(t), t = 0..n_months: pool balance with scheduled amortisation only."""
    required = {"current_upb", "current_interest_rate", "months_remaining"}
    missing = required.difference(loans.columns)
    if missing:
        raise ValueError(f"Loan snapshot missing fields: {sorted(missing)}")
    balance = pd.to_numeric(loans["current_upb"]).to_numpy(dtype=float)
    rate = pd.to_numeric(loans["current_interest_rate"]).to_numpy(dtype=float) / 12.0
    term = pd.to_numeric(loans["months_remaining"]).to_numpy(dtype=int)
    keep = balance > 0
    balance, rate, term = balance[keep], rate[keep], term[keep]
    if (term <= 0).any():
        raise ValueError("Positive-balance loans must have positive remaining term.")
    horizon = int(term.max()) if n_months is None else int(n_months)
    months = np.arange(horizon + 1, dtype=float)[np.newaxis, :]
    growth_t = np.expm1(months * np.log1p(rate)[:, np.newaxis])
    growth_n = np.expm1(term * np.log1p(rate))[:, np.newaxis]
    remaining_share = np.where(
        rate[:, np.newaxis] > 0,
        1.0 - growth_t / np.where(growth_n > 0, growth_n, 1.0),
        1.0 - months / term[:, np.newaxis],
    )
    remaining_share = np.clip(np.where(months <= term[:, np.newaxis], remaining_share, 0.0), 0.0, 1.0)
    return (balance[:, np.newaxis] * remaining_share).sum(axis=0)


def pool_cashflows(smm: np.ndarray, sched_balance: np.ndarray, coupon: float) -> dict:
    """Investor cash flows per path and month (paths x months)."""
    smm = np.asarray(smm, dtype=float)
    sched_balance = np.asarray(sched_balance, dtype=float)
    n_months = smm.shape[1]
    if sched_balance.shape != (n_months + 1,):
        raise ValueError("sched_balance must have one more entry than smm has months.")
    if np.any(smm < 0) or np.any(smm > 1):
        raise ValueError("SMM must lie in [0, 1].")
    survival = np.cumprod(1.0 - smm, axis=1)
    survival_open = np.concatenate([np.ones((smm.shape[0], 1)), survival[:, :-1]], axis=1)
    opening = survival_open * sched_balance[np.newaxis, :-1]
    interest = coupon / 12.0 * opening
    scheduled = survival_open * (sched_balance[:-1] - sched_balance[1:])[np.newaxis, :]
    prepaid = smm * survival_open * sched_balance[np.newaxis, 1:]
    return {
        "total": interest + scheduled + prepaid,
        "interest": interest,
        "scheduled": scheduled,
        "prepaid": prepaid,
        "opening": opening,
        "closing": survival * sched_balance[np.newaxis, 1:],
    }


def payment_schedule(settlement_date, n_months: int, delay_days: int = UMBS_DELAY_DAYS) -> pd.DataFrame:
    """Accrual months from the settlement month, their pay dates and times."""
    first = month_start(settlement_date)
    accrual = [first + pd.DateOffset(months=k) for k in range(n_months)]
    pay = [payment_date(month, delay_days) for month in accrual]
    return pd.DataFrame(
        {
            "accrual_month": accrual,
            "payment_date": pay,
            "time": [year_fraction(settlement_date, day) for day in pay],
        }
    )
