"""OAS, Z-spread, option cost and effective duration/convexity for one pool.

Price at spread s on Monte Carlo paths w = 1..N (full price, dollars):

    PV_w(s) = sum_k CF_k(w) D_k(w) exp(-s T_k),   P(s) = mean_w PV_w(s)

OAS solves P(OAS) = clean + accrued (the model values every future cash
flow, so it is matched to the invoice price). The Z-spread solves the same
equation on the zero-volatility path (sigma = 0: rates follow the forward
curve, D_k = P(0, T_k)), so option cost = Z - OAS compares the same curve,
the same mortgage-rate model and the same S-curve with and without rate
volatility.

Effective duration/convexity reprice at fixed OAS after a parallel shift of
the input Treasury par curve by +/-Delta (re-bootstrapped), with common
random numbers:

    D = (P_- - P_+) / (2 P_0 Delta),   C = (P_+ + P_- - 2 P_0) / (P_0 Delta^2)

Standard errors use the pathwise differences, so they reflect only the
Monte Carlo noise that does not cancel between the bumped runs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from cashflow import payment_schedule, pool_cashflows, scheduled_balance_curve
from curve import ZeroCurve
from dates import month_start, year_fraction
from hull_white import HullWhite
from mortgage_rate import MortgageRateModel
from prepayment import cpr_to_smm, scurve_cpr

DAYS_PER_WEEK = 7.0


@dataclass(frozen=True)
class PoolSetup:
    """Everything about the pool and the calendar that does not depend on rates."""

    trade_date: pd.Timestamp
    settlement_date: pd.Timestamp
    coupon: float
    wac: float
    current_face: float
    sched_balance: np.ndarray
    schedule: pd.DataFrame
    driving_observed: np.ndarray
    driving_time: np.ndarray

    @property
    def pay_times(self) -> np.ndarray:
        return self.schedule["time"].to_numpy(dtype=float)

    @property
    def n_months(self) -> int:
        return len(self.schedule)


@dataclass(frozen=True)
class Assumptions:
    hull_white: HullWhite
    mortgage: MortgageRateModel
    scurve: dict


def build_pool_setup(
    loans: pd.DataFrame,
    coupon: float,
    current_face: float,
    trade_date,
    settlement_date,
    mortgage_monthly: pd.Series,
    delay_days: int,
    lag_months: int = 1,
) -> PoolSetup:
    """Calendar, scheduled balances and which months' refinancing rates are known.

    Payoffs in accrual month k respond to the mortgage rate of month k - lag.
    A driving month that ended before the trade month uses its observed
    PMMS average; later months use the simulated rate at mid-month.
    """
    trade_date = pd.Timestamp(trade_date)
    settlement_date = pd.Timestamp(settlement_date)
    if month_start(settlement_date) != month_start(trade_date):
        raise ValueError("Trade and settlement must fall in the factor month being priced.")
    sched = scheduled_balance_curve(loans)
    face_from_loans = float(sched[0])
    if abs(face_from_loans - current_face) > 1.0:
        raise ValueError(f"Loan balances {face_from_loans:,.2f} != current face {current_face:,.2f}.")
    schedule = payment_schedule(settlement_date, len(sched) - 1, delay_days)
    observed = np.full(len(schedule), np.nan)
    sim_time = np.zeros(len(schedule))
    trade_month = month_start(trade_date)
    for k, accrual in enumerate(schedule["accrual_month"]):
        driving = accrual - pd.DateOffset(months=lag_months)
        if driving < trade_month:
            if driving not in mortgage_monthly.index:
                raise ValueError(f"Observed mortgage rate missing for {driving:%Y-%m}.")
            observed[k] = float(mortgage_monthly.loc[driving])
        else:
            mid_month = driving + pd.Timedelta(days=14)
            sim_time[k] = max(year_fraction(settlement_date, mid_month), 0.0)
    return PoolSetup(
        trade_date=trade_date,
        settlement_date=settlement_date,
        coupon=coupon,
        wac=float(np.average(loans["current_interest_rate"], weights=loans["current_upb"])),
        current_face=current_face,
        sched_balance=sched,
        schedule=schedule,
        driving_observed=observed,
        driving_time=sim_time,
    )


def simulate(setup: PoolSetup, curve: ZeroCurve, assumptions: Assumptions,
             n_paths: int, seed: int) -> dict:
    """Rates -> mortgage rate -> incentive -> CPR -> cash flows and discount factors."""
    hw = assumptions.hull_white
    n_paths = 1 if hw.sigma == 0.0 else n_paths
    pay_times = setup.pay_times
    unknown = np.isnan(setup.driving_observed)
    obs_times = setup.driving_time[unknown]
    grid = np.unique(np.concatenate([pay_times, obs_times[obs_times > 0]]))
    x, y_int = hw.simulate_state(grid, n_paths, seed)
    disc = hw.discount_factors(curve, pay_times, y_int[:, np.searchsorted(grid, pay_times)])

    x_obs = np.zeros((n_paths, obs_times.size))
    positive = obs_times > 0
    x_obs[:, positive] = x[:, np.searchsorted(grid, obs_times[positive])]
    cmt2 = hw.par_yields(curve, obs_times, 2.0, x_obs)
    cmt10 = hw.par_yields(curve, obs_times, 10.0, x_obs)
    anchor = pd.Timestamp(assumptions.mortgage.anchor_survey_mid)
    weeks = (setup.settlement_date - anchor).days / DAYS_PER_WEEK + obs_times * 365.0 / DAYS_PER_WEEK
    mortgage = np.empty((n_paths, setup.n_months))
    mortgage[:, ~unknown] = setup.driving_observed[~unknown][np.newaxis, :]
    mortgage[:, unknown] = assumptions.mortgage.project(cmt2, cmt10, weeks[np.newaxis, :])

    sc = assumptions.scurve
    cpr = scurve_cpr(setup.wac - mortgage, sc["floor"], sc["ceiling"], sc["midpoint"], sc["width"])
    smm = cpr_to_smm(cpr)
    flows = pool_cashflows(smm, setup.sched_balance, setup.coupon)
    return {
        "cashflows": flows["total"],
        "flows": flows,
        "discount": disc,
        "times": pay_times,
        "mortgage_rate": mortgage,
        "cpr": cpr,
        "smm": smm,
    }


# ── Spreads ────────────────────────────────────────────────────────────────────

def path_values(sim: dict, spread: float) -> np.ndarray:
    weight = np.exp(-spread * sim["times"])[np.newaxis, :]
    return np.sum(sim["cashflows"] * sim["discount"] * weight, axis=1)


def solve_spread(sim: dict, target: float, bracket=(-0.05, 0.15)) -> float:
    """Spread s with mean_w PV_w(s) = target (dollars)."""
    return float(brentq(lambda s: path_values(sim, s).mean() - target, *bracket, xtol=1e-12, maxiter=200))


def spread_standard_error(sim: dict, spread: float) -> float:
    """Delta-method Monte Carlo SE of a solved spread."""
    pv = path_values(sim, spread)
    if pv.size < 2:
        return 0.0
    weight = np.exp(-spread * sim["times"])[np.newaxis, :]
    dprice = -np.mean(np.sum(sim["cashflows"] * sim["discount"] * weight * sim["times"], axis=1))
    return float(pv.std(ddof=1) / np.sqrt(pv.size) / abs(dprice))


def zero_volatility(assumptions: Assumptions) -> Assumptions:
    return replace(assumptions, hull_white=HullWhite(assumptions.hull_white.kappa, 0.0))


def weighted_average_life(sim: dict) -> float:
    principal = sim["flows"]["scheduled"] + sim["flows"]["prepaid"]
    return float(np.mean(np.sum(principal * sim["times"], axis=1) / np.sum(principal, axis=1)))


def effective_risk(setup: PoolSetup, curve: ZeroCurve, assumptions: Assumptions, base_sim: dict,
                   oas: float, n_paths: int, seed: int, bump: float = 0.0025) -> dict:
    """Effective duration and convexity at fixed OAS under parallel par-curve shifts."""
    pv0 = path_values(base_sim, oas)
    up = path_values(simulate(setup, curve.shifted(+bump), assumptions, n_paths, seed), oas)
    dn = path_values(simulate(setup, curve.shifted(-bump), assumptions, n_paths, seed), oas)
    price0 = pv0.mean()
    dur_paths = (dn - up) / (2.0 * price0 * bump)
    conv_paths = (up + dn - 2.0 * pv0) / (price0 * bump**2)
    root_n = np.sqrt(pv0.size)
    to_points = 100.0 / setup.current_face
    return {
        "bump_bp": bump * 1e4,
        "price_base": price0 * to_points,
        "price_up": up.mean() * to_points,
        "price_down": dn.mean() * to_points,
        "eff_duration": float(dur_paths.mean()),
        "eff_duration_se": float(dur_paths.std(ddof=1) / root_n),
        "eff_convexity": float(conv_paths.mean()),
        "eff_convexity_se": float(conv_paths.std(ddof=1) / root_n),
    }


def value_pool(setup: PoolSetup, curve: ZeroCurve, assumptions: Assumptions, target_dollars: float,
               n_paths: int, seed: int, bump: float = 0.0025, with_risk: bool = True) -> dict:
    """OAS, Z-spread, option cost, duration/convexity and prepay summaries."""
    sim = simulate(setup, curve, assumptions, n_paths, seed)
    oas = solve_spread(sim, target_dollars)
    oas_se = spread_standard_error(sim, oas)
    det = simulate(setup, curve, zero_volatility(assumptions), 1, seed)
    z_spread = solve_spread(det, target_dollars)
    pv_zero = path_values(sim, 0.0)
    to_points = 100.0 / setup.current_face
    balance_open = sim["flows"]["opening"]
    result = {
        "oas": oas,
        "oas_se": oas_se,
        "z_spread": z_spread,
        "option_cost": z_spread - oas,
        "option_cost_se": oas_se,
        "model_price_zero_oas": float(pv_zero.mean() * to_points),
        "model_price_zero_oas_se": float(pv_zero.std(ddof=1) / np.sqrt(pv_zero.size) * to_points),
        "zero_vol_price_zero_spread": float(path_values(det, 0.0)[0] * to_points),
        "wal_years": weighted_average_life(sim),
        "wal_years_zero_vol": weighted_average_life(det),
        "month1_cpr": float(sim["cpr"][0, 0]),
        "month1_driving_rate_observed": bool(not np.isnan(setup.driving_observed[0])),
        "avg_cpr_12m": float(
            smm_weighted_cpr(sim["smm"][:, :12], balance_open[:, :12])
        ),
        "avg_cpr_life": float(smm_weighted_cpr(sim["smm"], balance_open)),
        "avg_cpr_life_zero_vol": float(smm_weighted_cpr(det["smm"], det["flows"]["opening"])),
    }
    if with_risk:
        result.update(effective_risk(setup, curve, assumptions, sim, oas, n_paths, seed, bump))
    return result, sim, det


def smm_weighted_cpr(smm: np.ndarray, opening: np.ndarray) -> float:
    """Balance-weighted average SMM over the window, quoted as CPR."""
    after = opening.sum()
    if after <= 0:
        return float("nan")
    return float(1.0 - (1.0 - np.sum(smm * opening) / after) ** 12)
