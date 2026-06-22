"""
src/pricing.py

MBS OAS solver, effective duration, and convexity (spec §9.1–9.2).

Implements:
- price_at_oas: expected PV with a constant OAS spread added to all discount rates
- solve_oas: Brent's method root-find to match market price
- mbs_price_at_rate: full MC reprice after shifting the initial short rate (for fd)
- effective_duration_convexity: symmetric finite-difference sensitivities
"""

import numpy as np
from scipy.optimize import brentq


def price_at_oas(
    oas_lambda: float,
    cashflows: np.ndarray,
    disc: np.ndarray,
    time_step: float = 1.0 / 12,
) -> float:
    """
    Expected PV of MBS cash flows with OAS spread added to the discount rate.

    P(λ) = E[ Σ_t CF(t,ω) · B(0,t,ω) · exp(-λ · t · Δt) ]

    Parameters
    ----------
    oas_lambda : float      OAS in decimal (e.g. 0.005 = 50bp).
    cashflows : np.ndarray  Shape (n_paths, n_steps) from generate_cashflows().
    disc : np.ndarray       Shape (n_paths, n_steps) from compute_discount_factors().
    time_step : float       Monthly time step in years (1/12).

    Returns
    -------
    float  Model price (dollars, not % of par).
    """
    n_steps = cashflows.shape[1]
    times   = np.arange(1, n_steps + 1) * time_step       # (n_steps,) in years
    oas_adj = np.exp(-oas_lambda * times)[np.newaxis, :]   # (1, n_steps)
    pv_paths = np.sum(cashflows * disc * oas_adj, axis=1)
    return float(np.mean(pv_paths))


def solve_oas(
    market_price: float,
    cashflows: np.ndarray,
    disc: np.ndarray,
    time_step: float = 1.0 / 12,
    bracket: tuple = (-0.05, 0.25),
) -> float:
    """
    Find OAS λ such that price_at_oas(λ) == market_price (spec §9.1).

    Uses Brent's method — guaranteed convergence for a bracketed root.

    Parameters
    ----------
    market_price : float    Observed market price in dollars.
    cashflows : np.ndarray  Shape (n_paths, n_steps).
    disc : np.ndarray       Shape (n_paths, n_steps).
    time_step : float       Monthly time step in years.
    bracket : tuple         (lo, hi) bracket for OAS search in decimal.

    Returns
    -------
    float  OAS in decimal (multiply by 10 000 for basis points).
    """
    def objective(lam: float) -> float:
        return price_at_oas(lam, cashflows, disc, time_step) - market_price

    return float(brentq(objective, *bracket, xtol=1e-8, maxiter=200))


def simulate_pool_cashflows(  # pylint: disable=too-many-arguments,too-many-locals
    initial_rate: float,
    kappa: float,
    theta: float,
    sigma: float,
    coupon: float,
    orig_upb: float,
    term: int,
    n_paths: int = 5_000,
    seed: int = 42,
    net_coupon: float = None,
    mdr: np.ndarray = None,
    mbs_spread: float = 0.015,
    prepay_params: dict = None,
) -> tuple:
    """
    Run the full pricing pipeline once and return cash flows and discount factors.

    simulate (Vasicek under the supplied θ) → 30Y mortgage rates → rate-dependent
    CPR → SMM → pool cash flows (net coupon + MDR buyout) → stochastic discount
    factors.  ``theta`` should be the *risk-neutral* θ_Q for pricing.

    Returns
    -------
    tuple (cashflows, disc, balances, rate_paths, cpr_paths).
    """
    # pylint: disable=import-outside-toplevel
    from vasicek import simulate_vasicek, mortgage_rate_from_short
    from prepayment import compute_cpr_paths, cpr_to_smm
    from cashflow import generate_cashflows, compute_discount_factors

    rate_paths = simulate_vasicek(
        initial_rate=initial_rate, kappa=kappa, theta=theta, sigma=sigma,
        n_paths=n_paths, n_steps=term, time_step=1.0 / 12, seed=seed,
    )
    mortgage_paths = mortgage_rate_from_short(
        rate_paths, kappa, theta, sigma, tau=30.0, mbs_spread=mbs_spread
    )
    cpr_paths = compute_cpr_paths(
        mortgage_paths[:, :-1], coupon, **(prepay_params or {})
    )
    smm_paths = cpr_to_smm(cpr_paths)
    cashflows, balances = generate_cashflows(
        smm_paths, coupon, orig_upb, term, net_coupon=net_coupon, mdr=mdr
    )
    disc = compute_discount_factors(rate_paths, time_step=1.0 / 12)
    return cashflows, disc, balances, rate_paths, cpr_paths


def mbs_price_at_rate(  # pylint: disable=too-many-arguments
    rate_shift: float,
    initial_rate: float,
    kappa: float,
    theta: float,
    sigma: float,
    coupon: float,
    orig_upb: float,
    term: int,
    n_paths: int = 5_000,
    seed: int = 42,
    net_coupon: float = None,
    mdr: np.ndarray = None,
    mbs_spread: float = 0.015,
    prepay_params: dict = None,
) -> float:
    """
    Full MC reprice of the MBS at a shifted initial short rate (spec §9.2).

    Used for finite-difference effective duration/convexity and rate-shock
    stress.  The shift is applied to r₀ only; all other parameters (and crucially
    the RNG seed and path count) are held fixed so repricings share common random
    numbers.

    Parameters
    ----------
    rate_shift : float   Shift in decimal added to initial_rate (e.g. +0.0025).
    initial_rate : float Base initial short rate r₀.
    kappa, theta, sigma : float  Vasicek parameters (θ = risk-neutral θ_Q).
    coupon : float       Pool WAC in decimal.
    orig_upb : float     Pool original balance.
    term : int           Loan term in months.
    n_paths, seed : int  Monte Carlo controls — keep identical across bumps (CRN).
    net_coupon : float   Investor pass-through coupon (WAC − strip).
    mdr : np.ndarray     Monthly default-rate schedule (buyout at par).
    mbs_spread : float   Primary/secondary mortgage spread over the Vasicek curve.
    prepay_params : dict Override S-curve params passed to compute_cpr_paths.

    Returns
    -------
    float  MBS model price in dollars.
    """
    cashflows, disc, *_ = simulate_pool_cashflows(
        initial_rate=initial_rate + rate_shift, kappa=kappa, theta=theta,
        sigma=sigma, coupon=coupon, orig_upb=orig_upb, term=term,
        n_paths=n_paths, seed=seed, net_coupon=net_coupon, mdr=mdr,
        mbs_spread=mbs_spread, prepay_params=prepay_params,
    )
    return float(np.mean(np.sum(cashflows * disc, axis=1)))


def effective_duration_convexity(  # pylint: disable=too-many-arguments,too-many-locals
    initial_rate: float,
    kappa: float,
    theta: float,
    sigma: float,
    coupon: float,
    orig_upb: float,
    term: int,
    rate_bump: float = 0.0025,
    n_paths: int = 5_000,
    seed: int = 42,
    net_coupon: float = None,
    mdr: np.ndarray = None,
    mbs_spread: float = 0.015,
    prepay_params: dict = None,
) -> dict:
    """
    Effective duration and convexity via symmetric finite differences (spec §9.2).

    EffDur  = (P_dn - P_up) / (2 · P₀ · rate_bump)
    EffConv = (P_up + P_dn - 2·P₀) / (P₀ · rate_bump²)

    The base and both bumped prices are repriced here with the SAME seed and path
    count (common random numbers), so the finite differences are not swamped by
    Monte Carlo noise — essential for the second-order convexity term, which is a
    difference of nearly-equal large numbers.

    Returns
    -------
    dict with keys: base_price, price_up, price_dn, eff_duration, eff_convexity.
    """
    shared = {
        "initial_rate": initial_rate, "kappa": kappa, "theta": theta,
        "sigma": sigma, "coupon": coupon, "orig_upb": orig_upb, "term": term,
        "n_paths": n_paths, "seed": seed, "net_coupon": net_coupon, "mdr": mdr,
        "mbs_spread": mbs_spread, "prepay_params": prepay_params,
    }
    base_price = mbs_price_at_rate(0.0, **shared)
    price_up   = mbs_price_at_rate(+rate_bump, **shared)
    price_dn   = mbs_price_at_rate(-rate_bump, **shared)
    return {
        "base_price":    base_price,
        "price_up":      price_up,
        "price_dn":      price_dn,
        "eff_duration":  (price_dn - price_up) / (2.0 * base_price * rate_bump),
        "eff_convexity": (price_up + price_dn - 2.0 * base_price) / (base_price * rate_bump ** 2),
    }
