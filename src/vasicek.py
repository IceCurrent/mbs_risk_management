"""
src/vasicek.py

Vasicek (Ornstein-Uhlenbeck) interest rate model (spec §7).

Implements:
- OLS calibration from monthly FRED Treasury data (spec §7.3)
- Exact Euler-Maruyama Monte Carlo simulation (spec §7.4)
- Closed-form zero-coupon bond pricing (spec §7.2)
- Yield curve computation at standard maturities
- 30-year mortgage rate paths from short rate paths
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_vasicek(
    rates: np.ndarray,
    time_step: float = 1.0 / 12,
) -> Tuple[float, float, float]:
    """
    OLS calibration of Vasicek parameters from a monthly rate time series (spec §7.3).

    The conditional distribution r_{t+dt} | r_t is Gaussian with mean
    linear in r_t, so OLS of r_{t+1} on r_t gives exact MLEs:

        r_{t+dt} = a + b * r_t + eps,   eps ~ N(0, s²)

    Recover:  kappa = -ln(b)/dt,  theta = a/(1-b),  sigma from residual variance.

    Parameters
    ----------
    rates : np.ndarray  Monthly short-rate observations (decimal, e.g. 0.05 for 5%).
    time_step : float   Time step in years (1/12 for monthly data).

    Returns
    -------
    Tuple (kappa, theta, sigma) — Vasicek parameters under physical measure P.
    """
    r_current = rates[:-1]
    r_next    = rates[1:]

    design = np.column_stack([np.ones_like(r_current), r_current])
    # Use index [0] directly to avoid unpacking unused rank/sv/residual outputs
    coeffs = np.linalg.lstsq(design, r_next, rcond=None)[0]
    intercept, slope = float(coeffs[0]), float(coeffs[1])

    s_squared = float(np.var(r_next - (intercept + slope * r_current), ddof=2))

    kappa = -np.log(slope) / time_step
    theta = intercept / (1.0 - slope)
    sigma = np.sqrt(s_squared * 2.0 * kappa / (1.0 - np.exp(-2.0 * kappa * time_step)))

    return float(kappa), float(theta), float(sigma)


def load_and_calibrate(
    start: str = "2000-01-01",
    end: str = "2024-12-31",
    cache_path: Optional[Path] = None,
) -> Tuple[float, float, float, float]:
    """
    Fetch 1-Year Treasury (GS1) from FRED, calibrate Vasicek, and print results.

    Parameters
    ----------
    start, end : str  Date range for FRED query.
    cache_path : Path  If provided and file exists, load cached CSV instead.

    Returns
    -------
    Tuple (kappa, theta, sigma, r0) where r0 is the most recent short rate.
    """
    # Prefer the cache: it lets the downstream notebooks (06-08) run with no
    # FRED API key. Only require a key when we actually have to hit the API.
    if cache_path is not None and Path(cache_path).exists():
        gs1_series = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze()
    else:
        try:
            from fredapi import Fred as _Fred  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError("fredapi not installed. Run: pip install fredapi") from exc

        import os  # pylint: disable=import-outside-toplevel
        from dotenv import load_dotenv  # pylint: disable=import-outside-toplevel
        load_dotenv()

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "FRED_API_KEY not set. Add it to .env and call load_dotenv() first."
            )

        fred = _Fred(api_key=api_key)
        gs1_series = fred.get_series("GS1", start=start, end=end).dropna()
        gs1_series = gs1_series.resample("MS").last()
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            gs1_series.to_csv(cache_path)

    rates = gs1_series.values / 100.0
    kappa, theta, sigma = calibrate_vasicek(rates)
    initial_rate = float(rates[-1])

    print(f"κ (mean-reversion speed):  {kappa:.4f}  [yr⁻¹]")
    print(f"θ (long-run mean):         {theta * 100:.2f}%")
    print(f"σ (volatility):            {sigma * 100:.3f}%  [yr⁻¹/²]")
    print(f"r₀ (current short rate):   {initial_rate * 100:.2f}%")
    print(f"Rate half-life:            {np.log(2) / kappa:.1f} years")

    return kappa, theta, sigma, initial_rate


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_vasicek(  # pylint: disable=too-many-arguments
    initial_rate: float,
    kappa: float,
    theta: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    time_step: float = 1.0 / 12,
    seed: int = 42,
) -> np.ndarray:
    """
    Exact Monte Carlo simulation of the Vasicek short rate (spec §7.4).

    Uses the exact Euler-Maruyama transition (not an approximation):
        r_{t+dt} = r_t * exp(-κ dt) + θ(1 - exp(-κ dt)) + σ_dt * Z_t
    where σ_dt = σ * sqrt((1 - exp(-2κ dt)) / (2κ)).

    Parameters
    ----------
    initial_rate : float  Current short rate r₀ (decimal).
    kappa : float         Mean-reversion speed.
    theta : float         Long-run mean.
    sigma : float         Volatility.
    n_paths : int         Number of Monte Carlo paths.
    n_steps : int         Number of monthly time steps.
    time_step : float     Time step in years (1/12 for monthly).
    seed : int            RNG seed for reproducibility.

    Returns
    -------
    np.ndarray  Shape (n_paths, n_steps + 1).
                Column 0 = initial_rate; column t = short rate at month t.
    """
    rng = np.random.default_rng(seed)
    rate_paths = np.zeros((n_paths, n_steps + 1))
    rate_paths[:, 0] = initial_rate

    exp_kdt  = np.exp(-kappa * time_step)
    mean_adj = theta * (1.0 - exp_kdt)
    vol_adj  = sigma * np.sqrt(
        (1.0 - np.exp(-2.0 * kappa * time_step)) / (2.0 * kappa)
    )

    noise = rng.standard_normal((n_paths, n_steps))
    for step in range(n_steps):
        rate_paths[:, step + 1] = (
            rate_paths[:, step] * exp_kdt + mean_adj + vol_adj * noise[:, step]
        )

    return rate_paths


# ── Bond pricing and yield curve ──────────────────────────────────────────────

def vasicek_bond_price(
    r_t: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    tau: float,
) -> np.ndarray:
    """
    Closed-form Vasicek zero-coupon bond price (spec §7.2).

    P(t, T) = A(t,T) * exp(-B(t,T) * r_t)

    where B = (1 - exp(-κτ)) / κ,
          A = exp((θ - σ²/2κ²)(B - τ) - σ²B²/4κ)

    Parameters
    ----------
    r_t : np.ndarray  Current short rate(s) — scalar or array.
    kappa, theta, sigma : float  Calibrated parameters.
    tau : float  Time to maturity in years.

    Returns
    -------
    np.ndarray  Bond price(s), same shape as r_t.
    """
    rate = np.asarray(r_t, dtype=float)
    bond_b = (1.0 - np.exp(-kappa * tau)) / kappa
    bond_a = np.exp(
        (theta - sigma ** 2 / (2.0 * kappa ** 2)) * (bond_b - tau)
        - sigma ** 2 / (4.0 * kappa) * bond_b ** 2
    )
    return bond_a * np.exp(-bond_b * rate)


def vasicek_yield(
    r_t: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    tau: float,
) -> np.ndarray:
    """
    Continuously compounded yield for maturity tau (spec §7.2).

    y(t, T) = -ln P(t, T) / τ

    Parameters
    ----------
    r_t : np.ndarray  Current short rate(s).
    kappa, theta, sigma : float
    tau : float  Time to maturity in years.

    Returns
    -------
    np.ndarray  Yield(s), same shape as r_t.
    """
    price = vasicek_bond_price(r_t, kappa, theta, sigma, tau)
    return -np.log(price) / tau


def vasicek_yield_curve(
    r_t: float,
    kappa: float,
    theta: float,
    sigma: float,
    maturities: Tuple[float, ...] = (0.25, 0.5, 1, 2, 5, 10, 20, 30),
) -> pd.DataFrame:
    """
    Yield curve at standard maturities for a given short rate.

    Parameters
    ----------
    r_t : float         Current short rate (decimal).
    kappa, theta, sigma : float
    maturities : tuple  Time-to-maturity values in years.

    Returns
    -------
    pd.DataFrame  Columns: maturity_yrs, yield (decimal), yield_pct.
    """
    mat_arr = np.array(maturities, dtype=float)
    yields  = vasicek_yield(r_t, kappa, theta, sigma, mat_arr)
    return pd.DataFrame({
        "maturity_yrs": mat_arr,
        "yield":        yields,
        "yield_pct":    yields * 100,
    })


def mortgage_rate_from_short(  # pylint: disable=too-many-arguments
    rate_paths: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    tau: float = 30.0,
    mbs_spread: float = 0.015,
) -> np.ndarray:
    """
    Convert short rate paths to 30-year mortgage rate paths (spec §7.5).

    Computes the Vasicek 30-year yield, then adds a constant MBS-Treasury
    spread (~150bp) to capture the premium mortgage borrowers pay above
    the risk-free curve.

    Parameters
    ----------
    rate_paths : np.ndarray  Short rate paths, any shape.
    kappa, theta, sigma : float  Vasicek parameters.
    tau : float  Yield maturity in years (default 30.0).
    mbs_spread : float  MBS-Treasury spread (default 0.015 = 150bp).

    Returns
    -------
    np.ndarray  Mortgage rate paths, same shape as rate_paths.
    """
    zcb_yield = vasicek_yield(rate_paths, kappa, theta, sigma, tau)
    return zcb_yield + mbs_spread


# ── Risk-neutral measure (market price of risk) ───────────────────────────────
#
# Calibration on GS1 recovers the *physical* (P) parameters that describe how
# rates actually evolve.  Discounting cash flows for PRICING must be done under
# the *risk-neutral* (Q) measure, whose drift carries a term premium.  With a
# constant market price of risk λ the Vasicek drift becomes
#     dr = κ(θ_P − r)dt − λσ dt + σ dW^Q = κ(θ_Q − r)dt + σ dW^Q,
# i.e. only the long-run mean changes:  θ_Q = θ_P − λσ/κ.
#
# Notebooks 06–07 (forecasting where rates will actually go) use θ_P.
# The pricer (notebook 08) uses θ_Q so the discount curve is realistic.

def calibrate_risk_neutral_theta(
    short_rate: float,
    kappa: float,
    sigma: float,
    target_yield: float,
    maturity: float = 10.0,
) -> float:
    """
    Solve the risk-neutral long-run mean θ_Q so the model zero yield at a chosen
    maturity matches a target (e.g. the current 10Y Treasury).

    The physical curve calibrated on the 1Y rate is far too low at the long end
    (the σ²/2κ² convexity drag pulls 30Y zeros below the short rate), which
    over-discounts and inflates MBS prices.  Pinning θ_Q to a realistic long
    yield restores an upward-sloping discount curve.

    Parameters
    ----------
    short_rate : float    Current short rate r₀.
    kappa, sigma : float  Calibrated mean-reversion speed and volatility.
    target_yield : float  Target zero yield at ``maturity`` (decimal).
    maturity : float      Maturity at which to match (years; default 10).

    Returns
    -------
    float  θ_Q (risk-neutral long-run mean, decimal).
    """
    low, high = -0.05, 0.20
    for _ in range(100):
        mid = 0.5 * (low + high)
        if vasicek_yield(short_rate, kappa, mid, sigma, maturity) < target_yield:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def implied_market_price_of_risk(
    theta_p: float, theta_q: float, kappa: float, sigma: float
) -> float:
    """
    Back out the constant market price of risk λ from θ_P and θ_Q.

    θ_Q = θ_P − λσ/κ  ⇒  λ = −(θ_Q − θ_P)·κ/σ.  A negative λ corresponds to a
    positive term premium (risk-neutral rates drift above physical).
    """
    return -(theta_q - theta_p) * kappa / sigma


# ── Summary statistics ────────────────────────────────────────────────────────

def long_run_distribution(
    kappa: float, theta: float, sigma: float
) -> Tuple[float, float, float]:
    """
    Analytical long-run (stationary) distribution of the Vasicek model (spec §7.1).

    As t → ∞: r_∞ ~ N(θ, σ²/2κ)

    Parameters
    ----------
    kappa, theta, sigma : float

    Returns
    -------
    Tuple (mean, std, var) = (θ, σ/sqrt(2κ), σ²/2κ).
    """
    variance = sigma ** 2 / (2.0 * kappa)
    return theta, float(np.sqrt(variance)), variance
