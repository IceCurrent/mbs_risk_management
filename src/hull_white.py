"""One-factor Hull-White (extended Vasicek) fitted exactly to today's curve.

Short rate r(t) = x(t) + phi(t), with dx = -kappa x dt + sigma dW and x(0)=0.
phi(t) is chosen so model zero-coupon prices equal the bootstrapped Treasury
curve P^M(0,T); it never has to be computed explicitly because

    exp(-int_0^T r ds) = P^M(0,T) exp(-V(T)/2 - Y(T)),     Y(T) = int_0^T x ds
    P(t,T) = P^M(0,T)/P^M(0,t) exp( [V(T-t) - V(T) + V(t)]/2 - B(T-t) x(t) )

with B(tau) = (1 - e^{-kappa tau})/kappa and V(tau) = Var[int_0^tau x ds].
(x, Y) is simulated exactly on any time grid, so E[path discount] = P^M(0,T)
with no discretisation bias. A parallel shift of the input curve moves every
future yield on every path by the same amount, holding the random numbers.

kappa and sigma are fitted to the realised term structure of daily
zero-yield volatility, sd(dy_tau) = sigma B(tau)/tau. The decay of vol
along the curve identifies the risk-neutral kappa used in bond pricing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from curve import ZeroCurve, bootstrap_par_curve

TRADING_DAYS = 252
VOL_FIT_TENORS = (2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)


def b_factor(kappa: float, tau) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    if kappa < 1e-10:
        return tau.copy()
    return -np.expm1(-kappa * tau) / kappa


def _b2_factor(kappa: float, tau) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    if kappa < 1e-10:
        return tau.copy()
    return -np.expm1(-2.0 * kappa * tau) / (2.0 * kappa)


def integral_variance(kappa: float, sigma: float, tau) -> np.ndarray:
    """V(tau) = Var[int_0^tau x ds] for x(0) = 0."""
    tau = np.asarray(tau, dtype=float)
    series = sigma**2 * (tau**3 / 3.0 - kappa * tau**4 / 4.0 + 7.0 * kappa**2 * tau**5 / 60.0)
    if sigma == 0.0:
        return np.zeros_like(tau)
    if kappa < 1e-10:
        return series
    exact = sigma**2 / kappa**2 * (tau - 2.0 * b_factor(kappa, tau) + _b2_factor(kappa, tau))
    return np.where(kappa * tau < 1e-3, series, exact)


@dataclass(frozen=True)
class HullWhite:
    kappa: float
    sigma: float

    def __post_init__(self) -> None:
        if self.kappa < 0 or self.sigma < 0:
            raise ValueError("kappa and sigma must be non-negative.")

    def zero_yield_vol(self, tau) -> np.ndarray:
        """Normal vol of the tau-year zero yield implied by the model."""
        tau = np.asarray(tau, dtype=float)
        return self.sigma * b_factor(self.kappa, tau) / tau

    def simulate_state(self, times, n_paths: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact joint draws of x(t) and Y(t) = int_0^t x on ``times`` (>0)."""
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times[0] <= 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must be positive and strictly increasing.")
        x = np.zeros((n_paths, times.size))
        y = np.zeros((n_paths, times.size))
        if self.sigma == 0.0:
            return x, y
        rng = np.random.default_rng(seed)
        shocks = rng.standard_normal((2, n_paths, times.size))
        prev_x = np.zeros(n_paths)
        prev_y = np.zeros(n_paths)
        steps = np.diff(np.concatenate([[0.0], times]))
        for j, step in enumerate(steps):
            decay = np.exp(-self.kappa * step)
            var_x = self.sigma**2 * _b2_factor(self.kappa, step)
            var_y = integral_variance(self.kappa, self.sigma, step)
            cov = 0.5 * self.sigma**2 * b_factor(self.kappa, step) ** 2
            l11 = np.sqrt(var_x)
            l21 = cov / l11
            l22 = np.sqrt(max(var_y - l21**2, 0.0))
            eps_x = l11 * shocks[0, :, j]
            eps_y = l21 * shocks[0, :, j] + l22 * shocks[1, :, j]
            prev_y = prev_y + prev_x * b_factor(self.kappa, step) + eps_y
            prev_x = prev_x * decay + eps_x
            x[:, j] = prev_x
            y[:, j] = prev_y
        return x, y

    def discount_factors(self, curve: ZeroCurve, times, y_integral: np.ndarray) -> np.ndarray:
        """Pathwise exp(-int_0^t r ds) at ``times``."""
        times = np.asarray(times, dtype=float)
        base = curve.discount(times) * np.exp(-0.5 * integral_variance(self.kappa, self.sigma, times))
        return base[np.newaxis, :] * np.exp(-y_integral)

    def bond_prices(self, curve: ZeroCurve, times, tau: float, x: np.ndarray) -> np.ndarray:
        """P(t, t+tau) on every path for observation ``times`` (x is paths x times)."""
        times = np.asarray(times, dtype=float)

        def var(tau_):
            return integral_variance(self.kappa, self.sigma, tau_)

        log_a = (
            np.log(curve.discount(times + tau))
            - np.log(curve.discount(times))
            + 0.5 * (var(tau) - var(times + tau) + var(times))
        )
        return np.exp(log_a[np.newaxis, :] - b_factor(self.kappa, tau) * x)

    def par_yields(self, curve: ZeroCurve, times, maturity: float, x: np.ndarray) -> np.ndarray:
        """Semiannual par (CMT-convention) yield of ``maturity`` years on every path."""
        count = int(round(2 * maturity))
        annuity = np.zeros_like(x, dtype=float)
        final = None
        for j in range(1, count + 1):
            price = self.bond_prices(curve, times, j / 2.0, x)
            annuity += price
            final = price
        return 2.0 * (1.0 - final) / annuity


# ── Calibration to the realised volatility term structure ─────────────────────

def zero_yield_history(cmt_panel: pd.DataFrame, tenors=VOL_FIT_TENORS) -> pd.DataFrame:
    """Bootstrap each day's CMT curve and read zero yields at ``tenors``."""
    rows = {}
    columns = cmt_panel.columns.to_numpy(dtype=float)
    for day, values in cmt_panel.iterrows():
        curve = bootstrap_par_curve(columns, values.to_numpy(dtype=float))
        rows[day] = curve.zero_rate(np.asarray(tenors, dtype=float))
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(tenors))


def fit_vol_term_structure(tenors, realized_vols) -> tuple[float, float]:
    """Least-squares (kappa, sigma) so sigma B(tau)/tau matches realised vols."""
    tenors = np.asarray(tenors, dtype=float)
    realized_vols = np.asarray(realized_vols, dtype=float)

    def residual(params: np.ndarray) -> np.ndarray:
        kappa, sigma = params
        return sigma * b_factor(kappa, tenors) / tenors - realized_vols

    start = np.array([0.03, float(realized_vols.max())])
    fit = least_squares(residual, start, bounds=([1e-6, 1e-6], [2.0, 0.2]), xtol=1e-14, ftol=1e-14)
    return float(fit.x[0]), float(fit.x[1])


def calibrate_hull_white(
    zero_yields: pd.DataFrame,
    n_boot: int = 500,
    block: int = 20,
    seed: int = 0,
) -> dict:
    """Fit kappa, sigma on daily zero-yield changes with a moving-block bootstrap."""
    changes = zero_yields.diff().dropna()
    tenors = changes.columns.to_numpy(dtype=float)
    realized = changes.std(ddof=1).to_numpy() * np.sqrt(TRADING_DAYS)
    kappa, sigma = fit_vol_term_structure(tenors, realized)

    rng = np.random.default_rng(seed)
    values = changes.to_numpy()
    n_obs = len(values)
    n_blocks = int(np.ceil(n_obs / block))
    draws = np.empty((n_boot, 2))
    for b in range(n_boot):
        starts = rng.integers(0, n_obs - block + 1, size=n_blocks)
        index = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n_obs]
        boot_vol = values[index].std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)
        draws[b] = fit_vol_term_structure(tenors, boot_vol)
    model = HullWhite(kappa, sigma)
    return {
        "kappa": kappa,
        "sigma": sigma,
        "tenors": tenors.tolist(),
        "realized_vol": realized.tolist(),
        "fitted_vol": model.zero_yield_vol(tenors).tolist(),
        "n_days": int(n_obs),
        "window_start": str(changes.index[0].date()),
        "window_end": str(changes.index[-1].date()),
        "bootstrap": draws,
        "kappa_ci95": np.percentile(draws[:, 0], [2.5, 97.5]).tolist(),
        "sigma_ci95": np.percentile(draws[:, 1], [2.5, 97.5]).tolist(),
        "block_days": block,
    }
