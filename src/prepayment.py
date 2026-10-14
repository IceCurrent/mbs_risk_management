"""Prepayment S-curve estimated from monthly pool factor histories.

Observation (pool p, activity month t):

    sched_t  = one month of level-payment amortisation at the pool WAC/WAM
    SMM_t    = (F_{t-1}(1 - sched_t) - F_t) / (F_{t-1}(1 - sched_t))
    I_t      = WAC_p - PMMS(month t - lag)          refinancing incentive

Model (CPR scale, logistic in incentive, pool-specific baseline):

    CPR_p(I) = c_p + (c_max - c_p) / (1 + exp(-(I - I_mid) / w))
    SMM      = 1 - (1 - CPR)^(1/12)

c_p is each pool's turnover/curtailment floor; I_mid and w (the refinancing
response) are shared. Each month's unscheduled balance is treated as
k = SMM_obs * n_eff successes out of n_eff effective loans, giving a
binomial quasi-likelihood; overdispersion phi = Pearson chi2 / (N - p).
c_max is not identified (no pool was deeply in the money) and is fixed;
its profile likelihood is reported. Intervals come from a moving-block
bootstrap over calendar months, which keeps both serial correlation and
the same-month correlation across pools.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit  # pylint: disable=no-name-in-module

from dates import factor_publication_date, month_start

DEFAULT_CEILING = 0.50
DEFAULT_LAG_MONTHS = 1
BOOT_BLOCK_MONTHS = 3


def cpr_to_smm(cpr):
    return 1.0 - (1.0 - np.asarray(cpr, dtype=float)) ** (1.0 / 12.0)


def smm_to_cpr(smm):
    return 1.0 - (1.0 - np.asarray(smm, dtype=float)) ** 12


def scurve_cpr(incentive, floor: float, ceiling: float, midpoint: float, width: float):
    """CPR at refinancing ``incentive`` (decimal, note rate minus mortgage rate)."""
    z = (np.asarray(incentive, dtype=float) - midpoint) / width
    return floor + (ceiling - floor) * expit(z)


def scurve_slope(incentive, floor: float, ceiling: float, midpoint: float, width: float):
    """dCPR/dI per unit (decimal) incentive."""
    lam = expit((np.asarray(incentive, dtype=float) - midpoint) / width)
    return (ceiling - floor) * lam * (1.0 - lam) / width


def scheduled_fraction(note_rate: float, remaining_months) -> np.ndarray:
    """Share of balance repaid by one level payment."""
    rate = note_rate / 12.0
    n = np.asarray(remaining_months, dtype=float)
    return rate / np.expm1(n * np.log1p(rate))


def monthly_mortgage_rate(pmms: pd.Series) -> pd.Series:
    """Mean weekly PMMS by release month."""
    return pmms.groupby(pmms.index.to_period("M")).mean().to_timestamp()


@dataclass(frozen=True)
class PoolHistorySpec:
    cusip: str
    wac: float
    wam_now: int
    loan_count_now: int


def build_prepay_panel(
    factors: pd.DataFrame,
    specs: dict,
    mortgage_monthly: pd.Series,
    trade_date,
    neff_ratio: float,
    lag_months: int = DEFAULT_LAG_MONTHS,
) -> pd.DataFrame:
    """One row per pool-month of realised prepayment, point-in-time at ``trade_date``.

    Bloomberg dates the month-M factor on the 25th of M; it is published on
    the fourth business day of M and reflects payoffs during month M-1.
    """
    trade_date = pd.Timestamp(trade_date)
    rows = []
    for cusip, spec in specs.items():
        hist = factors.loc[factors["cusip"] == cusip, ["date", "factor"]].copy()
        hist["factor_month"] = hist["date"].map(month_start)
        hist["published"] = hist["factor_month"].map(lambda m: pd.Timestamp(factor_publication_date(m)))
        hist = hist.loc[hist["published"] <= trade_date].sort_values("factor_month").reset_index(drop=True)
        if len(hist) < 2:
            continue
        now = hist["factor_month"].iloc[-1]
        f_now = float(hist["factor"].iloc[-1])
        for i in range(1, len(hist)):
            prev_month, month = hist["factor_month"].iloc[i - 1], hist["factor_month"].iloc[i]
            if month != prev_month + pd.DateOffset(months=1):
                raise ValueError(f"{cusip}: factor history gap between {prev_month:%Y-%m} and {month:%Y-%m}.")
            f0, f1 = float(hist["factor"].iloc[i - 1]), float(hist["factor"].iloc[i])
            months_back = (now.year - prev_month.year) * 12 + now.month - prev_month.month
            sched = float(scheduled_fraction(spec.wac, spec.wam_now + months_back))
            after_sched = f0 * (1.0 - sched)
            smm = (after_sched - f1) / after_sched
            activity = month - pd.DateOffset(months=1)
            driving = activity - pd.DateOffset(months=lag_months)
            if driving not in mortgage_monthly.index:
                raise ValueError(f"No mortgage rate for driving month {driving:%Y-%m}.")
            n_eff = spec.loan_count_now * f0 / f_now * neff_ratio
            rows.append(
                {
                    "cusip": cusip,
                    "factor_month": month,
                    "activity_month": activity,
                    "driving_month": driving,
                    "factor_open": f0,
                    "factor_close": f1,
                    "scheduled_fraction": sched,
                    "smm": smm,
                    "cpr": float(smm_to_cpr(max(smm, 0.0))),
                    "mortgage_rate": float(mortgage_monthly.loc[driving]),
                    "incentive": spec.wac - float(mortgage_monthly.loc[driving]),
                    "n_eff": n_eff,
                    "k": max(smm, 0.0) * n_eff,
                }
            )
    return pd.DataFrame(rows)


# ── Quasi-likelihood fit ──────────────────────────────────────────────────────

INCENTIVE_SCALE = 100.0  # optimise in percent so all parameters are O(1)


@dataclass
class ScurveFit:
    ceiling: float
    midpoint: float
    width: float
    floors: dict
    loglik: float
    phi: float
    n_obs: int
    n_params: int
    cov: np.ndarray
    param_names: list

    def params(self, cusip: str) -> dict:
        return {
            "floor": self.floors[cusip],
            "ceiling": self.ceiling,
            "midpoint": self.midpoint,
            "width": self.width,
        }

    def natural_vector(self) -> np.ndarray:
        return np.array([*self.floors.values(), self.midpoint, self.width])


class _PanelArrays:
    def __init__(self, panel: pd.DataFrame, cusips: list):
        self.cusips = cusips
        self.pool = panel["cusip"].map({c: i for i, c in enumerate(cusips)}).to_numpy()
        self.incentive = panel["incentive"].to_numpy(dtype=float) * INCENTIVE_SCALE
        self.k = panel["k"].to_numpy(dtype=float)
        self.n = panel["n_eff"].to_numpy(dtype=float)
        self.smm_obs = np.clip(panel["smm"].to_numpy(dtype=float), 0.0, None)

    def subset(self, index: np.ndarray) -> "_PanelArrays":
        out = object.__new__(_PanelArrays)
        out.cusips = self.cusips
        for name in ("pool", "incentive", "k", "n", "smm_obs"):
            setattr(out, name, getattr(self, name)[index])
        return out


def _model_smm(natural: np.ndarray, data: _PanelArrays, ceiling: float) -> np.ndarray:
    n_pools = len(data.cusips)
    floors = natural[:n_pools]
    midpoint, width = natural[n_pools], natural[n_pools + 1]
    cpr = scurve_cpr(data.incentive, floors[data.pool], ceiling, midpoint, width)
    return cpr_to_smm(np.clip(cpr, 1e-9, 0.999))


def _negloglik_natural(natural: np.ndarray, data: _PanelArrays, ceiling: float) -> float:
    n_pools = len(data.cusips)
    if np.any(natural[:n_pools] <= 0) or np.any(natural[:n_pools] >= ceiling) or natural[n_pools + 1] <= 0:
        return 1e12
    smm = _model_smm(natural, data, ceiling)
    return float(-np.sum(data.k * np.log(smm) + (data.n - data.k) * np.log1p(-smm)))


def _to_natural(theta: np.ndarray, n_pools: int, ceiling: float) -> np.ndarray:
    return np.concatenate([ceiling * expit(theta[:n_pools]), [theta[n_pools], np.exp(theta[n_pools + 1])]])


def _to_theta(natural: np.ndarray, n_pools: int, ceiling: float) -> np.ndarray:
    floors = logit(natural[:n_pools] / ceiling)
    return np.concatenate([floors, [natural[n_pools], np.log(natural[n_pools + 1])]])


def _optimise(data: _PanelArrays, ceiling: float, starts: list) -> tuple[np.ndarray, float]:
    n_pools = len(data.cusips)
    best = None
    for start in starts:
        theta0 = _to_theta(np.asarray(start, dtype=float), n_pools, ceiling)
        result = minimize(
            lambda th: _negloglik_natural(_to_natural(th, n_pools, ceiling), data, ceiling),
            theta0,
            method="Nelder-Mead",
            options={"maxiter": 20_000, "xatol": 1e-10, "fatol": 1e-12, "adaptive": True},
        )
        result = minimize(
            lambda th: _negloglik_natural(_to_natural(th, n_pools, ceiling), data, ceiling),
            result.x,
            method="BFGS",
            options={"gtol": 1e-9},
        )
        if best is None or result.fun < best.fun:
            best = result
    return _to_natural(best.x, n_pools, ceiling), float(best.fun)


def _default_starts(n_pools: int) -> list:
    starts = []
    for midpoint in (0.5, 1.0, 1.5):
        for width in (0.1, 0.25, 0.5):
            starts.append([*([0.04] * n_pools), midpoint, width])
    return starts


def _numerical_hessian(fun, x: np.ndarray) -> np.ndarray:
    n = len(x)
    step = 1e-4 * np.maximum(np.abs(x), 1e-2)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            e_i = np.zeros(n)
            e_j = np.zeros(n)
            e_i[i] = step[i]
            e_j[j] = step[j]
            value = (
                fun(x + e_i + e_j) - fun(x + e_i - e_j) - fun(x - e_i + e_j) + fun(x - e_i - e_j)
            ) / (4.0 * step[i] * step[j])
            hess[i, j] = hess[j, i] = value
    return hess


def fit_scurve(panel: pd.DataFrame, ceiling: float = DEFAULT_CEILING,
               cusips: list | None = None) -> ScurveFit:
    """Quasi-binomial MLE of pool floors and the shared refinancing response."""
    cusips = cusips or sorted(panel["cusip"].unique())
    data = _PanelArrays(panel, cusips)
    natural, nll = _optimise(data, ceiling, _default_starts(len(cusips)))
    smm = _model_smm(natural, data, ceiling)
    pearson = np.sum((data.k - data.n * smm) ** 2 / (data.n * smm * (1.0 - smm)))
    n_params = len(natural)
    phi = float(pearson / (len(data.k) - n_params))
    hess = _numerical_hessian(lambda v: _negloglik_natural(v, data, ceiling), natural)
    cov = phi * np.linalg.inv(hess)
    n_pools = len(cusips)
    scale = np.ones(n_params)
    scale[n_pools:] = 1.0 / INCENTIVE_SCALE  # midpoint and width back to decimals
    return ScurveFit(
        ceiling=ceiling,
        midpoint=float(natural[n_pools] / INCENTIVE_SCALE),
        width=float(natural[n_pools + 1] / INCENTIVE_SCALE),
        floors={c: float(v) for c, v in zip(cusips, natural[:n_pools])},
        loglik=-nll,
        phi=phi,
        n_obs=int(len(data.k)),
        n_params=n_params,
        cov=cov * np.outer(scale, scale),
        param_names=[f"floor_{c}" for c in cusips] + ["midpoint", "width"],
    )


def pearson_residuals(panel: pd.DataFrame, fit: ScurveFit) -> pd.Series:
    floors = panel["cusip"].map(fit.floors).to_numpy()
    cpr = scurve_cpr(panel["incentive"].to_numpy(), floors, fit.ceiling, fit.midpoint, fit.width)
    smm = cpr_to_smm(cpr)
    obs = np.clip(panel["smm"].to_numpy(), 0.0, None)
    return pd.Series((obs - smm) / np.sqrt(smm * (1 - smm) / panel["n_eff"].to_numpy()), index=panel.index)


def block_bootstrap(
    panel: pd.DataFrame,
    fit: ScurveFit,
    n_boot: int = 500,
    block: int = BOOT_BLOCK_MONTHS,
    seed: int = 0,
) -> np.ndarray:
    """Moving-block bootstrap over calendar activity months; rows are natural params."""
    cusips = list(fit.floors)
    data = _PanelArrays(panel, cusips)
    months = np.sort(panel["activity_month"].unique())
    rows_by_month = [np.flatnonzero(panel["activity_month"].to_numpy() == m) for m in months]
    n_months = len(months)
    n_blocks = int(np.ceil(n_months / block))
    start = [*fit.floors.values(), fit.midpoint * INCENTIVE_SCALE, fit.width * INCENTIVE_SCALE]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(start)))
    for b in range(n_boot):
        firsts = rng.integers(0, n_months - block + 1, size=n_blocks)
        chosen = (firsts[:, None] + np.arange(block)[None, :]).ravel()[:n_months]
        index = np.concatenate([rows_by_month[m] for m in chosen])
        natural, _ = _optimise(data.subset(index), fit.ceiling, [start])
        draws[b] = natural
    draws[:, len(cusips):] /= INCENTIVE_SCALE
    return draws


def profile_ceiling(panel: pd.DataFrame, ceilings, cusips: list | None = None) -> pd.DataFrame:
    """Quasi-likelihood ratio across fixed ceilings (identification check)."""
    fits = [fit_scurve(panel, c, cusips) for c in ceilings]
    best = max(fits, key=lambda f: f.loglik)
    return pd.DataFrame(
        {
            "ceiling": list(ceilings),
            "loglik": [f.loglik for f in fits],
            "quasi_lr": [2.0 * (best.loglik - f.loglik) / best.phi for f in fits],
            "midpoint": [f.midpoint for f in fits],
            "width": [f.width for f in fits],
        }
    )
