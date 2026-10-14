"""Treasury zero curve bootstrapped from CMT par yields.

CMT yields are on a bond-equivalent basis. Bills (tenors up to six months)
are single payments quoted as simple coupon-equivalent yields,
P(t) = 1 / (1 + y t). Longer tenors are par bonds paying y/2 every six
months. The two conventions coincide at six months. Continuously
compounded zero rates are linear in maturity between knots and flat beyond
the first and last knot, and each knot is solved so its par bond prices to
exactly 100.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

SHORT_END = 0.5


def _semiannual_times(maturity: float) -> np.ndarray:
    count = int(round(2 * maturity))
    if abs(count / 2.0 - maturity) > 1e-9:
        raise ValueError(f"Coupon tenor {maturity} is not a multiple of six months.")
    return np.arange(1, count + 1) / 2.0


@dataclass(frozen=True)
class ZeroCurve:
    knot_times: np.ndarray
    knot_zeros: np.ndarray
    par_tenors: np.ndarray
    par_yields: np.ndarray

    def zero_rate(self, t) -> np.ndarray:
        return np.interp(np.asarray(t, dtype=float), self.knot_times, self.knot_zeros)

    def discount(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero_rate(t) * t)

    def par_yield(self, maturity: float) -> float:
        """Par yield in the same convention as the CMT inputs."""
        if maturity <= SHORT_END:
            return float((1.0 / self.discount(maturity) - 1.0) / maturity)
        times = _semiannual_times(maturity)
        annuity = self.discount(times).sum() / 2.0
        return float((1.0 - self.discount(maturity)) / annuity)

    def shifted(self, shift: float) -> "ZeroCurve":
        """Re-bootstrap after a parallel shift of every input par yield."""
        return bootstrap_par_curve(self.par_tenors, self.par_yields + shift)


def _solve_coupon_knot(knot_t: list, knot_z: list, tenor: float, par: float) -> float:
    """Zero rate at ``tenor`` that prices a semiannual par bond to 100."""
    times = _semiannual_times(tenor)
    prev_t, prev_z = knot_t[-1], knot_z[-1]
    known = np.interp(times, knot_t, knot_z)
    bridge = (times - prev_t) / (tenor - prev_t)

    def price_error(z_new: float) -> float:
        zeros = np.where(times <= prev_t, known, prev_z + (z_new - prev_z) * bridge)
        disc = np.exp(-zeros * times)
        return par / 2.0 * disc.sum() + disc[-1] - 1.0

    return brentq(price_error, -0.10, 0.50, xtol=1e-14, maxiter=200)


def bootstrap_par_curve(tenors, par_yields) -> ZeroCurve:
    """Zero curve that reprices every input par yield exactly."""
    tenors = np.asarray(tenors, dtype=float)
    yields = np.asarray(par_yields, dtype=float)
    order = np.argsort(tenors)
    tenors, yields = tenors[order], yields[order]
    if tenors[0] > SHORT_END:
        raise ValueError("Curve needs at least one tenor of six months or less.")

    knot_t: list[float] = []
    knot_z: list[float] = []
    for tenor, par in zip(tenors, yields):
        if tenor <= SHORT_END:
            zero = np.log1p(par * tenor) / tenor
        else:
            zero = _solve_coupon_knot(knot_t, knot_z, tenor, par)
        knot_t.append(tenor)
        knot_z.append(zero)
    return ZeroCurve(np.array(knot_t), np.array(knot_z), tenors, yields)
