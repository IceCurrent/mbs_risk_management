import numpy as np
import pandas as pd
import pytest

from curve import bootstrap_par_curve
from hull_white import HullWhite, calibrate_hull_white, fit_vol_term_structure, integral_variance


def test_bootstrap_reprices_every_par_yield(upward_curve):
    for tenor, par in zip(upward_curve.par_tenors, upward_curve.par_yields):
        assert upward_curve.par_yield(tenor) == pytest.approx(par, abs=1e-12)


def test_parallel_par_shift_is_exact(upward_curve):
    shifted = upward_curve.shifted(0.0025)
    for tenor, par in zip(upward_curve.par_tenors, upward_curve.par_yields):
        assert shifted.par_yield(tenor) == pytest.approx(par + 0.0025, abs=1e-12)


def test_flat_par_curve_gives_flat_zero_curve_beyond_bills():
    tenors = np.array([0.25, 0.5, 1, 2, 5, 10, 30])
    curve = bootstrap_par_curve(tenors, np.full(7, 0.05))
    # Semiannual par bonds at 5% discount at the semiannual-compounded 5% zero.
    assert np.allclose(curve.knot_zeros[1:], 2 * np.log1p(0.025), atol=1e-10)
    # Bills are simple interest: P(0.25) = 1 / (1 + 0.05 * 0.25).
    assert curve.discount(0.25) == pytest.approx(1 / (1 + 0.05 * 0.25), rel=1e-12)


def test_monte_carlo_discount_factors_are_unbiased(upward_curve):
    hw = HullWhite(0.03, 0.01)
    times = np.array([0.5, 2.0, 10.0, 25.0])
    _, y_int = hw.simulate_state(times, 200_000, seed=7)
    disc = hw.discount_factors(upward_curve, times, y_int)
    z = (disc.mean(axis=0) - upward_curve.discount(times)) / (disc.std(axis=0) / np.sqrt(len(disc)))
    assert np.all(np.abs(z) < 4)


def test_model_bond_prices_match_curve_at_time_zero(upward_curve):
    hw = HullWhite(0.02, 0.009)
    for tau in (1.0, 10.0, 30.0):
        price = hw.bond_prices(upward_curve, np.array([0.0]), tau, np.zeros((1, 1)))
        assert price[0, 0] == pytest.approx(upward_curve.discount(tau), rel=1e-12)
    assert hw.par_yields(upward_curve, np.array([0.0]), 10.0, np.zeros((1, 1)))[0, 0] == pytest.approx(
        upward_curve.par_yield(10.0), abs=1e-12
    )


def test_curve_shift_moves_every_path_zero_yield_by_the_same_amount(upward_curve):
    hw = HullWhite(0.02, 0.009)
    times = np.array([1.0, 5.0])
    x, _ = hw.simulate_state(times, 500, seed=1)
    shifted = upward_curve.shifted(0.0025)

    def zero_yield(curve):
        return -np.log(hw.bond_prices(curve, times, 10.0, x)) / 10.0

    change = zero_yield(shifted) - zero_yield(upward_curve)
    assert np.ptp(change, axis=0).max() < 1e-12  # path-independent: x cancels exactly
    par_change = hw.par_yields(shifted, times, 10.0, x) - hw.par_yields(upward_curve, times, 10.0, x)
    assert par_change.mean() == pytest.approx(0.0025, abs=3e-4)


def test_simulated_yield_vol_matches_closed_form(upward_curve):
    hw = HullWhite(0.05, 0.01)
    x, _ = hw.simulate_state(np.array([1.0]), 100_000, seed=3)
    y10 = -np.log(hw.bond_prices(upward_curve, np.array([1.0]), 10.0, x)) / 10.0
    theory = hw.zero_yield_vol(10.0) * np.sqrt((1 - np.exp(-2 * 0.05)) / (2 * 0.05))
    assert y10.std() == pytest.approx(theory, rel=0.02)


def test_integral_variance_small_kappa_limit():
    assert integral_variance(1e-12, 0.01, 2.0) == pytest.approx(0.01**2 * 8 / 3, rel=1e-9)
    assert integral_variance(0.05, 0.01, 1e-5) == pytest.approx(0.01**2 * 1e-15 / 3, rel=1e-4)


def test_vol_calibration_recovers_kappa_and_sigma():
    tenors = np.array([2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    true = HullWhite(0.04, 0.009)
    kappa, sigma = fit_vol_term_structure(tenors, true.zero_yield_vol(tenors))
    assert kappa == pytest.approx(0.04, rel=1e-6)
    assert sigma == pytest.approx(0.009, rel=1e-6)

    rng = np.random.default_rng(0)
    shocks = rng.standard_normal(2_000) * 0.009 / np.sqrt(252)
    changes = shocks[:, None] * (true.zero_yield_vol(tenors) / 0.009)[None, :]
    days = pd.bdate_range("2018-01-01", periods=len(changes))
    levels = pd.DataFrame(np.cumsum(changes, axis=0), columns=tenors, index=days)
    cal = calibrate_hull_white(levels, n_boot=50)
    assert cal["sigma_ci95"][0] < 0.009 < cal["sigma_ci95"][1]
    assert cal["kappa"] == pytest.approx(0.04, abs=0.01)
