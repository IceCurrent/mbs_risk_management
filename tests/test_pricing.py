import numpy as np
import pandas as pd
import pytest

from hull_white import HullWhite
from pricing import (
    Assumptions,
    build_pool_setup,
    path_values,
    simulate,
    solve_spread,
    value_pool,
)

FACE = 1_000_000.0


def _setup(small_pool, monthly_rates):
    return build_pool_setup(small_pool, 0.055, FACE, "2026-10-09", "2026-10-14", monthly_rates, 55)


def _assumptions(mortgage_model, scurve, sigma=0.0085):
    return Assumptions(HullWhite(0.01, sigma), mortgage_model, scurve)


FLAT = {"floor": 0.06, "ceiling": 0.06, "midpoint": 0.0, "width": 0.002}
STEEP = {"floor": 0.03, "ceiling": 0.60, "midpoint": -0.004, "width": 0.0025}


def test_oas_roundtrip(small_pool, monthly_rates, upward_curve, mortgage_model):
    setup = _setup(small_pool, monthly_rates)
    sim = simulate(setup, upward_curve, _assumptions(mortgage_model, STEEP), 2_000, 3)
    target = path_values(sim, 0.0030).mean()
    assert solve_spread(sim, target) == pytest.approx(0.0030, abs=1e-10)


def test_option_cost_is_zero_when_prepayment_ignores_rates(small_pool, monthly_rates, upward_curve,
                                                          mortgage_model):
    """Regression test: Z and OAS must use the same curve, so no option => no option cost."""
    setup = _setup(small_pool, monthly_rates)
    assumptions = _assumptions(mortgage_model, FLAT)
    det = simulate(setup, upward_curve, Assumptions(HullWhite(0.01, 0.0), mortgage_model, FLAT), 1, 0)
    target = path_values(det, 0.004).mean()
    val, _, _ = value_pool(setup, upward_curve, assumptions, target, 20_000, 9, with_risk=False)
    assert val["z_spread"] == pytest.approx(0.004, abs=1e-10)
    assert abs(val["option_cost"]) < 4 * val["option_cost_se"]


def test_option_cost_is_positive_with_a_refinancing_option(small_pool, monthly_rates, upward_curve,
                                                           mortgage_model):
    setup = _setup(small_pool, monthly_rates)
    assumptions = _assumptions(mortgage_model, STEEP, sigma=0.012)
    det = simulate(setup, upward_curve, Assumptions(HullWhite(0.01, 0.0), mortgage_model, STEEP), 1, 0)
    target = path_values(det, 0.004).mean()
    val, _, _ = value_pool(setup, upward_curve, assumptions, target, 5_000, 2, with_risk=False)
    assert val["option_cost"] > 5 * val["option_cost_se"]


def test_duration_matches_deterministic_repricing_when_cash_flows_are_fixed(
    small_pool, monthly_rates, upward_curve, mortgage_model
):
    setup = _setup(small_pool, monthly_rates)
    assumptions = _assumptions(mortgage_model, FLAT)
    det = simulate(setup, upward_curve, Assumptions(HullWhite(0.01, 0.0), mortgage_model, FLAT), 1, 0)
    target = path_values(det, 0.004).mean()
    val, _, _ = value_pool(setup, upward_curve, assumptions, target, 20_000, 4)

    def deterministic_pv(curve):
        return float(np.sum(det["cashflows"][0] * curve.discount(det["times"])
                            * np.exp(-val["oas"] * det["times"])))

    up, dn = deterministic_pv(upward_curve.shifted(0.0025)), deterministic_pv(upward_curve.shifted(-0.0025))
    expected = (dn - up) / (2 * deterministic_pv(upward_curve) * 0.0025)
    assert val["eff_duration"] == pytest.approx(expected, abs=4 * val["eff_duration_se"] + 1e-3)


def test_first_month_prepay_uses_the_observed_rate_and_ignores_shocks(
    small_pool, monthly_rates, upward_curve, mortgage_model
):
    setup = _setup(small_pool, monthly_rates)
    assert setup.driving_observed[0] == pytest.approx(0.069)  # September, already printed
    assert np.isnan(setup.driving_observed[1])  # October is the trade month: simulated
    assumptions = _assumptions(mortgage_model, STEEP)
    base = simulate(setup, upward_curve, assumptions, 200, 1)
    bumped = simulate(setup, upward_curve.shifted(0.01), assumptions, 200, 1)
    assert np.ptp(base["cpr"][:, 0]) == 0.0
    assert np.allclose(base["cpr"][:, 0], bumped["cpr"][:, 0])
    assert np.all(bumped["mortgage_rate"][:, 1] > base["mortgage_rate"][:, 1])


def test_setup_rejects_settlement_outside_the_factor_month(small_pool, monthly_rates):
    with pytest.raises(ValueError, match="factor month"):
        build_pool_setup(small_pool, 0.055, FACE, "2026-09-30", "2026-10-02", monthly_rates, 55)
    with pytest.raises(ValueError, match="current face"):
        build_pool_setup(small_pool, 0.055, FACE + 5, "2026-10-09", "2026-10-14", monthly_rates, 55)
