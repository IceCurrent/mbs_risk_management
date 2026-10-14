import numpy as np
import pandas as pd
import pytest

from prepayment import (
    PoolHistorySpec,
    block_bootstrap,
    build_prepay_panel,
    cpr_to_smm,
    fit_scurve,
    scheduled_fraction,
    scurve_cpr,
    smm_to_cpr,
)


def test_cpr_smm_roundtrip():
    cpr = np.array([0.0, 0.06, 0.5])
    assert np.allclose(smm_to_cpr(cpr_to_smm(cpr)), cpr)
    assert cpr_to_smm(0.06) == pytest.approx(1 - 0.94 ** (1 / 12))


def test_panel_smm_incentive_lag_and_publication_cutoff():
    dates = pd.to_datetime(["2026-01-25", "2026-02-25", "2026-03-25", "2026-04-25"])
    factors = pd.DataFrame({"cusip": "P1", "date": dates, "factor": [1.0, 0.99, 0.985, 0.97]})
    rates = pd.Series([0.070, 0.068, 0.066, 0.064], index=pd.date_range("2025-12-01", periods=4, freq="MS"))
    spec = {"P1": PoolHistorySpec("P1", wac=0.065, wam_now=300, loan_count_now=400)}

    panel = build_prepay_panel(factors, spec, rates, "2026-04-30", neff_ratio=0.8)
    assert len(panel) == 3
    first = panel.iloc[0]
    sched = scheduled_fraction(0.065, 300 + 3)  # January is three months before April
    assert first["scheduled_fraction"] == pytest.approx(sched)
    assert first["smm"] == pytest.approx((1.0 * (1 - sched) - 0.99) / (1.0 * (1 - sched)))
    # February factor = January payoffs, driven by the December mortgage rate.
    assert first["activity_month"] == pd.Timestamp("2026-01-01")
    assert first["incentive"] == pytest.approx(0.065 - 0.070)
    assert first["n_eff"] == pytest.approx(400 * 1.0 / 0.97 * 0.8)

    # April's factor is published on the 4th business day of April (6 Apr 2026).
    early = build_prepay_panel(factors, spec, rates, "2026-04-03", neff_ratio=0.8)
    assert early["factor_month"].max() == pd.Timestamp("2026-03-01")


def _simulated_panel(seed: int, months: int = 72, n_eff: float = 3_000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    truth = {"floors": {"A": 0.04, "B": 0.05, "C": 0.10}, "midpoint": 0.010, "width": 0.003}
    rows = []
    calendar = pd.date_range("2020-01-01", periods=months, freq="MS")
    for cusip, floor in truth["floors"].items():
        incentive = rng.uniform(-0.015, 0.02, months)
        cpr = scurve_cpr(incentive, floor, 0.5, truth["midpoint"], truth["width"])
        k = rng.binomial(int(n_eff), cpr_to_smm(cpr))
        for month, inc, kk in zip(calendar, incentive, k):
            rows.append({"cusip": cusip, "activity_month": month, "incentive": inc,
                         "n_eff": n_eff, "k": float(kk), "smm": kk / n_eff})
    return pd.DataFrame(rows), truth


def test_quasi_likelihood_recovers_simulated_scurve():
    panel, truth = _simulated_panel(seed=11)
    fit = fit_scurve(panel, 0.5, ["A", "B", "C"])
    se = dict(zip(fit.param_names, np.sqrt(np.diag(fit.cov))))
    for cusip, floor in truth["floors"].items():
        assert fit.floors[cusip] == pytest.approx(floor, abs=4 * se[f"floor_{cusip}"])
    assert fit.midpoint == pytest.approx(truth["midpoint"], abs=4 * se["midpoint"])
    assert fit.width == pytest.approx(truth["width"], abs=4 * se["width"])
    assert 0.6 < fit.phi < 1.5  # binomial data: no overdispersion


def test_block_bootstrap_brackets_the_estimate():
    panel, _ = _simulated_panel(seed=5, months=48)
    fit = fit_scurve(panel, 0.5, ["A", "B", "C"])
    draws = block_bootstrap(panel, fit, n_boot=40, seed=1)
    assert draws.shape == (40, 5)
    lo, hi = np.percentile(draws[:, 3], [2.5, 97.5])
    assert lo < fit.midpoint < hi
