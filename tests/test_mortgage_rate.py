import numpy as np
import pandas as pd
import pytest

from mortgage_rate import fit_mortgage_rate_model, survey_panel


def test_survey_window_is_monday_to_wednesday():
    days = pd.bdate_range("2026-09-28", "2026-10-09")
    cmt = pd.DataFrame({2.0: np.arange(len(days)) / 100.0, 10.0: 0.05}, index=days)
    pmms = pd.Series([0.07], index=[pd.Timestamp("2026-10-08")])  # Thursday
    panel = survey_panel(pmms, cmt)
    # Mon 5 - Wed 7 Oct are positions 5, 6, 7 in the business-day index.
    assert panel["cmt2"].iloc[0] == pytest.approx(0.06)
    assert panel["survey_mid"].iloc[0] == pd.Timestamp("2026-10-06")


def test_dols_recovers_long_run_relation_and_anchors_today():
    rng = np.random.default_rng(4)
    n = 400
    index = pd.date_range("2018-01-04", periods=n, freq="W-THU")
    cmt10 = 0.03 + np.cumsum(rng.normal(0, 0.0012, n))
    cmt2 = cmt10 - 0.004 + np.cumsum(rng.normal(0, 0.0006, n))
    resid = np.zeros(n)
    for t in range(1, n):
        resid[t] = 0.85 * resid[t - 1] + rng.normal(0, 0.0008)
    pmms = 0.025 + 0.55 * cmt2 + 0.40 * cmt10 + resid
    panel = pd.DataFrame(
        {"pmms": pmms, "cmt2": cmt2, "cmt10": cmt10, "survey_mid": index - pd.Timedelta(days=2)},
        index=index,
    )
    fit = fit_mortgage_rate_model(panel, index[-1], start="2018-01-01")
    model = fit["model"]
    assert model.beta_2y == pytest.approx(0.55, abs=4 * fit["coef_se"]["cmt2"])
    assert model.beta_10y == pytest.approx(0.40, abs=4 * fit["coef_se"]["cmt10"])
    assert model.rho_weekly == pytest.approx(0.85, abs=0.06)
    today = model.project(cmt2[-1], cmt10[-1], 0.0)
    assert today == pytest.approx(pmms[-1], abs=1e-12)
    far = model.project(cmt2[-1], cmt10[-1], 520.0)
    assert far == pytest.approx(model.equilibrium(cmt2[-1], cmt10[-1]), abs=1e-9)
