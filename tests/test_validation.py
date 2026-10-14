import numpy as np
import pandas as pd
import pytest

from prepayment import cpr_to_smm, scurve_cpr
from validation import empirical_duration, rolling_origin_prepay_test


def test_empirical_duration_recovers_known_sensitivity():
    rng = np.random.default_rng(2)
    days = pd.bdate_range("2025-01-01", "2026-10-09")
    dy = rng.normal(0, 0.0005, len(days))
    level = 0.045 + np.cumsum(dy)
    cmt = pd.DataFrame({tenor: level for tenor in (2.0, 5.0, 10.0, 30.0)}, index=days)
    noise = rng.normal(0, 0.0002, len(days))
    prices = 96 * np.exp(-6.0 * (level - level[0]) + noise)
    quotes = pd.DataFrame({"date": days, "price": prices})
    out = empirical_duration(quotes, cmt, days[-1])
    assert out["full"]["y10"]["duration"] == pytest.approx(6.0, abs=4 * out["full"]["y10"]["se"])
    assert out["last_26w"]["n_weeks"] == 26


def test_rolling_origin_test_only_uses_earlier_months():
    rng = np.random.default_rng(8)
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    rows = []
    for cusip, floor in (("A", 0.04), ("B", 0.08)):
        inc = rng.uniform(-0.01, 0.015, len(months))
        smm = cpr_to_smm(scurve_cpr(inc, floor, 0.5, 0.01, 0.003))
        k = rng.binomial(2_000, smm)
        rows += [{"cusip": cusip, "activity_month": m, "incentive": i, "n_eff": 2_000.0,
                  "k": float(kk), "smm": kk / 2_000} for m, i, kk in zip(months, inc, k)]
    panel = pd.DataFrame(rows)
    result = rolling_origin_prepay_test(panel, ["A", "B"], 0.5, test_months=4)
    assert result["summary"]["forecasts"] == 8
    assert set(result["table"]["activity_month"]) == set(months[-4:])
    assert result["summary"]["model"]["deviance"] < result["summary"]["naive"]["deviance"]


def test_month_ahead_score_uses_the_forecast_quantiles():
    from validation import score_month_ahead

    quantiles = np.linspace(0.860, 0.880, 99)
    forecast = {"factor_quantiles": quantiles.tolist(), "factor_pi90": [0.861, 0.879], "point_factor": 0.870}
    score = score_month_ahead(forecast, 0.870)
    assert score["pit"] == pytest.approx(0.5, abs=0.01)
    assert score["inside_pi90"]
    assert score_month_ahead(forecast, 0.90)["pit"] == 1.0
