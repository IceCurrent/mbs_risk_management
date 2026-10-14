import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curve import bootstrap_par_curve  # noqa: E402
from mortgage_rate import MortgageRateModel  # noqa: E402

TENORS = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)


@pytest.fixture
def upward_curve():
    """A realistic upward-sloping Treasury par curve (decimals)."""
    pars = np.array([4.13, 4.25, 4.32, 4.47, 4.80, 4.89, 5.02, 5.13, 5.24, 5.65, 5.60]) / 100
    return bootstrap_par_curve(TENORS, pars)


@pytest.fixture
def mortgage_model():
    return MortgageRateModel(
        const=0.0263, beta_2y=0.57, beta_10y=0.385, rho_weekly=0.85,
        anchor_residual=0.0, anchor_survey_mid="2026-10-06", anchor_pmms=0.074,
    )


@pytest.fixture
def small_pool():
    """Three 30-year loans that sum to a round current face."""
    return pd.DataFrame(
        {
            "loan_id": ["A", "B", "C"],
            "cusip": ["TESTCUSIP"] * 3,
            "current_upb": [400_000.0, 350_000.0, 250_000.0],
            "current_interest_rate": [0.0650, 0.0660, 0.0640],
            "months_remaining": [335, 334, 240],
            "loan_age": [25, 26, 25],
        }
    )


@pytest.fixture
def monthly_rates():
    months = pd.date_range("2024-01-01", "2026-09-01", freq="MS")
    return pd.Series(0.069, index=months)
