from datetime import date

import pandas as pd
import pytest

from cashflow import payment_schedule
from dates import factor_publication_date, payment_date, settlement_date
from security import agency_mbs_accrued_per_100, dirty_price_from_clean, payment_delay_days


def test_settlement_skips_weekend_and_columbus_day():
    # Fri 9 Oct 2026 + 2 bond-market days: Mon 12 Oct is Columbus Day.
    assert settlement_date("2026-10-09") == date(2026, 10, 14)
    with pytest.raises(ValueError, match="not a bond-market business day"):
        settlement_date("2026-10-10")


def test_factor_publication_is_fourth_business_day():
    assert factor_publication_date("2026-10-01") == date(2026, 10, 6)
    assert factor_publication_date("2026-11-01") == date(2026, 11, 5)


def test_umbs_pays_on_the_25th_rolled_forward():
    assert payment_date("2026-10-01") == date(2026, 11, 25)
    # 25 Jul 2026 is a Saturday.
    assert payment_date("2026-06-01") == date(2026, 7, 27)
    assert payment_delay_days("FNMA", "CL") == 55
    with pytest.raises(ValueError, match="No payment-delay rule"):
        payment_delay_days("FNMA", "XX")


def test_first_cash_flow_is_42_days_after_settlement_not_85():
    schedule = payment_schedule(pd.Timestamp("2026-10-14"), 3)
    assert schedule["payment_date"].iloc[0] == date(2026, 11, 25)
    assert schedule["time"].iloc[0] == pytest.approx(42 / 365)


def test_accrued_is_30_360_from_the_first():
    settle = date(2026, 10, 14)
    accrued = agency_mbs_accrued_per_100(0.055, settle)
    assert accrued == pytest.approx(5.5 * 13 / 360)
    assert dirty_price_from_clean(96.3, 0.055, settle) == pytest.approx(96.3 + accrued)
    assert agency_mbs_accrued_per_100(0.055, date(2026, 10, 1)) == 0.0
