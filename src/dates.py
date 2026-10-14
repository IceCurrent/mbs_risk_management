"""US bond-market business days and Agency MBS date conventions."""

from __future__ import annotations

from datetime import date

import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, GoodFriday, USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


class BondMarketCalendar(AbstractHolidayCalendar):
    """Federal holidays (incl. Columbus and Veterans Day) plus Good Friday.

    This is the SIFMA full-close set to within early closes, which do not
    move a settlement or payment date.
    """

    rules = USFederalHolidayCalendar.rules + [GoodFriday]


BUSINESS_DAY = CustomBusinessDay(calendar=BondMarketCalendar())

# Fannie Mae publishes monthly factors and loan-level disclosures on the
# fourth business day of the factor month.
FACTOR_PUBLICATION_BUSINESS_DAY = 4
# A d-day delay counts from the 1st of the accrual month, so the investor is
# paid on day (d - 30) of the following month: 55 days -> the 25th (UMBS).
UMBS_DELAY_DAYS = 55


def _ts(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def is_business_day(value) -> bool:
    day = _ts(value)
    return bool(BUSINESS_DAY.is_on_offset(day))


def roll_forward(value) -> pd.Timestamp:
    """Next business day on or after ``value``."""
    day = _ts(value)
    return day if is_business_day(day) else day + BUSINESS_DAY


def add_business_days(value, count: int) -> pd.Timestamp:
    return _ts(value) + count * BUSINESS_DAY


def settlement_date(trade_date, lag_business_days: int = 2) -> date:
    """Trade date plus ``lag`` bond-market business days."""
    if not is_business_day(trade_date):
        raise ValueError(f"Trade date {trade_date} is not a bond-market business day.")
    return add_business_days(trade_date, lag_business_days).date()


def nth_business_day(year: int, month: int, n: int) -> date:
    first = roll_forward(pd.Timestamp(year=year, month=month, day=1))
    return (first + (n - 1) * BUSINESS_DAY).date()


def factor_publication_date(factor_month) -> date:
    """Public availability of the factor / loan tape for ``factor_month``."""
    month = _ts(factor_month)
    return nth_business_day(month.year, month.month, FACTOR_PUBLICATION_BUSINESS_DAY)


def month_start(value) -> pd.Timestamp:
    return _ts(value).to_period("M").to_timestamp()


def payment_date(accrual_month, delay_days: int = UMBS_DELAY_DAYS) -> date:
    """Investor payment date for interest accrued during ``accrual_month``."""
    day = delay_days - 30
    if not 1 <= day <= 28:
        raise ValueError(f"Unsupported payment delay of {delay_days} days.")
    nxt = month_start(accrual_month) + pd.DateOffset(months=1)
    return roll_forward(nxt.replace(day=day)).date()


def year_fraction(start, end) -> float:
    """ACT/365F year fraction used for continuous-compounding discounting."""
    return (_ts(end) - _ts(start)).days / 365.0
