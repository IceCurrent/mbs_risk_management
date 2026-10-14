"""Agency MBS market conventions: payment delay and accrued interest."""

from __future__ import annotations

from datetime import date

# Days from the first of the accrual month to the investor payment. UMBS
# single-class pools (CL/CI) pay on the 25th of the following month. Unknown
# prefixes fail rather than silently assuming a generic convention.
PAYMENT_DELAY_DAYS = {
    ("FNMA", "CL"): 55,
    ("FNMA", "CI"): 55,
    ("FHLMC", "CL"): 55,
    ("FHLMC", "CI"): 55,
}


def payment_delay_days(agency: str, prefix: str) -> int:
    """Official cash-window delay for a disclosed agency/prefix pair."""
    key = (str(agency).upper(), str(prefix).upper())
    if key not in PAYMENT_DELAY_DAYS:
        raise ValueError(
            f"No payment-delay rule for {key}. Source the delay from the "
            "security disclosure rather than assuming a generic convention."
        )
    return PAYMENT_DELAY_DAYS[key]


def agency_mbs_accrued_per_100(coupon: float, settlement_date: date) -> float:
    """30/360 accrued interest per 100 current face from the 1st to settlement."""
    if not 0 <= coupon <= 0.20:
        raise ValueError("coupon must be a decimal rate.")
    accrued_days = min(settlement_date.day - 1, 29)
    return float(coupon * 100.0 * accrued_days / 360.0)


def dirty_price_from_clean(clean_price: float, coupon: float, settlement_date: date) -> float:
    """Invoice price per 100 current face = clean + 30/360 accrued."""
    return clean_price + agency_mbs_accrued_per_100(coupon, settlement_date)
