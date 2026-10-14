"""Fannie Mae aligned single-class loan/security disclosure parser.

The post-June-2019 aligned loan-level files are ordered by the official L-001,
L-002, ... attribute sequence and do not contain headers.  PoolTalk collateral
exports contain headers; both forms normalize to the same compact schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .common import (
    coalesce_aliases,
    first_record,
    numeric_series,
    parse_date_series,
    rate_series,
    read_pipe_records,
)

# Official aligned fields needed by the cash-flow engine, in L-001 onward order.
# Additional fields in the source are retained as ``fannie_extra_NNN``.
FANNIE_ALIGNED_LOAN_FIELDS = [
    "loan_id",                       # L-001
    "correction_indicator",          # L-002
    "prefix",                        # L-003
    "pool_number",                   # L-004 Security Identifier
    "cusip",                         # L-005
    "orig_upb",                      # L-006 Mortgage Loan Amount
    "issuance_upb",                  # L-007 Issuance Investor Loan UPB
    "current_upb",                   # L-008 Current Investor Loan UPB
    "amortization_type",             # L-009
    "orig_interest_rate",            # L-010
    "issuance_interest_rate",        # L-011
    "current_interest_rate",         # L-012
    "issuance_net_interest_rate",    # L-013
    "current_net_interest_rate",     # L-014
    "first_payment_date",            # L-015
    "maturity_date",                 # L-016
    "orig_term",                     # L-017
    "months_remaining",              # L-018
    "loan_age",                      # L-019
    "ltv",                           # L-020
    "cltv",                          # L-021
    "borrowers",                     # L-022
    "dti",                           # L-023
    "credit_score",                  # L-024
    "first_time_homebuyer",          # L-025
    "loan_purpose",                  # L-026
    "occupancy_status",              # L-027
    "property_type",                 # L-028
    "units",                         # L-029
    "property_state",                # L-030
    "msa",                           # L-031
    "zipcode",                       # L-032
    "channel",                       # L-033
    "mi_pct",                        # L-034
    "seller_name",                   # L-035
    "servicer_name",                 # L-036
]

LOAN_ALIASES = {
    "loan_id": ["loan identifier", "loan id", "loan_identifier"],
    "correction_indicator": ["loan correction indicator", "correction indicator"],
    "prefix": ["prefix", "pool prefix"],
    "pool_number": ["security identifier", "pool number", "security id"],
    "cusip": ["cusip", "cusip number"],
    "orig_upb": ["mortgage loan amount", "original unpaid principal balance upb"],
    "issuance_upb": ["issuance investor loan upb", "issuance upb"],
    "current_upb": [
        "current investor loan upb",
        "current scheduled unpaid principal balance upb",
        "current upb",
    ],
    "orig_interest_rate": ["original interest rate", "original note rate"],
    "current_interest_rate": ["current interest rate", "current note rate"],
    "current_net_interest_rate": ["current net interest rate", "net interest rate"],
    "first_payment_date": ["first payment date"],
    "maturity_date": ["maturity date"],
    "orig_term": ["loan term", "original loan term"],
    "months_remaining": ["remaining months to maturity", "remaining term"],
    "loan_age": ["loan age"],
    "ltv": ["original loan to value ltv", "ltv"],
    "cltv": ["original combined loan to value cltv", "cltv"],
    "borrowers": ["number of borrowers", "borrowers"],
    "dti": ["debt to income ratio", "dti"],
    "credit_score": ["credit score", "borrower credit score"],
    "loan_purpose": ["loan purpose"],
    "occupancy_status": ["occupancy status", "occupancy type"],
    "property_type": ["property type"],
    "units": ["number of units", "units"],
    "property_state": ["property state", "state"],
    "zipcode": ["zip code", "zipcode"],
    "channel": ["origination channel", "channel"],
    "seller_name": ["seller name", "seller"],
    "servicer_name": ["servicer name", "servicer"],
}

SECURITY_ALIASES = {
    "prefix": ["prefix", "pool prefix"],
    "pool_number": ["security identifier", "pool number", "security id"],
    "cusip": ["cusip", "cusip number"],
    "factor_date": ["security factor date", "factor date", "current date"],
    "factor": ["security factor", "current factor", "factor"],
    "coupon": [
        "weighted average net interest rate",
        "pass through rate",
        "security interest rate",
        "coupon",
    ],
    "issue_date": ["issue date", "issuance date"],
    "maturity_date": ["maturity date", "final maturity"],
    "original_upb": [
        "issuance investor security upb",
        "original aggregate balance",
        "original balance",
    ],
    "current_upb": ["current investor security upb", "current balance", "pool upb"],
    "wac": ["weighted average coupon", "wac"],
    "wam": ["weighted average remaining maturity", "wam"],
    "wala": ["weighted average loan age", "wala"],
    "loan_count": ["number of mortgage loans", "loan count", "loans"],
}


def _has_header(first: list[str]) -> bool:
    text = "|".join(str(value).lower() for value in first)
    return any(token in text for token in ("loan identifier", "cusip", "security identifier"))


def _aligned_columns(count: int) -> list[str]:
    if count < len(FANNIE_ALIGNED_LOAN_FIELDS):
        raise ValueError(
            f"Aligned Fannie loan record has {count} fields; expected at least "
            f"{len(FANNIE_ALIGNED_LOAN_FIELDS)}."
        )
    extras = [
        f"fannie_extra_{index:03d}"
        for index in range(len(FANNIE_ALIGNED_LOAN_FIELDS) + 1, count + 1)
    ]
    return [*FANNIE_ALIGNED_LOAN_FIELDS, *extras]


def parse_loan_file(
    path: Path,
    as_of_date: str,
    available_date: str,
    selected_cusips: Optional[Iterable[str]] = None,
    member: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize a post-2019 FNM_ILLD/FNM_MLLD or PoolTalk export."""
    first = first_record(path, member)
    if first and first[0] in {"1", "2"}:
        raise ValueError(
            "Legacy mixed pool/loan record layout detected. Use post-June-2019 "
            "aligned FNM_ILLD/FNM_MLLD files for point-in-time validation."
        )
    if _has_header(first):
        raw = read_pipe_records(path, member=member)
        loans = coalesce_aliases(
            raw,
            LOAN_ALIASES,
            required=[
                "loan_id",
                "pool_number",
                "cusip",
                "current_upb",
                "current_interest_rate",
                "months_remaining",
                "loan_age",
            ],
        )
    else:
        raw = read_pipe_records(path, columns=_aligned_columns(len(first)), member=member)
        loans = raw.copy()

    for column in ("orig_upb", "issuance_upb", "current_upb", "orig_term",
                   "months_remaining", "loan_age", "ltv", "cltv", "borrowers",
                   "dti", "credit_score", "units"):
        if column in loans:
            loans[column] = numeric_series(loans[column])
    for column in ("orig_interest_rate", "issuance_interest_rate",
                   "current_interest_rate", "issuance_net_interest_rate",
                   "current_net_interest_rate"):
        if column in loans:
            loans[column] = rate_series(loans[column])
    for column in ("first_payment_date", "maturity_date"):
        if column in loans:
            loans[column] = parse_date_series(loans[column])

    loans["agency"] = "FNMA"
    loans["as_of_date"] = pd.Timestamp(as_of_date)
    loans["available_date"] = pd.Timestamp(available_date)
    loans["cusip"] = loans["cusip"].astype("string").str.upper().str.strip()
    loans["loan_id"] = loans["loan_id"].astype("string").str.strip()
    loans["pool_number"] = loans["pool_number"].astype("string").str.strip()
    if "correction_indicator" not in loans:
        loans["correction_indicator"] = "N"
    if selected_cusips is not None:
        chosen = {str(cusip).upper() for cusip in selected_cusips}
        loans = loans.loc[loans["cusip"].isin(chosen)]
    loans = loans.loc[:, ~loans.columns.duplicated()].copy()
    if loans.duplicated(["cusip", "loan_id"]).any():
        raise ValueError("Duplicate (CUSIP, loan_id) rows in one disclosure snapshot.")
    return loans.reset_index(drop=True)


def parse_security_export(
    path: Path,
    as_of_date: str,
    available_date: str,
    member: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize a header-bearing PoolTalk/security export."""
    raw = read_pipe_records(path, member=member)
    securities = coalesce_aliases(
        raw,
        SECURITY_ALIASES,
        required=["pool_number", "cusip", "coupon", "original_upb", "current_upb"],
    )
    for column in ("factor", "original_upb", "current_upb", "wac", "wam", "wala",
                   "loan_count"):
        if column in securities:
            securities[column] = numeric_series(securities[column])
    for column in ("coupon", "wac"):
        if column in securities:
            securities[column] = rate_series(securities[column])
    for column in ("factor_date", "issue_date", "maturity_date"):
        if column in securities:
            securities[column] = parse_date_series(securities[column])
    if "factor" not in securities:
        securities["factor"] = securities["current_upb"] / securities["original_upb"]
    securities["agency"] = "FNMA"
    securities["as_of_date"] = pd.Timestamp(as_of_date)
    securities["available_date"] = pd.Timestamp(available_date)
    securities["cusip"] = securities["cusip"].astype("string").str.upper().str.strip()
    return securities.reset_index(drop=True)
