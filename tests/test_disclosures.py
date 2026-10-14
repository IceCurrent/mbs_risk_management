from pathlib import Path

import pandas as pd
import pytest

from disclosures.common import apply_point_in_time_corrections, build_manifest
from disclosures.fannie import FANNIE_ALIGNED_LOAN_FIELDS, parse_loan_file, parse_security_export
from disclosures.quality import reconcile_factor_runoff, reconcile_security_snapshot


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_fannie_header_exports_and_reconciliation(tmp_path):
    loans_path = _write(
        tmp_path / "loans.txt",
        "|".join(
            [
                "Loan Identifier",
                "Loan Correction Indicator",
                "Pool Prefix",
                "Security Identifier",
                "CUSIP",
                "Mortgage Loan Amount",
                "Current Investor Loan UPB",
                "Current Interest Rate",
                "First Payment Date",
                "Maturity Date",
                "Loan Term",
                "Remaining Months to Maturity",
                "Loan Age",
            ]
        )
        + "\n"
        + "L1|N|CL|TEST01|123456789|120000|60000|6.5|20240101|20531201|360|300|24\n"
        + "L2|N|CL|TEST01|123456789|80000|40000|6.5|20240101|20531201|360|300|24\n",
    )
    security_path = _write(
        tmp_path / "security.txt",
        "Pool Prefix|Pool Number|CUSIP|Pass Through Rate|Issue Date|Final Maturity|"
        "Original Balance|Current Balance|Factor|WAC|WAM|WALA|Loan Count\n"
        "CL|TEST01|123456789|5.5|20240101|20531201|200000|100000|0.5|6.5|300|24|2\n",
    )
    loans = parse_loan_file(
        loans_path,
        as_of_date="2026-09-30",
        available_date="2026-10-06",
    )
    securities = parse_security_export(
        security_path,
        as_of_date="2026-09-30",
        available_date="2026-10-06",
    )
    assert len(loans) == 2
    assert loans["current_interest_rate"].eq(0.065).all()
    assert securities.iloc[0]["coupon"] == pytest.approx(0.055)
    result = reconcile_security_snapshot(securities.iloc[0], loans)
    assert result.passed
    assert result.upb_difference == 0
    assert result.wac_difference_bps == pytest.approx(0)


def test_aligned_fannie_file_without_headers(tmp_path):
    values = [""] * len(FANNIE_ALIGNED_LOAN_FIELDS)
    values[0] = "L9"
    values[1] = "N"
    values[2] = "CL"
    values[3] = "TEST01"
    values[4] = "3140QVLM4"
    values[5] = "100000"
    values[6] = "100000"
    values[7] = "90000"
    values[9] = "6.5"
    values[11] = "6.5"
    values[17] = "300"
    values[18] = "24"
    path = _write(tmp_path / "aligned.txt", "|".join(values) + "\n")
    loans = parse_loan_file(path, as_of_date="2026-09-30", available_date="2026-10-06")
    assert len(loans) == 1
    assert loans.iloc[0]["loan_id"] == "L9"
    assert loans.iloc[0]["current_interest_rate"] == pytest.approx(0.065)
    assert loans.iloc[0]["agency"] == "FNMA"


def test_factor_runoff_reconciles_consecutive_months():
    prior_loans = pd.DataFrame({"current_upb": [60_000.0, 40_000.0]})
    current_loans = pd.DataFrame({"current_upb": [59_000.0, 39_000.0]})
    result = reconcile_factor_runoff(
        prior_loans,
        current_loans,
        {"current_upb": 100_000.0},
        {"current_upb": 98_000.0},
    )
    assert result["passed"]
    assert result["security_runoff"] == 2_000.0
    assert result["loan_runoff"] == 2_000.0


def test_point_in_time_corrections_do_not_look_ahead():
    base = pd.DataFrame([{"loan_id": "A", "current_upb": 100.0}])
    corrections = pd.DataFrame(
        [
            {
                "loan_id": "A",
                "current_upb": 90.0,
                "correction_indicator": "Y",
                "available_date": "2026-02-01",
            },
            {
                "loan_id": "A",
                "current_upb": 80.0,
                "correction_indicator": "Y",
                "available_date": "2026-03-01",
            },
        ]
    )
    result = apply_point_in_time_corrections(
        base,
        corrections,
        ["loan_id"],
        pd.Timestamp("2026-02-15"),
    )
    assert result.iloc[0]["current_upb"] == 90.0


def test_manifest_hashes_immutable_source(tmp_path):
    source = _write(tmp_path / "sample.txt", "a|b\n1|2\n")
    manifest = build_manifest(
        source,
        agency="FNMA",
        file_type="monthly_loan_level",
        source_url="https://example.invalid/source",
        publication_date="2026-10-06",
        schema_version="test-v1",
        out_path=tmp_path / "manifest.json",
    )
    assert len(manifest.sha256) == 64
    assert manifest.byte_size == source.stat().st_size
    assert (tmp_path / "manifest.json").exists()
