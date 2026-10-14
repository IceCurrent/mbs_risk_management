"""Build point-in-time actual-security snapshots from issuer disclosures."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disclosures.common import build_manifest  # noqa: E402
from disclosures.quality import reconcile_security_snapshot  # noqa: E402
from disclosures import fannie, freddie  # noqa: E402
from dates import factor_publication_date  # noqa: E402

CONFIG = ROOT / "config" / "securities.yaml"


def _locked_cusips(agency: str) -> list[str]:
    with open(CONFIG, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return [
        str(security["cusip"]).upper()
        for security in config["securities"]
        if security["agency"].upper() == agency.upper()
    ]


def _requested_cusips(agency: str, cusips: list[str] | None) -> list[str]:
    locked = _locked_cusips(agency)
    if not cusips:
        return locked
    requested = [str(cusip).upper() for cusip in cusips]
    unknown = sorted(set(requested).difference(locked))
    if unknown:
        raise ValueError(f"CUSIPs are not in the locked {agency} universe: {unknown}")
    return requested


def build_snapshots(
    agency: str,
    loan_file: Path,
    security_file: Path,
    as_of_date: str,
    available_date: str,
    output_dir: Path,
    source_url: str,
    schema_version: str = "aligned-single-class-v1",
    cusips: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse, reconcile and persist selected CUSIP snapshots."""
    parser = fannie if agency.upper() == "FNMA" else freddie
    cusips = _requested_cusips(agency, cusips)
    loans = parser.parse_loan_file(
        loan_file,
        as_of_date=as_of_date,
        available_date=available_date,
        selected_cusips=cusips,
    )
    securities = parser.parse_security_export(
        security_file,
        as_of_date=as_of_date,
        available_date=available_date,
    )
    securities = securities.loc[securities["cusip"].isin(cusips)].reset_index(drop=True)
    if set(securities["cusip"]) != set(cusips):
        missing = sorted(set(cusips).difference(securities["cusip"]))
        raise ValueError(f"Security disclosure missing locked CUSIPs: {missing}")

    reports = []
    for _, security in securities.iterrows():
        reports.append(
            asdict(
                reconcile_security_snapshot(
                    security,
                    loans,
                    raise_on_failure=True,
                )
            )
        )
    quality = pd.DataFrame(reports)

    partition = pd.Timestamp(as_of_date).strftime("%Y%m")
    output_dir.mkdir(parents=True, exist_ok=True)
    loans.to_parquet(output_dir / f"loan_snapshots_{partition}.parquet", index=False)
    securities.to_parquet(output_dir / f"security_master_{partition}.parquet", index=False)
    quality.to_parquet(output_dir / f"reconciliation_{partition}.parquet", index=False)

    manifest_dir = output_dir / "manifests" / partition
    build_manifest(
        loan_file,
        agency,
        "monthly_loan_level",
        source_url,
        publication_date=available_date,
        schema_version=schema_version,
        out_path=manifest_dir / f"{loan_file.name}.manifest.json",
    )
    build_manifest(
        security_file,
        agency,
        "monthly_security",
        source_url,
        publication_date=available_date,
        schema_version=schema_version,
        out_path=manifest_dir / f"{security_file.name}.manifest.json",
    )
    return securities, loans, quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agency", choices=["FNMA", "FHLMC"], required=True)
    parser.add_argument("--loan-file", type=Path, required=True)
    parser.add_argument("--security-file", type=Path, required=True)
    parser.add_argument("--as-of", required=True, help="Collateral month end, YYYY-MM-DD.")
    parser.add_argument(
        "--available",
        help="Issuer publication date. Default: fourth business day of the factor month.",
    )
    parser.add_argument("--source-url", required=True)
    parser.add_argument(
        "--cusip",
        action="append",
        dest="cusips",
        help="Locked CUSIP to ingest. Repeatable. Default: all locked CUSIPs for the agency.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "market" / "processed" / "disclosures",
    )
    args = parser.parse_args()
    available = args.available or str(factor_publication_date(args.as_of))
    securities, loans, quality = build_snapshots(
        args.agency,
        args.loan_file,
        args.security_file,
        args.as_of,
        available,
        args.output,
        args.source_url,
        cusips=args.cusips,
    )
    print(
        json.dumps(
            {
                "securities": len(securities),
                "loans": len(loans),
                "passed": int(quality["passed"].sum()),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
