"""Deterministic security selection and Bloomberg market-coverage gate.

The CUSIPs in ``config/securities.yaml`` are locked before model validation.
This module scores only observable data quality; it never uses model price or
OAS residuals.  A locked security that fails Bloomberg coverage remains in the
report as a failed go/no-go candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "securities.yaml"


def cusip_is_valid(cusip: str) -> bool:
    """Validate the nine-character CUSIP check digit."""
    value = cusip.strip().upper()
    if len(value) != 9:
        return False

    def char_value(char: str) -> int:
        if char.isdigit():
            return int(char)
        if "A" <= char <= "Z":
            return ord(char) - ord("A") + 10
        return {"*": 36, "@": 37, "#": 38}.get(char, -100)

    total = 0
    for index, char in enumerate(value[:8]):
        number = char_value(char)
        if number < 0:
            return False
        if index % 2 == 1:
            number *= 2
        total += number // 10 + number % 10
    return value[8].isdigit() and (10 - total % 10) % 10 == int(value[8])


def load_security_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load and minimally validate the locked security configuration."""
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    securities = config.get("securities", [])
    if len(securities) != 3:
        raise ValueError("Exactly one primary and two holdouts must be locked.")
    if sum(row.get("role") == "primary" for row in securities) != 1:
        raise ValueError("Security config must contain exactly one primary.")
    if sum(row.get("role") == "holdout" for row in securities) != 2:
        raise ValueError("Security config must contain exactly two holdouts.")
    cusips = [str(row["cusip"]).upper() for row in securities]
    if len(set(cusips)) != len(cusips):
        raise ValueError("CUSIPs must be unique.")
    invalid = [cusip for cusip in cusips if not cusip_is_valid(cusip)]
    if invalid:
        raise ValueError(f"Invalid CUSIP check digit: {invalid}")
    return config


def _coverage_by_cusip(market_data: Optional[pd.DataFrame]) -> Dict[str, Dict[str, int]]:
    if market_data is None or market_data.empty:
        return {}
    required = {"cusip", "date", "source", "price"}
    missing = required.difference(market_data.columns)
    if missing:
        raise ValueError(f"Market data missing columns: {sorted(missing)}")
    frame = market_data.copy()
    frame["cusip"] = frame["cusip"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["source"] = frame["source"].astype(str).str.lower()
    frame = frame.dropna(subset=["price"])

    result: Dict[str, Dict[str, int]] = {}
    for cusip, group in frame.groupby("cusip"):
        result[cusip] = {
            "bval_observations": int(group.loc[group["source"] == "bval", "date"].nunique()),
            "quote_observations": int(
                group.loc[group["source"].isin(["bid_ask", "dealer_mid"]), "date"].nunique()
            ),
            "trade_observations": int(
                group.loc[group["source"].isin(["trace", "trade"]), "date"].nunique()
            ),
        }
    return result


def score_securities(
    securities: Iterable[Dict[str, Any]],
    market_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Score observable disclosure and market coverage without model outputs."""
    coverage = _coverage_by_cusip(market_data)
    rows = []
    for security in securities:
        cusip = str(security["cusip"]).upper()
        issue_date = pd.Timestamp(security["issue_date"])
        disclosure_score = 0
        disclosure_score += 15 if cusip_is_valid(cusip) else 0
        disclosure_score += 15 if issue_date >= pd.Timestamp("2019-06-01") else 0
        disclosure_score += 15 if not security.get("is_resecuritization", True) else 0
        disclosure_score += 10 if float(security.get("current_upb", 0)) >= 5_000_000 else 0
        disclosure_score += 10 if int(security.get("loan_count", 0)) >= 50 else 0
        disclosure_score += 10 if 0 < float(security.get("factor", 0)) <= 1 else 0
        disclosure_score += 10 if float(security.get("wac", 0)) > 0 else 0
        disclosure_score += 5 if int(security.get("wam", 0)) > 0 else 0
        disclosure_score += 5 if int(security.get("wala", -1)) >= 0 else 0
        disclosure_score += 5 if str(security.get("prefix", "")) == "CL" else 0

        market = coverage.get(
            cusip,
            {"bval_observations": 0, "quote_observations": 0, "trade_observations": 0},
        )
        dense_bval = market["bval_observations"] >= 252
        dense_quotes = market["quote_observations"] >= 252
        observed_market = (
            market["quote_observations"] >= 20 or market["trade_observations"] >= 10
        )
        # BVAL is not a field on the locked CL pools. Daily bid/ask is sufficient.
        market_ok = observed_market and (dense_bval or dense_quotes)
        if dense_quotes and not dense_bval:
            market_status = "pass_quotes_no_bval"
        elif market_ok:
            market_status = "pass"
        else:
            market_status = "pending_or_fail"
        market_score = (
            min(market["bval_observations"] / 252, 1.0) * 25
            + min(market["quote_observations"] / 252, 1.0) * 10
            + min(market["trade_observations"] / 10, 1.0) * 5
        )
        rows.append(
            {
                "role": security["role"],
                "agency": security["agency"],
                "cusip": cusip,
                "pool_number": security["pool_number"],
                "coupon": float(security["coupon"]),
                "issue_date": issue_date.date().isoformat(),
                "current_upb": float(security["current_upb"]),
                "loan_count": int(security["loan_count"]),
                "disclosure_score": disclosure_score,
                "market_score": round(market_score, 2),
                **market,
                "dense_bval": dense_bval,
                "dense_quotes": dense_quotes,
                "observed_market": observed_market,
                "go_no_go": bool(disclosure_score >= 90 and market_ok),
                "market_status": market_status,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--market-data",
        type=Path,
        help="Optional market observations (parquet or CSV) with cusip,date,source,price.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    config = load_security_config(args.config)
    market = None
    if args.market_data:
        reader = pd.read_parquet if args.market_data.suffix == ".parquet" else pd.read_csv
        market = reader(args.market_data)
    report = score_securities(config["securities"], market)
    print(report.to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(report.to_dict(orient="records"), stream, indent=2)


if __name__ == "__main__":
    main()
