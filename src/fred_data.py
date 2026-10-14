"""Download and load the FRED series used for the curve and mortgage-rate model.

Treasury constant-maturity (CMT) par yields are daily; the Freddie Mac
Primary Mortgage Market Survey (PMMS) 30-year rate is weekly (Thursday
release, survey taken Monday-Wednesday). Values are stored in percent as
published and returned in decimals.
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FRED_DIR = ROOT / "data" / "fred"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"

CMT_TENORS = {
    "DGS1MO": 1.0 / 12.0,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}
PMMS_SERIES = "MORTGAGE30US"
ALL_SERIES = (*CMT_TENORS, PMMS_SERIES)


def download_series(
    series_ids=ALL_SERIES,
    out_dir: Path = FRED_DIR,
    start: str = "2010-01-01",
) -> dict:
    """Fetch FRED CSVs (no API key needed) and record when they were pulled."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for series in series_ids:
        url = FRED_CSV_URL.format(series=series, start=start)
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode("utf-8")
        frame = pd.read_csv(io.StringIO(text))
        if list(frame.columns) != ["observation_date", series]:
            raise ValueError(f"Unexpected FRED layout for {series}: {list(frame.columns)}")
        (out_dir / f"{series}.csv").write_text(text, encoding="utf-8")
        manifest[series] = {
            "url": url,
            "rows": int(len(frame)),
            "last_date": str(frame["observation_date"].iloc[-1]),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def load_series(series: str, fred_dir: Path = FRED_DIR) -> pd.Series:
    """One FRED series in decimals, indexed by observation date, gaps dropped."""
    path = fred_dir / f"{series}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python src/fred_data.py")
    frame = pd.read_csv(path, parse_dates=["observation_date"], na_values=".")
    values = frame.set_index("observation_date")[series].dropna() / 100.0
    values.index.name = "date"
    values.name = series
    return values


def load_cmt_panel(fred_dir: Path = FRED_DIR) -> pd.DataFrame:
    """Daily CMT par yields (decimal); columns are tenors in years.

    Only dates on which every tenor printed are kept, so each row is a
    complete curve.
    """
    panel = pd.concat({tenor: load_series(series, fred_dir) for series, tenor in CMT_TENORS.items()}, axis=1)
    return panel.dropna().sort_index()


def load_pmms(fred_dir: Path = FRED_DIR) -> pd.Series:
    """Weekly PMMS 30-year fixed rate (decimal), indexed by Thursday release."""
    return load_series(PMMS_SERIES, fred_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--out", type=Path, default=FRED_DIR)
    args = parser.parse_args()
    manifest = download_series(out_dir=args.out, start=args.start)
    for series, meta in manifest.items():
        print(f"{series:13s} rows={meta['rows']:6d} last={meta['last_date']}")


if __name__ == "__main__":
    main()
