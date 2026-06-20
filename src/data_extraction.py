"""
src/data_extraction.py

Parse Freddie Mac SF LLD origination and performance files directly
from their quarterly zip archives into parquet files in data/processed/.

Supports all years 2018-2025. Each quarter is saved individually, then
concatenated into full origination_all.parquet and performance_all.parquet.
The final step creates model-specific filtered panels (spec §4.6).
"""

import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Project root — this file lives at src/, one level below project root.
PROJECT_ROOT = Path(__file__).parent.parent

# ── Column schemas — spec §4.2 (origination) ─────────────────────────────────

ORIG_COLS: List[str] = [
    "credit_score",        # 0  FICO at origination; 9999 = missing
    "first_payment_date",  # 1  YYYYMM
    "first_time_homebuyer",# 2  Y/N/U
    "maturity_date",       # 3  YYYYMM
    "msa_code",            # 4  MSA code
    "mi_pct",              # 5  MI coverage %
    "n_units",             # 6  Number of units
    "occupancy_status",    # 7  O/S/I
    "cltv",                # 8  Combined LTV at origination
    "dti",                 # 9  DTI; 999 = missing
    "orig_upb",            # 10 Original UPB ($)
    "ltv",                 # 11 LTV at origination
    "orig_interest_rate",  # 12 Note rate (%)
    "channel",             # 13 R/B/C/T
    "prepay_penalty",      # 14 Y/N
    "amortization_type",   # 15 FRM/ARM
    "property_state",      # 16 2-letter state
    "property_type",       # 17 SF/CO/PU/MH
    "zipcode",             # 18 3-digit prefix
    "loan_id",             # 19 12-char sequence number (primary key)
    "loan_purpose",        # 20 P/C/N
    "orig_loan_term",      # 21 Months (typically 360)
    "n_borrowers",         # 22 Number of borrowers
    "seller_name",         # 23
    "servicer_name",       # 24
    "super_conforming",    # 25
    "pre_harp_loan_id",    # 26
    "program_indicator",   # 27
    "harp_indicator",      # 28
    "property_val_method", # 29
    "interest_only",       # 30 Y/N
    "mi_cancellation",     # 31
]

# ── Column schemas — spec §4.3 (performance) ──────────────────────────────────

PERF_COLS: List[str] = [
    "loan_id",                      # 0  FK to origination
    "reporting_period",             # 1  YYYYMM
    "current_upb",                  # 2  Current actual UPB ($)
    "delinquency_status",           # 3  '0','1','2','3','RA'
    "loan_age",                     # 4  Months since first payment
    "months_remaining",             # 5  Remaining months to maturity
    "repurchase_date",              # 6  YYYYMM
    "modification_flag",            # 7  Y/N
    "zero_balance_code",            # 8  '01','02','03','06','09'
    "zero_balance_date",            # 9  YYYYMM
    "current_interest_rate",        # 10 Current note rate (%)
    "current_deferred_upb",         # 11 Deferred UPB (modified loans)
    "due_date_last_paid",           # 12 MMYYYY
    "mi_recoveries",                # 13
    "net_sale_proceeds",            # 14
    "non_mi_recoveries",            # 15
    "expenses",                     # 16
    "legal_costs",                  # 17
    "maintenance_costs",            # 18
    "taxes_insurance",              # 19
    "misc_costs",                   # 20
    "actual_loss",                  # 21
    "modification_cost",            # 22
    "step_modification",            # 23
    "deferred_payment_mod",         # 24
    "eltv",                         # 25 Estimated LTV (AVM; delinquent only)
    "zero_balance_removal_upb",     # 26
    "delinquency_accrued_interest", # 27
    "disaster_flag",                # 28
    "borrower_assistance",          # 29
    "monthly_mod_cost",             # 30
    "interest_bearing_upb",         # 31 UPB excluding deferred principal
]

EXPECTED_N_COLS: int = 32

# ── Per-column dtype overrides — reduce memory footprint ─────────────────────

ORIG_DTYPES: Dict[str, str] = {
    "credit_score":       "Int32",   # nullable int; sentinel 9999 replaced later
    "msa_code":           "Int32",
    "n_units":            "Int8",
    "cltv":               "float32",
    "dti":                "float32", # sentinel 999 replaced later
    "orig_upb":           "float32",
    "ltv":                "float32",
    "orig_interest_rate": "float32",
    "orig_loan_term":     "Int16",
}

PERF_DTYPES: Dict[str, str] = {
    "current_upb":           "float32",
    "delinquency_status":    "str",    # can be '0','1','2','3','RA' — must be str
    "loan_age":              "Int16",
    "months_remaining":      "Int16",
    "zero_balance_code":     "str",    # '01','02','03','06','09' — leading zeros
    "current_interest_rate": "float32",
    "current_deferred_upb":  "float32",
    "eltv":                  "float32",
    "interest_bearing_upb":  "float32",
    "mi_recoveries":         "float32",
    "net_sale_proceeds":     "float32",
    "actual_loss":           "float32",
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_pipe_delimited(
    archive: zipfile.ZipFile,
    member_name: str,
    col_names: List[str],
    dtypes: Dict[str, str],
    label: str,
) -> pd.DataFrame:
    """
    Read a pipe-delimited member of an open ZipFile into a DataFrame.

    Validates that the column count matches EXPECTED_N_COLS before returning.
    Raises ValueError on schema mismatch — caller must not proceed.

    Parameters
    ----------
    archive : zipfile.ZipFile
    member_name : str  Name of the file inside the zip archive.
    col_names : list   Ordered column names to assign.
    dtypes : dict      Column-level dtype overrides.
    label : str        Human-readable identifier for error messages.

    Returns
    -------
    pd.DataFrame
    """
    try:
        with archive.open(member_name) as file_handle:
            frame = pd.read_csv(
                file_handle,
                sep="|",
                header=None,
                names=col_names,
                dtype=dtypes,
                low_memory=False,
                na_values=["", " ", "X"],
                keep_default_na=True,
            )
    except KeyError as exc:
        raise FileNotFoundError(
            f"Member '{member_name}' not found in archive. "
            "Confirm you downloaded Full Standard Files (not Sample Files)."
        ) from exc

    actual = frame.shape[1]
    if actual != EXPECTED_N_COLS:
        raise ValueError(
            f"Column count mismatch in {label}: "
            f"expected {EXPECTED_N_COLS}, got {actual}. "
            "Stop — do not proceed with misaligned columns (spec §4.2/§4.3)."
        )

    return frame


def _find_zip_files(years: range) -> List[Tuple[int, int, Path]]:
    """
    Locate all quarterly zip files for the given years.

    Looks in PROJECT_ROOT/historical_data_{YYYY}/ for each year.
    Logs a warning for missing years; silently skips missing quarters
    (e.g., 2025Q4 not yet released).

    Parameters
    ----------
    years : range  e.g. range(2018, 2026) for 2018-2025.

    Returns
    -------
    List of (year, quarter, zip_path) sorted chronologically.
    """
    results = []
    for year in years:
        year_dir = PROJECT_ROOT / f"historical_data_{year}"
        if not year_dir.exists():
            log.warning("Directory not found, skipping year %d: %s", year, year_dir)
            continue
        for qtr in range(1, 5):
            zip_path = year_dir / f"historical_data_{year}Q{qtr}.zip"
            if zip_path.exists():
                results.append((year, qtr, zip_path))
            else:
                log.debug("Not found (may not be released): %s", zip_path)
    return results


# ── Phase 2a — Origination parsing ───────────────────────────────────────────

def parse_origination_quarter(
    year: int,
    quarter: int,
    zip_path: Path,
    out_dir: Path,
) -> Path:
    """
    Parse the origination file for one quarter and save as parquet.

    Replaces sentinel values: credit_score 9999 → NaN, dti 999 → NaN.
    Skips writing if the output file already exists.

    Parameters
    ----------
    year : int
    quarter : int
    zip_path : Path  Path to the quarterly zip archive.
    out_dir : Path   Directory to write the output parquet file.

    Returns
    -------
    Path  Path to the written (or pre-existing) parquet file.
    """
    tag = f"{year}Q{quarter}"
    out_path = out_dir / f"origination_{tag}.parquet"

    if out_path.exists():
        log.info("Already exists, skipping: %s", out_path)
        return out_path

    member = f"historical_data_{year}Q{quarter}.txt"
    label = f"origination_{tag}"

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            frame = _read_pipe_delimited(archive, member, ORIG_COLS, ORIG_DTYPES, label)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse {label} from {zip_path}: {exc}"
        ) from exc

    # Replace sentinel values with NaN
    frame["credit_score"] = frame["credit_score"].where(
        frame["credit_score"] != 9999, other=pd.NA
    )
    frame["dti"] = frame["dti"].where(frame["dti"] != 999, other=np.nan)

    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    log.info("Saved %s: %d rows → %s", label, len(frame), out_path.name)
    return out_path


def parse_all_origination(
    years: range = range(2018, 2026),
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Parse all origination files for the given years into quarterly parquets.

    Saves each quarter individually, then concatenates into
    origination_all.parquet. Already-processed quarters are skipped.

    Parameters
    ----------
    years : range  Years to process (default 2018-2025).
    out_dir : Path  Output directory (default data/processed/).

    Returns
    -------
    Path  Path to origination_all.parquet.
    """
    if out_dir is None:
        out_dir = PROJECT_ROOT / "data" / "processed"

    zip_files = _find_zip_files(years)
    if not zip_files:
        raise FileNotFoundError(
            f"No zip files found for years {list(years)}. "
            "Expected: PROJECT_ROOT/historical_data_YYYY/historical_data_YYYYQn.zip"
        )

    paths = []
    for year, qtr, zip_path in zip_files:
        paths.append(parse_origination_quarter(year, qtr, zip_path, out_dir))

    combined = pd.concat(
        [pd.read_parquet(p) for p in paths], ignore_index=True
    )
    all_path = out_dir / "origination_all.parquet"
    combined.to_parquet(all_path, index=False)
    log.info(
        "origination_all.parquet: %d rows, %d cols → %s",
        *combined.shape, all_path.name,
    )
    return all_path


# ── Phase 2b — Performance parsing ───────────────────────────────────────────

def parse_performance_quarter(
    year: int,
    quarter: int,
    zip_path: Path,
    out_dir: Path,
) -> Path:
    """
    Parse the performance file for one quarter and save as parquet.

    Skips writing if the output file already exists.

    Parameters
    ----------
    year : int
    quarter : int
    zip_path : Path  Path to the quarterly zip archive.
    out_dir : Path   Directory to write the output parquet file.

    Returns
    -------
    Path  Path to the written (or pre-existing) parquet file.
    """
    tag = f"{year}Q{quarter}"
    out_path = out_dir / f"performance_{tag}.parquet"

    if out_path.exists():
        log.info("Already exists, skipping: %s", out_path)
        return out_path

    member = f"historical_data_time_{year}Q{quarter}.txt"
    label = f"performance_{tag}"

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            frame = _read_pipe_delimited(
                archive, member, PERF_COLS, PERF_DTYPES, label
            )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse {label} from {zip_path}: {exc}"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    log.info("Saved %s: %d rows → %s", label, len(frame), out_path.name)
    return out_path


def parse_all_performance(
    years: range = range(2018, 2026),
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Parse all performance files for the given years into quarterly parquets.

    Saves each quarter individually, then concatenates into
    performance_all.parquet. Already-processed quarters are skipped.

    Parameters
    ----------
    years : range  Years to process (default 2018-2025).
    out_dir : Path  Output directory (default data/processed/).

    Returns
    -------
    Path  Path to performance_all.parquet.
    """
    if out_dir is None:
        out_dir = PROJECT_ROOT / "data" / "processed"

    zip_files = _find_zip_files(years)
    if not zip_files:
        raise FileNotFoundError(
            f"No zip files found for years {list(years)}. "
            "Expected: PROJECT_ROOT/historical_data_YYYY/historical_data_YYYYQn.zip"
        )

    paths = []
    for year, qtr, zip_path in zip_files:
        paths.append(parse_performance_quarter(year, qtr, zip_path, out_dir))

    combined = pd.concat(
        [pd.read_parquet(p) for p in paths], ignore_index=True
    )
    all_path = out_dir / "performance_all.parquet"
    combined.to_parquet(all_path, index=False)
    log.info(
        "performance_all.parquet: %d rows, %d cols → %s",
        *combined.shape, all_path.name,
    )
    return all_path


# ── Phase 2c — Merge panel and model-specific subsets ────────────────────────

def build_merged_panel(
    orig_path: Optional[Path] = None,
    perf_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Join origination (static, one row per loan) with performance
    (monthly, many rows per loan) on loan_id.

    The join is a left join on the performance side — every performance
    row is kept and origination fields are broadcast to all its monthly
    observations. Saves as merged_panel.parquet.

    Parameters
    ----------
    orig_path : Path  origination_all.parquet (default data/processed/)
    perf_path : Path  performance_all.parquet (default data/processed/)
    out_dir : Path    Output directory (default data/processed/)

    Returns
    -------
    Path  Path to merged_panel.parquet.
    """
    if out_dir is None:
        out_dir = PROJECT_ROOT / "data" / "processed"
    if orig_path is None:
        orig_path = out_dir / "origination_all.parquet"
    if perf_path is None:
        perf_path = out_dir / "performance_all.parquet"

    try:
        orig = pd.read_parquet(orig_path)
        perf = pd.read_parquet(perf_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Input file not found: {exc}. "
            "Run parse_all_origination() and parse_all_performance() first."
        ) from exc

    # Normalize loan_id to plain string in both tables
    orig["loan_id"] = orig["loan_id"].astype(str).str.strip()
    perf["loan_id"] = perf["loan_id"].astype(str).str.strip()

    panel = perf.merge(orig, on="loan_id", how="left", suffixes=("", "_orig"))

    out_path = out_dir / "merged_panel.parquet"
    panel.to_parquet(out_path, index=False)
    log.info("merged_panel.parquet: %d rows, %d cols", *panel.shape)
    return out_path


def create_model_subsets(
    panel_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Create model-specific filtered panels from the merged panel (spec §4.6).

    Applies date filters immediately after loading, before any feature
    engineering or modeling. Each notebook loads its own pre-filtered file.

    Saves:
      panel_logistic_2021_2025.parquet  — logistic regression + PSA model
        (2021–2025 originations; richest macro-regime variation)
      panel_cph_2018_2025.parquet       — Cox PH survival model
        (2018–2025 originations; enough time to observe full default hump)

    Parameters
    ----------
    panel_path : Path  merged_panel.parquet (default data/processed/)
    out_dir : Path     Output directory (default data/processed/)

    Returns
    -------
    dict  {'logistic': Path, 'cph': Path}
    """
    if out_dir is None:
        out_dir = PROJECT_ROOT / "data" / "processed"
    if panel_path is None:
        panel_path = out_dir / "merged_panel.parquet"

    try:
        panel = pd.read_parquet(panel_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Merged panel not found: {exc}. Run build_merged_panel() first."
        ) from exc

    panel["reporting_period"] = pd.to_datetime(
        panel["reporting_period"].astype(str), format="%Y%m", errors="coerce"
    )
    panel["orig_quarter"] = pd.to_datetime(
        panel["first_payment_date"].astype(str), format="%Y%m", errors="coerce"
    ).dt.to_period("Q")

    # Logistic regression + PSA: 2021-2025 originations (spec §4.6)
    panel_logistic = panel[panel["orig_quarter"] >= "2021Q1"].copy()
    logistic_path = out_dir / "panel_logistic_2021_2025.parquet"
    panel_logistic.to_parquet(logistic_path, index=False)
    log.info("panel_logistic_2021_2025.parquet: %d rows", len(panel_logistic))

    # Cox PH: 2018-2025 originations (spec §4.6)
    panel_cph = panel[panel["orig_quarter"] >= "2018Q1"].copy()
    cph_path = out_dir / "panel_cph_2018_2025.parquet"
    panel_cph.to_parquet(cph_path, index=False)
    log.info("panel_cph_2018_2025.parquet: %d rows", len(panel_cph))

    return {"logistic": logistic_path, "cph": cph_path}
