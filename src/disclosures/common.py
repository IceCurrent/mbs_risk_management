"""Shared utilities for immutable, point-in-time Agency MBS disclosures."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class DisclosureManifest:
    """Provenance recorded beside every raw disclosure file."""

    agency: str
    file_type: str
    source_url: str
    local_path: str
    sha256: str
    byte_size: int
    publication_date: str
    downloaded_at_utc: str
    schema_version: str


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    path: Path,
    agency: str,
    file_type: str,
    source_url: str,
    publication_date: str,
    schema_version: str,
    out_path: Optional[Path] = None,
) -> DisclosureManifest:
    """Create and optionally persist the immutable-file manifest.

    Rebuilding a manifest for an unchanged file keeps its original download
    time; only a new hash gets a new timestamp.
    """
    digest = sha256_file(path)
    downloaded_at = datetime.now(timezone.utc).isoformat()
    if out_path and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous.get("sha256") == digest:
            downloaded_at = previous["downloaded_at_utc"]
    manifest = DisclosureManifest(
        agency=agency.upper(),
        file_type=file_type,
        source_url=source_url,
        local_path=str(path),
        sha256=digest,
        byte_size=path.stat().st_size,
        publication_date=pd.Timestamp(publication_date).date().isoformat(),
        downloaded_at_utc=downloaded_at,
        schema_version=schema_version,
    )
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as stream:
            json.dump(asdict(manifest), stream, indent=2)
    return manifest


def _open_text_source(path: Path, member: Optional[str] = None):
    """Open a plain or zipped text source; callers close the returned handles."""
    # pylint: disable=consider-using-with
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        names = archive.namelist()
        if member is None:
            text_members = [name for name in names if name.lower().endswith((".txt", ".csv"))]
            if len(text_members) != 1:
                archive.close()
                raise ValueError(
                    f"{path} contains {len(text_members)} text files; specify member explicitly."
                )
            member = text_members[0]
        return archive, archive.open(member, "r")
    return None, open(path, "rb")


def first_record(path: Path, member: Optional[str] = None) -> list[str]:
    archive, stream = _open_text_source(path, member)
    try:
        raw = stream.readline().decode("utf-8-sig", errors="replace").rstrip("\r\n")
        return raw.split("|")
    finally:
        stream.close()
        if archive:
            archive.close()


def read_pipe_records(
    path: Path,
    columns: Optional[Sequence[str]] = None,
    member: Optional[str] = None,
    header: str | int | None = "infer",
    dtype: str | dict = "string",
) -> pd.DataFrame:
    """Read a pipe-delimited plain/zip source without silently dropping strings."""
    archive, stream = _open_text_source(path, member)
    try:
        if columns is not None:
            frame = pd.read_csv(
                stream,
                sep="|",
                header=None,
                names=list(columns),
                dtype=dtype,
                keep_default_na=False,
                na_values=["", " ", "null", "NULL"],
                low_memory=False,
            )
        else:
            frame = pd.read_csv(
                stream,
                sep="|",
                header=header,
                dtype=dtype,
                keep_default_na=False,
                na_values=["", " ", "null", "NULL"],
                low_memory=False,
            )
    finally:
        stream.close()
        if archive:
            archive.close()
    return frame


def normalized_column(name: str) -> str:
    value = str(name).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def normalize_headers(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [normalized_column(column) for column in result.columns]
    return result


def coalesce_aliases(
    frame: pd.DataFrame,
    aliases: dict[str, Iterable[str]],
    required: Iterable[str] = (),
) -> pd.DataFrame:
    """Rename the first available issuer/export alias to each canonical name."""
    result = normalize_headers(frame)
    rename = {}
    for canonical, candidates in aliases.items():
        normalized = [normalized_column(candidate) for candidate in candidates]
        available = [candidate for candidate in normalized if candidate in result.columns]
        if available:
            rename[available[0]] = canonical
    result = result.rename(columns=rename)
    missing = set(required).difference(result.columns)
    if missing:
        raise ValueError(f"Disclosure missing required fields: {sorted(missing)}")
    return result


def parse_date_series(series: pd.Series) -> pd.Series:
    """Parse aligned disclosure dates, accepting YYYYMMDD and ISO forms."""
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, errors="coerce")
    compact = parsed.isna() & text.str.fullmatch(r"\d{8}", na=False)
    parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return parsed


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.replace(",", "", regex=False), errors="coerce")


def rate_series(series: pd.Series) -> pd.Series:
    """Convert disclosed rates to decimals; values over one are interpreted as percent."""
    values = numeric_series(series)
    return values.where(values.abs() <= 1.0, values / 100.0)


def apply_point_in_time_corrections(
    base: pd.DataFrame,
    corrections: pd.DataFrame,
    key_columns: Sequence[str],
    available_date: pd.Timestamp,
    indicator_column: str = "correction_indicator",
) -> pd.DataFrame:
    """Apply only corrections published on or before ``available_date``.

    Supported indicators: A (add), D (delete), Y (replace/update), N (unchanged).
    Later corrections remain excluded, preserving the historical information set.
    """
    if corrections.empty:
        return base.copy()
    required = {*key_columns, indicator_column, "available_date"}
    missing = required.difference(corrections.columns)
    if missing:
        raise ValueError(f"Corrections missing fields: {sorted(missing)}")
    eligible = corrections.loc[
        pd.to_datetime(corrections["available_date"]) <= pd.Timestamp(available_date)
    ].copy()
    if eligible.empty:
        return base.copy()
    eligible["available_date"] = pd.to_datetime(eligible["available_date"])
    eligible = eligible.sort_values("available_date")

    result = base.copy().set_index(list(key_columns), drop=False)
    for _, correction in eligible.iterrows():
        key = tuple(correction[column] for column in key_columns)
        if len(key_columns) == 1:
            key = key[0]
        indicator = str(correction[indicator_column]).upper()
        if indicator == "D":
            result = result.drop(index=key, errors="ignore")
            continue
        if indicator not in {"A", "Y", "N"}:
            raise ValueError(f"Unknown correction indicator: {indicator}")
        if indicator == "N":
            continue
        values = correction.drop(
            labels=[indicator_column, "available_date"], errors="ignore"
        ).to_dict()
        result.loc[key, list(values)] = list(values.values())
    return result.reset_index(drop=True)
