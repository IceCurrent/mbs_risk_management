"""Freddie Mac aligned single-class disclosure parser.

Freddie and Fannie aligned loan-level disclosures share the L-attribute order
under the Single Security initiative.  This module reuses the normalized aligned
parser and changes issuer identity; Freddie-specific legacy files are excluded
from the v1 actual-security universe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .fannie import parse_loan_file as _parse_aligned_loan_file
from .fannie import parse_security_export as _parse_aligned_security_export


def parse_loan_file(
    path: Path,
    as_of_date: str,
    available_date: str,
    selected_cusips: Optional[Iterable[str]] = None,
    member: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize post-Single-Security Freddie loan-level disclosures."""
    result = _parse_aligned_loan_file(
        path=path,
        as_of_date=as_of_date,
        available_date=available_date,
        selected_cusips=selected_cusips,
        member=member,
    )
    result["agency"] = "FHLMC"
    return result


def parse_security_export(
    path: Path,
    as_of_date: str,
    available_date: str,
    member: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize a header-bearing Freddie Security Lookup/Clarity export."""
    result = _parse_aligned_security_export(
        path=path,
        as_of_date=as_of_date,
        available_date=available_date,
        member=member,
    )
    result["agency"] = "FHLMC"
    return result
