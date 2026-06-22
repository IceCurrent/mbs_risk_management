"""Precompute small EDA aggregate tables from the large performance panels.

The logistic (329M rows) and CPH (592M rows) panels are far too large to load
into a Jupyter kernel on a 16 GB machine — doing so OOM-kills the kernel. This
script aggregates them out-of-core with DuckDB and writes small parquet files
that notebook 02_eda.ipynb can load cheaply. DuckDB ``COPY ... TO`` streams the
result straight to disk, so this process itself stays well under ~2 GB of real
(anonymous) memory even though it scans ~1 billion rows total.

Run from the repo root (or anywhere):

    python src/build_eda_aggregates.py

Outputs (to data/processed/):
    eda_loan_level.parquet     one row per loan (2021-2025 logistic panel)
    eda_delinq_counts.parquet  delinquency_status value counts (logistic panel)
    eda_zbc_counts.parquet     zero_balance_code value counts (logistic panel)
    eda_vintage.parquet        loan-level default rate by orig quarter (CPH panel)
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / 'data' / 'processed'

LOG = (PROCESSED / 'panel_logistic_2021_2025.parquet').as_posix()
CPH = (PROCESSED / 'panel_cph_2018_2025.parquet').as_posix()

# Strict default flag (spec §4.3): terminal exit code OR 90+ DPD on any monthly row.
DEFAULT_EXPR = (
    "CASE WHEN zero_balance_code IN ('03','06','09') "
    "OR COALESCE(TRY_CAST(delinquency_status AS INTEGER), 0) >= 3 "
    "THEN 1 ELSE 0 END"
)
# Origination quarter from first_payment_date (BIGINT YYYYMM) -> 'YYYYQn'.
QTR = (
    "(first_payment_date // 100)::VARCHAR || 'Q' || "
    "(((first_payment_date % 100) - 1) // 3 + 1)::VARCHAR"
)


def main() -> None:
    con = duckdb.connect()
    # Cap memory hard and spill to disk; keep the scans off the kernel's RAM.
    con.execute("SET memory_limit='1500MB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")

    jobs = {
        'eda_loan_level.parquet': f"""
            COPY (
                WITH base AS (
                    SELECT
                        loan_id,
                        first_payment_date // 100 AS orig_year,
                        {QTR}                     AS orig_quarter,
                        credit_score, ltv, dti, orig_loan_term, property_state,
                        amortization_type, occupancy_status, loan_age,
                        {DEFAULT_EXPR}            AS is_default
                    FROM read_parquet('{LOG}')
                )
                SELECT
                    loan_id,
                    any_value(credit_score)      AS credit_score,
                    any_value(ltv)               AS ltv,
                    any_value(dti)               AS dti,
                    any_value(orig_loan_term)    AS orig_loan_term,
                    any_value(property_state)    AS property_state,
                    any_value(amortization_type) AS amortization_type,
                    any_value(occupancy_status)  AS occupancy_status,
                    any_value(orig_year)         AS orig_year,
                    any_value(orig_quarter)      AS orig_quarter,
                    max(is_default)              AS is_default,
                    max(loan_age)                AS max_loan_age
                FROM base
                GROUP BY loan_id
            ) TO '{{out}}' (FORMAT parquet)
        """,
        'eda_delinq_counts.parquet': f"""
            COPY (
                SELECT delinquency_status, count(*) AS n
                FROM read_parquet('{LOG}') GROUP BY delinquency_status
            ) TO '{{out}}' (FORMAT parquet)
        """,
        'eda_zbc_counts.parquet': f"""
            COPY (
                SELECT zero_balance_code, count(*) AS n
                FROM read_parquet('{LOG}')
                WHERE zero_balance_code IS NOT NULL
                GROUP BY zero_balance_code
            ) TO '{{out}}' (FORMAT parquet)
        """,
        'eda_vintage.parquet': f"""
            COPY (
                WITH loan AS (
                    SELECT loan_id, {QTR} AS orig_quarter, max({DEFAULT_EXPR}) AS deflt
                    FROM read_parquet('{CPH}')
                    GROUP BY loan_id, {QTR}
                )
                SELECT orig_quarter, avg(deflt) * 100 AS default_rate
                FROM loan GROUP BY orig_quarter ORDER BY orig_quarter
            ) TO '{{out}}' (FORMAT parquet)
        """,
    }

    for name, sql in jobs.items():
        out = (PROCESSED / name).as_posix()
        t = time.time()
        con.execute(sql.format(out=out))
        rows = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
        print(f"  wrote {name:<28} {rows:>12,} rows  ({time.time() - t:.0f}s)")

    con.close()
    print("Done.")


if __name__ == '__main__':
    main()
