"""Precompute small loan-level / pool-level model inputs from the large panels.

The logistic panel (329M rows, 3.7 GB) and the CPH panel (592M rows, 6.7 GB)
cannot be loaded into a Jupyter kernel on a 16 GB machine — ``pd.read_parquet``
followed by a pandas ``groupby`` OOM-kills the kernel. This script performs the
same loan-level / pool-level aggregations **out-of-core** with DuckDB and writes
small parquet files that notebooks 03-05 can load cheaply. DuckDB ``COPY ... TO``
streams results to disk with a hard memory cap, so this process stays well under
~2 GB of real memory even though it scans ~1 billion rows total.

This is the modelling-stage counterpart to ``build_eda_aggregates.py``.

Run from anywhere:

    python src/build_model_aggregates.py

Outputs (to data/processed/):
    loan_level_logistic.parquet   one row per loan, raw fields for logistic FE
                                  (notebook 03 / feature_engineering)
    loan_level_survival.parquet   one row per loan, survival tuples + covariates
                                  (notebook 04 / survival_model)
    pool_cpr_monthly.parquet      monthly pool-level CPR series, FRM loans
                                  (notebook 05 / prepayment)
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

LOG = (PROCESSED / "panel_logistic_2021_2025.parquet").as_posix()
CPH = (PROCESSED / "panel_cph_2018_2025.parquet").as_posix()

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

# Last reporting period as a YYYYMM integer (matches feature_engineering parsing).
LAST_PERIOD = "year(max(reporting_period)) * 100 + month(max(reporting_period))"

# Delinquency ordinal severity (spec §5.3): 0=current .. 3=90+/REO/foreclosure.
# Mirrors feature_engineering.delinquency_ordinal().
DELINQ_ORD = (
    "CASE WHEN delinquency_status IN ('RA','RF','RE','999') THEN 3 "
    "WHEN TRY_CAST(delinquency_status AS INTEGER) IS NOT NULL "
    "THEN LEAST(TRY_CAST(delinquency_status AS INTEGER), 3) "
    "ELSE 0 END"
)


def _loan_level_logistic_sql(out: str) -> str:
    """One row per loan with the raw fields logistic feature engineering needs."""
    return f"""
        COPY (
            WITH base AS (
                SELECT
                    loan_id,
                    credit_score, ltv, cltv, dti, orig_interest_rate,
                    orig_loan_term, property_state, occupancy_status, n_borrowers,
                    {QTR}            AS orig_quarter,
                    loan_age, reporting_period,
                    {DEFAULT_EXPR}   AS is_default
                FROM read_parquet('{LOG}')
            )
            SELECT
                loan_id,
                any_value(credit_score)       AS credit_score,
                any_value(ltv)                AS ltv,
                any_value(cltv)               AS cltv,
                any_value(dti)                AS dti,
                any_value(orig_interest_rate) AS orig_interest_rate,
                any_value(orig_loan_term)     AS orig_loan_term,
                any_value(property_state)     AS property_state,
                any_value(occupancy_status)   AS occupancy_status,
                any_value(n_borrowers)        AS n_borrowers,
                any_value(orig_quarter)       AS orig_quarter,
                max(loan_age)                 AS max_loan_age,
                {LAST_PERIOD}                 AS last_period,
                max(is_default)               AS default
            FROM base
            GROUP BY loan_id
        ) TO '{out}' (FORMAT parquet)
    """


def _loan_level_survival_sql(out: str) -> str:
    """One row per loan with Cox PH survival tuples + raw covariates.

    LOCF imputation for eltv (populated only when delinquent) is done with a
    running ``last_value(... IGNORE NULLS)`` window, then coalesced with cltv —
    matching feature_engineering.build_survival_features().
    """
    return f"""
        COPY (
            WITH base AS (
                SELECT
                    loan_id, loan_age,
                    credit_score, cltv, dti, orig_upb, orig_loan_term,
                    current_interest_rate, current_upb,
                    (occupancy_status = 'I')::INTEGER AS occ_investment,
                    (occupancy_status = 'S')::INTEGER AS occ_second,
                    {DELINQ_ORD} AS delinq_ord,
                    COALESCE(
                        last_value(eltv IGNORE NULLS) OVER (
                            PARTITION BY loan_id ORDER BY loan_age
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ),
                        cltv
                    ) AS eltv_locf
                FROM read_parquet('{CPH}')
            )
            SELECT
                loan_id,
                max(loan_age)                          AS duration,
                (max(delinq_ord) >= 3)::INTEGER        AS event,
                any_value(credit_score)                AS fico,
                any_value(cltv)                        AS orig_cltv,
                any_value(dti)                         AS orig_dti,
                any_value(orig_upb)                    AS orig_upb,
                any_value(orig_loan_term)              AS orig_term,
                any_value(occ_investment)              AS occ_investment,
                any_value(occ_second)                  AS occ_second,
                arg_max(delinq_ord, loan_age)          AS delinq_last,
                arg_max(eltv_locf, loan_age)           AS eltv_last,
                arg_max(current_interest_rate, loan_age) AS current_rate,
                arg_max(current_upb, loan_age)         AS current_upb
            FROM base
            GROUP BY loan_id
        ) TO '{out}' (FORMAT parquet)
    """


def _pool_cpr_monthly_sql(out: str) -> str:
    """Monthly pool-level CPR series for FRM loans (spec §6.2).

    Replicates prepayment.compute_scheduled_principal() + pool_cpr():
    previous-period balance via lag, annuity payment, scheduled principal,
    prepayment = max(balance after scheduled amort - actual balance, 0),
    zeroed on terminal (zero_balance_code set) months; then aggregated by
    reporting period into beg_bal / sched_p / prepay / smm / cpr.
    """
    return f"""
        COPY (
            WITH frm AS (
                SELECT
                    loan_id, loan_age, reporting_period,
                    interest_bearing_upb, orig_upb, current_interest_rate,
                    orig_loan_term, zero_balance_code,
                    lag(interest_bearing_upb) OVER w AS prev_ibupb,
                    row_number() OVER w              AS rn
                FROM read_parquet('{LOG}')
                WHERE amortization_type = 'FRM'
                WINDOW w AS (PARTITION BY loan_id ORDER BY loan_age)
            ),
            calc AS (
                SELECT
                    reporting_period,
                    interest_bearing_upb,
                    zero_balance_code,
                    CASE WHEN rn = 1 THEN orig_upb ELSE prev_ibupb END AS bal_prev,
                    (current_interest_rate / 100.0) / 12.0             AS mrate,
                    greatest(orig_loan_term - loan_age, 1)             AS rem
                FROM frm
            ),
            sched AS (
                SELECT
                    reporting_period,
                    bal_prev,
                    CASE
                        WHEN bal_prev IS NULL THEN NULL
                        WHEN mrate = 0.0 THEN bal_prev / rem
                        ELSE least(
                            bal_prev * (mrate * power(1.0 + mrate, rem))
                                     / (power(1.0 + mrate, rem) - 1.0)
                                     - bal_prev * mrate,
                            bal_prev
                        )
                    END AS sched_prin,
                    interest_bearing_upb,
                    zero_balance_code
                FROM calc
            ),
            prepay AS (
                SELECT
                    reporting_period,
                    bal_prev,
                    sched_prin,
                    CASE
                        WHEN zero_balance_code IS NOT NULL THEN 0.0
                        ELSE greatest(
                            (bal_prev - sched_prin) - interest_bearing_upb, 0.0
                        )
                    END AS prepayment
                FROM sched
            ),
            monthly AS (
                SELECT
                    reporting_period,
                    sum(bal_prev)   AS beg_bal,
                    sum(sched_prin) AS sched_p,
                    sum(prepayment) AS prepay
                FROM prepay
                GROUP BY reporting_period
            )
            SELECT
                year(reporting_period) * 100 + month(reporting_period) AS reporting_period,
                beg_bal, sched_p, prepay,
                prepay / greatest(beg_bal - sched_p, 1.0)              AS smm,
                1.0 - power(1.0 - prepay / greatest(beg_bal - sched_p, 1.0), 12) AS cpr
            FROM monthly
            ORDER BY reporting_period
        ) TO '{out}' (FORMAT parquet)
    """


def main() -> None:
    con = duckdb.connect()
    # Cap memory hard and spill to disk; keep the scans off the kernel's RAM.
    con.execute("SET memory_limit='2000MB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")

    jobs = {
        "loan_level_logistic.parquet": _loan_level_logistic_sql,
        "loan_level_survival.parquet": _loan_level_survival_sql,
        "pool_cpr_monthly.parquet":    _pool_cpr_monthly_sql,
    }

    for name, sql_fn in jobs.items():
        out = (PROCESSED / name).as_posix()
        t = time.time()
        con.execute(sql_fn(out))
        rows = con.execute(
            f"SELECT count(*) FROM read_parquet('{out}')"
        ).fetchone()[0]
        print(f"  wrote {name:<30} {rows:>12,} rows  ({time.time() - t:.0f}s)")

    con.close()
    print("Done.")


if __name__ == "__main__":
    main()
