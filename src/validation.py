"""Out-of-sample checks that do not reuse the data they test.

1. Prepayment, rolling origin: for each of the last ``test_months`` activity
   months, refit on earlier months only and forecast that month for every
   pool. Compared with a pooled S-curve (no pool floors) and with each
   pool's trailing six-month speed.
2. Rate sensitivity: weekly regression of the pool's clean-price return on
   Treasury yield changes, HAC errors. -beta is an empirical duration; it
   also absorbs any co-movement of spreads with rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from prepayment import cpr_to_smm, fit_scurve, scurve_cpr, smm_to_cpr

LEVEL_TENORS = (2.0, 5.0, 10.0, 30.0)


def _binomial_deviance(k: np.ndarray, n: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(k > 0, k * np.log(k / (n * p)), 0.0)
        t2 = np.where(n - k > 0, (n - k) * np.log((n - k) / (n * (1 - p))), 0.0)
    return float(2.0 * np.sum(t1 + t2))


def _predict(fit, rows: pd.DataFrame) -> np.ndarray:
    floors = rows["cusip"].map(fit.floors).to_numpy(dtype=float)
    cpr = scurve_cpr(rows["incentive"].to_numpy(), floors, fit.ceiling, fit.midpoint, fit.width)
    return cpr_to_smm(cpr)


def rolling_origin_prepay_test(panel: pd.DataFrame, cusips: list, ceiling: float,
                               test_months: int = 6, naive_window: int = 6) -> dict:
    months = np.sort(panel["activity_month"].unique())
    records = []
    for month in months[-test_months:]:
        train = panel.loc[panel["activity_month"] < month]
        test = panel.loc[panel["activity_month"] == month]
        model = fit_scurve(train, ceiling, cusips)
        pooled = fit_scurve(train.assign(cusip="pooled"), ceiling, ["pooled"])
        pooled_pred = _predict(pooled, test.assign(cusip="pooled"))
        model_pred = _predict(model, test)
        for (_, row), p_model, p_pooled in zip(test.iterrows(), model_pred, pooled_pred):
            hist = train.loc[train["cusip"] == row["cusip"]].tail(naive_window)
            naive = float(np.clip(hist["k"].sum() / hist["n_eff"].sum(), 1e-6, None))
            records.append(
                {
                    "activity_month": row["activity_month"],
                    "cusip": row["cusip"],
                    "incentive": row["incentive"],
                    "n_eff": row["n_eff"],
                    "k": row["k"],
                    "cpr_actual": float(smm_to_cpr(max(row["smm"], 0.0))),
                    "cpr_model": float(smm_to_cpr(p_model)),
                    "cpr_pooled": float(smm_to_cpr(p_pooled)),
                    "cpr_naive": float(smm_to_cpr(naive)),
                    "smm_model": p_model,
                    "smm_pooled": p_pooled,
                    "smm_naive": naive,
                }
            )
    table = pd.DataFrame(records)
    summary = {"test_months": int(test_months), "forecasts": int(len(table))}
    for name in ("model", "pooled", "naive"):
        summary[name] = {
            "deviance": _binomial_deviance(table["k"].to_numpy(), table["n_eff"].to_numpy(),
                                           table[f"smm_{name}"].to_numpy()),
            "mae_cpr": float((table[f"cpr_{name}"] - table["cpr_actual"]).abs().mean()),
            "bias_cpr": float((table[f"cpr_{name}"] - table["cpr_actual"]).mean()),
        }
    return {"table": table, "summary": summary}


def score_month_ahead(forecast: dict, realized_factor: float) -> dict:
    """Probability integral transform of the published factor under the forecast.

    PIT near 0 or 1 means the realised factor fell in a tail; across many
    months a calibrated forecast gives uniform PITs.
    """
    quantiles = np.asarray(forecast["factor_quantiles"], dtype=float)
    levels = np.arange(1, 100) / 100.0
    pit = float(np.interp(realized_factor, quantiles, levels, left=0.0, right=1.0))
    lo, hi = forecast["factor_pi90"]
    return {
        "realized_factor": realized_factor,
        "point_factor": forecast["point_factor"],
        "error": realized_factor - forecast["point_factor"],
        "pit": pit,
        "inside_pi90": bool(lo <= realized_factor <= hi),
    }


def _weekly(series: pd.Series) -> pd.Series:
    return series.sort_index().resample("W-FRI").last().dropna()


def empirical_duration(quotes: pd.DataFrame, cmt: pd.DataFrame, trade_date,
                       windows=(26, 52, None)) -> dict:
    """-d ln(P) / dy from weekly clean mids, against the 10Y and a level factor."""
    prices = quotes.set_index("date")["price"]
    prices = prices.loc[prices.index <= pd.Timestamp(trade_date)]
    common = prices.index.intersection(cmt.index)
    frame = pd.DataFrame(
        {
            "log_price": np.log(prices.loc[common]),
            "y10": cmt.loc[common, 10.0],
            "level": cmt.loc[common, list(LEVEL_TENORS)].mean(axis=1),
        }
    )
    weekly = frame.apply(_weekly).diff().dropna()
    out = {}
    for window in windows:
        sample = weekly if window is None else weekly.tail(window)
        label = "full" if window is None else f"last_{window}w"
        out[label] = {"n_weeks": int(len(sample)), "start": str(sample.index[0].date()),
                      "end": str(sample.index[-1].date())}
        for factor in ("y10", "level"):
            reg = sm.OLS(sample["log_price"], sm.add_constant(sample[factor])).fit(
                cov_type="HAC", cov_kwds={"maxlags": 4}
            )
            beta, se = float(reg.params[factor]), float(reg.bse[factor])
            out[label][factor] = {
                "duration": -beta,
                "se": se,
                "ci95": [-beta - 1.96 * se, -beta + 1.96 * se],
                "r2": float(reg.rsquared),
            }
    return out
