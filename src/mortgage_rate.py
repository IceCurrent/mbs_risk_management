"""Primary 30-year mortgage rate as a function of the 2Y and 10Y Treasury.

Borrowers refinance against the rate lenders quote (Freddie Mac PMMS), not
against a Treasury. Street prepayment models drive that rate off the 2Y and
10Y; the 30Y Treasury is not used because a mortgage's cash flows (average
life ~6-9 years) sit around the 5-10 year sector.

Long-run relation (weekly, survey-aligned), estimated by dynamic OLS with
Newey-West errors so the coefficient intervals are valid for cointegrated
I(1) series:

    PMMS_t = a + b2 CMT2_t + b10 CMT10_t + e_t,      e_t = rho e_{t-1} + u_t

Projection: m(t) = a + b2 CMT2(t) + b10 CMT10(t) + e_0 rho^(weeks since survey),
where e_0 is today's observed spread residual, so m(0) equals the latest
PMMS and the spread mean-reverts at the estimated speed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# Fed MBS runoff (QT) began June 2022: the regime without a price-insensitive
# buyer in agency MBS, which sets today's primary/secondary spread.
DEFAULT_SAMPLE_START = "2022-06-01"
DOLS_LEADS_LAGS = 4


def survey_panel(pmms: pd.Series, cmt: pd.DataFrame) -> pd.DataFrame:
    """Align each Thursday PMMS print with mean 2Y/10Y CMT over Mon-Wed.

    PMMS is surveyed Monday-Wednesday and released Thursday, so the matching
    Treasury yields are the survey-window averages, not the release day.
    """
    rows = []
    for release, rate in pmms.items():
        window = cmt.loc[release - pd.Timedelta(days=3): release - pd.Timedelta(days=1), [2.0, 10.0]]
        if window.empty:
            continue
        rows.append(
            {
                "release": release,
                "survey_mid": release - pd.Timedelta(days=2),
                "pmms": rate,
                "cmt2": window[2.0].mean(),
                "cmt10": window[10.0].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("release")


@dataclass(frozen=True)
class MortgageRateModel:
    const: float
    beta_2y: float
    beta_10y: float
    rho_weekly: float
    anchor_residual: float
    anchor_survey_mid: str
    anchor_pmms: float

    def equilibrium(self, cmt2, cmt10):
        return self.const + self.beta_2y * cmt2 + self.beta_10y * cmt10

    def project(self, cmt2, cmt10, weeks_since_anchor):
        decay = self.rho_weekly ** np.asarray(weeks_since_anchor, dtype=float)
        return self.equilibrium(cmt2, cmt10) + self.anchor_residual * decay

    def with_coefficients(self, const: float, beta_2y: float, beta_10y: float,
                          rho_weekly: float, anchor_cmt2: float, anchor_cmt10: float) -> "MortgageRateModel":
        """Same anchor PMMS, new coefficients (used for parameter-uncertainty draws)."""
        residual = self.anchor_pmms - (const + beta_2y * anchor_cmt2 + beta_10y * anchor_cmt10)
        return MortgageRateModel(const, beta_2y, beta_10y, rho_weekly, residual,
                                 self.anchor_survey_mid, self.anchor_pmms)


def _dols_design(panel: pd.DataFrame, leads_lags: int) -> tuple[pd.Series, pd.DataFrame]:
    frame = panel[["pmms", "cmt2", "cmt10"]].copy()
    regressors = {"const": np.ones(len(frame)), "cmt2": frame["cmt2"], "cmt10": frame["cmt10"]}
    for name in ("cmt2", "cmt10"):
        diff = frame[name].diff()
        for shift in range(-leads_lags, leads_lags + 1):
            regressors[f"d_{name}_{shift:+d}"] = diff.shift(shift)
    design = pd.DataFrame(regressors, index=frame.index).dropna()
    return frame.loc[design.index, "pmms"], design


def fit_mortgage_rate_model(
    panel: pd.DataFrame,
    as_of,
    start: str = DEFAULT_SAMPLE_START,
    leads_lags: int = DOLS_LEADS_LAGS,
) -> dict:
    """Estimate the long-run relation and spread decay with data known by ``as_of``."""
    as_of = pd.Timestamp(as_of)
    sample = panel.loc[(panel.index >= pd.Timestamp(start)) & (panel.index <= as_of)]
    if len(sample) < 104:
        raise ValueError("Need at least two years of weekly PMMS observations.")
    y, design = _dols_design(sample, leads_lags)
    n_obs = len(y)
    hac_lags = int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
    dols = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    const, b2, b10 = (float(dols.params[name]) for name in ("const", "cmt2", "cmt10"))
    long_run_cov = dols.cov_params().loc[["const", "cmt2", "cmt10"], ["const", "cmt2", "cmt10"]]

    residual = sample["pmms"] - (const + b2 * sample["cmt2"] + b10 * sample["cmt10"])
    ar = sm.OLS(residual.iloc[1:].to_numpy(), residual.iloc[:-1].to_numpy()).fit()
    rho = float(ar.params[0])
    eg_stat, eg_pvalue, _ = coint(sample["pmms"], sample[["cmt2", "cmt10"]], trend="c")

    last = sample.iloc[-1]
    model = MortgageRateModel(
        const=const,
        beta_2y=b2,
        beta_10y=b10,
        rho_weekly=rho,
        anchor_residual=float(residual.iloc[-1]),
        anchor_survey_mid=str(pd.Timestamp(last["survey_mid"]).date()),
        anchor_pmms=float(last["pmms"]),
    )
    return {
        "model": model,
        "sample_start": str(sample.index[0].date()),
        "sample_end": str(sample.index[-1].date()),
        "n_weeks": int(len(sample)),
        "n_dols": int(n_obs),
        "hac_lags": hac_lags,
        "coef_se": {name: float(dols.bse[name]) for name in ("const", "cmt2", "cmt10")},
        "long_run_cov": long_run_cov.to_numpy(),
        "sum_betas": b2 + b10,
        "sum_betas_se": float(np.sqrt(long_run_cov.loc["cmt2", "cmt2"] + long_run_cov.loc["cmt10", "cmt10"]
                                      + 2 * long_run_cov.loc["cmt2", "cmt10"])),
        "rho_weekly": rho,
        "rho_se": float(ar.bse[0]),
        "half_life_weeks": float(np.log(0.5) / np.log(rho)) if 0 < rho < 1 else float("inf"),
        "residual_sd": float(residual.std(ddof=3)),
        "r_squared_static": float(1 - residual.var() / sample["pmms"].var()),
        "engle_granger_stat": float(eg_stat),
        "engle_granger_pvalue": float(eg_pvalue),
        "anchor_cmt2": float(last["cmt2"]),
        "anchor_cmt10": float(last["cmt10"]),
        "anchor_release": str(sample.index[-1].date()),
    }


def model_summary(fit: dict) -> dict:
    """JSON-safe view of a fit."""
    out = {key: value for key, value in fit.items() if key not in {"model", "long_run_cov"}}
    out["model"] = asdict(fit["model"])
    out["long_run_cov"] = np.asarray(fit["long_run_cov"]).tolist()
    return out
