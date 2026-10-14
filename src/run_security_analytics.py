"""Price one Agency MBS pool: OAS, Z-spread, option cost, duration, convexity.

Every input is point-in-time at the trade date (the last two-sided quote):

  market      Bloomberg bid/ask mid on the trade date (clean, per 100 face)
  curve       Treasury CMT par curve on the trade date -> zero curve
  rate vol    Hull-White kappa, sigma from the trailing two years of
              daily zero-yield changes
  mortgage    PMMS ~ 2Y + 10Y (DOLS, June 2022 onward), anchored to the
              latest PMMS print
  prepay      S-curve fitted on the three locked pools' factor histories
              published by the trade date
  collateral  issuer loan tape for the settlement month

Usage:
    python src/run_security_analytics.py --cusip 3140QVLM4
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curve import bootstrap_par_curve  # noqa: E402
from dates import factor_publication_date, month_start, settlement_date  # noqa: E402
from disclosures.quality import reconcile_security_snapshot  # noqa: E402
from fred_data import load_cmt_panel, load_pmms  # noqa: E402
from hull_white import HullWhite, calibrate_hull_white, zero_yield_history  # noqa: E402
from mortgage_rate import fit_mortgage_rate_model, model_summary, survey_panel  # noqa: E402
from prepayment import (  # noqa: E402
    DEFAULT_CEILING,
    PoolHistorySpec,
    block_bootstrap,
    build_prepay_panel,
    fit_scurve,
    monthly_mortgage_rate,
    profile_ceiling,
    scheduled_fraction,
    scurve_cpr,
    scurve_slope,
    smm_to_cpr,
    cpr_to_smm,
)
from pricing import Assumptions, build_pool_setup, value_pool  # noqa: E402
from security import agency_mbs_accrued_per_100, payment_delay_days  # noqa: E402
from validation import empirical_duration, rolling_origin_prepay_test  # noqa: E402

PROCESSED = ROOT / "data" / "market" / "processed"
DISCLOSURES = PROCESSED / "disclosures"
CONFIG = ROOT / "config" / "securities.yaml"
VOL_WINDOW_YEARS = 2
CEILING_SENSITIVITY = (0.35, 0.65)
INCENTIVE_GRID = (-0.010, -0.005, 0.0, 0.005, 0.010, 0.015)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())
    return value


def _ci(values, level=95) -> list:
    lo = (100 - level) / 2
    return np.percentile(np.asarray(values, dtype=float), [lo, 100 - lo]).tolist()


def load_inputs(cusip: str, trade_date=None) -> dict:
    """Read every source and cut it at the trade date."""
    cusip = cusip.upper()
    market = pd.read_parquet(PROCESSED / "market_observations.parquet")
    market["date"] = pd.to_datetime(market["date"])
    quotes = market.loc[(market["cusip"] == cusip) & (market["source"] == "bid_ask")].sort_values("date")
    if trade_date is None:
        trade_date = quotes["date"].max()
    trade_date = pd.Timestamp(trade_date)
    quote = quotes.loc[quotes["date"] == trade_date]
    if quote.empty:
        raise ValueError(f"No two-sided quote for {cusip} on {trade_date.date()}.")
    quote = quote.iloc[0]
    settle = pd.Timestamp(settlement_date(trade_date))

    partition = month_start(settle).strftime("%Y%m")
    securities = pd.read_parquet(DISCLOSURES / f"security_master_{partition}.parquet")
    loans = pd.read_parquet(DISCLOSURES / f"loan_snapshots_{partition}.parquet")
    security = securities.loc[securities["cusip"] == cusip].iloc[0]
    if pd.Timestamp(security["available_date"]) > trade_date:
        raise ValueError("Loan tape was not public on the trade date.")
    loans = loans.loc[loans["cusip"] == cusip].copy()
    reconciliation = reconcile_security_snapshot(security, loans)

    cmt = load_cmt_panel()
    if trade_date not in cmt.index:
        raise ValueError(f"No complete CMT curve on {trade_date.date()}.")
    pmms = load_pmms()
    pmms = pmms.loc[pmms.index <= trade_date]

    factors = pd.read_parquet(PROCESSED / "bloomberg_factor_history.parquet")
    factors["date"] = pd.to_datetime(factors["date"])
    static = pd.read_parquet(PROCESSED / "bloomberg_static.parquet")
    with open(CONFIG, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return {
        "cusip": cusip,
        "trade_date": trade_date,
        "settlement_date": settle,
        "quote": quote,
        "security": security,
        "loans": loans,
        "reconciliation": reconciliation,
        "cmt": cmt.loc[cmt.index <= trade_date],
        "pmms": pmms,
        "factors": factors,
        "static": static,
        "config": config,
        "market_history": quotes,
    }


def prepay_specs(inputs: dict) -> dict:
    """WAC / WAM / loan count of each locked pool at its latest factor month."""
    security = inputs["security"]
    specs = {
        inputs["cusip"]: PoolHistorySpec(
            inputs["cusip"], float(security["wac"]), int(security["wam"]), int(security["loan_count"])
        )
    }
    static = inputs["static"].set_index("ID_CUSIP")
    for row in inputs["config"]["securities"]:
        cusip = row["cusip"].upper()
        if cusip in specs:
            continue
        specs[cusip] = PoolHistorySpec(
            cusip,
            float(static.loc[cusip, "MTG_WAC"]) / 100.0,
            int(static.loc[cusip, "MTG_WAM"]),
            int(row["loan_count"]),
        )
    return specs


def estimate_prepayment(inputs: dict, n_boot: int, seed: int) -> dict:
    loans = inputs["loans"]
    balance = loans["current_upb"].to_numpy(dtype=float)
    neff_ratio = balance.sum() ** 2 / np.sum(balance**2) / len(balance)
    mortgage_monthly = monthly_mortgage_rate(inputs["pmms"])
    specs = prepay_specs(inputs)
    cusips = list(specs)
    panel = build_prepay_panel(inputs["factors"], specs, mortgage_monthly, inputs["trade_date"], neff_ratio)
    fit = fit_scurve(panel, DEFAULT_CEILING, cusips)
    draws = block_bootstrap(panel, fit, n_boot=n_boot, seed=seed)
    ceilings = profile_ceiling(panel, [0.25, 0.30, 0.35, 0.40, 0.50, 0.65, 0.80, 0.95], cusips)
    lag_fits = {}
    for lag in (0, 1, 2):
        lag_panel = build_prepay_panel(inputs["factors"], specs, mortgage_monthly, inputs["trade_date"],
                                       neff_ratio, lag_months=lag)
        lag_fits[lag] = fit_scurve(lag_panel, DEFAULT_CEILING, cusips).loglik
    pooled_ll = fit_scurve(panel.assign(cusip="pooled"), DEFAULT_CEILING, ["pooled"]).loglik
    return {
        "panel": panel,
        "fit": fit,
        "draws": draws,
        "ceiling_profile": ceilings,
        "lag_loglik": lag_fits,
        "pooled_loglik": pooled_ll,
        "neff_ratio": float(neff_ratio),
        "specs": specs,
        "mortgage_monthly": mortgage_monthly,
    }


def summarize_prepayment(est: dict, cusip: str) -> dict:
    fit, draws = est["fit"], est["draws"]
    names = fit.param_names
    i_floor = names.index(f"floor_{cusip}")
    i_mid, i_width = names.index("midpoint"), names.index("width")
    point = fit.params(cusip)
    grid = {}
    for incentive in INCENTIVE_GRID:
        boot = scurve_cpr(incentive, draws[:, i_floor], fit.ceiling, draws[:, i_mid], draws[:, i_width])
        slope = scurve_slope(incentive, draws[:, i_floor], fit.ceiling, draws[:, i_mid], draws[:, i_width])
        grid[f"{incentive * 1e4:+.0f}bp"] = {
            "cpr": float(scurve_cpr(incentive, **point)),
            "cpr_ci95": _ci(boot),
            "slope_cpr_per_100bp": float(scurve_slope(incentive, **point) * 0.01),
            "slope_ci95": _ci(slope * 0.01),
        }
    pooled_lr = 2.0 * (fit.loglik - est["pooled_loglik"]) / fit.phi
    return {
        "observations": fit.n_obs,
        "pools": {c: int((est["panel"]["cusip"] == c).sum()) for c in fit.floors},
        "negative_smm_months": int((est["panel"]["smm"] < 0).sum()),
        "neff_ratio": est["neff_ratio"],
        "ceiling_fixed": fit.ceiling,
        "floors": fit.floors,
        "floors_ci95": {c: _ci(draws[:, names.index(f"floor_{c}")]) for c in fit.floors},
        "midpoint": fit.midpoint,
        "midpoint_ci95": _ci(draws[:, i_mid]),
        "width": fit.width,
        "width_ci95": _ci(draws[:, i_width]),
        "wald_se": dict(zip(names, np.sqrt(np.diag(fit.cov)).tolist())),
        "phi": fit.phi,
        "loglik": fit.loglik,
        "bootstrap_draws": int(len(draws)),
        "cpr_by_incentive_primary": grid,
        "ceiling_profile": est["ceiling_profile"].to_dict(orient="list"),
        "lag_loglik": {str(k): v for k, v in est["lag_loglik"].items()},
        "pool_floor_vs_pooled_quasi_lr": float(pooled_lr),
    }


def assumption_draws(rng, n: int, cusip: str, hw_cal: dict, mortgage_fit: dict, prepay: dict) -> list:
    """Independent draws of rate-vol, mortgage-model and S-curve parameters."""
    base = mortgage_fit["model"]
    coef = np.array([base.const, base.beta_2y, base.beta_10y])
    fit, draws = prepay["fit"], prepay["draws"]
    names = fit.param_names
    out = []
    for _ in range(n):
        kappa, sigma = hw_cal["bootstrap"][rng.integers(len(hw_cal["bootstrap"]))]
        c, b2, b10 = rng.multivariate_normal(coef, mortgage_fit["long_run_cov"])
        rho = float(np.clip(rng.normal(base.rho_weekly, mortgage_fit["rho_se"]), 0.0, 0.999))
        mortgage = base.with_coefficients(c, b2, b10, rho, mortgage_fit["anchor_cmt2"],
                                          mortgage_fit["anchor_cmt10"])
        row = draws[rng.integers(len(draws))]
        scurve = {
            "floor": float(row[names.index(f"floor_{cusip}")]),
            "ceiling": fit.ceiling,
            "midpoint": float(row[names.index("midpoint")]),
            "width": float(row[names.index("width")]),
        }
        out.append(Assumptions(HullWhite(float(kappa), float(sigma)), mortgage, scurve))
    return out


def month_ahead_forecast(inputs: dict, prepay: dict, setup, rng, n_sims: int = 20_000) -> dict:
    """Pre-registered forecast of the next factor (payoffs in the settlement month).

    The driving mortgage rate is already observed, so the only uncertainty is
    S-curve parameters (bootstrap) and binomial payoff noise on n_eff loans.
    """
    cusip = inputs["cusip"]
    fit, draws = prepay["fit"], prepay["draws"]
    names = fit.param_names
    spec = prepay["specs"][cusip]
    rate = float(setup.driving_observed[0])
    incentive = spec.wac - rate
    sched = float(scheduled_fraction(spec.wac, spec.wam_now))
    factor_now = float(inputs["security"]["factor"])
    n_eff = spec.loan_count_now * prepay["neff_ratio"]
    rows = draws[rng.integers(len(draws), size=n_sims)]
    cpr = scurve_cpr(incentive, rows[:, names.index(f"floor_{cusip}")], fit.ceiling,
                     rows[:, names.index("midpoint")], rows[:, names.index("width")])
    smm = cpr_to_smm(cpr)
    n_int = int(round(n_eff))
    if fit.phi > 1.0:
        # Beta-binomial with the fitted overdispersion: Var = phi * n p (1 - p).
        rho = (fit.phi - 1.0) / (n_int - 1.0)
        concentration = 1.0 / rho - 1.0
        smm = rng.beta(smm * concentration, (1.0 - smm) * concentration)
    smm_obs = rng.binomial(n_int, smm) / n_int
    factor_next = factor_now * (1.0 - sched) * (1.0 - smm_obs)
    point_smm = float(cpr_to_smm(scurve_cpr(incentive, **fit.params(cusip))))
    accrual = setup.schedule["accrual_month"].iloc[0]
    next_factor_month = accrual + pd.DateOffset(months=1)
    return {
        "payoff_month": str(accrual.date())[:7],
        "factor_month": str(next_factor_month.date())[:7],
        "publication_date": str(factor_publication_date(next_factor_month)),
        "driving_mortgage_rate": rate,
        "incentive": incentive,
        "point_cpr": float(smm_to_cpr(point_smm)),
        "point_factor": factor_now * (1.0 - sched) * (1.0 - point_smm),
        "factor_pi90": _ci(factor_next, 90),
        "cpr_pi90": _ci(smm_to_cpr(smm_obs), 90),
        "prob_zero_full_payoffs": float(np.mean(smm_obs == 0)),
        "factor_quantiles": np.percentile(factor_next, np.arange(1, 100)).tolist(),
        "scheduled_fraction_pool_level": sched,
        "n_eff": n_eff,
        "note": "Score with validation.score_month_ahead once the factor is published.",
    }


REPORT_KEYS = ("oas", "z_spread", "option_cost", "eff_duration", "eff_convexity",
               "month1_cpr", "avg_cpr_life")


def market_target(inputs: dict) -> dict:
    """Clean mid, 30/360 accrued to settlement and the dirty dollar target."""
    security, quote = inputs["security"], inputs["quote"]
    coupon = float(security["coupon"])
    current_face = float(security["current_upb"])
    clean = float(quote["price"])
    accrued = agency_mbs_accrued_per_100(coupon, inputs["settlement_date"].date())
    return {
        "coupon": coupon,
        "current_face": current_face,
        "bid": float(quote["bid"]),
        "ask": float(quote["ask"]),
        "clean_mid": clean,
        "accrued_per_100": accrued,
        "dirty_mid": clean + accrued,
        "target_dollars": (clean + accrued) / 100.0 * current_face,
    }


def estimate_models(inputs: dict, n_boot: int, seed: int) -> dict:
    """Curve, Hull-White vol, mortgage-rate model and S-curve, all at the trade date."""
    trade, cmt = inputs["trade_date"], inputs["cmt"]
    curve = bootstrap_par_curve(cmt.columns.to_numpy(dtype=float), cmt.loc[trade].to_numpy(dtype=float))
    window = cmt.loc[cmt.index > trade - pd.DateOffset(years=VOL_WINDOW_YEARS)]
    hw_cal = calibrate_hull_white(zero_yield_history(window), n_boot=n_boot, seed=seed)
    return {
        "curve": curve,
        "hw_cal": hw_cal,
        "hw": HullWhite(hw_cal["kappa"], hw_cal["sigma"]),
        "mortgage_fit": fit_mortgage_rate_model(survey_panel(inputs["pmms"], cmt), trade),
        "prepay": estimate_prepayment(inputs, n_boot=n_boot, seed=seed),
    }


def specification_sensitivity(setup, models: dict, cusip: str, target: float,
                              n_paths: int, seed: int) -> dict:
    """Revalue under modelling choices the data cannot settle.

    S-curve ceiling (not identified) and a pooled S-curve; the curve without
    the 20Y CMT (the 20-year bond trades cheap to the curve); rate vol scaled
    by 1.2 (implied vol usually exceeds realised vol).
    """
    panel, fit = models["prepay"]["panel"], models["prepay"]["fit"]
    base_hw, mortgage, curve = models["hw"], models["mortgage_fit"]["model"], models["curve"]
    base_scurve = fit.params(cusip)
    pooled = fit_scurve(panel.assign(cusip="pooled"), DEFAULT_CEILING, ["pooled"]).params("pooled")
    keep = curve.par_tenors != 20.0
    variants = {
        **{f"ceiling_{c:.2f}": (fit_scurve(panel, c, list(fit.floors)).params(cusip), base_hw, curve)
           for c in CEILING_SENSITIVITY},
        "pooled_scurve": (pooled, base_hw, curve),
        "curve_without_20y": (base_scurve, base_hw,
                              bootstrap_par_curve(curve.par_tenors[keep], curve.par_yields[keep])),
        "sigma_x1.2": (base_scurve, HullWhite(base_hw.kappa, 1.2 * base_hw.sigma), curve),
    }
    out = {}
    for name, (scurve, hw, variant_curve) in variants.items():
        val, _, _ = value_pool(setup, variant_curve, Assumptions(hw, mortgage, scurve), target, n_paths, seed)
        out[name] = {"scurve": scurve, "sigma": hw.sigma, **{key: val[key] for key in REPORT_KEYS}}
    return out


def parameter_uncertainty(setup, models: dict, cusip: str, target: float, rng,
                          param_draws: int, param_paths: int, seed: int) -> pd.DataFrame:
    """Change in each output when the estimated parameters are redrawn.

    Every draw and the reference run use the same ``param_paths`` random
    numbers, so a difference isolates parameter uncertainty from Monte Carlo
    noise. Intervals are point estimate + percentiles of these differences.
    """
    base = Assumptions(models["hw"], models["mortgage_fit"]["model"],
                       models["prepay"]["fit"].params(cusip))
    reference, _, _ = value_pool(setup, models["curve"], base, target, param_paths, seed)
    draws = assumption_draws(rng, param_draws, cusip, models["hw_cal"], models["mortgage_fit"],
                             models["prepay"])
    rows = []
    for assumptions in draws:
        val, _, _ = value_pool(setup, models["curve"], assumptions, target, param_paths, seed)
        rows.append({key: val[key] - reference[key] for key in REPORT_KEYS})
    return pd.DataFrame(rows)


def build_report(inputs: dict, market: dict, models: dict, setup, valuation: dict,
                 extras: dict) -> dict:
    security, trade = inputs["security"], inputs["trade_date"]
    curve, uncertainty = models["curve"], extras["uncertainty"]
    quote = inputs["quote"]
    bbg_oas, bbg_z = float(quote["external_oas_bps"]), float(quote["external_zspread_bps"])
    return {
        "security": {
            "cusip": inputs["cusip"],
            "coupon": market["coupon"],
            "wac": setup.wac,
            "current_face": market["current_face"],
            "factor": float(security["factor"]),
            "loan_count": int(security["loan_count"]),
            "wam": int(security["wam"]),
            "wala": int(security["wala"]),
        },
        "dates": {
            "trade_date": trade,
            "settlement_date": inputs["settlement_date"],
            "loan_tape_factor_month": str(pd.Timestamp(security["as_of_date"]).date())[:7],
            "loan_tape_available": pd.Timestamp(security["available_date"]),
            "first_payment_date": str(setup.schedule["payment_date"].iloc[0]),
            "first_payment_time_years": float(setup.pay_times[0]),
            "curve_date": trade,
            "pmms_latest_release": models["mortgage_fit"]["anchor_release"],
        },
        "market": market,
        "data_quality": asdict(inputs["reconciliation"]),
        "curve": {
            "par_tenors": curve.par_tenors.tolist(),
            "par_yields": curve.par_yields.tolist(),
            "zero_rates": curve.knot_zeros.tolist(),
        },
        "hull_white": {k: v for k, v in models["hw_cal"].items() if k != "bootstrap"},
        "mortgage_rate_model": model_summary(models["mortgage_fit"]),
        "prepayment": summarize_prepayment(models["prepay"], inputs["cusip"]),
        "valuation": valuation,
        "specification_sensitivity": extras["sensitivity"],
        "parameter_uncertainty": {
            "draws": int(len(uncertainty)),
            "paths_per_draw": extras["param_paths"],
            "method": "point + percentiles of (draw - reference) on common random numbers",
            **{col: {"ci95": (valuation[col] + np.asarray(_ci(uncertainty[col]))).tolist(),
                     "sd": float(uncertainty[col].std(ddof=1))} for col in uncertainty},
        },
        "month_ahead_forecast": extras["forecast"],
        "prepay_out_of_sample": extras["oos"]["summary"],
        "empirical_duration": extras["empirical_duration"],
        "bloomberg_benchmark": {
            "oas_bps": bbg_oas,
            "zspread_bps": bbg_z,
            "option_cost_bps": bbg_z - bbg_oas,
        },
    }


def run_security_analytics(
    cusip: str,
    trade_date=None,
    n_paths: int = 5_000,
    seed: int = 42,
    param_draws: int = 200,
    param_paths: int = 2_000,
    n_boot: int = 500,
) -> tuple[dict, dict]:
    """Full point-in-time valuation; returns the JSON report and the working objects."""
    cusip = cusip.upper()
    inputs = load_inputs(cusip, trade_date)
    market = market_target(inputs)
    models = estimate_models(inputs, n_boot, seed)
    fit = models["prepay"]["fit"]
    delay = payment_delay_days(inputs["security"]["agency"], inputs["security"]["prefix"])
    setup = build_pool_setup(inputs["loans"], market["coupon"], market["current_face"],
                             inputs["trade_date"], inputs["settlement_date"],
                             models["prepay"]["mortgage_monthly"], delay)
    assumptions = Assumptions(models["hw"], models["mortgage_fit"]["model"], fit.params(cusip))
    valuation, sim, det = value_pool(setup, models["curve"], assumptions, market["target_dollars"],
                                     n_paths, seed)
    valuation.update({"n_paths": n_paths, "seed": seed})

    rng = np.random.default_rng(seed)
    extras = {
        "sensitivity": specification_sensitivity(setup, models, cusip, market["target_dollars"],
                                                 n_paths, seed),
        "uncertainty": parameter_uncertainty(setup, models, cusip, market["target_dollars"], rng,
                                             param_draws, param_paths, seed),
        "param_paths": param_paths,
        "forecast": month_ahead_forecast(inputs, models["prepay"], setup, rng),
        "oos": rolling_origin_prepay_test(models["prepay"]["panel"], list(fit.floors), DEFAULT_CEILING),
        "empirical_duration": empirical_duration(inputs["market_history"], inputs["cmt"],
                                                 inputs["trade_date"]),
    }
    report = build_report(inputs, market, models, setup, valuation, extras)
    working = {
        "inputs": inputs,
        "models": models,
        "setup": setup,
        "assumptions": assumptions,
        "sim": sim,
        "zero_vol_sim": det,
        **extras,
    }
    return _json_ready(report), working


def write_results(results: dict, output: Path | None = None) -> Path:
    if output is None:
        name = f"{results['security']['cusip']}_{results['dates']['trade_date']}_valuation.json"
        output = ROOT / "artifacts" / "results" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cusip", required=True)
    parser.add_argument("--trade-date", help="Defaults to the last two-sided quote.")
    parser.add_argument("--n-paths", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--param-draws", type=int, default=200)
    parser.add_argument("--param-paths", type=int, default=2_000)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results, _ = run_security_analytics(
        args.cusip, args.trade_date, args.n_paths, args.seed,
        args.param_draws, args.param_paths, args.n_boot,
    )
    path = write_results(results, args.output)
    val = results["valuation"]
    print(
        f"{args.cusip} {results['dates']['trade_date']}: OAS {val['oas'] * 1e4:.1f} bp "
        f"(MC se {val['oas_se'] * 1e4:.2f}), Z {val['z_spread'] * 1e4:.1f} bp, "
        f"option cost {val['option_cost'] * 1e4:.1f} bp, EffDur {val['eff_duration']:.2f} "
        f"(se {val['eff_duration_se']:.3f}), EffConv {val['eff_convexity']:.2f} "
        f"(se {val['eff_convexity_se']:.2f})"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
