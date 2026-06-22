"""
src/run_mbs_analytics.py

End-to-end MBS analytics driver (Phases 6–9). Reproduces the headline pricing
results from the calibrated source modules and serialises them to
``artifacts/results/mbs_pricing_results.json``.

Pipeline
--------
1. Calibrate Vasicek on FRED GS1 (physical measure P).
2. Solve a market price of risk → risk-neutral θ_Q so the model 10Y zero matches
   an assumed current 10Y UST (the discount curve used for pricing).
3. Calibrate the rate-dependent prepayment S-curve floor to the observed pool CPR.
4. Load the MDR (default buyout) schedule produced by the Cox model.
5. Price a representative 6.5% WAC / 6.0% net pass-through:
   - model price (OAS = 0 over the Q curve),
   - Z-spread (no rate vol) vs OAS (full MC) → prepayment OPTION COST,
   - OAS table across market prices,
   - effective duration & convexity via common-random-number finite differences,
   - parallel rate-shock stress table.

Run:  python3 src/run_mbs_analytics.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vasicek import (  # noqa: E402  pylint: disable=wrong-import-position
    calibrate_vasicek,
    calibrate_risk_neutral_theta,
    implied_market_price_of_risk,
    vasicek_yield,
)
from prepayment import (  # noqa: E402  pylint: disable=wrong-import-position
    calibrate_scurve_from_pool_cpr,
)
from pricing import (  # noqa: E402  pylint: disable=wrong-import-position
    simulate_pool_cashflows,
    price_at_oas,
    effective_duration_convexity,
)

# ── Pool & calibration assumptions ────────────────────────────────────────────
WAC          = 0.065      # weighted-average note rate (borrower)
SERVICE_GFEE = 0.005      # servicing (25bp) + guarantee fee (25bp) strip
NET_COUPON   = WAC - SERVICE_GFEE   # investor pass-through coupon = 6.0%
ORIG_UPB     = 1_000_000.0
TERM         = 360
N_PATHS      = 5_000
SEED         = 42
TARGET_10Y   = 0.043      # assumed current 10Y UST (only GS1 is in the dataset)
MBS_SPREAD   = 0.015      # primary/secondary mortgage spread over the Vasicek curve


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Run the full pricing pipeline and write the results JSON."""
    # 1. Vasicek calibration (physical) ----------------------------------------
    gs1 = pd.read_csv(ROOT / "data/fred/gs1_monthly.csv",
                      index_col=0, parse_dates=True).squeeze()
    rates = gs1.values / 100.0
    kappa, theta_p, sigma = calibrate_vasicek(rates)
    short_rate = float(rates[-1])

    # 2. Risk-neutral θ_Q -------------------------------------------------------
    theta_q = calibrate_risk_neutral_theta(short_rate, kappa, sigma, TARGET_10Y, maturity=10.0)
    mpr = implied_market_price_of_risk(theta_p, theta_q, kappa, sigma)
    y10 = vasicek_yield(short_rate, kappa, theta_q, sigma, 10.0)
    y30 = vasicek_yield(short_rate, kappa, theta_q, sigma, 30.0)

    # 3. Prepayment S-curve floor from observed pool CPR -----------------------
    pool_cpr = pd.read_parquet(ROOT / "data/processed/pool_cpr_monthly.parquet")
    obs_floor = float(pool_cpr.loc[pool_cpr["cpr"] > 0, "cpr"].mean())
    scurve = calibrate_scurve_from_pool_cpr(obs_floor)

    # 4. MDR (default buyout) schedule from the Cox model ----------------------
    mdr = pd.read_csv(ROOT / "data/processed/mdr_schedule.csv")["mdr"].values

    common = dict(  # pylint: disable=use-dict-literal
        initial_rate=short_rate, kappa=kappa, theta=theta_q, sigma=sigma,
        coupon=WAC, orig_upb=ORIG_UPB, term=TERM,
        net_coupon=NET_COUPON, mdr=mdr, mbs_spread=MBS_SPREAD,
        prepay_params=scurve,
    )

    # 5a. Model price (full MC, OAS = 0) ---------------------------------------
    cflows, disc, _, _, cpr = simulate_pool_cashflows(n_paths=N_PATHS, seed=SEED, **common)
    pv_paths = np.sum(cflows * disc, axis=1)
    model_price = float(pv_paths.mean())

    # 5b. Z-spread (deterministic, no rate vol) vs OAS (full MC) → option cost --
    cf_det, disc_det, _, _, _ = simulate_pool_cashflows(
        n_paths=1, seed=SEED, **{**common, "sigma": 0.0}
    )

    def price_oas_mc(lam: float) -> float:
        return price_at_oas(lam, cflows, disc)

    def price_oas_det(lam: float) -> float:
        return price_at_oas(lam, cf_det, disc_det)

    oas_table = []
    for mp_pct in (96, 97, 98, 99, 100, 101, 102):
        target = mp_pct / 100.0 * ORIG_UPB
        oas = brentq(lambda l, t=target: price_oas_mc(l) - t, -0.05, 0.25, xtol=1e-8)
        zsp = brentq(lambda l, t=target: price_oas_det(l) - t, -0.05, 0.25, xtol=1e-8)
        oas_table.append({
            "market_price_pct": mp_pct,
            "oas_bps": round(oas * 1e4, 1),
            "z_spread_bps": round(zsp * 1e4, 1),
            "option_cost_bps": round((zsp - oas) * 1e4, 1),
        })

    # 6. Effective duration & convexity (common random numbers) ----------------
    greeks = effective_duration_convexity(
        initial_rate=short_rate, kappa=kappa, theta=theta_q, sigma=sigma,
        coupon=WAC, orig_upb=ORIG_UPB, term=TERM, rate_bump=0.0025,
        n_paths=N_PATHS, seed=SEED, net_coupon=NET_COUPON, mdr=mdr,
        mbs_spread=MBS_SPREAD, prepay_params=scurve,
    )
    base = greeks["base_price"]

    # 7. Parallel rate-shock stress (CRN) --------------------------------------
    from pricing import mbs_price_at_rate  # pylint: disable=import-outside-toplevel
    stress = []
    for shock in (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02):
        price_shock = mbs_price_at_rate(
            shock, initial_rate=short_rate, kappa=kappa, theta=theta_q, sigma=sigma,
            coupon=WAC, orig_upb=ORIG_UPB, term=TERM, n_paths=N_PATHS, seed=SEED,
            net_coupon=NET_COUPON, mdr=mdr, mbs_spread=MBS_SPREAD, prepay_params=scurve,
        )
        lin = -greeks["eff_duration"] * shock          # duration-only prediction
        stress.append({
            "rate_shock_bps": int(shock * 1e4),
            "price_pct": round(price_shock / ORIG_UPB * 100, 4),
            "price_chg_pct": round((price_shock / base - 1) * 100, 4),
            "duration_only_chg_pct": round(lin * 100, 4),
        })

    results = {
        "pool": {"wac": WAC, "net_coupon": NET_COUPON, "servicing_gfee_strip": SERVICE_GFEE,
                 "orig_upb": ORIG_UPB, "term": TERM},
        "vasicek": {
            "kappa": round(kappa, 6), "theta_physical": round(theta_p, 6),
            "theta_risk_neutral": round(theta_q, 6), "sigma": round(sigma, 8),
            "initial_rate": round(short_rate, 6), "market_price_of_risk": round(mpr, 5),
            "model_10y_zero": round(float(y10), 5), "model_30y_zero": round(float(y30), 5),
            "target_10y_assumed": TARGET_10Y,
        },
        "prepayment": {"observed_cpr_floor": round(obs_floor, 5),
                       **{k: round(v, 4) for k, v in scurve.items()},
                       "mean_cpr_life": round(float(cpr.mean()), 4),
                       "mean_cpr_yr1": round(float(cpr[:, :12].mean()), 4)},
        "pricing": {
            "n_paths": N_PATHS,
            "model_price": round(model_price, 2),
            "model_price_pct": round(model_price / ORIG_UPB * 100, 4),
            "price_std_pct": round(float(pv_paths.std()) / ORIG_UPB * 100, 4),
        },
        "risk_metrics": {
            "rate_bump_bps": 25, "n_paths": N_PATHS,
            "base_price_pct": round(base / ORIG_UPB * 100, 4),
            "price_up_pct": round(greeks["price_up"] / ORIG_UPB * 100, 4),
            "price_dn_pct": round(greeks["price_dn"] / ORIG_UPB * 100, 4),
            "eff_duration": round(greeks["eff_duration"], 4),
            "eff_convexity": round(greeks["eff_convexity"], 4),
            "common_random_numbers": True,
        },
        "oas_table": oas_table,
        "stress_table": stress,
    }

    out = ROOT / "artifacts/results/mbs_pricing_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=2)

    # Console summary ----------------------------------------------------------
    print(f"Vasicek P : kappa={kappa:.4f} theta_P={theta_p*100:.2f}% "
          f"sigma={sigma*100:.3f}% r0={short_rate*100:.2f}%")
    print(f"Vasicek Q : theta_Q={theta_q*100:.2f}% (mpr={mpr:.4f})  "
          f"10Y={y10*100:.2f}%  30Y={y30*100:.2f}%")
    print(f"Prepay    : floor CPR={obs_floor*100:.2f}%  k_min={scurve['k_min']:.3f}  "
          f"mean life CPR={cpr.mean()*100:.1f}%")
    print(f"Price     : {model_price/ORIG_UPB*100:.2f}  "
          f"(path std {pv_paths.std()/ORIG_UPB*100:.1f})")
    print(f"Greeks    : EffDur={greeks['eff_duration']:.2f}  "
          f"EffConv={greeks['eff_convexity']:.1f}  (CRN)")
    print("OAS table (market price -> OAS / Z-spread / option cost, bps):")
    for row in oas_table:
        print(f"  {row['market_price_pct']:>3} -> OAS {row['oas_bps']:6.1f} | "
              f"Z {row['z_spread_bps']:6.1f} | optcost {row['option_cost_bps']:5.1f}")
    print("Stress (shock -> price%, chg%, duration-only%):")
    for row in stress:
        print(f"  {row['rate_shock_bps']:+5d} -> {row['price_pct']:7.3f}  "
              f"{row['price_chg_pct']:+6.2f}%  (lin {row['duration_only_chg_pct']:+6.2f}%)")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
