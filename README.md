# Agency MBS: OAS, rate sensitivity and prepayment on a real pool

Prices one Fannie Mae 30-year pool (CB9331, `3140QVLM4`, 5.5% coupon, 445 loans, $181mm) from:

- its own loan tape;
- the Treasury curve;
- a mortgage-rate model;
- a prepayment S-curve **estimated** from realised pool speeds.

Every number comes with its Monte Carlo error and its parameter uncertainty.

## Results: trade 9 Oct 2026, settle 14 Oct 2026

Market: bid/ask 96.270 / 96.332, so clean mid 96.301 + accrued 0.199 = **96.499** full.
5,000 Hull–White paths, seed 42.

| | Estimate | MC se | 95% parameter interval | Bloomberg, same day |
| --- | ---: | ---: | --- | ---: |
| OAS | **34.5 bp** | 1.8 | 8 – 59 | 40.4 |
| Z-spread | **57.8 bp** | – | 41 – 69 | 76.6 |
| Option cost (Z − OAS) | **+23.3 bp** | 1.8 | 9 – 34 | +36.2 |
| Effective duration | **6.07** | 0.03 | 5.55 – 7.10 | |
| Effective convexity | **−60.7** | 4.6 | −105 – +67 | |
| CPR, first month | **4.1%** | | 2.1% – 6.1% | |

- **Duration.** A weekly regression of price on the 10Y over the last 26 weeks gives 5.27 ± 0.62 (HAC). The
  model's 6.07 is inside that interval.
- **Option cost.** Positive, and close to Bloomberg once vol is scaled to implied levels: with σ × 1.2 it is
  35.6 bp vs Bloomberg's 36.2.
- **Prepayment.** The pool is 86 bp out of the money (WAC 6.54% vs a 7.40% mortgage rate). It prepays at
  its turnover floor (≈4% CPR) unless rates rally.
- **Where the uncertainty comes from.** The interval widths are mostly the S-curve's refinancing response,
  which two years of data pin down only near the money.

## How it works

1. **Prepayment S-curve** (`prepayment.py`, notebook 01).
   - Monthly SMM is backed out of three pools' factor histories, with incentive = WAC − PMMS one month
     earlier.
   - A logistic S-curve with pool-specific floors is fitted by binomial quasi-likelihood. Intervals come from
     a moving-block bootstrap.
   - Data decisions: no seasoning ramp (rejected by the data); pool floors (quasi-LR 13.3, p = 0.002); the
     plateau is not identified, so it is fixed at 50% CPR with sensitivity runs.
2. **Treasury curve and Hull–White** (`curve.py`, `hull_white.py`, notebook 02).
   - The CMT par curve is bootstrapped exactly.
   - One-factor Hull–White is fitted to that curve. κ and σ come from the realised term structure of
     zero-yield vol: κ = 0.006, σ = 81.5 bp/yr.
   - Exact simulation, so pathwise discount factors are unbiased against the curve.
3. **Mortgage rate** (`mortgage_rate.py`, notebook 03).
   - PMMS = 2.63% + 0.571·CMT2 + 0.385·CMT10, by dynamic OLS with Newey–West errors since June 2022.
   - Today's spread residual decays with a 4.4-week half-life.
   - Uses the 2Y and 10Y, not the 30Y: the 30Y alone explains 28% of PMMS levels, the 2Y + 10Y 87%.
4. **Valuation** (`cashflow.py`, `pricing.py`, notebook 04).
   - Each loan amortises on its own terms, with pool-level SMM on each path.
   - Cash is paid on the 25th after each accrual month.
   - OAS is matched to the full price. Z-spread uses the same curve with vol off.
   - Duration and convexity come from ±25 bp parallel par-curve shifts at fixed OAS on common random numbers.
5. **Validation** (`validation.py`, notebook 05).
   - Rolling-origin prepayment forecasts: deviance 18.1 vs 24.0 for a trailing-speed benchmark.
   - Empirical duration regression.
   - A pre-registered forecast of the November factor: 0.87193, 90% interval 0.86788–0.87496, scored on
     5 Nov 2026.

All inputs are cut at the trade date:

- the price and the curve are from the same day;
- the loan tape and factors count only once published (4th business day);
- the first month's prepayment uses the already observed September mortgage rate.

## Reproduce

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python src/fred_data.py                                  # Treasury curve + PMMS (no key needed)
python src/build_security_snapshots.py --agency FNMA \
    --loan-file data/market/raw/fnma/202610/CB9331_3140QVLM4_COLLAT_CURRENT.txt \
    --security-file data/market/raw/fnma/202610/CB9331_3140QVLM4_security.txt \
    --as-of 2026-10-01 --cusip 3140QVLM4 \
    --source-url https://fanniemae.mbs-securities.com/fannie/details/3140QVLM4
python src/market_data.py data/market/raw/bloomberg/bloomberg_raw_data.xlsx
python src/run_security_analytics.py --cusip 3140QVLM4   # ~2 minutes; writes artifacts/results/

python -m pytest tests -q
```

The Bloomberg workbook is licensed and not committed.
