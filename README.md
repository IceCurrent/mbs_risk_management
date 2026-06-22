# MBS Risk Management — Loan-Level Pricing & Risk Framework

An end-to-end, loan-level agency-MBS analytics stack built on the **Freddie Mac Single-Family Loan-Level Dataset (2018–2025, ~15M loans)**. It estimates default and prepayment behaviour from the raw servicing data, simulates interest rates with a calibrated Vasicek model, and prices a representative 30-year pass-through with a Monte Carlo **OAS engine** that reports option-adjusted spread, prepayment option cost, effective duration/convexity, and a rate-shock stress profile.

> **What this is:** a transparent, reproducible demonstration of the full agency-MBS valuation chain — credit → prepayment → rates → cash flows → OAS/risk — calibrated to real data with every assumption stated.
>
> **What this is not:** a desk-ready mark. It is calibrated under the physical measure from a single rate series (GS1) with no vol surface or TBA prices, so absolute levels are model values, not tradeable quotes. The **relative** machinery (OAS, option cost, duration, convexity, scenario P&L) is the deliverable. See [Limitations](#limitations).

---

## Headline Results

Representative pool: **\$1.0M UPB, 6.50% WAC, 6.00% net pass-through, 30-year FRM, 5,000 Monte Carlo paths.** Full numbers in [artifacts/results/mbs_pricing_results.json](artifacts/results/mbs_pricing_results.json) and [artifacts/results/credit_metrics.json](artifacts/results/credit_metrics.json).

| Block | Metric | Value | Read |
| --- | --- | --- | --- |
| **Pricing** | Model price (OAS=0, Q-curve) | **107.74** | Premium: 6.0% net coupon vs ~4.5% discount curve |
| | Par OAS / Z-spread / **option cost** | 127 / 142 / **16 bp** | Z≈ assumed 150bp MBS spread; option cost is the prepay-option charge |
| | Effective duration | **3.34 yr** | Prepayment-shortened |
| | Effective convexity | **−30.4** | Genuinely negative (CRN finite differences) |
| | Stress −200 / +200 bp | +6.1% / **−7.2%** | Downside > upside ⇒ negative convexity |
| **Default (logistic)** | Test ROC-AUC / Brier | **0.608 / 0.021** | Honest; weak discrimination in a benign 2021–25 window |
| **Survival (Cox PH)** | Test C-index (leakage-free) | **0.766** | vs **0.907** with leaked end-of-life covariates |
| | Lifetime default (ever 90+DPD) | ~12% | Feeds the pricer as a buyout-at-par leg |
| **Rates (Vasicek)** | κ / θ_P / θ_Q / σ | 0.090 / 4.85% / 5.76% / 1.35% | Physical vs risk-neutral (market price of risk −0.061) |

**The single most useful output for an MBS investor here is the OAS → option-cost decomposition and the negatively-convex stress profile**, both of which behave correctly and are explained below.

---

## Why the numbers make sense (senior-review walkthrough)

1. **Risk-neutral discounting.** Vasicek is calibrated on GS1 under the *physical* measure (κ=0.090, θ_P=4.85%). Used directly, the σ²/2κ² convexity drag pushes the model 30Y zero to ~3.7% — far below the 6% coupon — which over-discounts and inflates price. A **market price of risk** (λ=−0.061) shifts the long-run mean to θ_Q=5.76%, pinning the model 10Y zero to an assumed 4.30% (proxy for the current 10Y UST, the only curve input not in the dataset) and giving a realistic upward-sloping discount curve (10Y 4.30%, 30Y 4.53%). Forecasting notebooks use θ_P; the pricer uses θ_Q.
2. **Pass-through economics.** Borrowers amortise at the 6.50% note rate; the investor receives a **6.00% net coupon** (50 bp servicing + guarantee-fee strip) plus 100% of principal. A 6.0% coupon against a ~4.5% curve is genuinely a premium, so **107.74 is the correct order of magnitude**, not a bug.
3. **OAS, decomposed.** With a 150 bp assumed MBS-Treasury spread baked into the borrower's refi rate, the engine recovers a **Z-spread of 142 bp** (≈ the input — a clean internal check), strips **~16 bp of prepayment option cost** via the Monte Carlo, and leaves an **OAS of 127 bp** at par. Option cost rises monotonically with dollar price (5.8 bp at 98 → 25 bp at 102) — the textbook signature of premium-MBS negative convexity.
4. **Negative convexity is now real.** The original prepayment S-curve was effectively rate-insensitive (logistic steepness `beta=8` on a refi incentive measured in *decimals* ⇒ a 200 bp move barely shifted the curve). Recalibrating it (floor from observed CPR; `beta=200`) makes CPR respond to rates: a 100 bp rally roughly triples CPR. The stress table then shows the correct asymmetry — a −200 bp rally adds +6.1% (less than the +6.7% duration-only line) while a +200 bp sell-off costs −7.2% (more than −6.7%).
5. **Credit feeds the price.** The Cox hazard produces a default (MDR) term structure that enters the cash flows as an involuntary **buyout at par** (agency convention: a default is a guarantor buyout, not a loss to the investor). It shortens a premium by ~0.10 pts — small, correctly signed, and coherent end-to-end.

---

## Methodology

```text
Freddie Mac SF LLD (pipe-delimited, 32 cols)         FRED: GS1 + macro
   Origination 2018–25 ─┐                                   │
   Performance 2018–25 ─┘                                   │
            │ parse · 32-col assert · merge                 │
            ▼                                               │
   loan-level panels (out-of-core aggregation)              │
      ├─ logistic table (9.1M loans, 2021–25)               │
      └─ survival table (15.3M loans, 2018–25)              │
            │                                               │
   ┌────────┼───────────────────────────────┐              │
   ▼        ▼                                ▼              ▼
 EDA   Default: Logistic           Survival: Cox PH    Vasicek calibration
       (origination + macro,       (origination-only,  (OLS AR(1) on GS1)
        temporal split,             leakage-free)        P-measure κ,θ,σ
        Platt-calibrated)           C-index 0.77             │
        AUC 0.61                        │ baseline hazard      │ market price of risk
                                        ▼                      ▼
                                  MDR term structure      θ_Q (risk-neutral)
                                        │                      │
   Prepayment S-curve (CPR-calibrated) ─┤                      │
                                        ▼                      ▼
                          ┌─────────────────────────────────────────┐
                          │  MBS Monte Carlo pricer (5,000 paths)     │
                          │  Vasicek(Q) → 30Y mortgage rate → CPR     │
                          │  → cash flows (net coupon + MDR buyout)   │
                          │  → stochastic discount → price            │
                          │  → Z-spread / OAS / option cost           │
                          │  → eff. duration & convexity (CRN)        │
                          │  → rate-shock stress (−200…+200 bp)       │
                          └─────────────────────────────────────────┘
```

Reproduce the full pricing block from the calibrated modules:

```bash
python3 src/run_mbs_analytics.py        # → artifacts/results/mbs_pricing_results.json
```

---

## Results in detail

### 1. Default model — logistic regression  ([03_default_logistic.ipynb](notebooks/03_default_logistic.ipynb))

One row per loan (2021–25), origination + origination-quarter macro features, **temporal** split (train <2024-01, valid →2024-07, test ≥2024-07), `class_weight="balanced"`, Platt-calibrated.

| Metric | Validation | Test |
| --- | --- | --- |
| ROC-AUC | 0.651 | **0.608** |
| PR-AUC | 0.043 | 0.026 |
| Brier score | — | 0.021 |
| Default rate | 2.65% | 1.75% (overall 1.72%) |

**Honest read:** discrimination is weak. The 2021–25 window is a benign, low-default regime (high home-price appreciation, low rates, cured COVID forbearance), so there is little default signal to learn, and macro features are fixed at origination. Coefficient **signs are all economically correct** — strongest drivers are loan age (+), second-home occupancy (−), high LTV (+), FICO (−), and note rate (+). This is an honestly weak model on a hard, imbalanced problem, not a broken one.

### 2. Survival model — Cox proportional hazards  ([04_cox_ph.ipynb](notebooks/04_cox_ph.ipynb))

15.3M loans (2018–25, 2.48% event rate), event = ever 90+ DPD or terminal zero-balance (03/06/09). **Origination-only covariates** — fitting on a 2M event-stratified subsample.

| Feature set | Test C-index |
| --- | --- |
| **Origination-only (used)** | **0.766** |
| With end-of-life covariates (`delinq_last`, `eltv_last`, `current_*`) | 0.907 |

The 0.907 figure is **leakage**: those covariates are measured at the loan's last observation, which for a defaulted loan is the default month itself — predicting default with the default. Removing them gives an honest **0.766**, strong for origination-only mortgage default. Hazard ratios are all correctly signed:

| Covariate | Hazard ratio | Effect |
| --- | --- | --- |
| FICO (z) | 0.857 | higher FICO → lower default |
| Orig CLTV (z) | 1.068 | higher leverage → higher |
| Orig DTI (z) | 1.087 | higher DTI → higher |
| log(orig UPB) | 1.045 | larger loans slightly higher |
| Investor / second home | 0.96 / 0.91 | lower than owner-occupied (strong agency underwriting) |

The fitted baseline hazard is turned into a monthly default-rate (**MDR**) term structure — an SDA-style curve anchored to the empirical age-hazard for months 1–48 (risk set >6M), peaking at ~1.44% CDR in year 4 and declining with seasoning. (The late-duration empirical hazard, e.g. 9%/mo at month 92 from a 194-loan risk set, is discarded as a thin-risk-set artifact.) This MDR drives the pricer's buyout leg.

### 3. Prepayment — rate-dependent CPR  ([05_psa_prepayment.ipynb](notebooks/05_psa_prepayment.ipynb))

`CPR = k(refi incentive) × PSA_base(age)`, with `k` a logistic S-curve in the refi incentive (WAC − prevailing mortgage rate).

- **Floor calibrated to data:** the observed pool CPR (~1.27%, dominated by deeply out-of-the-money 2020–21 vintages during the 2022–25 lock-in) pins `k_min = 0.21`.
- **At-the-money** anchored to 100% PSA (`k(0)=1`).
- **In-the-money response** (`k_max=6`, `beta=200`) set from PSA convention, because the sample contains **no in-the-money refi episodes** to fit against — a stated assumption, not a fitted result.

Result: a realistic S-curve (a 100 bp rally roughly triples CPR), mean life CPR ~9.4% on the representative pool.

### 4. Interest rates — Vasicek  ([06](notebooks/06_vasicek_calibration.ipynb) · [07](notebooks/07_vasicek_simulation.ipynb))

OLS AR(1) calibration on monthly GS1 (1973–2026): κ=0.090, θ_P=4.85%, σ=1.35%, r₀=3.79%. Exact (not Euler-approximate) transition; closed-form bond pricing; terminal-distribution KS check. Pricing uses the **risk-neutral** θ_Q=5.76% (market price of risk −0.061) as described above.

### 5. MBS pricer & risk  ([08_mbs_pricer.ipynb](notebooks/08_mbs_pricer.ipynb))

| Metric | Value |
| --- | --- |
| Model price (OAS=0) | 107.74 (path std 8.2%) |
| Effective duration | 3.34 yr |
| Effective convexity | −30.4 |

**OAS table** (option cost = Z-spread − OAS):

| Market price | OAS (bp) | Z-spread (bp) | Option cost (bp) |
| ---: | ---: | ---: | ---: |
| 98 | 162.1 | 167.9 | 5.8 |
| 100 | 126.6 | 142.2 | 15.6 |
| 102 | 92.3 | 117.3 | 25.1 |

**Rate-shock stress** (Monte Carlo vs. duration-only line — the gap *is* the convexity):

| Shock | Price | Δ (MC) | Δ (duration-only) |
| ---: | ---: | ---: | ---: |
| −200 bp | 114.30 | +6.08% | +6.69% |
| 0 bp | 107.74 | — | — |
| +200 bp | 99.97 | −7.22% | −6.69% |

Effective duration/convexity are computed with **common random numbers** (identical seed and path count for base and both bumps), so the second-order convexity term is a clean signal rather than Monte Carlo noise — the flaw that made the original −118 convexity meaningless.

---

## Key assumptions

| # | Assumption | Rationale / impact |
| --- | --- | --- |
| 1 | Current 10Y UST ≈ **4.30%** (sets θ_Q) | Only GS1 is in the dataset; this anchors the risk-neutral discount curve. Changing it shifts price/OAS levels. |
| 2 | Market price of risk is **constant** | Standard single-factor simplification; no vol-surface calibration. |
| 3 | **150 bp** constant MBS-Treasury spread | Drives the refi rate and ≈ the recovered Z-spread; a single point estimate, not a fitted curve. |
| 4 | **50 bp** servicing + g-fee strip (net 6.0%) | Typical agency economics. |
| 5 | **Agency** collateral ⇒ default = **buyout at par**, no loss severity | Correct for Freddie Mac guaranteed loans; non-agency would need a severity model. |
| 6 | Prepayment in-the-money response from **PSA convention** | Sample has no in-the-money episodes to fit; only the floor is data-calibrated. |
| 7 | MDR (default) term structure is **deterministic** across paths | Default modelled as credit/idiosyncratic, not rate-path-dependent. |
| 8 | Representative pool, not actual issued pools | Single WAC/term; no loan-level heterogeneity within the priced pool. |

---

## Limitations

Honest scope boundaries, several driven by compute/data constraints (16 GB RAM, no market-data feeds):

1. **No true risk-neutral OAS calibration.** A market OAS requires an interest-rate **vol surface** and observed **TBA/pool prices** to calibrate against — neither is in the dataset. OAS levels here are wider than tight agency OAS (0–50 bp) because the single-factor model with 1Y-calibrated vol attributes little spread to optionality; the figures are best read as a **Z-spread-like** measure with an explicit option-cost decomposition.
2. **Single-factor, single-series rate model.** Vasicek on GS1 alone cannot reproduce the full curve shape or term-structure dynamics (and admits negative rates — ~6% stationary mass below zero). A 2-factor or Hull-White model fit to the whole curve would be the next step.
3. **Benign credit window.** 2021–25 has almost no defaults, capping logistic discrimination. The Cox model (2018–25) is stronger because it spans more stress, but neither is validated through a full housing downturn.
4. **Survival validation is a random stratified split, not out-of-time.** Right-censoring makes recent vintages nearly event-free, so an out-of-time test would be unstable; the C-index measures covariate ranking, not a temporal backtest.
5. **Full monthly panels (6.3 GB CPH / 3.5 GB logistic) OOM a 16 GB kernel.** Models are fit on precomputed loan-level tables and a 2M-loan subsample; the codebase aggregates out-of-core ([build_model_aggregates.py](src/build_model_aggregates.py)) to make this tractable.
6. **No prepayment burnout / turnover seasonality / loan-level dispersion** in the priced pool. A `burnout` hook exists ([prepayment.py](src/prepayment.py)) but is not wired into the representative-pool run.

### What would make this production-grade

Multi-factor rate model calibrated to the full curve + a vol surface · OAS calibrated to live TBA prices · loan-level pool composition · time-varying macro in the credit models · burnout & turnover in prepayment · out-of-time / through-the-cycle credit validation.

---

## Project structure

```text
src/
  data_extraction.py       parse & merge SF LLD (32-col assert, sentinel handling)
  feature_engineering.py   logistic + Cox features, FRED macro join, default flag
  build_model_aggregates.py  out-of-core loan-level aggregation (avoids OOM)
  default_model.py         logistic regression, Platt calibration, SHAP
  survival_model.py        Cox PH (leakage-free), C-index, PH check
  prepayment.py            PSA, SMM↔CPR, CPR-calibrated rate-dependent S-curve
  vasicek.py               calibration, MC simulation, market price of risk (θ_Q)
  cashflow.py              pool cash flows: net coupon + MDR buyout-at-par
  pricing.py               OAS / Z-spread, effective duration & convexity (CRN)
  run_mbs_analytics.py     end-to-end driver → results JSON
notebooks/                 01 extract · 02 EDA · 03 logistic · 04 Cox · 05 prepay
                           · 06–07 Vasicek · 08 pricer
artifacts/
  models/                  logistic_model.pkl, cox_ph_model.pkl
  figures/                 all chart PNGs
  results/                 mbs_pricing_results.json, credit_metrics.json
data/
  raw/ processed/ fred/    inputs & parquet panels (gitignored)
```

---

## Setup & reproduction

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                 # add your FRED API key (only needed to refresh caches)

# Notebooks 01→08 run top-to-bottom. The pricing block is also a one-shot script:
python3 src/run_mbs_analytics.py     # regenerates artifacts/results/mbs_pricing_results.json
```

Place Freddie Mac quarterly zips under `historical_data_YYYY/historical_data_YYYYQN.zip` (2018Q1 … 2025Q3).

### Data sources

- **Freddie Mac SF LLD** — Single-Family Loan-Level Dataset, 2018–2025 ([freddiemac.com](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)). Pipe-delimited, 32 origination + 32 performance columns.
- **FRED GS1** (1Y Treasury) and macro series — via `fredapi`, cached to `data/fred/`.

**Engineering guarantees:** `.env` gitignored; column count asserted == 32 before any parse; raw/processed data and large parquets gitignored; reproducible RNG seeds throughout; finite-difference Greeks use common random numbers.
