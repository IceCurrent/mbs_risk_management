# MBS Risk Management

End-to-end loan-level MBS risk framework built on Freddie Mac Single-Family Loan-Level Dataset (SF LLD, 2018–2025). Covers default prediction, prepayment modelling, Vasicek interest rate simulation, and a full Monte Carlo MBS pricer with OAS, effective duration, and convexity.

---

## Methodology

```
Freddie Mac SF LLD (pipe-delimited, 32 cols each)
  ├── Origination files (2018–2025 Q1–Q4)
  └── Performance files (2018–2025 Q1–Q4)
         │
         ▼
  [Phase 2] Data extraction & merge → panel_logistic_2021_2025.parquet
                                    → panel_cph_2018_2025.parquet
         │
         ├── [Phase 3]  EDA — vintage charts, default rate, delinquency breakdown
         │
         ├── [Phase 4a] Feature engineering (20 logistic features, FRED macro join)
         │
         ├── [Phase 4b] Logistic Regression (class_weight=balanced, Platt calibration)
         │              ROC-AUC ≈ 0.83, default rate ≈ 0.32%  •  SHAP feature importance
         │
         ├── [Phase 4c] Cox PH Survival Model (penalizer=0.1, Schoenfeld residual check)
         │              C-index > 0.70  •  Baseline survival curve  •  Partial hazard plot
         │
         ├── [Phase 5]  PSA CPR schedule  •  Observed pool CPR  •  Rate-dependent PSA k(RI)
         │
         ├── [Phase 6]  Vasicek calibration (OLS AR(1) on GS1, 2000–2025)
         │              κ, θ, σ  •  5,000-path simulation  •  Terminal distribution KS test
         │
         └── [Phase 7–8] MBS Pricer
                Vasicek short rates → 30Y mortgage rates → CPR paths → cash flows
                → stochastic discount factors → model price
                → OAS (Brent's method)  •  Eff. Duration  •  Eff. Convexity
                → Stress table (−200bp to +200bp)
```

---

## Key Results (Representative Pool)

Pool: \$1M UPB, 6.50% WAC, 30-year FRM, 5,000 MC paths

| Metric | Value |
| --- | --- |
| Model Price (OAS=0) | see `artifacts/results/mbs_pricing_results.json` |
| Effective Duration | ~3–7 years (prepayment-shortened) |
| Effective Convexity | negative (MBS negative convexity) |
| OAS at 98 cents | ~27 bp |

### Stress Test (rate shock → model price)

| Shock | Expected effect |
| --- | --- |
| −200 bp | Price rises less than duration predicts (negative convexity) |
| 0 bp | Base model price |
| +200 bp | Price falls more than duration predicts |

### Default Model

| Metric | Target (spec §5.2) |
| --- | --- |
| ROC-AUC | ~0.83 |
| Default rate | ~0.32% |
| Strict default | 90+ DPD or ZBC in {03, 06, 09} |

### Cox PH Survival

| Metric | Target |
| --- | --- |
| C-index | > 0.70 |
| Training data | 2018–2025 (7 years for resolved events) |

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # add your FRED API key
```

Place Freddie Mac zip archives in the expected directory structure:

```
historical_data_2018/historical_data_2018Q1.zip
historical_data_2018/historical_data_2018Q2.zip
...
historical_data_2025/historical_data_2025Q1.zip
```

Then run notebooks in order (01 → 08).

---

## Project Structure

```
.
├── src/
│   ├── data_extraction.py      # Phases 2a-2c: parse & merge SF LLD
│   ├── feature_engineering.py  # Phase 4a: features, FRED macro, default flag
│   ├── default_model.py        # Phase 4b: logistic regression, SHAP
│   ├── survival_model.py       # Phase 4c: Cox PH, C-index, PH assumption check
│   ├── prepayment.py           # Phase 5: PSA, SMM↔CPR, rate-dependent k(RI)
│   ├── vasicek.py              # Phase 6: calibration, MC simulation, bond pricing
│   ├── cashflow.py             # Phase 7a: pool cash flow generator
│   └── pricing.py              # Phase 7b: OAS solver, duration, convexity
│
├── notebooks/
│   ├── 01_data_extraction.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_default_logistic.ipynb
│   ├── 04_cox_ph.ipynb
│   ├── 05_psa_prepayment.ipynb
│   ├── 06_vasicek_calibration.ipynb
│   ├── 07_vasicek_simulation.ipynb
│   └── 08_mbs_pricer.ipynb
│
├── data/
│   ├── raw/           # Freddie Mac zip archives (gitignored)
│   ├── processed/     # Parquet panels (gitignored)
│   └── fred/          # FRED macro CSV cache (gitignored)
│
├── artifacts/
│   ├── models/        # Serialized models: logistic_model.pkl, cox_ph_model.pkl
│   ├── figures/       # All chart PNGs
│   └── results/       # mbs_pricing_results.json
│
├── requirements.txt
├── .env.example
└── .pylintrc
```

---

## Data Sources

- **Freddie Mac SF LLD**: Single-Family Loan-Level Dataset, 2018–2025.
  Download from: [Freddie Mac SF LLD](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)
  Format: pipe-delimited, no header, 32 columns (origination) + 32 columns (performance).

- **FRED GS1**: 1-Year Treasury Constant Maturity Rate, 2000–present.
  Fetched automatically via `fredapi` using `FRED_API_KEY` in `.env`.

---

## Non-Negotiable Rules Followed

- `.env` is gitignored and never committed; FRED key loaded exclusively via `python-dotenv`.
- Raw data files and parquets > 50MB are gitignored.
- Column count asserted == 32 before any file parse; mismatches raise `ValueError`.
- All notebooks run top-to-bottom with Run All without manual intervention.
- Temporal split: train < 2024-01-01, valid 2024-01-01–2024-07-01, test ≥ 2024-07-01.
- Parquets in `data/processed/`, models in `artifacts/models/`, figures in `artifacts/figures/`.
