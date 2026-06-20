# MBS Risk Management

Loan-level MBS risk framework using real Freddie Mac SF LLD data (2018–2025).

## Modules

| Module | Description | Status |
|--------|-------------|--------|
| A — Default Risk | Logistic regression + Cox PH survival model | Planned |
| B — Prepayment Risk | PSA benchmark + rate-dependent CPR | Planned |
| C — Interest Rate | Vasicek (Ornstein-Uhlenbeck) calibrated to FRED Treasury yields | Planned |
| D — MBS Pricer | Monte Carlo cash flow simulation, OAS solver, duration/convexity | Planned |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in your FRED API key
```

## Project Structure

```
data/raw/          Freddie Mac pipe-delimited text files (gitignored)
data/processed/    Parquet modeling panels (gitignored)
data/fred/         FRED macro series (gitignored)
src/               Production Python modules
notebooks/         Jupyter notebooks (one per phase)
artifacts/         Serialized models, figures, results JSON
```

## Data

Freddie Mac Single-Family Loan-Level Dataset (SF LLD), 2018–2025.
Download from: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
