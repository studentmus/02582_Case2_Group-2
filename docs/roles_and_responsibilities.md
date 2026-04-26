# Roles and responsibilities

## Role 1 — Data & Preprocessing Lead

- Collect and organize the EmoPairCompete data locally under `data/raw/` (HRdata/HRdata2 and time-series folders).
- Implement data loading utilities in `src/data/load_data.py` for:
  - feature-level CSVs (e.g. HRdata2.csv)
  - optional time-series signals (EDA, HR, TEMP, BVP) per subject/phase.
- Design and run the preprocessing pipeline in `src/data/preprocess.py`:
  - handle missing values and obvious artefacts
  - standardize / normalize features where appropriate
  - encode categorical variables and merge labels / metadata.
- Provide a clean, documented feature matrix for modelling:
  - `X_processed` (numpy array or DataFrame)
  - `y` / condition labels (e.g. resting vs puzzling vs recovery, subjects, rounds)
  - indices/IDs for traceability back to raw data.
- Maintain basic EDA notebooks (`01_eda.ipynb`, `02_preprocessing.ipynb`) to:
  - check distributions, outliers, and correlations
  - verify that preprocessing decisions are reasonable.
- Coordinate with the Modelling and Evaluation roles to ensure that:
  - the pipelines receive consistent inputs
  - any data leakage or label contamination is avoided.

## Role 2 — Modelling Lead

(Describe responsibilities here.)

## Role 3 — Evaluation & Interpretation Lead

(Describe responsibilities here.)

## Role 4 — Report & Writing Lead

(Describe responsibilities here.)

## Role 5 — Visuals & Video Lead

(Describe responsibilities here.)
