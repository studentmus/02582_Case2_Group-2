# 02582 Case 2 — Group 2

Repository for DTU 02582 Case II project based on the EmoPairCompete dataset.

## Project focus
We are building an exploratory machine learning pipeline for physiological data analysis, with emphasis on preprocessing, representation learning, clustering/anomaly detection, evaluation, and critical interpretation.

## Repository structure
- `data/` — local data storage (not tracked in git)
- `notebooks/` — exploratory analysis and prototyping
- `src/` — reusable pipeline code
- `configs/` — experiment configuration files
- `results/` — saved outputs, logs, embeddings, evaluations
- `reports/` — report, figures, poster, and video materials
- `docs/` — project planning and team coordination
- `tests/` — lightweight tests for preprocessing and evaluation code

## Setup
### Option 1: venv
```bash
bash setup.sh
source .venv/bin/activate
```

### Option 2: conda
```bash
conda env create -f environment.yml
conda activate case2-env
```

## Notes
- Do not commit raw or processed dataset files.
- Keep reusable logic in `src/`, not only in notebooks.
- Save final figures for the report in `reports/figures/`.
- Log important decisions in `docs/decision_log.md`.

## Suggested workflow
1. EDA and preprocessing checks.
2. Build baseline representation methods (e.g. PCA/UMAP).
3. Run clustering or anomaly detection experiments.
4. Evaluate robustness and limitations.
5. Export figures/tables for the final report and video poster.
