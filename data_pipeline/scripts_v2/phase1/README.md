# Phase 1: the pipeline (not yet written)

Skeleton only. Implements `notes/data_plan.pdf` Section 5: six stages, each its
own script, reading only earlier stages' outputs and the raw tables, writing
one parquet file and one report CSV, re-runnable on its own.

Planned layout (one script per stage, matching Section 5.1-5.6):

| Stage | Script (planned) | Out |
|---|---|---|
| 1 | `01_visits.py` | `visits.parquet` |
| 2 | `02_vitals_long.py` | `vitals_long.parquet` |
| 3 | `03_vitals_triage.py` | `vitals_triage.parquet` |
| 4 | `04_outcomes.py` | `outcome_death.parquet`, `outcome_diag_long.parquet` |
| 5 | `05_split.py` | `split.parquet` |
| 6 | `06_model_table.py` | `model_table.parquet`, `data_dictionary.csv` |

Plus `config.yaml` (every rule and number: cohort_end, the triage cutoff from
phase0 Question B, plausible-value limits from phase0, split_seed, ...) and
`checks/` (the "must hold" assertions listed per stage in the plan, as tests).

Blocked on phase0 finishing first: the triage cutoff (B), the hospital-source
map (A), plausible limits, and the consciousness alert mapping are all inputs
`config.yaml` needs before Stage 2/3 can be written correctly.
