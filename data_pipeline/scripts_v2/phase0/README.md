# Phase 0: exploration

Implements `notes/data_plan.pdf` Section 4. Reads the raw `clean_parquet` tables
directly (the phase1 pipeline doesn't exist yet when this runs). Everything here
writes small aggregate CSVs to `$SWED_OUT/explore/phase0/<NN_script_name>/`, each
passed through `apply_export_rule` (>= 10 patients/cell, counts rounded to 5)
before it's safe to copy off the cluster.

## Subset

No new subset script: reuse `scripts_v1/01_subset.py` with `-n 200000`, per the
plan ("made with `data_pipeline/01_subset.py -n 200000`"):

```
python ../scripts_v1/01_subset.py --n 200000
export SWED_DATA=$SWED_OUT/subset
```

Run everything below against `$SWED_DATA=$SWED_OUT/subset` first; repeat the
required questions (A-D) once on the full data to confirm.

## Run order

| # | Script | Produces | Feeds |
|---|---|---|---|
| 1 | `01_map_vital_terms.py` | `vital_term_candidates.csv`, `unmapped_terms.csv` | review by hand -> `$SWED_WORK/out/vitals_concept_map.csv`, used by every script below |
| 2 | `02_question_a_vitals_coverage.py` | coverage by hospital x year, hospital-source candidates | Appendix B (hospital-source map) |
| 3 | `03_question_b_vitals_timing.py` | first-value timing curves, cutoff recommendation | the triage cutoff (Stage 3 window) |
| 4 | `04_question_c_missingness_bias.py` | mortality/diagnosis rates by has_vital x priority x arrival type | whether vitals-missing visits are excludable |
| 5 | `05_question_d_priority_vitals.py` | vital distribution by priority, R² of priority on vitals | whether priority is used as a predictor |
| 6 | `06_units_table.py` | source x term x unit percentiles | reviewed `vital_units.csv` |
| 7 | `07_value_distributions.py` | concept x source percentile ladder | config.yaml plausible-value limits |
| 8 | `08_consciousness_scales.py` | scale x hospital, value counts | alert/not-alert mapping per scale |
| 9 | `09_diagnosis_prevalence.py` | 3-char code prevalence, short-code (candidate block) counts | NODIAG/block handling -- **see the script's docstring: block-coded detection is unverified** |
| 10 | `10_revisits_and_inexact_times.py` | revisit rates, inexact-time share by hospital/year | the "Inexact arrival time" exclusion |

Script 1 must run (and be reviewed) before 2-5 and 8 for the concept mapping to
be trusted; the rest are independent of each other.

## Known gaps, to close before trusting the output

- SNOMED codes in `common.py::VITAL_CONCEPTS` are from the plan text, not
  independently verified against the SNOMED browser.
- `09_diagnosis_prevalence.py` cannot roll 3-char codes up to ICD-10
  blocks/chapters -- no block/chapter reference table exists in `data_info/`
  yet (data_plan.pdf Section 7, open question 6).
- All "4h window" and "±30min" choices used for A/C/D are the plan's own
  definitions where stated; where the plan doesn't pin a window for a
  question (C), these scripts reuse the Question A definition and say so in
  the docstring -- revisit this once B's cutoff is fixed.
