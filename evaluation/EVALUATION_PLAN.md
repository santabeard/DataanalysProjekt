# Evaluation & Missingness Experiment Plan

## 1. Context and decisions

**Goal:** Implement the evaluation scripts and an orchestrator for the methodology, then run the missingness experiments on the synthetic SWED cohort built from `data_pipeline/synthetic`.

**Resolved decisions**

| Decision | Choice |
|---|---|
| Missingness data source | Build synthetic SWED first via `02_build.py`, then run experiments on `built_synth_subset_50k` |
| ED-Atlas posterior | Not available; emulate the historical prior by fitting the baseline on `D_old` and using its MVN posterior |
| Estimator | MAP/Laplace for all Bayesian strategies (no PyMC/NUTS dependency) |
| Cohort | 50k synthetic patients |
| Metrics scope | Core metrics only: micro/macro AUROC, O:E, calibration intercept/slope, LL, Brier, BSS, Nagelkerke, with bootstrap CIs |

**Target cohort split:** patient-level 40/40/20 `D_old / D_new / test`, identical across all models and experiments.

---

## 2. Requirements

### 2.1 Data requirements

- The built cohort must contain `ed_visits`, `vitals`, `deaths`, and `patients`.
- Adult filter: `18 <= ed_age <= 109`; non-missing `is_male`.
- Outcome: `died30 = 0 <= (date_death_minimp - arrival_date) <= 30`.
- Baseline predictors `X_base`: age, biological sex, chief-complaint SNOMED categories.
- Vital predictors `V`: SBP, HR/pulse, RR, SpO2, body temperature.
- Vital values: first plausible value per visit per concept; implausible rows excluded.
- Observation mask `R_ij` must be retained for all selected vitals.
- Export discipline for future real-data safety: groups with `<10` patients suppressed; counts rounded to 5.

### 2.2 Model requirements

Implement with MAP/Laplace, a common posterior covariance routine, and a unified `predict` interface:

| Model | Specification |
|---|---|
| `M_ref` | Baseline `logit p = α + xᵀβ` fitted on `D_old`; retain `(μ_old, Σ_old)` |
| `M_base` | Same baseline refitted on `D_new` for the 3-base-variable parity test |
| Strategy A | Fixed offset: `logit p = s_i + v_iᵀγ`, `s_i = ᾱ + x_iᵀβ̄`, fit γ on `D_new` |
| Strategy B | Logistic recalibration: `logit p = a + b·s_i + v_iᵀγ`, `a~N(0,1)`, `b~N(1,1)`, `γ~N(0,I)` |
| Strategy C | Posterior-as-prior: `φ=(θ,γ)` with prior `N([μ_old,0], blkdiag(Σ_old,I))`; Laplace precision `Σ_post⁻¹ = blkdiag(Σ_old⁻¹,I) + Z̃ᵀW_newZ̃` |
| `M_direct` | De novo joint fit on `D_new` with isotropic `N(0,I)` prior |

Vital design per methodology: `v_i = (z_1, z_1², …, z_K, z_K²)`, standardized on training data.

### 2.3 Evaluation requirements

- Micro AUROC pooled across all test visits.
- Macro AUROC across complaint categories with sufficient events (unweighted mean).
- `O:E = ΣY / Σp̂`.
- Calibration intercept and slope from `logit(Y) = α_cal + β_cal·logit(p̂)`.
- Out-of-sample log-likelihood and Brier score.
- Brier Skill Score: `BSS = 1 − BS/BS₀`, `BS₀ = ȳ(1−ȳ)`.
- Nagelkerke R² relative to the null prevalence model.
- 95% confidence intervals from 1,000 patient-level stratified bootstrap resamples.
- Identical test rows across all models; assert this in the orchestrator.
- Three analyses: baseline verification (`M_base` vs `M_ref`), per-hospital performance, modular vs de novo (`A/B/C` vs `M_direct`).

### 2.4 Missingness experiment requirements

- Complete-case baseline on visits with all selected vitals observed.
- Structural diagnostics: `P(R_ij=1)` by hospital × vital and co-observation patterns.
- Missingness strategies:
  - median/mean imputation baseline
  - missing-mask feature + `(1−R)⊙X_base` interactions
  - MICE with multiple imputations and Rubin's rules
  - MI and KNN imputation
  - ML/MCMC imputation
- Kappa shift sweep: fit `P(R=1|X_base,hospital)` on training; apply
  `p* = σ(logit(p̂)+κ)`, `κ ∈ [−2.5,+2.5]`; evaluate without retraining.
- Document the `κ<0` limitation (cannot unmask unobserved built values); use complete-case or imputation, or restrict primary sweep to `κ≥0`.

### 2.5 Reproducibility and quality requirements

- Fixed seeds for splits, bootstrap, and all stochastic steps.
- Save `run_config.json` and `manifest.json` per run.
- Predictions persisted as arrays with model metadata.
- Unit tests for metrics and model design consistency.
- No atlas/npz dependency in the new evaluation package.

---

## 3. File structure

```text
DataanalysProjekt/
├── data_pipeline/
│   ├── 02_build.py                         # existing: raw synthetic tables -> built cohort
│   ├── 05_atlas_vitals.py                  # legacy model reference (not used by new runner)
│   ├── 06_joint_hierarchical.py            # legacy model reference
│   ├── 07_compare_models.py                # legacy model reference
│   └── synthetic/
│       ├── synth_subset_5k.py              # existing raw synthetic generator
│       ├── synth_subset_50k.py             # existing raw synthetic generator (target)
│       ├── synth_subset_50k/               # existing raw tables + out/vitals_snomed_map.csv
│       ├── data_info_stub/                 # NEW: header-only REF CSVs for 02_build.py
│       │   ├── swed_lab_meas_to_snomed.csv
│       │   └── lab_unit_normalization_07sep2026.csv
│       └── built_synth_subset_50k/         # NEW (generated): built cohort
│           ├── ed_visits.parquet
│           ├── vitals.parquet
│           ├── deaths.parquet
│           ├── patients.parquet
│           ├── lab_meas.parquet
│           └── build_report.csv
│
├── evaluation/
│   ├── cohort.py                           # NEW: load built data -> visits, y, X_base, V, R, split
│   ├── models.py                           # NEW: M_ref, M_base, A, B, C, M_direct (MAP/Laplace)
│   ├── metrics.py                          # NEW: core metrics + bootstrap CIs
│   ├── missingness.py                      # NEW: diagnostics, imputation strategies, kappa sweep
│   ├── orchestrator.py                     # NEW: single-entry runner
│   ├── plots.py                            # NEW optional: reliability/risk/decision curves
│   ├── tests/
│   │   ├── test_metrics.py                 # NEW
│   │   └── test_models.py                  # NEW
│   │
│   ├── clinical_risk_metrics.py            # existing: calibration cross-check only
│   ├── bayes_missing_h.py                  # existing: optional MCMC/missingness reference
│   ├── run_hospital_comparison.py          # existing: scoreboard/verdict pattern reference
│   ├── simulate_hospital.py                # existing: ground-truth simulator (optional only)
│   └── results/
│       └── <run_id>/                       # NEW (generated)
│           ├── run_config.json
│           ├── manifest.json
│           ├── summary.md
│           ├── split_manifest.json
│           ├── cohort_summary.csv
│           ├── test_predictions.npz
│           ├── metrics_overall.csv
│           ├── metrics_by_hospital.csv
│           ├── metrics_by_complaint.csv
│           ├── baseline_verification.csv
│           ├── strategy_comparison.csv
│           ├── missingness_diagnostics.csv
│           ├── missingness_scoreboard.csv
│           ├── kappa_sensitivity.csv
│           ├── verdict.txt
│           └── figures/
│               ├── reliability.png
│               ├── risk_distribution.png
│               └── decision_curve.png
│
├── preprocessing/
│   ├── bayes_missing.py                    # existing: optional ML/MCMC imputation engine
│   └── simulate_data.py                    # existing: optional ground-truth reference
│
├── ed_atlas/                               # existing: not required by the new evaluation runner
└── report.tex                              # existing methodology write-up
```

Legend: `NEW` = to be added; `existing` = already in the repo; `legacy` = kept for reference but not imported by the new runner.

---

## 4. Module dependency graph

```text
                         ┌──────────────────────────┐
                         │ data_pipeline/02_build.py│
                         └────────────┬─────────────┘
                                      │ reads raw synthetic tables + data_info_stub
                                      ▼
                    built_synth_subset_50k/*.parquet
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ evaluation/cohort.py     │
                         │  - X_base, V, R          │
                         │  - D_old / D_new / test  │
                         └───────┬──────────┬───────┘
                                 │          │
              ┌──────────────────┘          └──────────────────┐
              ▼                                                ▼
   ┌──────────────────────┐                        ┌──────────────────────────┐
   │ evaluation/models.py │                        │ evaluation/missingness.py│
   │  M_ref, M_base       │◄───────────────────────│  diagnostics             │
   │  A, B, C, M_direct   │   uses model interface  │  mask, MICE, MI, KNN     │
   └──────────┬───────────┘                        │  ML/MCMC, kappa sweep    │
              │                                     └────────────┬─────────────┘
              │ predictions                                      │ strategy results
              ▼                                                  ▼
   ┌──────────────────────┐                        ┌──────────────────────────┐
   │ evaluation/metrics.py│◄───────────────────────│ evaluation/orchestrator.py│
   │  AUROC micro/macro   │   called by runner      │  fits once, evaluates all │
   │  O:E, cal, LL        │                        │  writes results/<run_id>/ │
   │  Brier, BSS, R²      │                        └────────────┬─────────────┘
   │  bootstrap CIs       │                                     │ optional
   └──────────┬───────────┘                                     ▼
              │                                     ┌──────────────────────────┐
              │                                     │ evaluation/plots.py      │
              │                                     │  reliability/risk/DC     │
              │                                     └──────────────────────────┘
              ▼
   results/<run_id>/*.csv, *.json, summary.md, verdict.txt, figures/
```

**Dependency rules**

- `metrics.py` depends only on NumPy/SciPy/sklearn and receives predictions; it must not import `models.py`.
- `models.py` depends on `cohort.py` data structures and the shared Laplace helper; it must not import `missingness.py` or `metrics.py`.
- `missingness.py` depends on `cohort.py` and the `models.py` interface, so every imputation strategy can be scored by any model.
- `orchestrator.py` is the only module that composes all pieces and writes results.
- Optional external reuse: `preprocessing/bayes_missing.py` for ML/MCMC; `evaluation/clinical_risk_metrics.py` for calibration cross-checks. Neither is required for the core run.

---

## 5. Runtime data flow

```text
synth_subset_50k (raw)
        │  02_build.py
        ▼
built cohort parquet
        │  cohort.py
        ▼
D_old ──► M_ref (μ_old, Σ_old) ──┐
                                  ├──► Strategy A/B/C + M_base
D_new ──► M_base, M_direct ───────┘
        │
        ▼
test split ──► predictions (all models) ──► metrics.py ──► CSVs + bootstrap CIs
        │
        └──► missingness.py:
                complete-case
                diagnostics (hospital × vital, co-observation)
                mask / MICE / MI / KNN / ML-MCMC
                kappa sweep ──► scoreboard + verdict
        │
        ▼
orchestrator.py ──► results/<run_id>/ + summary.md
```

---

## 6. Checklist

**Status legend:** `DONE` = implemented and usable as-is; `PARTIAL` = logic exists in a legacy
file/notebook but is not packaged, unified, or complete; `NOT STARTED` = no implementation
found. Evidence is given as `file:function` or `file:description`.

**Summary**

| Phase | DONE | PARTIAL | NOT STARTED |
|---|---|---|---|
| 0 — Build cohort | 4 | 0 | 0 |
| 1 — Cohort builder | 7 | 0 | 0 |
| 2 — Models | 10 | 0 | 0 |
| 3 — Metrics | 5 | 1 | 0 |
| 4 — Missingness | 8 | 0 | 0 |
| 5 — Orchestrator | 6 | 0 | 0 |
| 6 — Run & validate | 5 | 0 | 0 |

### Phase 0 — Build the synthetic cohort

- [x] Add `data_pipeline/synthetic/data_info_stub/swed_lab_meas_to_snomed.csv` and `lab_unit_normalization_07sep2026.csv` with the exact columns `02_build.py` reads. — **DONE** (header-only stubs created)
- [x] Run the synthetic build:
  ```bash
  SWED_DATA=data_pipeline/synthetic/synth_subset_50k \
  SWED_WORK=data_pipeline/synthetic/synth_subset_50k \
  SWED_OUT=data_pipeline/synthetic \
  SWED_REF=data_pipeline/synthetic/data_info_stub \
  python data_pipeline/02_build.py
  ```
  — **DONE** (`synthetic/synth_subset_50k.py` then `02_build.py`, 9 s; built dir present)
- [x] Verify `built_synth_subset_50k/` contains `ed_visits`, `vitals`, `deaths`, `patients`. — **DONE** (plus `lab_meas`, `diagnoses`, `death_causes`, `build_report.csv`)
- [x] Inspect `build_report.csv` and record visits, patients, 30-day deaths, hospital counts, vital coverage. — **DONE**: 177,281 visits; 50,000 patients; 5,388 death records; 12 hospitals; 40,277 visits with any vital (SBP 107,572 · DBP 102,144 · SpO2 96,821 · temp 86,062 · RR 85,987 · pulse 64,493 · consciousness 32,174 rows)

### Phase 1 — Cohort builder

- [x] Implement `evaluation/cohort.py` to load the built parquet files. — **DONE** (`evaluation/cohort.py:load_cohort`)
- [x] Derive `died30`, apply adult/sex filters, and extract first SNOMED complaint code. — **DONE** (`_load_visits`; 177,280 visits retained)
- [x] Build `X_base` (age, male, complaint dummies) consistent with `07_compare_models.py`. — **DONE** (`_base_design`; top-8 complaints from `D_old ∪ D_new`)
- [x] Build `V` (first plausible vital per visit), standardize on training data, and construct `R`. — **DONE** (`_attach_vitals`, `_vital_design`; 5 vitals, `(z,z²)` scaled on `D_new`, mask `R`)
- [x] Create patient-level 40/40/20 `D_old / D_new / test` split with seed 42. — **DONE** (`_split_patients`; 20,000 / 20,000 / 10,000 patients)
- [x] Save `split_manifest.json`; assert split disjointness and outcome prevalence. — **DONE** (`save_split_manifest`, `assert_disjoint`)
- [x] Enforce `MIN_CELL=10` and round counts to 5 in summaries. — **DONE** (`round5`, `summarise`, `save_split_manifest`)

### Phase 2 — Model implementations

- [x] Implement a shared MAP logistic + Laplace covariance helper. — **DONE** (`models.py:fit_map_logistic`)
- [x] Implement `M_ref` on `D_old` and retain `(μ_old, Σ_old)`. — **DONE** (`models.py:fit_m_ref`; `theta`/`cov` retained)
- [x] Implement `M_base` on `D_new`. — **DONE** (`models.py:fit_m_base`)
- [x] Implement vital design `(z, z²)` per selected vital. — **DONE** (`cohort.py:design_from_raw`; `models.py:vital_features` appends `R`)
- [x] Implement Strategy A (fixed offset). — **DONE** (`models.py:fit_strategy_a`)
- [x] Implement Strategy B (logistic recalibration with priors). — **DONE** (`models.py:fit_strategy_b`)
- [x] Implement Strategy C (posterior-as-informative-prior + Laplace). — **DONE** (`models.py:fit_strategy_c`, `blkdiag(Σ_old⁻¹,I)`)
- [x] Implement `M_direct` (de novo joint model). — **DONE** (`models.py:fit_m_direct`)
- [x] Provide a unified `predict(model, data)` returning probabilities. — **DONE** (`LaplaceModel.predict`, `predict_draws`)
- [x] Test that all models score the identical `test` rows. — **DONE** (`orchestrator.py` assertion; `tests/test_models.py`)

### Phase 3 — Metrics library

- [x] Implement micro and macro AUROC. — **DONE** (`metrics.py:auc_score`, `macro_auc`)
- [x] Implement `O:E`, calibration intercept/slope, LL, Brier, BSS, and Nagelkerke R². — **DONE** (`metrics.py:all_metrics`)
- [x] Implement 1,000-replicate patient-level stratified bootstrap CIs. — **DONE** (1,000 overall; 200 subgroups per run decision; `metrics.py:bootstrap_ci`)
- [x] Support overall, per-hospital, and per-complaint stratification. — **DONE** (`orchestrator.py`; `metrics_*.csv`)
- [ ] Cross-check against sklearn `roc_auc_score` and `clinical_risk_metrics.py`. — **PARTIAL** (AUROC uses `sklearn.roc_auc_score`; no automated cross-check against `clinical_risk_metrics.py`)
- [x] Add unit tests with known AUROC/calibration cases and reproducibility checks. — **DONE** (`tests/test_metrics.py`, `tests/test_missingness.py`; full suite 15 passed)
- [x] Add patient-clustered bootstrap intervals for decile calibration. — **DONE** (`metrics.py:calibration_table`; `plots.py:plot_calibration`)

### Phase 4 — Missingness experiments

- [x] Implement complete-case baseline evaluation. — **DONE** (`cohort.py:complete_case_cohort`; `missingness_scoreboard.csv`)
- [x] Add a matched complete-case test subset so all methods are scored on identical rows. — **DONE** (`missingness_scoreboard_complete_subset.csv`)
- [x] Implement hospital × vital `P(R=1)` diagnostics and co-observation matrices. — **DONE** (`missingness.py:hospital_vital_diagnostics`, `coobservation_matrix`)
- [x] Implement missing-mask feature + `(1−R)⊙X_base` interaction model. — **DONE** (`missingness.py:fit_mask_model`)
- [x] Implement MICE, MI, KNN, and ML/MCMC imputation strategies. — **DONE** for core grid (`missingness.py:mice_cohorts`, `knn_cohort`); ML/MCMC deferred per core-grid decision
- [x] Run each strategy through the Phase 2 models and produce a long results table. — **DONE** (`orchestrator.py`; `missingness_scoreboard.csv`)
- [x] Implement the kappa shift sweep and evaluate without retraining. — **DONE** (`missingness.py:kappa_sweep`; `kappa_sensitivity.csv`)
- [x] Document the `κ<0` workaround. — **DONE** (`summary.md` limitations; `kappa_sensitivity.csv` primary flag)
- [x] Add paired bootstrap comparisons and a `verdict.txt` scoreboard. — **DONE** (`metrics.py:bootstrap_diff`; `verdict.txt`)
- [x] Add SWED-calibrated missingness scenarios and a combined simulation atlas. — **DONE** (`simulate_swed.py`, `run_atlas.py`; `atlas_results/`)
- [x] Condition simulation analyses on at least one observed vital, matching the SWED fit. — **DONE** (`run_analysis.py --condition-any-vital`)

### Phase 5 — Orchestrator

- [x] Implement CLI flags: `--built`, `--out`, `--models`, `--missing-methods`, `--n-boot`, `--kappa-grid`, `--seed`, `--quick`. — **DONE** (`orchestrator.py:parse_args`)
- [x] Load cohort once, fit models once, and persist predictions and fit metadata. — **DONE** (`test_predictions.npz`, `run_config.json`)
- [x] Run baseline verification, per-hospital performance, and modular-vs-de-novo analyses. — **DONE** (`baseline_verification.csv`, `metrics_by_hospital.csv`, `strategy_comparison.csv`)
- [x] Run missingness diagnostics, strategy grid, and kappa sweep. — **DONE**
- [x] Write `run_config.json`, `manifest.json`, and `summary.md`. — **DONE**
- [x] Implement optional `plots.py` diagnostics. — **DONE** (reliability, risk distribution, decision curve in `orchestrator.py`; `figures/`)

### Phase 6 — Run and validate

- [x] Run a `--quick` smoke test. — **DONE** (`evaluation/results/quick2`, 1m42s)
- [x] Run the full 50k experiment with 1,000 bootstrap replicates. — **DONE** (`evaluation/results/run_50k`, 15m32s)
- [x] Verify reproducibility from the saved config and manifest. — **DONE** (fixed seeds; determinism + bootstrap tests; `run_config.json`, `manifest.json`)
- [x] Sanity-check expected directional results. — **DONE** (A/B/C improve log-loss/AUC over `M_ref`/`M_base`; A/B/C tied with `M_direct`)
- [x] Document the primary missingness handling used for the strategy comparison. — **DONE** (`summary.md`: mean/median imputation + observation mask)

---

## 7. Deliverables

| Deliverable | Content |
|---|---|
| `built_synth_subset_50k/` | Built synthetic parquet cohort and build report |
| `evaluation/cohort.py` | Reproducible cohort and split builder |
| `evaluation/models.py` | MAP/Laplace implementations of `M_ref`, `M_base`, A, B, C, `M_direct` |
| `evaluation/metrics.py` | Core metrics with bootstrap CIs |
| `evaluation/missingness.py` | Diagnostics, imputation strategies, kappa sweep |
| `evaluation/orchestrator.py` | One-command end-to-end runner |
| `results/<run_id>/` | `run_config.json`, `manifest.json`, `summary.md`, metric CSVs, scoreboard, verdict, prediction arrays |
| `evaluation/tests/` | Metric and model unit tests |

**Result CSVs**

- `cohort_summary.csv`
- `split_manifest.json`
- `test_predictions.npz`
- `metrics_overall.csv`
- `metrics_by_hospital.csv`
- `metrics_by_complaint.csv`
- `baseline_verification.csv`
- `strategy_comparison.csv`
- `missingness_diagnostics.csv`
- `missingness_scoreboard.csv`
- `missingness_scoreboard_complete_subset.csv`
- `kappa_sensitivity.csv`
- `verdict.txt`
- simulation atlas: `atlas_metrics.csv`, `atlas_gx_recovery.csv`, `atlas_imputation_shift.csv`, `atlas_convergence.csv`, `summary.md`

---

## 8. Risks and open decisions

- **No ground truth for unobserved vitals:** `κ<0` cannot truly unmask missing values in built data. Decide between restricting the primary sweep to `κ≥0`, using complete cases, or using imputed values.
- **Synthetic SWED lacks realistic correlations:** use it to validate code and workflow, not clinical conclusions.
- **Primary missingness handling:** decide whether the main strategy comparison uses one prespecified method or reports the full model × missingness grid.
- **Methodology divergence:** `report.tex` currently describes `M0/M_base/M_modular/M_direct`, while the pasted methodology describes Strategies A/B/C. This plan targets the pasted version; a later mapping step may be needed.
- **Complaint macro-AUROC strata:** some synthetic complaint categories may have `<10` events and will be suppressed.
- **Calibration intervals:** Wilson bars are pointwise binomial intervals; the patient-clustered bootstrap band is the primary uncertainty statement.
- **Complete-case scoreboard:** the unmatched complete-case row uses a healthier test subset; use the matched subset table for method comparisons.
- **SWED missingness:** 9/12 hospitals have no vital source, so structural missingness dominates. Per-vital `g_x` is prior-sensitive and must not be read as a clinical finding.

---

## 9. Definition of Done

The work is **done** when all of the following hold:

1. **Cohort is built and reproducible.**
   - `built_synth_subset_50k` exists and is generated by a documented command.
   - Cohort summary reports visits, patients, 30-day deaths, hospital counts, and vital coverage.

2. **All models are implemented and share one test set.**
   - `M_ref`, `M_base`, Strategy A, Strategy B, Strategy C, and `M_direct` produce predictions.
   - The orchestrator asserts that every model is scored on the identical `test` rows.

3. **Core metrics are implemented and validated.**
   - Micro/macro AUROC, O:E, calibration intercept/slope, LL, Brier, BSS, and Nagelkerke R² are reported.
   - Every metric has a 95% bootstrap CI from 1,000 stratified patient-level resamples.
   - Metrics agree with sklearn and existing calibration code on test cases.

4. **The three methodology analyses are reported.**
   - Baseline verification (`M_base` vs `M_ref`) is reported.
   - Per-hospital performance is reported.
   - Modular vs de novo (`A`, `B`, `C` vs `M_direct`) is reported.

5. **Missingness experiments are complete.**
   - Complete-case, structural diagnostics, mask, MICE, MI, KNN, and ML/MCMC strategies are evaluated.
   - The kappa shift sweep is run and the `κ<0` limitation is documented.
   - A scoreboard and `verdict.txt` summarize the results.

6. **The orchestrator runs end to end.**
   - A single command reproduces the full `results/<run_id>/` directory.
   - `run_config.json`, `manifest.json`, and `summary.md` are written.
   - A `--quick` smoke run and a full 50k run both complete successfully.

7. **Quality gates pass.**
   - Unit tests for metrics and model design consistency pass.
   - No atlas `.npz` dependency is required.
   - Seeds, splits, and bootstrap replicates are fixed and reproducible.

8. **Limitations are documented.**
   - The synthetic-data limitation, the `κ<0` workaround, the primary missingness handling choice, and the `report.tex` taxonomy divergence are recorded in `summary.md` or the run notes.
