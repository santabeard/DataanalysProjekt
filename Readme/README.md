# DataanalysProjekt

Bayesian updating of a baseline emergency-department (ED) risk model with routinely recorded
vital signs, under realistic missing data. Group 36 project, Uppsala University.

The outcome is **30-day mortality** per ED visit. A baseline model on triage information
(`X_base`: age, sex, chief complaint) is updated with the five vital signs that are routinely
recorded at triage (`V`: SBP, heart rate, respiratory rate, SpO2, temperature). The questions are
whether the vitals add predictive value, whether it pays to keep the baseline model's calibration
(modular updating) instead of refitting everything jointly, and how much the answers depend on how
missing vitals are handled.

## Repository layout

| Path | What it is |
|---|---|
| `data_pipeline/` | The data pipeline, flat as it runs on Bianca: `01_subset` → `02_build` → `03_export_stats` (+ `03b` missingness, `03c` joint stats) → `04_synthesize`. See [`data_pipeline/README.md`](data_pipeline/README.md). |
| `evaluation/` | Cohort building, models, metrics, missingness experiments, the end-to-end runner and the synth-vs-real fidelity check. See [`evaluation/EVALUATION_PLAN.md`](evaluation/EVALUATION_PLAN.md). |
| `preprocessing/` | A pure numpy/scipy Gibbs sampler for the selection-model view of missing vitals, plus SWED-calibrated simulation scenarios. See [`preprocessing/README.md`](preprocessing/README.md). |
| `ed_atlas/`, `bayesianUpdating/`, `bayesian_examples/`, `mortality_imputation/` | Earlier exploration: ED-Atlas material, updating demos and model notes. Reference only. |
| `report.tex` | Methodology write-up. |
| `requirements.txt` | Python dependencies. |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# tests
python3 -m pytest evaluation/tests -q

# full end-to-end evaluation on the committed synthetic cohort
python3 -m evaluation.orchestrator \
  --built data_pipeline/synthetic/built_synth_subset_50k \
  --out evaluation/results/run_50k --n-boot 1000 --n-boot-subgroup 200

# fast smoke run while developing
python3 -m evaluation.orchestrator \
  --built data_pipeline/synthetic/built_synth_subset_50k --out /tmp/quick --quick
```

A run writes one self-contained directory: `run_config.json`, `manifest.json`, `summary.md`,
`verdict.txt`, the metric CSVs, `test_predictions.npz` and `figures/`. `--models`,
`--missing-methods`, `--kappa-grid`, `--seed` and `--no-figures` narrow it down.

To rebuild the synthetic cohort from the raw synthetic tables:

```bash
SWED_DATA=data_pipeline/synthetic/synth_subset_50k \
SWED_WORK=data_pipeline/synthetic/synth_subset_50k \
SWED_OUT=data_pipeline/synthetic \
SWED_REF=data_pipeline/synthetic/data_info_stub \
python3 data_pipeline/02_build.py
```

## Two data environments

The full SWED extract (2.17M patients, 30 parquet files, 9.8 GB) lives on **Bianca** and never
leaves it. The workflow is:

1. On Bianca: subset patients, build the harmonised cohort, and export **disclosure-safe
   aggregates only** — every group needs ≥10 distinct patients, counts are rounded to 5,
   percentiles need ≥50 values (≥1000 for the 1st/99th), and no IDs, free text or exact dates are
   written.
2. Off Bianca: synthesize a cohort from those aggregates. The synthetic files copy the built
   cohort's file names, columns and types, so **code developed locally runs unchanged on the real
   build**.

The committed synthetic cohort (`built_synth_subset_50k`) has 177,281 visits, 50,000 patients and
12 hospitals. Missingness now follows fitted hospital × variable mechanisms from
`03b_export_missingness.py`; `--missingness legacy` restores independent missingness and
`--gx-scale` sweeps the non-ignorable assumption. Details and Bianca paths are in
[`data_pipeline/README.md`](data_pipeline/README.md).

## Models

All Bayesian models use MAP/Laplace rather than MCMC, so the whole comparison runs in minutes
without a PyMC dependency.

| Model | Specification |
|---|---|
| `M_ref` | Baseline `logit p = α + xᵀβ` fitted on `D_old`; keeps `(μ_old, Σ_old)` for the update |
| `M_base` | The same baseline refitted on `D_new`, for the parity check |
| Strategy A | Fixed offset: `logit p = s_i + v_iᵀγ` with `s_i` from the historical fit; `γ` fitted on `D_new` |
| Strategy B | Logistic recalibration: `logit p = a + b·s_i + v_iᵀγ` with priors centred at `a=0, b=1` |
| Strategy C | Posterior-as-prior: `φ = (θ, γ)` with `N([μ_old, 0], blkdiag(Σ_old, I))` |
| `M_direct` | De novo joint fit on `D_new` with an isotropic prior |

Patient-level split: 40/40/20 `D_old / D_new / test`, seed 42, identical test rows for every
model — the runner asserts it. Vitals enter as `(z, z²)` per sign, standardized on training data.

Metrics: micro/macro AUROC, O:E, calibration intercept and slope, log-loss, Brier score, Brier
skill score and Nagelkerke R², each with 95% patient-level bootstrap intervals.

## What the 50k run found

Test set: 35,660 visits, 545 deaths. `evaluation/results/run_50k/summary.md` has the full output.

| Model | Micro-AUROC | BSS | Log-loss |
|---|---|---|---|
| `M_ref` | 0.793 | 0.023 | 0.0702 |
| `M_base` | 0.794 | 0.023 | 0.0701 |
| A | 0.830 | 0.057 | 0.0662 |
| B | 0.831 | 0.056 | 0.0662 |
| C | 0.831 | 0.057 | 0.0661 |
| `M_direct` | 0.832 | 0.056 | 0.0662 |

Adding the vitals is worth roughly 4 AUROC points and halves the Brier skill deficit. The modular
strategies (A/B/C) are **statistically tied** with the de novo joint model on log-loss under a
paired patient bootstrap — which is the useful result: the baseline model's calibration can be
kept without paying for it in accuracy.

## Missing data

- Primary handling is median/mean imputation plus an observation mask; the other strategies
  (complete case, MICE, KNN, mask interactions) are sensitivity analyses in
  `missingness_scoreboard.csv`.
- `kappa_sensitivity.csv` shifts the logit by `κ` without refitting models. `κ < 0` cannot unmask a
  vital that was never built, so the primary sweep is `κ ≥ 0`.
- In the real SWED data, **9 of 12 hospitals have no vital source at all**, so structural
  missingness dominates and per-vital `g_x` from the selection model is prior-dependent. Treat it
  as an assumption, not a clinical finding.

## Limitations

- The synthetic cohort validates **code and workflow**, not clinical conclusions. Distributions
  come from rounded, suppressed aggregates, so rare groups are absent.
- The latent values behind missing rows are model-based assumptions; check `--gx-scale` before
  drawing conclusions from the missingness experiments.
- `report.tex` still uses the older `M0 / M_base / M_modular / M_direct` taxonomy, while the code
  implements Strategies A/B/C. Unifying the two is an open item.
- `03b`/`03c` outputs must be reviewed against the disclosure rules before they leave Bianca.

## Project board

Current work, next steps and the week-to-week timeline:
<https://github.com/users/santabeard/projects/1>

## Team

Emma Marti, Theo and Marwan. Contact: emma.marti.7837@student.uu.se
