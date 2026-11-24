# Design: BLR vitals model (train_clean → died_30d + Hdia)

> Maintained by the /grill-top-baldy interviewer. Researchers read it but never edit it.
> Status: R1+R2 settled; plan written, awaiting Top Baldy's approval

## Goal

Design a Bayesian logistic regression (BLR) model that uses `500K_train_clean.parquet`
(output of `data_pipeline/scripts_v2/phase1/`) to predict 30-day mortality (`died_30d`) and
30-day diagnosis (`Hdia` / `diag_class`), the same two outcomes `ed_atlas` predicts, and to make
use of any clever technique `ed_atlas` itself uses (see **Attribution rule** below).

The simple template to start from is model **3.4 ("new data only")** from
`bayesianUpdating/demos/demo_model_update/demo_model_update.ipynb`: Top Baldy only has access to
SWED (essentially what `ed_atlas` itself was trained on — `D_old` in that notebook's framing),
not a separate, non-overlapping new dataset. So none of the atlas-posterior-as-prior update
tricks in that notebook (3.1, 3.2, 3.3, 3.5, 3.6) apply without double-counting the same
patients' information and introducing bias Top Baldy explicitly wants to avoid.

A new `training/` directory has been created on Bianca. `train_clean` will be copied there, then
statistical/ML preprocessing (standardization, age splines, etc.) and the model fit happen inside
it.

Top Baldy asked to move fast — keep rounds tight, don't over-research.

### Attribution rule (hard constraint for this whole project)

`ed_atlas_impl/` (Top Baldy's existing, separate implementation repo) is read-only reference
material. Anything sourced from it — a technique, a formula, a prior, a code pattern — must be
explicitly labelled "per `ed_atlas_impl`'s `<file>`" everywhere it appears: in this document, in
researcher replies, in quiz option text, and later in any actual code and its comments. Nothing
from it is ever presented as independently derived. Confirmed sourced-from-`ed_atlas_impl`
techniques now in the plan:
- the weighted-collapsed-rows trick + custom weighted-Bernoulli likelihood MECHANISM
  (`generate_dataset.py`, `train_mort.py`, `train_diag.py`) — reused in R1 Q3, but applied to
  this project's own `complaint_weight` column, not ed_atlas's row-counts
- the patsy formula shape `sex * cr(age, knots=..., lower_bound=..., upper_bound=...) * symptoms`
  (`train_mort.py`, `train_diag.py`, `ed_atlas/convert_traces.py`)
- the age spline knots `[22, 43, 65, 88]` (`train_diag.py`) — reused as-is in R1 Q4
- the per-row structured-sparsity trick (10 base dims + 10 per-complaint dims; `base_cov`/
  `sym_covs`/`cross_covs`) from `ed_atlas/convert_traces.py` and `_lookup.py` — ed_atlas uses this
  only at prediction time; R1 Q1 reuses it *inside training* too (the researcher's own extension,
  not itself from `ed_atlas_impl`)
- the two-complaint probability-space mixing rule `0.5·σ(μ1)+0.5·σ(μ2)` from `ed_atlas/_lookup.py`
  — R1 Q3's runner-up option (mixture likelihood) would have applied this at training time too;
  not the option chosen (R1 Q3 chose the weighted-likelihood option instead, see Settled decisions)
- the Laplace/MAP-approximation fallback's covariance shape (block-diagonal: base + per-complaint
  + cross blocks, no complaint×complaint term) mirrors `ed_atlas/convert_traces.py`'s own shipped
  structure — chosen in R1 Q7
- NOT reused: the one-vs-rest-per-diagnosis-code architecture (`train_diag.py`) — R1 Q2 chose a
  tree-shaped nested categorical instead, since `diag_class` is single-label, unlike ed_atlas's
  multi-label targets
- NOT reused: ed_atlas's own unweighted double-counting of multi-symptom visits
  (`generate_dataset.py`'s explode+count) — R1 Q3 explicitly chose to weight instead (this
  project's own decision 11, not ed_atlas's practice)
- NOT reused: ed_atlas's exact intercept prior `N(-log(9), 0.5)` — R1 Q5 chose a hierarchical
  prior on complaint deviations instead (override, see Settled decisions)

### Scope notes
- No code is written and nothing is touched on Bianca during this design loop (hard rule 2).
  I (the interviewer) never run anything on Bianca myself, per Top Baldy's standing rule.
- `train_clean`'s actual schema (verified on real 500K-patient Bianca data: 1,805,088 rows × 329
  columns total): **19 fixed columns** — `ed_visit_id`, `lopnr_swed` (key only, never a feature),
  `ed_age`, `is_male`, `complaint_sctid`/`complaint_role`/`complaint_weight`,
  `vital_systolic_bp`/`diastolic_bp`/`spo2`/`temperature`/`pulse`/`resp_rate` (all nullable, no
  imputation applied upstream), `died_30d`, `death_days_to_death`, `death_date_inexact`,
  `diag_hdia_first`, `diag_hdia_any`, `diag_class` (NODIAG / top-K exact `Hdia` codes /
  `other_<chapter>` buckets) — **plus ~310 `complaint_bucket_*` one-hot columns**, i.e. the real
  complaint vocabulary size is S≈300.
- Note on attribution: the "collapse identical rows" trick in `demo_model_update.ipynb` cell 25
  is itself attributed there to a *different* notebook, `demo_adv` — cite that specific idea to
  `demo_adv`, not to `demo_model_update.ipynb`.
- Real observed rates on the 500K-patient subset: `died_30d` ≈ 1.9%, `diag_class != NODIAG` ≈
  85.6% (existence semantics over *any* specialist/inpatient registry row in 30 days, not just
  ED-related — a broad, majority-positive target, not a rare-event one).
- Two-complaint visits are duplicated rows (`complaint_role` primary/secondary,
  `complaint_weight` 0.5/0.5), per the earlier pipeline design (Section 3.1,
  `ed_atlas_vitals_pipeline_discussion.tex`) — not an `ed_atlas_impl` technique, this project's
  own pipeline decision.
- `diag_class` is **not** the same target ed_atlas predicts: single Hdia only (first by date),
  R*/Z* codes kept (ed_atlas excludes them), death registry not included. Claims of direct
  comparability with ed_atlas's own diagnosis models should stay weak (R1 researcher's caveat).
- All R1 benchmark numbers (gradient timings, memory, NUTS extrapolations) were measured on Top
  Baldy's laptop (i7-10750H, 6 cores, 15GB RAM, CPU-only JAX), **not Bianca** — treat as directional,
  not as Bianca-accurate, until re-measured there.

## Settings
- max_options: 5 (counting "Need more information"; default)
- max_questions: 8 (per quiz; default)

## Team
| Researcher | Kind (agent/session) | ID or name | Model | Effort | Area by round (R1: …, R2: …) |
|---|---|---|---|---|---|
| R1 | agent | af8b54c9c329e6d31 | opus | high | R1: all 8 questions |

Note: Top Baldy asked to move fast from R2 onward — effort steps down to **medium** for every
subsequent round (R1 stayed `high`).

## Settled decisions
| Q | Stakes | Decision | Choice (score) | Top score(s) | Override? (Top Baldy's reason) | Round |
|---|---|---|---|---|---|---|
| Q1 | 🔒 | Fitting strategy at scale | A. Gather-NUTS on all ~1.8M rows, index-gather likelihood instead of dense patsy matrix (9/10) | A (9) | No | R1 |
| Q2 | 🔒 | Diagnosis multiclass architecture | A. Tree-shaped nested categorical: NODIAG → chapter → exact top-K code (8/10) | A (8) | No | R1 |
| Q3 | 🔒 | Weighted-likelihood reuse for `complaint_weight` | A. Weighted Bernoulli/categorical likelihood using `complaint_weight`, mechanism per `ed_atlas_impl` (8/10) | A (8) | No | R1 |
| Q4 | 🔒 | Age spline knots | B. Reuse ed_atlas's exact `[22,43,65,88]` (8/10) | A, B (8 tied) | No (top-tied) | R1 |
| Q5 | 🔒 | Prior for intercept/coefficients | B. Hierarchical prior on complaint deviations, non-centred (7/10) | A (8) | **Yes** — "Gut feeling" | R1 |
| Q6 | ↩ | Standardization scheme | A. Z-score, fit on the Stage-05 training split (8/10) | A (8) | No | R1 |
| Q7 | 🔒 | Sampler stack feasibility + fallback | B. Laplace approximation (MAP + inverse Hessian), numpy+scipy only (8/10) | A, B (8 tied) | No (top-tied) | R1 |
| Q8 | ↩ | Codebase location/organization | A. Phase1 conventions + one shared module for design-matrix/likelihood/knots/scaling (8/10) | A (8) | No | R1 |
| Q9 | 🔒 | Model formula / dimensionality | D. Formula chosen per model: full `ed_atlas_impl`-shaped interaction (`sex*cr(age)*complaint`) for mortality + diagnosis-tree level 1 (NODIAG vs any), additive (complaint main effects only) for tree levels 2/3 (8/10) | D (8) | No | R2 |
| Q10 | 🔒 | Missing-vitals handling | B. Complete-case by hospital site at the modelling stage: fit the vitals model only on visits from the 3 hospitals that record vitals (karolinska, danderyd, södersjukhuset; ≈989K of 1,805,088 rows, estimate); `train_clean` itself stays unchanged (8/10) | B (8) | No | R2 |

## Open questions (next round)
(none — loop closed, see Plan)

## Waiting on other decisions

## Need more information queue

## Cut by limits
- Q7: Plain PyMC with its own NUTS, no JAX/numpyro (score 4/10) — cut by max_options=5 (4 real
  options + Need more information). Still viable via Other if Top Baldy wants it; drops
  jax/jaxlib/numpyro but keeps pytensor (+ numba, a C compiler at runtime), arviz. Still needs
  Python ≥3.12 for current PyMC, and slower than the numpyro/JAX option (estimate).

## Facts still to check
Corrected this round: R2's researcher caught that this file previously said scipy on Bianca was
"already confirmed present" — it was not; R1 never checked it. That line is fixed below.

- **Bianca's actual Python/package situation.** R1 verified Bianca's `python_ML_packages` module
  is Python 3.9.5, and that current pymc/jax/arviz need Python ≥3.12 (numpyro ≥3.11) — so the
  stack as `ed_atlas_impl` pins it (Python 3.11, PyMC 5.17, JAX 0.4.31) almost certainly can't run
  there as-is. Q7's chosen fallback (Laplace/MAP) only needs numpy+scipy. **Only Top Baldy can
  check** (I never touch Bianca; the researcher only has local/laptop access):
  - Whether `scipy` is in `python_ML_packages/3.9.5-cpu` (numpy/pyarrow confirmed; scipy
    **not** confirmed — verify-first build step, before writing the Laplace fallback).
  - If Q1's gather-NUTS/numpyro approach is still wanted as a stretch goal: `module avail python`
    for anything ≥3.11, and `module spider jax numpyro`.
  - Whether Apptainer/Singularity runs on Bianca (only relevant if the container option from Q7's
    cut-by-limits list is revisited).
- **Laplace/hierarchical-prior clash (Q5 × Q7, flagged by R2's researcher).** A Laplace
  approximation at the joint mode is degenerate under Q5's hierarchical prior on complaint
  deviations: the density is unbounded as the hierarchical scale σ→0 with all deviations near 0.
  Verify-first build step: fix σ (empirical Bayes / type-II ML, or carried over from a short NUTS
  run on a subsample) before writing the Laplace fallback, or integrate it out instead of taking a
  joint MAP.
- **Partial within-site vitals gaps (Q10 follow-up).** Q10 restricts fitting to the 3 covered
  hospitals, but even there not every vital is recorded on every visit. Build default: reuse the
  missing-indicator + training-split-mean fill from Q10's option A, applied only within the
  covered-site rows. Verify-first: read `02_vitals_join.py`'s Bianca log for the real per-vital
  coverage % at the covered sites (only Top Baldy can read Bianca logs).
- **`ed_hospital` is not in `train_clean`** (verified by R2: absent from both `01_subset_v2.py`
  and `04_train_clean.py`). Q10's complete-case-by-site option needs it. Build step: add
  `ed_hospital` to `04_train_clean.py`'s `keep_cols` (it already exists upstream in
  `swed_ed_core`/`train_raw`), or join it back by `ed_visit_id` at modelling time. Either is a
  small change; prefer adding it to `04` since it's a cheap column to keep.
- Plausibility-clipping bounds for vitals (needed before scaling/squaring) — a build-time default,
  not a design decision; whoever builds should pick clinically sensible bounds and note them.
- Age range/extrapolation at the spline boundary: no age filter found in `01_subset_v2.py`;
  patsy's `cr()` extrapolates (verified, doesn't error) outside `[22,88]`. Decide during build
  whether to clip or accept extrapolation.
- Bianca job resource limits (cores/RAM/walltime per node, GPU availability) — needed to turn R1's
  laptop-measured timings into real Bianca estimates. Only Top Baldy can check.
- Coding/identifiability scheme (ed_atlas's rank-deficient coding vs. reference/sum-to-zero) —
  pick whichever patsy/the shared module defaults to; revisit only if it causes a fitting problem.
- Convergence/output criteria (ed_atlas's own mortality fit uses 1 chain, so no R-hat; decide
  whether to export an ed_atlas-style posterior-summary file) — build-time decision.

## Question log
### R1
- Q1: Fitting strategy at scale → A. Gather-NUTS on all ~1.8M rows (9/10) (options: A/B/C/D, A top)
- Q2: Diagnosis multiclass architecture → A. Tree-shaped nested categorical (8/10) (options: A/B/C/D, A top)
- Q3: Weighted-likelihood reuse for complaint_weight → A. Weighted likelihood (8/10) (options: A/B/C/D, A top)
- Q4: Age spline knots → B. Reuse ed_atlas's exact [22,43,65,88] (8/10) (options: A/B/C/D, A&B tied top)
- Q5: Prior for intercept/coefficients → B. Hierarchical prior on complaint deviations (7/10), override of top A (8/10), reason: gut feeling (options: A/B/C/D)
- Q6: Standardization scheme → A. Z-score on Stage-05 training split (8/10) (options: A/B/C/D, A top)
- Q7: Sampler stack feasibility + fallback → B. Laplace approximation (8/10) (options: A/B/C/D, A&B tied top; E cut by limit)
- Q8: Codebase location/organization → A. Phase1 conventions + shared module (8/10) (options: A/B/C/D, A top)

### R2
- Q9: Model formula / dimensionality → D. Formula chosen per model: full ed_atlas interaction for mortality + tree level 1, additive for tree levels 2/3 (8/10) (options: A/B/C/D, D top; vitals×age interaction cut by limit)
- Q10: Missing-vitals handling → B. Complete-case by hospital site at modelling (8/10) (options: A/B/C/D, B top; two-model reporting cut by limit)

## Plan

### Decisions table
See **Settled decisions** above — Q1–Q10, one override (Q5, reason "Gut feeling").

### Architecture
One shared module (Q8) holding: design-matrix/formula builder (two variants per Q9 — full
interaction for mortality/level-1, additive for levels 2/3), the weighted-likelihood mechanism
(Q3, per `ed_atlas_impl`), age-spline knot constants (Q4, per `ed_atlas_impl`'s `[22,43,65,88]`),
and the standardization fit/apply pair (Q6).

Three models, fit independently, all using Q1's gather-NUTS/index-gather likelihood with Q7's
Laplace/MAP fallback if NUTS proves infeasible on Bianca:
1. **Mortality** (binary, `died_30d`) — full interaction formula, all ~1.8M train_clean rows for
   the non-vitals terms; vitals terms only identified on the Q10-restricted covered-site subset
   (~989K rows) — see the open "partial gaps" build step for exactly how the two row-sets combine.
2. **Diagnosis tree level 1** (binary, NODIAG vs any) — full interaction formula, same row/vitals
   split as mortality.
3. **Diagnosis tree levels 2/3** (chapter, then exact top-K code within chapter; Q2's tree
   architecture) — additive formula (Q9), same row/vitals split.

All three share Q5's hierarchical prior on complaint deviations (non-centred) and Q3's
complaint-weighted likelihood for two-complaint visits.

### Data and files
- Source: `training/<N>K_train_clean.parquet` on Bianca (copied there by Top Baldy from
  `phase1/out_data/subset/`), read-only to the model code.
- `04_train_clean.py` gets one small addition: keep `ed_hospital` in `keep_cols` (build step,
  needed for Q10).
- Nothing from this model-fitting stage goes in git except the code itself (shared module, one
  fit script per model, config). No data, no fitted posteriors, no logs.

### Build order (independent, separately testable steps)
1. Add `ed_hospital` to `04_train_clean.py`'s output; re-run on Bianca; confirm the covered-site
   row count (~989K estimate) against the real column.
2. Verify-first: confirm `scipy` is importable under `python_ML_packages/3.9.5-cpu` on Bianca.
3. Build the shared module (formula builder for both Q9 variants, weighted likelihood, spline
   knots, standardization) with a small synthetic smoke test, off Bianca.
4. Decide the Laplace/hierarchical-prior fix (fix σ vs. integrate out) and implement it in the
   shared module's Laplace path; test on synthetic data.
5. Decide and implement the partial-within-site-gap fill (Q10 follow-up) inside the vitals-prep
   step.
6. Fit the mortality model on Bianca (smallest, most informative first); report.
7. Fit diagnosis tree level 1, then levels 2/3; report after each.
8. Export posterior/MAP summaries in an `ed_atlas`-comparable format (decide convergence/output
   criteria as part of this step, since ed_atlas's own mortality fit uses 1 chain too).

### How the result will be checked
- Synthetic-data smoke tests on the shared module before touching Bianca (per rule 2: nothing
  built or run until this plan is approved; afterwards, dry runs stay off real data until step 1
  is confirmed).
- Real train_clean sanity numbers already verified in R1/R2 (1.9% mortality, 85.6% diagnosed,
  S≈300, 1,805,088 total rows, ~989K covered-site rows) as a check against the fitted cohort size.
- Compare fitted mortality coefficients' sign/shape against `ed_atlas`'s own published shapes
  where comparable (same caveat as the Scope notes: diagnosis target isn't directly comparable).

### Risks and unverified facts the first build steps must check
- scipy availability on Bianca (step 2) — blocks the whole Laplace fallback path if missing.
- Level-2 diagnosis-tree wall time under the full-interaction option was explicitly avoided by
  choosing D (additive at levels 2/3) — if that later proves too coarse, re-opening it means
  re-fitting, so don't revisit lightly.
- Bianca job resource limits (cores/RAM/walltime) are still unmeasured; R1/R2's timings are
  laptop-estimates only. First real Bianca fit (step 6) is also the first real timing check.
- Everything under **Facts still to check** above that's marked "only Top Baldy can check."

### Out of scope / for later
- Any atlas-posterior-as-prior update (demo models 3.1/3.2/3.3/3.5/3.6) — ruled out at the start
  of this project (no genuine `D_new`).
- The one-vs-rest-per-code diagnosis architecture ed_atlas itself uses — explicitly not reused
  (Q2).
- GPU/container/Apptainer sampler stacks — cut by limits at Q7, revisit only if the Laplace
  fallback and gather-NUTS both prove infeasible on Bianca.
- Two-model reporting (vitals vs. no-vitals comparison) and the vitals×age interaction — both
  logged under Cut by limits, available later via a follow-up round if Top Baldy wants them.

### Cost and usage notes
- This design loop used 1 researcher (`af8b54c9c329e6d31`), high effort for R1 (8 questions),
  medium effort for R2 (2 questions), 2 rounds, 0 send-backs, 1 Top Baldy override.
- No Bianca time was used during design (hard rule 2) — the first Bianca cost is build step 1.
