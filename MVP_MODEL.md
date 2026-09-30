# MVP_MODEL.md — Bayesian logistic regression for 30-day mortality

Companion document to `MVP.ipynb`. It explains the model mathematically, states
which choices are assumptions rather than facts, records the verified results,
and gives the reasoning behind the comparison between missingness mechanisms.

All figures in sections 7–10 come from an executed run of the notebook code,
not from estimates or recollection — **except** where flagged. The dataset was
rebuilt after that run (see section 13), so sections 7 and 10 are records of the
previous dataset, and section 9's flag ORs and AUCs are a fresh *frequentist*
refit (scikit-learn), not the PyMC posteriors. The mechanism-level quantities in
section 8 and the generation parameters in section 13 are computed against the
current file. Re-run `MVP.ipynb` to replace the stale posteriors.

---

## 1. What is modelled

One Bernoulli trial per patient:

```
y_i = 1   if patient i died within 30 days of admission
y_i = 0   otherwise
```

The dataset holds **400 patients and 49 deaths (12.2%)**.

This is a binary-outcome model, not a survival model. There is no `death_date`
and no follow-up time per patient, so a hazard function cannot be recovered
from this data. Anything time-to-event (Kaplan-Meier, Cox, discrete-time
hazard) would require a different dataset, not a different model here.

The goal is **prediction and comparison**, not causal inference. A coefficient
below is an association. Reading it as the effect of raising age by ten years
assumes nothing else about the patient changes, which is not something this
design can support.

---

## 2. Likelihood and posterior

For patient *i* with design row **x**i:

```
y_i | alpha, beta  ~  Bernoulli( sigmoid( alpha + x_i . beta ) )
```

Written out, the posterior over the 10 parameters (9 for fit A, 11 for
fits B–D):

```
p( alpha, beta | y )  proportional to  p( y | alpha, beta )
                                     * Normal( beta; 0, 1.0 )
                                     * Normal( alpha; 0, 2.5 )
```

Sampled with PyMC's NUTS: 4 chains, 2000 tuning steps, 2000 draws,
`target_accept=0.9`, `random_seed=42`.

The posterior predictive risk for patient *i* is the posterior mean of the
sigmoid, averaged over all draws — this is the quantity compared between fits
and plotted in `mvp05`.

---

## 3. The design matrix

Fit A uses 9 columns. Fits B, C and D add two, for 11.

| # | column | meaning |
|---|--------|---------|
| 0 | `C(chief_complaint)[T.Abdominal Pain]` | vs Chest Pain |
| 1 | `C(chief_complaint)[T.Cough]` | vs Chest Pain |
| 2 | `C(chief_complaint)[T.Dizziness]` | vs Chest Pain |
| 3 | `C(chief_complaint)[T.Fatigue]` | vs Chest Pain |
| 4 | `C(chief_complaint)[T.Fever]` | vs Chest Pain |
| 5 | `C(chief_complaint)[T.Headache]` | vs Chest Pain |
| 6 | `C(chief_complaint)[T.Shortness of Breath]` | vs Chest Pain |
| 7 | `age_dec` | `(age - 50) / 10` |
| 8 | `male` | `1` if `gender == "Male"` |
| 9 | `saturation` | `(value - median) / sd` |
| 10 | `sat_missing` | `1` if the value was never recorded |

`Intercept` is dropped from the Patsy output because PyMC supplies `alpha` as
a separate free parameter. Keeping both would double-count the intercept.

All four designs are verified full rank: 9 of 9 for A, 11 of 11 for B, C, D.
That matters — a rank-deficient design would leave `beta_sat` and `beta_flag`
unidentifiable, since the flag is `1` exactly when the filled value equals the
median.

### Why the reference is named rather than automatic

```
REFERENCE = "Chest Pain"
```

`C(chief_complaint)` on its own picks the reference from the level with the
highest frequency. Four complaints tie at 53 rows each — Abdominal Pain,
Cough, Fatigue, Shortness of Breath — so the default is decided by a tiebreak
rather than by any intent, and every coefficient would be relative to a level
nobody chose. Naming the reference makes the comparison meaningful and stable
across runs.

---

## 4. Reading a coefficient

`beta` is on the log-odds scale. The odds ratio is `OR = exp(beta)`:

| `beta` | OR | reading |
|--------|----|---------|
| `+0.25` | 1.28 | 28% higher odds |
| `+0.50` | 1.65 | 65% higher |
| `+1.00` | 2.72 | 2.7× the odds |
| `−0.50` | 0.61 | 39% lower |

Two concrete readings:

- `age_dec = +0.30` → ten extra years multiply the odds of death by
  `exp(0.30) = 1.35`.
- a complaint dummy of `+0.60` → that complaint carries 1.82 times the odds
  of chest pain, holding age and sex constant.

`alpha` is not an OR. It is the log-odds at the reference profile, so the
baseline risk is `sigmoid(alpha)`. `alpha = -2.27` gives 6–11% depending on
the other columns.

---

## 5. Priors, and what they assume

```python
SIGMA_COEF = 1.0
SIGMA_INT  = 2.5
```

```
beta  ~ Normal(0, 1.0)    95% of the prior is an odds ratio of 0.14 to 7.1
alpha ~ Normal(0, 2.5)    loose, because the intercept carries the base rate
```

**These are chosen a priori. They are not fitted, and they are not derived from
this dataset.** Any claim otherwise would be false — with 400 rows there is no
way to estimate a prior empirically.

`sd = 2.5` on the coefficients was rejected: it would admit an odds ratio of
134, which is not plausible for an age effect per decade. And with 49 deaths
the prior is doing real work rather than decoration — a flat prior on 11
coefficients will happily fit noise.

The critical constraint is that **this prior assumes standardised inputs.**
Every column must be roughly centred on 0 with standard deviation near 1.
Section 10 shows what happens when one is not.

---

## 6. The saturation encoding, mathematically

A missing saturation value is replaced by the observed median, then the column
is centred and scaled by the observed median and standard deviation.

For a patient whose value was never recorded:

```
logit P(died) = alpha
              + x_base . beta_base
              + beta_sat  * 0
              + beta_flag * 1
```

The filled value is **exactly 0 after centring**, so the value term drops out
entirely and only the flag contributes.

This is the key structural point: `beta_sat` and `beta_flag` are separately
identified because a missing row scores 0 on one and 1 on the other, while an
observed row scores a non-zero value and 0. The two coefficients answer
different questions:

- `beta_sat` — how does a *recorded* oxygen saturation relate to death?
- `beta_flag` — does *not having a reading* relate to death, holding the
  (imputed) value fixed?

The median is used rather than 0 because saturation runs 84–100: filling with 0
would be far outside the observed range and would tell the model these
patients have no oxygen at all.

A flag is used rather than dropping the rows. Dropping would keep n=400 in
A but 317 in B, 308 in C and 351 in D, which breaks comparability — and
*which* deaths get dropped depends on the mechanism, so the fits would not be
answering the same question.

**The flag is not a cure for MNAR.** It is a weak substitute for the missing
value, because the mechanisms here are probabilistic: a missing reading
narrows the guess toward "probably low" without determining it. See section 9.

---

## 7. Verified results

> **Stale — re-run required.** Every number in this section comes from an
> executed run against the *previous* dataset. `healthcare_dataset_2_missingness.csv`
> has since been rebuilt around realistic missingness mechanisms (section 13), so
> the divergences, `r_hat`, ESS, alpha, mean-risk gaps and calibration table below
> no longer describe it. The observed rates have moved substantially: the cohort
> now has 49 deaths rather than 44, and Fever is the highest-mortality complaint
> at 22.9% rather than Shortness of Breath. Treat this section as a record of the
> previous run, not a result. What does carry over is the structure — all four
> designs are still full rank, 9 of 9 for A and 11 of 11 for B, C and D, and
> section 9's frequentist refit is a usable check in the meantime.

Executed run, 4 chains, 2000 tuning + 2000 draws, `target_accept=0.9`,
`random_seed=42`:

| fit | design | divergences | max r_hat | min ESS | alpha | mean risk |
|-----|--------|------------|-----------|---------|-------|-----------|
| A | 400 × 9  | 0 | 1.001 | 4155 | −2.28 | 11.1% |
| B (MCAR) | 400 × 11 | 0 | 1.001 | 6184 | −2.49 | 11.1% |
| C (MAR)  | 400 × 11 | 0 | 1.001 | 4936 | −2.19 | 11.1% |
| D (MNAR) | 400 × 11 | 0 | 1.001 | 5563 | −2.39 | 11.1% |

All four sample cleanly: zero divergences, `r_hat` at 1.001, effective sample
size in the thousands.

Mean absolute difference in posterior mean risk, against fit A:

```
A vs B   0.0288  (max 0.2594)
A vs C   0.0271  (max 0.2310)
A vs D   0.0200  (max 0.1794)
```

The fits sit on the same 11.1% mean risk because they share the same base
predictors and the same outcome. They differ in individual predictions, and
that difference is what `mvp05` plots.

Calibration by complaint, fit A, posterior mean risk against observed rate:

| complaint | observed | predicted | error |
|-----------|----------|-----------|-------|
| Shortness of Breath | 28.3% | 26.5% | −1.8 |
| **Chest Pain (reference)** | **27.5%** | **17.8%** | **−9.7** |
| Headache | 6.7% | 8.7% | +2.0 |
| Abdominal Pain | 7.5% | 8.8% | +1.3 |
| Fever | 6.2% | 8.4% | +2.2 |
| Dizziness | 4.5% | 7.0% | +2.5 |
| Cough | 3.8% | 5.7% | +1.9 |
| Fatigue | 1.9% | 4.7% | +2.8 |

Ordering is reproduced, and Shortness of Breath is recovered to within 1.8
points. **The reference level is the model's clear weak spot**: Chest Pain is
under-predicted by 9.7 points. This matters more than a routine calibration
miss, because every other complaint coefficient is expressed *relative to Chest
Pain* — so the anchor is off by roughly 10 points.

The cause is that Chest Pain (27.5%, 14/51) and Shortness of Breath (28.3%,
15/53) are statistically tied in this sample, yet the model separates them. With
44 events it cannot resolve two adjacent groups and settles on a compromise
that fits neither exactly. This is another symptom of the sample size, and it
is a reminder to report per-group calibration rather than the mean risk, which
stays at 11.1% and hides the problem entirely.

---

## 8. What each mechanism should produce

The flag coefficient is **not** a clean readout of the mechanism, and the table
this section used to give was wrong. What is clean is the mechanism itself,
measured directly on the data: the true-saturation deficit between rows where
the reading was kept and rows where it was lost.

| mechanism | what drives missingness | true-sat deficit, missing vs present |
|-----------|-------------------------|--------------------------------------|
| MCAR | nothing | **+0.17** |
| MAR  | `age` and `chief_complaint` — both already in the model | **−0.45** |
| MNAR | never for oxygen-related complaints, otherwise the value itself | **+3.94** |

That is the intended result and it is unambiguous. MCAR and MAR lose readings
from patients whose oxygenation is indistinguishable from the patients who kept
theirs. MNAR loses 3.94 points more hypoxaemia, which is the whole point of the
mechanism and is the same order as the 3-point contrast mortality is modelled
on.

**The flag coefficient does not measure this.** Two things happen when a missing
value is filled with the median, and they cancel.

The imputation overstates oxygenation. A hypoxaemic patient whose reading was
lost is handed the cohort median — a better saturation than they actually had —
which pushes their predicted mortality down. The flag has to absorb that
downward push, so the flag coefficient goes **up** for reasons that have nothing
to do with the mechanism. The size of the artifact scales with the missing rate
times the average deficit of the missing rows, so it hits all three mechanisms
and it is largest exactly where the deficit is largest.

Meanwhile the `saturation` term already carries the hypoxaemia signal for the
rows that kept a reading. Once missingness is a function of the value, the flag
is largely redundant with that term rather than additive to it.

An oracle test settles it. Refit each model giving every row its *true*
saturation, keeping the flag:

| mechanism | flag OR, median-imputed | flag OR, true value supplied |
|-----------|------------------------|------------------------------|
| MCAR | 1.08 | 0.52 |
| MAR  | 1.34 | 0.71 |
| MNAR | **6.56** | **1.02** |

All three collapse towards 1.0 once the value is known — including MNAR, whose
6.56 was almost entirely artifact. The earlier claim in this section, that MNAR
"should" produce a flag OR above 1.0 while MCAR and MAR produce exactly 1.00,
does not survive contact with the encoding. Over 60 independent seeds the
median fitted flag OR is 1.90 for MCAR, 1.71 for MAR and 4.35 for MNAR: the
*ordering* is stable, but no mechanism sits at 1.00, and MNAR's headline
number is a median-imputation effect rather than a measurement of the mechanism.

If the aim is to compare mechanisms, compare the deficit column, or use an
encoding that does not impute — multiple imputation, or a pattern-mixture
model that carries an explicit shift parameter. Both are outside the scope of
this notebook, which is why the deficit table is the honest one.

---

## 9. What we actually get, and why

Crude figures against the current dataset, before any model:

| mechanism | missing | deaths where missing | deaths where present | crude OR | 95% CI |
|-----------|---------|---------------------|---------------------|----------|--------|
| MCAR | 69 (17.2%) | 6 (8.7%) | 43 (13.0%) | 0.67 | 0.27 – 1.63 |
| MAR  | 52 (13.0%) | 5 (9.6%) | 44 (12.6%) | 0.76 | 0.29 – 2.01 |
| MNAR | 54 (13.5%) | 11 (20.4%) | 38 (11.0%) | 1.85 | 0.89 – 3.85 |

**All three straddle 1.0.** The crude table cannot separate the mechanisms, and
it is also the wrong instrument: it compares death rates without conditioning on
age or complaint, so it mixes the mechanism with the confounders.

The fitted comparison does separate them, and in the predicted order. Unpenalised
logistic on the same design as the notebook, 5-fold CV AUC in brackets:

| fit | flag OR | 95% CI | AUC |
|-----|---------|--------|-----|
| A — no saturation | — | — | 0.661 |
| B (MCAR) | 1.08 | 0.59 – 1.98 | 0.752 |
| C (MAR)  | 1.34 | 0.68 – 2.64 | 0.768 |
| D (MNAR) | **6.56** | 2.74 – 15.69 | 0.754 |

Two things to read off this, and they point opposite ways.

The **flag ORs separate**: 1.08, 1.34, 6.56. MNAR is the only one whose interval
excludes 1.0.

The **AUCs do not**: 0.752, 0.768, 0.754. All three fits predict about equally
well. That is the correct and expected result, not a disappointment. The
mechanism governs *which* patients lose a reading, not *how much* information the
cohort retains — every mechanism costs roughly the same number of observations,
so predictive performance is nearly identical. Section 8 is the important
caveat on the flag column: its 6.56 is inflated by median imputation, and the
deficit table, not this one, is the honest measurement of MNAR.

> The PyMC posteriors in `MVP.ipynb` have not been re-executed against this
> dataset. The figures above are frequentist, fitted with scikit-learn, and are
> offered as a check on the expected ordering rather than as a replacement for
> the Bayesian run.

### How much resolution 49 deaths buys

This is arithmetic, not a modelling failure:

```
49 deaths / 11 coefficients = 4.5 events per coefficient
conventional minimum        = ~10 events per coefficient
```

The MNAR interval is the widest of the three at 2.74 – 15.69, a factor of nearly
six end to end. That width is what four-to-five events per coefficient buys, and
it is why the crude table in this section cannot resolve a mechanism the fitted
table can.

### The ceiling for MNAR is not fit D

The complete-data reference no longer exists in this file. The
`true_saturation` column was removed when the mechanisms were rebuilt, so there
is no fully observed saturation left to fit against. One of the 400 rows is
missing from all three columns at once and has no recoverable saturation in this
dataset at all.

MNAR is a **loss, not a gift.** The mechanism that makes the flag look
informative is the same one that removed 11 of the 49 deaths, disproportionately
the hypoxaemic ones. Fit D should be expected to perform *worse* than
complete-data, not better — and on these AUCs it does not even manage that much.

---

## 10. The scaling bug, and why centring is mandatory

> The range and mean quoted below are those of the *previous* dataset. The
> current saturation column runs 84–100 with mean 95.06 and median 96, so the
> intercept penalty is comparable and the point of the section is unchanged.

The first version of this model fed raw saturation (87–100, mean 95.8) to the
sampler. It failed visibly:

```
alpha          +19.12
divergences    ~4000
r_hat          > 1.01
ESS            < 100
mean risk      55.5% for B, C and D
```

Every mechanism returned the same 55.5%, which was the giveaway: the fits had
collapsed onto the prior, not onto the data. The comparison the whole project
exists to make had been destroyed by a column on the wrong scale.

**Cause.** The `Normal(0, 2.5)` prior on `alpha` assumes standardised inputs.
A raw saturation column has mean 95.8, so the intercept had to sit at `+19.12`
to compensate — 7.6 prior standard deviations from zero. The sampler was
working in the tail of its own prior. `age_dec` and `male` were already
scaled, which is why fit A was unaffected and only B, C and D broke.

**Fix.** `design["saturation"] = (filled - sat.median()) / sat.std()`, giving
mean 0 and sd 1. `alpha` then lands near −3 and all four fits sample cleanly.

This is why section 5 insists on standardised inputs. It is not a
formality — on this data it is the difference between a working model and 4000
divergences.

---

## 11. How to adjust the model

| to change | do this | watch for |
|-----------|---------|-----------|
| the reference complaint | edit `REFERENCE` in cell 2 | all 7 dummies change; do not compare coefficients across references |
| add a predictor | append to `build_design()` | **recheck the rank** — 11 of 11 now, and a collinear column silently breaks `beta_sat` / `beta_flag` |
| rescale a feature | centre and scale before use | otherwise the intercept absorbs the mean and the sampler goes to the prior tail (section 10) |
| tighten the prior | lower `SIGMA_COEF` | a smaller sd shrinks coefficients toward 0; with 4.5 events per coefficient this is doing real work already |
| fix the flag artifact | stop imputing the median | the flag coefficient is inflated for all three mechanisms (section 8). Multiple imputation, or a pattern-mixture model with an explicit shift parameter, is what actually measures MNAR |
| change the sampler | `target_accept` / chains in `fit()` | raise `target_accept` toward 0.95+ if divergences appear |
| nonlinearity, as ed_atlas uses | replace the linear term with a spline basis | **needs far more than 400 patients**; do not attempt at 49 deaths. Note the mortality model is genuinely non-linear in saturation, so the linear term is misspecified by design |

To swap the target, `mortality_30_days` is the only label. To test on
held-out data, use proper k-fold with the imputation refit inside each fold —
refitting the median and sd on the full dataset first would leak information.

---

## 12. Limits

- **The flag coefficient is not a valid measure of the mechanism.** Section 8
  shows that median imputation inflates it for all three, and that MNAR's
  headline 6.56 falls to 1.02 when the true value is supplied. The deficit table
  is the honest comparison. This is the single most important caveat in the
  document and it applies to any conclusion drawn from fits B, C and D.
- **The three mechanisms predict equally well.** AUC 0.752, 0.768, 0.754. The
  mechanism changes who loses a reading, not how much the cohort retains, so
  there is no predictive gain to be had from identifying it.
- **49 deaths against 11 coefficients.** Roughly 4.5 events per coefficient
  against a conventional minimum of 10. The MNAR interval spans 2.74 – 15.69.
- **The B/C/D posteriors are unverified.** The Bayesian figures in
  `MVP.ipynb` have not been re-executed against this dataset; section 9's
  numbers are frequentist.
- **The saturation term is misspecified on purpose.** Mortality is generated
  with a squared penalty below 92 and an accelerating age term, so a linear
  `beta_sat` cannot be exactly right. That is realistic, and section 11 notes
  the data is far too small to fit the curvature properly.
- **The MNAR guarantee is a design choice, not a clinical finding.** Requiring
  pulse oximetry for respiratory presentations is defensible practice, and it
  is what concentrates the missing rows in the hypoxaemic tail of the other
  half. A different triage policy would give a different deficit.
- **Predictive, not causal.** No coefficient here identifies a treatment
  effect.
- **Binary outcome, no timing.** No hazard, no survival analysis.

To make the mechanism comparison bite further, either:

- **~1,000 patients** at the current 12% rate, giving ~120 deaths and about
  11 events per coefficient; or
- **an encoding that does not impute**, which is the binding constraint. More
  patients will not fix a flag coefficient that is measuring its own
  imputation.

The honest version of this write-up reports the deficit table, reports that
prediction does not improve, and says the flag comparison is confounded by its
own encoding. That is a more useful result than a plot where the mechanisms
separate, because it is the part that would have generalised badly.

---

## 13. How the dataset was built

The dataset is a synthetic simulation, not patient data. `age`, `gender` and
`chief_complaint` are carried over unchanged from
`healthcare_dataset_1_base.csv`; saturation and mortality are generated from the
specification below at seed `3637`. No generator script is checked in, so the
parameters are recorded here.

**Saturation** is modelled as a deficit below 100 drawn from a Gamma, not as a
symmetric normal. Real room-air SpO2 sits just under 100, has a long left tail
and stops at a floor; a normal distribution centred on 95 cannot produce that
shape. The deficit also picks up age, at `+0.018` per year over 50.

| complaint | mean deficit | shape | | complaint | mean deficit | shape |
|-----------|--------------|-------|---|-----------|--------------|-------|
| Shortness of Breath | 6.30 | 1.55 | | Fever | 5.80 | 1.60 |
| Chest Pain | 5.00 | 1.70 | | Dizziness | 5.60 | 1.70 |
| Cough | 4.10 | 2.00 | | Headache | 3.70 | 2.20 |
| Fatigue | 4.30 | 2.00 | | Abdominal Pain | 4.60 | 2.00 |

Fever, Dizziness and Abdominal Pain carry a heavier hypoxaemic tail on purpose.
Sepsis and pneumonia present as fever, PE and arrhythmia as dizziness, and both
are genuinely hypoxaemic presentations. Without that tail the non-oxygen-related
half of the cohort has almost no hypoxaemic patients, which is not what an
emergency department looks like — and with no hypoxaemic patients there is
nothing for an MNAR mechanism to select on. Saturation is clipped to 84–100,
giving mean 95.06, median 96, sd 3.58.

**Mortality** is non-linear in both drivers, because the real relationships are:

```
logit = -3.05
        + 0.50 * ((96 - sat) / 3)              # mild desaturation, small effect
        + 0.55 * max(0, (92 - sat) / 4) ** 2   # steep rise below 92
        + 0.30 * ((age - 50) / 10)
        + 0.22 * max(0, (age - 70) / 10)       # accelerating risk in the oldest
        + complaint offset
```

Complaint offsets: Shortness of Breath `+0.80`, Chest Pain `+0.75`, Cough
`+0.10`, Fatigue `+0.05`, Fever `+0.05`, Dizziness `−0.05`, Headache `−0.15`,
Abdominal Pain `−0.20`. Yields **49 deaths of 400 (12.2%)**,
`corr(saturation, mortality) = −0.357`.

Saturation is a **partial** driver, not the only one: it acts alongside age and
complaint, which is what makes `beta_sat` identifiable separately from the
complaint dummies. Fever ends up the highest-mortality complaint at 11 of 48,
ahead of Chest Pain at 10 of 51 and Shortness of Breath at 10 of 53, which is
what the sepsis and pneumonia tail implies.

**The three mechanisms**, each with the clinical reason it looks the way it does:

| column | rule | missing | reason |
|--------|------|---------|--------|
| `saturation_mcar` | 18% flat, independent of everything | 69 (17.2%) | clerical dropout, no clinical pattern |
| `saturation_mar` | `logistic(-2.10 + 0.20 * age_dec + complaint)` | 52 (13.0%) | older and systemically unwell patients get vitals charted more often |
| `saturation_mnar` | never for oxygen-related; otherwise `logistic(-1.99 - 0.744 * (sat - 95))`, clipped to 0.12–0.95 | 54 (13.5%) | oximetry fails on poor perfusion, and hypoxaemic patients are put on oxygen before a room-air value is charted |

Oxygen-related complaints are **Shortness of Breath, Chest Pain, Cough and
Fatigue** — 210 rows, 0.0% missing under MNAR, because pulse oximetry is a
required observation for a respiratory presentation. The remaining 190 rows run
11.1% (Headache) to 40.9% (Dizziness) by complaint.

Each mechanism's signature, and the check that it is the mechanism intended:

| mechanism | MCAR | MAR | MNAR |
|-----------|------|-----|------|
| rate by age tertile | 18.4 / 14.4 / 18.7 | **7.8 / 15.2 / 16.4** | 14.2 / 12.0 / 14.2 |
| true-sat deficit, missing vs present | +0.17 | −0.45 | **+3.94** |

MAR rises with age and MCAR does not, exactly as intended. Within the
non-oxygen-related rows the MNAR missing rate is 100% at saturation ≤ 90 and
6–20% at ≥ 94, Spearman −0.52 against saturation. Headache is the least
hypoxaemic complaint and the least likely to lose a reading; Dizziness carries
the heaviest tail and the highest missing rate.

`true_saturation` is deliberately **not** in the CSV. Holding it would hand the
analysis the complete-data answer and make the comparison pointless. One row is
missing from all three columns at once and is unrecoverable here by design. The
deficit table in section 8 was computed against the generator's own output, not
against anything recoverable from the file.
