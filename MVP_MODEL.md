# MVP_MODEL.md — Bayesian logistic regression for 30-day mortality

Companion document to `MVP.ipynb`. It explains the model mathematically, states
which choices are assumptions rather than facts, records the verified results,
and gives the reasoning behind the comparison between missingness mechanisms.

All figures in sections 7–10 come from an executed run of the notebook code,
not from estimates or recollection.

---

## 1. What is modelled

One Bernoulli trial per patient:

```
y_i = 1   if patient i died within 30 days of admission
y_i = 0   otherwise
```

The dataset holds **400 patients and 44 deaths (11.0%)**.

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
134, which is not plausible for an age effect per decade. And with 44 deaths
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

The median is used rather than 0 because saturation runs 87–100: filling with 0
would be far outside the observed range and would tell the model these
patients have no oxygen at all.

A flag is used rather than dropping the rows. Dropping would keep n=400 in
A but 318 in B, 318 in C and 335 in D, which breaks comparability — and
*which* deaths get dropped depends on the mechanism, so the fits would not be
answering the same question.

**The flag is not a cure for MNAR.** It is a weak substitute for the missing
value, because the mechanisms here are probabilistic: a missing reading
narrows the guess toward "probably low" without determining it. See section 9.

---

## 7. Verified results

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

The flag coefficient has a known expected value under each mechanism, from how
that mechanism was constructed:

| mechanism | what drives missingness | expected flag OR |
|-----------|-------------------------|------------------|
| MCAR | nothing | exactly 1.00 |
| MAR  | `chief_complaint` — already a column in the model | exactly 1.00 |
| MNAR | the saturation value itself | **> 1** |

MAR is the subtle one. Missingness depends on the complaint, but the model
already contains the complaint, so once you condition on it the flag carries no
further information and should also sit at 1.0.

MNAR should exceed 1.0 because a missing reading indicates a patient who was
probably hypoxic, and hypoxia raises mortality.

---

## 9. What we actually get, and why

```
B  mcar   flag OR 1.55   94% HDI [0.69, 3.39]
C  mar    flag OR 0.81   94% HDI [0.29, 2.09]
D  mnar   flag OR 1.23   94% HDI [0.49, 2.96]
```

**All three straddle 1.0.** The crude figures, before any model:

| mechanism | deaths where missing | deaths where present | crude OR | 95% CI |
|-----------|---------------------|---------------------|----------|--------|
| MCAR | 9/82 (11.0%) | 35/318 (11.0%) | 1.00 | 0.46 – 2.17 |
| MAR  | 4/82 (4.9%)  | 40/318 (12.6%) | 0.36 | 0.12 – 1.03 |
| MNAR | 11/65 (16.9%) | 33/335 (9.9%)  | 1.86 | 0.89 – 3.91 |

MCAR reproduces its expected value of exactly 1.00. MAR runs low, and MNAR
runs high in the right direction — the mechanism is doing what theory predicts,
just not detectably so.

### Why the model cannot separate them

This is arithmetic, not a modelling failure:

```
44 deaths / 11 coefficients = 4.0 events per coefficient
conventional minimum        = ~10 events per coefficient
```

For the MNAR flag specifically, the effect is `+0.62` on the log-odds with a
standard error of `0.38`. That is a ratio of **1.66**, where roughly **2.0** is
needed to separate from zero.

**So the intervals crossing 1.0 is the model reporting correctly.** Given 44
deaths it genuinely cannot tell which mechanism it is looking at, and a
confident-looking separation would mean the model had fitted noise.

### The ceiling for MNAR is not fit D

It is a fit on `true_saturation`, which is complete — 400 of 400 values, no
missingness at all.

MNAR is a **loss, not a gift.** The mechanism that makes the flag informative is
the same one that removed 11 of the 44 deaths, disproportionately the hypoxic
ones. Fit D should be expected to perform *worse* than complete-data, not
better.

---

## 10. The scaling bug, and why centring is mandatory

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
| tighten the prior | lower `SIGMA_COEF` | a smaller sd shrinks coefficients toward 0; with 4 events per coefficient this is doing real work already |
| change the sampler | `target_accept` / chains in `fit()` | raise `target_accept` toward 0.95+ if divergences appear |
| nonlinearity, as ed_atlas uses | replace the linear term with a spline basis | **needs far more than 400 patients**; do not attempt at 44 deaths |

To swap the target, `mortality_30_days` is the only label. To test on
held-out data, use proper k-fold with the imputation refit inside each fold —
refitting the median and sd on the full dataset first would leak information.

---

## 12. Limits

- **44 deaths against 11 coefficients.** Roughly 4 events per coefficient
  against a conventional minimum of 10. Every interval in this document is
  correspondingly wide, and no mechanism is distinguishable.
- **The B/C/D differences are within noise.** Mean shifts of 0.02–0.03 with
  94% intervals crossing 1.0. The comparison is a pipeline demonstration, not
  an empirical finding.
- **The flag is a weak substitute for the value.** The mechanisms are
  probabilistic, so a missing reading narrows the guess rather than determining
  it.
- **Predictive, not causal.** No coefficient here identifies a treatment
  effect.
- **Binary outcome, no timing.** No hazard, no survival analysis.

To make the mechanism comparison bite, either:

- **~1,000 patients** at the current 11% rate, giving ~110 deaths and about
  10 events per coefficient; or
- **a more deterministic mechanism**, so the flag carries more signal — at the
  cost of no longer being a realistic MNAR simulation.

The honest version of this write-up is to report that all three intervals
straddle 1.0 and show section 9's arithmetic as the reason. That is a real
finding about sample size, and more defensible than tuning the simulation until
the plot separates.
