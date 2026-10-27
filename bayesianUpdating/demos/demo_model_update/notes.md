# Notes

## What is β in the base model?

- β = the coefficients for the base features: age, male, shortness of breath, headache, abdominal pain (5 numbers). The intercept α is separate.
- In a Bayesian model β is not one fixed number but a random variable:
  - before fitting: prior, each β_j ~ N(0, 1)
  - after fitting: posterior p(α, β | D_old)
- `pm.sample` represents the posterior as ~4000 draws: `idata_base.posterior["beta"]` has shape (6 chains, 667 draws, 5 features).
- Summaries: mean of the draws = estimate, sd = uncertainty, 2.5%–97.5% = 95% credible interval.
- The base score `base_lp` uses only the posterior mean β̄, so 3.1 and 3.2 drop the uncertainty in β; 3.3 keeps it (mean + covariance).

## Base model prior, with β named c1–c5

| coefficient | feature | prior |
|---|---|---|
| α | intercept | N(0, 1) |
| c1 | age (per sd ≈ 18 years) | N(0, 1) |
| c2 | male | N(0, 1) |
| c3 | shortness of breath (vs chest pain) | N(0, 1) |
| c4 | headache (vs chest pain) | N(0, 1) |
| c5 | abdominal pain (vs chest pain) | N(0, 1) |

- All independent: p(α, c1, …, c5) = N(α; 0, 1) · N(c1; 0, 1) · … · N(c5; 0, 1)
- Meaning: before seeing data, each effect is most likely near 0 and 95% likely between −2 and 2 on the log-odds scale, i.e. an odds ratio between e^−2 ≈ 0.14 and e^2 ≈ 7.4.
- Weakly informative: rules out absurd effects, but the old patients (N_OLD) easily overrule it.

## Base model: from prior to posterior, as a vector

Collect the intercept and the coefficients in one vector, and give each patient a matching feature vector with a leading 1:

```
θ   = (α, c1, c2, c3, c4, c5)ᵀ                   6 parameters
x̃_i = (1, age_i, male_i, sob_i, headache_i, abdominal_i)ᵀ

logit p_i = x̃_iᵀ θ = α + c1·age_i + … + c5·abdominal_i
```

### Prior: before any data

```
θ ~ N(0, I₆)

mean        μ_prior = (0, 0, 0, 0, 0, 0)ᵀ
covariance  Σ_prior = I₆   (variance 1 on the diagonal, 0 elsewhere)
```

Every parameter is centred at 0 with sd 1, and they are independent (no correlations).

### Bayes' rule: what the old data does

```
p(θ | D_old)  ∝  L(θ) · N(θ; 0, I₆)

L(θ) = ∏_{i ∈ old} p_i^y_i · (1 − p_i)^(1 − y_i)
```

There is no closed form (the logistic likelihood and the normal prior don't combine into a known distribution), so `pm.sample` draws from it instead. With this much data, the posterior is very close to a normal distribution:

```
θ | D_old ≈ N(μ_post, Σ_post)
```

### What changes: the mean

The centre moves from 0 to where the data points (values from the notebook run, rounded):

```
          prior   →  posterior mean    (true)
α           0     →     −3.68          (−4.2)
c1 age      0     →      0.73          ( 0.8)
c2 male     0     →      0.28          ( 0.3)
c3 sob      0     →      0.41          ( 0.5)
c4 head     0     →     −1.06          (−0.7)
c5 abd      0     →     −0.43          (−0.2)
```

The posterior is centred on what a model *without vitals* should learn. That's why α is less negative than the truth: the intercept absorbs the vitals' average extra risk.

### What changes: the uncertainty (covariance)

Using the normal approximation (Laplace), uncertainty is easiest to see as **precision**, the inverse of the covariance. The posterior precision is the prior precision plus the information from the data:

```
Σ_post⁻¹ ≈ Σ_prior⁻¹ + X̃ᵀ W X̃ = I₆ + X̃ᵀ W X̃

X̃ = the old patients' feature vectors stacked as rows
W = diag( p_i (1 − p_i) )   each patient's contribution
```

- **The diagonal shrinks:** each sd goes from 1 to about 0.04–0.15. Every patient adds information through X̃ᵀWX̃, which quickly outweighs the prior's I₆.
- **Off-diagonal values appear:** the parameters become correlated. For example, α and the complaint coefficients c3–c5 trade off, because every complaint shares the intercept as its baseline (chest pain).
- **The prior barely matters:** the prior contributes 1 to each diagonal entry of the precision, while the data contributes something proportional to the number of patients.

### Summary

```
prior:      θ ~ N( 0,       I₆     )     centred at 0, wide, independent
posterior:  θ ~ N( μ_post,  Σ_post )     centred on the data, narrow, correlated
```

`μ_post` and `Σ_post` estimated from the 4000 draws are exactly the `μ_old` and `Σ_old` that model 3.3 uses as its prior.

## Model 3.3: the base posterior becomes the prior

### The new parameter vector

The new model keeps the 6 old parameters and adds 6 vital coefficients, named g1–g6 (γ in the notebook):

```
φ = (θ, g)ᵀ = (α, c1, …, c5,  g1, …, g6)ᵀ          12 parameters
              └── old block ──┘ └ new block ┘

g = (g1, …, g6) for (V1, V1², V2, V2², V3, V3²)

z_i = (x̃_i, v_i) = (1, age_i, …, abdominal_i,  V1_i, V1_i², …, V3_i²)ᵀ

logit p_i = z_iᵀ φ = x̃_iᵀ θ + v_iᵀ g
```

### Prior: the old posterior for θ, N(0, 1) for g

```
φ ~ N( μ_prior, Σ_prior )

μ_prior = ( μ_old ,  0₆ )ᵀ

Σ_prior = ⎡ Σ_old   0  ⎤        block diagonal
          ⎣  0      I₆ ⎦
```

- **Old block:** centred on what the base model learned (μ_old ≈ μ_post from the base section), with its narrow sds and correlations (Σ_old).
- **New block:** the vitals were never seen, so they start like the base model's parameters did: centred at 0, sd 1, independent.
- **The 0 blocks:** before the new data there is nothing linking the old parameters to the vitals.

In the code:
```
old   = pm.MvNormal("old", mu=prev.mean(axis=0), cov=np.cov(prev, rowvar=False))   # θ ~ N(μ_old, Σ_old)
added = pm.Normal("added", mu=0, sigma=1, shape=6)                                 # g ~ N(0, I₆)
```

### Bayes' rule with the new data

```
p(φ | D_old, D_new)  ∝  L_new(φ) · N(θ; μ_old, Σ_old) · N(g; 0, I₆)

L_new(φ) = ∏_{i ∈ new} p_i^y_i · (1 − p_i)^(1 − y_i)
```

**Why this counts the old patients too:** N(θ; μ_old, Σ_old) ≈ p(θ | D_old) ∝ L_old(θ) · N(θ; 0, I₆). Substituting gives

```
p(φ | D_old, D_new)  ∝  L_new(φ) · L_old(θ) · N(φ; 0, I₁₂)
```

which is the same as fitting both batches at once. The old patients are gone, but their likelihood lives on inside the prior. This is exact only if:
1. the normal approximation of the old posterior is accurate (it is, with this much data), and
2. **θ means the same thing in both models.** It doesn't: L_old came from a model without vitals, where α and c1–c5 also absorbed the vitals' effects. This is the weak point of the method.

### What changes: the uncertainty (precision)

With the same Laplace approximation as before, precisions add up:

```
Σ_post⁻¹ ≈ Σ_prior⁻¹ + Z̃ᵀ W_new Z̃

         ≈ ⎡ I₆ + X̃_oldᵀW_oldX̃_old + X̃_newᵀW_newX̃_new      X̃_newᵀ W_new V     ⎤
           ⎣ Vᵀ W_new X̃_new                                 I₆ + Vᵀ W_new V    ⎦

Z̃ = [X̃_new  V]   the new patients' base features and vitals, as rows
W_new = diag( p_i (1 − p_i) ) for the new patients
```

- **Old block:** prior + old data + new data. These parameters end up the most certain (sds roughly 1/√2 of the base model's sds, since the two batches are the same size).
- **New block:** prior + new data only. The vitals are learned from the new patients alone.
- **Off-diagonal blocks:** they were 0 in the prior and now become non-zero. The new data links the old parameters to the vitals; for example, α and the squared vitals trade off, because each V² term has mean 1 and acts partly like an intercept.

For comparison, model 3.4 (new data only) has Σ_prior⁻¹ = I₁₂, so its old block gets only I₆ + X̃_newᵀW_newX̃_new, with no old information.

### What changes: the mean

Approximately (exact if everything were normal), the posterior mean is a **precision-weighted average** of the prior mean and what the new data alone says:

```
μ_post ≈ Σ_post ( Σ_prior⁻¹ μ_prior  +  Λ_new φ̂_new )

φ̂_new = the estimate from the new data alone (≈ model 3.4)
Λ_new = Z̃ᵀ W_new Z̃, the new data's information
```

- **Vitals:** their prior precision is only 1, so g ≈ what the new data says.
- **Old block:** a compromise between μ_old and the new data, pulled toward whichever is more precise.

**Illustration for α** (rough numbers, not from a run): the prior says −3.68 with sd ≈ 0.10, and the new data with vitals points toward −4.2 with sd ≈ 0.12. With weights 1/0.10² = 100 and 1/0.12² ≈ 69:

```
α_post ≈ (100 · (−3.68) + 69 · (−4.2)) / 169 ≈ −3.89
```

So the posterior is pulled away from the truth by the old model's bias, and confidently so, because the prior is narrow. The vitals and other parameters partly compensate, which is why 3.3 ranks patients well (same AUC) but its probabilities are less accurate (mean |p − p_true| 0.0065 vs 0.0042 for the recalibrated model).

### Summary

```
base prior:          θ ~ N( 0,                 I₆                        )
base posterior:      θ ~ N( μ_old,             Σ_old                     )
3.3 prior:           φ ~ N( (μ_old, 0₆),       blockdiag(Σ_old, I₆)      )
3.3 posterior:       φ ~ N( μ_post,            Σ_post                    )
                     Σ_post⁻¹ ≈ blockdiag(Σ_old⁻¹, I₆) + Z̃ᵀ W_new Z̃
```

Each step adds information (precision) on top of the previous one. The old patients' information enters only through Σ_old⁻¹ and μ_old.

---

# Meeting notes: one-step vs two-step (2026-10-13)

## The question

What I asked: is the goal a joint hierarchical BLR model for D_old and D_new that estimates the target ω directly and takes possible bias in model 1 into account, instead of the two-step model (model 1 estimates β, the updated model estimates θ) that inherits model 1's bias?

The supervisor's answer, in short:
- **Goal:** estimate the risk of death or diagnosis with **well-calibrated uncertainty**, using Bayesian statistics.
- The two-step model is natural and probably fine, but it is an **approximation**, not a "principled Bayesian approach". What is the more genuine Bayesian model? Maybe a hierarchical one.
- **Then compare** the more exact one-step model with the approximate two-step model: **how much worse is the two-step model?**
- Follow-up: **how does 3.6 compare with 3.3?**

## Which models answer it

| | Model | Uses old data as | Role |
|---|---|---|---|
| 3.1 | fixed base + vitals | a fixed score (posterior mean only) | two-step, crudest |
| 3.2 | recalibrated base + vitals | a score with a new intercept and slope | two-step |
| **3.3** | **old posterior as prior** | N(μ_old, Σ_old) as prior | **the two-step model** |
| 3.4 | new data only | not at all | reference |
| **3.5** | **joint, imputed vitals** | raw old records; their missing vitals integrated out | **one-step, one generative model** |
| **3.6** | **joint V2, hierarchical** | raw old records in their own model, linked by τ | **one-step, from the .tex** |

## Model 3.6: the hierarchical model from the .tex

Notation as in the .tex (note: in the sections above θ meant the base model's parameters; here β is the old model and θ the new one):

```
β = (α, c1, …, c5)                 old model: intercept + base features         6 parameters
θ = (α, c1, …, c5, g1, …, g6)       new model: intercept + base features + vitals  12 parameters

old patients:   logit p_i = x̃_iᵀ β        x̃_i = (1, x_i)
new patients:   logit p_j = w̃_jᵀ θ        w̃_j = (1, x_j, v_j)

β ~ N(0, I₆)
θ | β, τ ~ N( Aβ, diag(τ² I₆, I₆) )        A = ⎡ I₆ ⎤   (shared coefficients centred on β, vitals on 0)
                                               ⎣ 0  ⎦
τ ~ Half-Normal(1)

p(β, θ, τ | D_old, D_new) ∝ L_old(β) · L_new(θ) · N(θ; Aβ, Σ_τ) · N(β; 0, I₆) · p(τ)

risk:  ω⋆ = σ(w̃⋆ᵀ θ), one value per posterior draw
```

- **One deliberate change from the .tex:** the .tex uses τ² I for all of θ. Taken literally, the vitals would be centred on 0 with sd τ ≈ 0.16, shrinking them hard towards 0 for no reason. So τ applies only to the 6 shared coefficients; the vitals keep the N(0, 1) prior of every other model. **Ask whether this is acceptable.**
- **Sampling:** centred parameterisation with `target_accept=0.99` gives 0 divergences, r_hat ≤ 1.004, ESS ≥ 1500 per parameter, ~3 min. Default settings gave 2 divergences, 0.95 gave 1, and the non-centred form (θ = Aβ + sd·δ) gave 5: with this much data the centred form suits better.
- **τ posterior: 0.16 ± 0.09**, clearly above 0. The old and new base coefficients really do differ.

## Key insight: 3.3 is 3.6 with τ = 0

With τ = 0 the shared part of θ equals β, so the joint posterior becomes

```
p(θ | D_old, D_new) ∝ L_new(θ) · L_old(θ_shared) · priors
```

which is exactly what 3.3 computes, except that 3.3 replaces L_old · prior by its normal approximation N(μ_old, Σ_old). So **the two-step model's error splits into two parts:**

1. **The normal approximation of the old posterior:** probably negligible with 20 000 old patients (not tested yet; test by fitting 3.6 with τ fixed near 0).
2. **The assumption τ = 0, i.e. that model 1's coefficients mean the same thing in the new model.** The data reject it (τ ≈ 0.16). This is the real cost.

Why the coefficients differ: β is the effect *without* the vitals, θ the effect *with* them (omitted variables; logistic regression is also non-collapsible). For example, the old intercept −3.68 vs the true −4.2: the intercept absorbs the vitals' average extra risk.

## 3.5 vs 3.6

Both fit old and new patients together; they differ in how they handle the old patients' **missing vitals**.

**3.5, imputed vitals: one model for everyone**
- Old and new patients share the same coefficients.
- Also learns a model of the vitals given age, sex and complaint (from the new patients) and averages each old patient's risk over the vitals they probably had (Gauss–Hermite grid, 125 points).
- Assumes the old and new patients come from the **same population**. Models the bias instead of just tolerating it.
- Slower (grid + dense mass matrix), and the grid grows as 5^(number of vitals): 125 for 3 vitals, 3 125 for 5, ~390 000 for 8.

**3.6, hierarchical link: two models tied together**
- Old patients get their own model without vitals (β), new patients the full model (θ), with θ pulled towards β by τ.
- Doesn't recover the missing vitals; any difference (left-out vitals or real population drift) is absorbed by τ. **Allows for the bias but doesn't model it** (the .tex says the same in assumption 5).
- τ is estimated from only 6 coefficient differences, so its prior matters.
- Simpler and faster; the safer choice when old and new data might really differ.

**In short:** 3.5 asks "what were the old patients' vitals probably?", 3.6 asks "how similar is the old model to the new one?". On this simulated data (same population by construction) 3.5's stronger assumptions hold, so it should win here.

## Have we hit the goal?

**Mostly yes.**
- 3.6 is a genuine one-step model: one joint posterior over β, θ, τ from both datasets, and the risk ω comes straight from it. It doesn't simply inherit model 1, since τ lets θ move away from β.
- But **3.5 is arguably the more "äkta" Bayesian model**: one generative model for the whole chain, with the missing vitals integrated out. It also scores best.
- So the answer to "what is the principled model?" has two candidates: 3.5 (same population, models the bias) and 3.6 (hierarchical, tolerates the bias). **Ask which one the supervisor has in mind.**

## Results

Single run: seed 42, N_old = N_new = N_test = 20 000, ~3% mortality (test set: 569 deaths). All models sampled with 6 chains × 667 draws.

### 4.1 Point predictions

| Model | AUC | Brier | Log loss | Mean \|p − p_true\| |
|---|---|---|---|---|
| base (old data, no vitals) | 0.7050 | 0.0271 | 0.1215 | 0.0122 |
| 3.1 fixed base + vitals | 0.7509 | 0.0265 | 0.1167 | 0.0074 |
| 3.2 recalibrated base + vitals | 0.7526 | 0.0264 | 0.1158 | 0.0039 |
| 3.3 old posterior as prior | 0.7540 | 0.0264 | 0.1161 | 0.0053 |
| 3.4 new data only | 0.7560 | 0.0264 | 0.1156 | 0.0031 |
| 3.5 joint, imputed vitals | 0.7555 | 0.0264 | 0.1156 | **0.0030** |
| 3.6 joint, hierarchical | 0.7554 | 0.0264 | 0.1156 | 0.0033 |
| true probabilities | 0.7593 | 0.0263 | 0.1152 | 0 |

AUC, Brier and log loss barely separate the models (only 569 test deaths, so they are noisy). Mean |p − p_true| is cleaner because the simulation knows each patient's true risk.

### 4.2 Uncertainty: true risk per patient

| Model | Cover 50% | Cover 80% | Cover 95% | 95% width | z RMS | 95% interval score | Verdict |
|---|---|---|---|---|---|---|---|
| base | 12% | 22% | 34% | 0.0112 | 4.87 | 0.311 | overconfident |
| 3.1 fixed | 11% | 22% | 32% | 0.0099 | 8.56 | 0.157 | overconfident |
| 3.2 recalibrated | 27% | 48% | 65% | 0.0120 | 1.91 | 0.032 | overconfident |
| 3.3 old posterior as prior | 22% | 41% | **57%** | 0.0134 | **2.39** | **0.059** | overconfident |
| 3.4 new data only | 54% | 80% | 91% | 0.0155 | 1.06 | 0.020 | calibrated |
| 3.5 joint, imputed | 44% | 73% | 88% | **0.0129** | 1.21 | **0.019** | overconfident (just) |
| 3.6 joint, hierarchical | 50% | 75% | 86% | 0.0147 | 1.24 | 0.022 | overconfident (just) |

- **Cover ℓ:** share of test patients whose true risk is inside their ℓ credible interval; calibrated ≈ ℓ.
- **z RMS:** error in posterior sds on the log-odds scale; calibrated ≈ 1, 2 = errors twice as large as claimed.
- **Interval score:** width + 40 × how far the truth falls outside; lower is better.
- **Verdict:** overconfident if 95% coverage < 90%, too cautious if > 99%. 3.5 and 3.6 are just under the cut-off; with one dataset that is borderline.

**Coverage curve** (notebook 4.2, coverage at every level 5–95% vs the diagonal): three clear groups. Base and 3.1 far below the diagonal (~⅓ coverage at 95%); 3.2 and 3.3 well below; 3.4, 3.5 and 3.6 close to the diagonal.

### 4.2 Uncertainty: coefficients vs the true values

Posterior mean (sd); * = 95% credible interval misses the true value.

| | 3.3 prior | 3.4 new only | 3.5 joint | 3.6 hierarchical | true |
|---|---|---|---|---|---|
| intercept | −3.78 (0.07) * | −3.92 (0.10) * | −3.97 (0.08) * | −3.90 (0.09) * | −4.20 |
| age | 0.76 (0.03) | 0.80 (0.05) | 0.81 (0.03) | 0.79 (0.04) | 0.80 |
| male | 0.25 (0.06) | 0.20 (0.08) | 0.25 (0.06) | 0.21 (0.07) | 0.30 |
| shortness of breath | 0.37 (0.07) | 0.30 (0.11) | 0.37 (0.07) | 0.31 (0.09) * | 0.50 |
| headache | −0.98 (0.10) * | −0.91 (0.14) | −0.97 (0.10) * | −0.96 (0.11) * | −0.70 |
| abdominal pain | −0.40 (0.09) * | −0.38 (0.12) | −0.39 (0.09) * | −0.39 (0.10) | −0.20 |
| V1 | 0.10 (0.05) | 0.10 (0.05) | 0.10 (0.05) | 0.10 (0.05) | 0.10 |
| V1² | 0.03 (0.01) | 0.03 (0.01) | 0.03 (0.01) | 0.03 (0.01) | 0.04 |
| V2 | −0.26 (0.04) | −0.29 (0.05) | −0.29 (0.05) | −0.29 (0.05) | −0.30 |
| V2² | 0.04 (0.02) * | 0.08 (0.02) | 0.08 (0.03) | 0.07 (0.03) | 0.10 |
| V3 | 0.15 (0.05) | 0.14 (0.05) | 0.14 (0.05) | 0.14 (0.05) | 0.20 |
| V3² | 0.08 (0.02) | 0.11 (0.02) | 0.11 (0.02) | 0.11 (0.02) | 0.10 |
| **inside 95% CrI** | **8/12** | **11/12** | **9/12** | **9/12** | |
| **z RMS** | **2.35** | **1.31** | **1.52** | **1.56** | |

- 3.3 locks the base coefficients near model 1's values and compensates elsewhere: **V2² comes out at 0.04 vs a true 0.10.**
- Every model misses the true intercept, including the correctly specified 3.4. The new batch happened to have higher mortality (3.17%) than the test set (2.85%): sampling noise in this one dataset.
- Part of model 1's "bias" here is also sampling noise in the old data (headache −1.06 vs true −0.7 is ~2.4 sd off), not only the systematic effect of leaving out the vitals.

### 4.2 Uncertainty: check without the true risk (usable on real data)

Observed deaths in 10 risk groups vs each model's 95% predictive interval:

| Model | Groups inside 95% PI | z RMS |
|---|---|---|
| base | 8/10 | 1.33 |
| 3.1 fixed | **4/10** | 1.96 |
| 3.2 recalibrated | 10/10 | 1.04 |
| 3.3 old posterior as prior | 7/10 | 1.58 |
| 3.4 new data only | 10/10 | 1.32 |
| 3.5 joint, imputed | 9/10 | 1.56 |
| 3.6 joint, hierarchical | 10/10 | 1.34 |

Weak test, as expected: the predictive interval includes the chance of who dies (±15 deaths per group of ~57), which hides most of the parameter uncertainty. Only 3.1 clearly fails. Passing says little; failing is a clear sign.

## Conclusions

1. **How much worse is two-step (3.3) than one-step (3.5 / 3.6)?**
   - **Point predictions: a little.** Log loss 0.1161 vs 0.1156; mean |p − p_true| 0.0053 vs 0.0030–0.0033.
   - **Uncertainty: a lot.** 95% intervals contain the true risk for 57% of patients vs 86–88%; errors are 2.4× the claimed sd vs 1.2×; interval score ~3× worse (0.059 vs 0.019–0.022). The two-step model is **confidently wrong**.
2. **The cost isn't having two steps as such, but assuming the old coefficients mean the same thing in the new model** (τ = 0). The normal approximation is probably negligible at this sample size.
3. **The cruder two-step variants are worse still:** 3.1 treats the base score as exactly known, so model 1's uncertainty never enters its intervals (32% coverage, z RMS 8.6).
4. **3.5 is best overall:** most accurate and narrowest intervals with almost the right coverage. But the data were simulated under its assumptions (same population, vitals depend on the base features), which favours it.
5. **3.6 is close to 3.5** and doesn't need the same-population assumption.
6. **With 20 000 new patients the old data barely helps:** new-data-only is calibrated (91%) and nearly as good as anything. Reusing old data matters when the new batch is small, which is exactly where the models will differ most.

## Caveats

- **One simulated dataset.** All test patients share one posterior per model, so their misses are correlated, and even the correctly specified 3.4 reaches only 91% rather than 95%. Coverage needs repeated simulations before relying on it.
- 3.5's convergence was not rechecked in the latest run (check divergences and r_hat when the notebook is run).
- The simulation favours 3.5 (same population by construction).
- On real data, only the group check (and calibration intercept/slope) is available; the true risk is unknown.

## Next steps

1. **Vary the new batch size** (500, 2 000, 5 000, 20 000) with ~20 repetitions each, and report mean and spread of coverage, interval width and mean |p − p_true|. This shows when two-step costs you, and how much. The biggest lever.
2. **Add a scenario where old and new populations really differ** (e.g. a different baseline risk in the new batch). That breaks 3.5's assumption and is where 3.6 should do best.
3. **Fit 3.6 with τ fixed near 0** to measure the normal-approximation error in 3.3 separately.
4. **Stronger bias without more vitals:** give the existing vitals larger effects or link them more strongly to the base features. More vitals would also increase 3.3's bias but makes 3.5 impractical (grid size 5^d).
5. Use mean |p − p_true| and coverage as the main scores, not AUC / log loss (too few deaths to separate the models).
6. For real data: calibration by risk group, calibration intercept/slope (outcome ~ logit p̂: ≈ 0 and ≈ 1 when calibrated), posterior predictive checks.

## Questions for the supervisor

- Which model does he mean by "the principled Bayesian model": 3.5 (one generative model, same population) or 3.6 (hierarchical link)?
- Is applying τ only to the shared coefficients (vitals keep N(0, 1)) acceptable, or should the literal τ² I from the .tex be used?
- What new-batch size and how much population drift are realistic for the real data? This decides which scenario the comparison should focus on.
- Is coverage of the true risk (simulation) plus the group check (real data) the right way to judge "well-calibrated uncertainty"?
