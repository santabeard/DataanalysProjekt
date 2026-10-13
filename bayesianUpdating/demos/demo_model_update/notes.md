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
