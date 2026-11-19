# Bayesian logistic regression with missing vitals (selection-model approach)

Pure numpy/scipy Gibbs sampler – no PyMC/Stan needed.

## Files
| file | purpose |
|---|---|
Code lives in `src/preprocessing/`, data in `Data/preprocessing/`, outputs in `Results/preprocessing/`.

| file | purpose |
|---|---|
| `src/preprocessing/simulate_data.py` | synthetic data: 3 base vars (never missing) + 11 vitals, binary outcome, a different missingness mechanism per vital |
| `src/preprocessing/bayes_missing.py` | the model (`fit`, `run_chain`, `prepare`, R-hat/ESS) |
| `src/preprocessing/run_analysis.py` | baselines, both Bayesian fits, diagnostics, sensitivity analysis, figures |

```bash
PYTHONPATH=src python3 src/preprocessing/simulate_data.py
PYTHONPATH=src python3 src/preprocessing/run_analysis.py
```

## Model
For patient *i*, base variables **z**ᵢ (age, sex, comorbidity), vitals **x**ᵢ (11, partly missing), outcome *y*ᵢ:

1. **Vitals:** xᵢ | zᵢ ~ MVN(Bᵀ[1, zᵢ], Σ) with conjugate matrix-normal / inverse-Wishart priors.
2. **Outcome:** yᵢ ~ Bernoulli(logit⁻¹(α + zᵢβ_z + xᵢβ_x)), N(0, 2.5²) priors, Pólya-gamma augmentation.
3. **Missingness (selection model), per vital j you list:** P(R_ij = 1) = Φ(γ₀ + zᵢγ_z + yᵢγ_y + γ_x·x_ij).
   * `γ_x ≠ 0` is the non-ignorable (MNAR) part; prior N(0, 1), or fixed at δ for sensitivity analysis.
   * `γ_y` lets recording depend on the outcome ("sicker patients get lactate").
   * Vitals not listed are treated as MCAR/MAR (no missingness model).
4. Missing x_ij are latent parameters drawn each sweep from a Gaussian full conditional that combines (1), (2) and (3), so
   imputation uncertainty flows into the coefficient intervals and the outcome informs the imputations.

Optional: `prior_obs_rate={"lactate": 0.6}` puts an informative prior on a vital's observation rate (Ford et al. idea).

## Using it on real data
```python
import pandas as pd
from preprocessing import bayes_missing as bm
data = bm.prepare(df, ["age","sex","comorbidity"], vital_cols, "outcome", log_vars=("lactate",))
res  = bm.fit(data, selection={"lactate": None, "glucose": None}, chains=4, n_iter=6000, n_burn=2500)
bm.summarize(res["beta"], bm.coef_labels(res))        # per model-unit; multiply by data["x_scale"] for per-SD of the raw scale
```
`selection={}` gives the plain MAR/ignorable joint model. Always check `γ_x` posteriors and run the δ sensitivity analysis.

## Caveats
* `γ_x` is identified only through the parametric form (normal vitals, linear probit) plus the prior – treat it as an assumption, not a finding. Hence the sensitivity analysis.
* The simulation is well-specified (same functional forms as the model); real data will be messier. Misspecification (skewed vitals, non-linear selection) is untested here.
* Results are from a single simulated dataset; they illustrate, not prove, performance.
* Logistic outcome uses a truncated-series Pólya-gamma sampler (validated against exact mean/variance); missingness uses probit.
* Latent-profile (BPMI-style) structure and a multinomial/partially-missing outcome from your notes are not implemented.
