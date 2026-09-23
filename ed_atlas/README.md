# ED Atlas

Bayesian logistic regression priors for ED triage. Predicts 30-day diagnosis (930 ICD-10 codes) and 30-day mortality from age, sex, and presenting complaint (SNOMED CT).

## Usage

```python
from ed_atlas import EdAtlas

atlas = EdAtlas()
probs = atlas.predict(age=65, is_male=1, sctids=["271594007"])        # (n_icds,)
p_death = atlas.predict_death(age=65, is_male=1, sctids=["271594007"]) # float
mu, sigma = atlas.predict_death_distribution(age=65, is_male=1, sctids=["271594007"])
```

On first run, prediction grids are computed from the shipped posteriors and cached in `~/.cache/ed_atlas/`. Subsequent runs load from cache.

## Files

| File | What |
|------|------|
| `_lookup.py` | Module code. `EdAtlas` (grid lookup) + `ParameterPosterior` (exact structured covariance) |
| `data/*.npz` | Shipped posteriors: MVN approximation of PyMC MCMC posteriors (<1% error vs samples) |
| `convert_traces.py` | One-off script to produce the npz files from raw PyMC NetCDF traces into `data/`. Not needed at runtime. Run with `uv run --with arviz --with netcdf4 --with patsy python ed_atlas/convert_traces.py` |

## Dependencies

Runtime: `numpy`, `polars`. Conversion only: `arviz`, `netcdf4`, `patsy`.
