#!/usr/bin/env python
"""Convert PyMC NetCDF posterior traces into prediction grids and posterior files.

Background
----------
The ED Atlas is a collection of Bayesian logistic regression models trained
on the SwED database. Each model predicts a binary outcome (a specific ICD-10
diagnosis within 30 days, or death within 30 days) for a patient presenting
to the ED, given their age, sex, and presenting complaint (SNOMED CT ID).

The statistical model for each outcome is:

    logit(P(outcome)) = β · x

where x is a row of the design matrix built from the patsy formula:

    sex * cr(age, knots=[43, 65], lower_bound=22, upper_bound=88) * symptoms

Here cr() is a natural cubic regression spline with 2 interior knots (4 basis
functions). Patsy expands the full interaction into these column groups:

    Dims 0-9:       Base (Intercept, sex, cr[0..3], sex:cr[0..3])
    Dims 10..10+S:  symptoms[0..S-1]           (main effects)
    Dims 10+S..10+2S: sex:symptoms[0..S-1]     (sex interaction)
    Dims 10+2S..10+6S: cr[0..3]:symptoms       (4 per symptom)
    Dims 10+6S..10+10S: sex:cr[0..3]:symptoms  (4 per symptom)

    Total: 10 × (1 + S) dims, where S = number of symptoms

For any single prediction, x is nonzero in exactly 20 of these dims:
10 base dims + 10 dims for the active symptom. This sparsity allows an
exact structured covariance decomposition (see below).

Outputs
-------
1. **Prediction grid** (atlas_*_logitnormals.parquet):
   LogitNormal(mu, sigma) per (sctid, age, sex) cell, where mu and sigma
   are the mean and std of β·x across posterior samples.

2. **Structured posterior** (atlas_*_posterior.npz):
   Exact decomposition of the posterior covariance exploiting the 20-dim
   sparsity structure.

   Mortality (atlas_mortality_posterior.npz):
     - beta_mean: (n_dims,) posterior mean
     - base_cov: (10, 10) covariance of the base block (shared)
     - sym_covs: (S, 10, 10) per-symptom covariance blocks
     - cross_covs: (S, 10, 10) cross-covariance Cov(base, symptom_i)
     - sctids: (S,) symptom SCTID labels
     - cr_basis: (92, 4) precomputed natural cubic spline basis for ages 18-109

   Diagnosis (atlas_diagnosis_posterior.npz) — single file, all ICD codes stacked:
     - icd_codes: (K,) ICD code strings
     - beta_means: (K, n_dims) posterior means
     - base_covs: (K, 10, 10) base covariance per model
     - sym_covs: (K, S, 10, 10) per-symptom covariance blocks
     - cross_covs: (K, S, 10, 10) cross-covariance blocks
     - sctids: (S,) symptom SCTID labels (shared across all diagnosis models)
     - cr_basis: (92, 4) precomputed natural cubic spline basis for ages 18-109

   Predictive variance at (age, sex, symptom i):
     x = [x_base(10), x_sym(10)]
     Var = x_base @ base_cov @ x_base
         + x_sym @ sym_covs[i] @ x_sym
         + 2 * x_base @ cross_covs[i] @ x_sym

   This is EXACT — no approximation. Verified against full covariance.

Input files (in edatlas_stuff/):
    trace.nc                    — mortality posterior (with symptoms)
    mortality_dummies.parquet   — training data (symptom column names/order)
    trace_diag_*.nc             — diagnosis posteriors (one per ICD code)

Usage (from project root):
    uv run --with arviz --with netcdf4 --with patsy python edatlas_stuff/convert_traces.py

Dependencies (NOT in project venv — only needed for this one-off conversion):
    arviz, netcdf4, patsy    (plus polars, numpy which are already in the project)
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import patsy
import polars as pl

EDATLAS_DIR = Path(__file__).parent
RESOURCES_DIR = EDATLAS_DIR.parent / "resources"
DATA_DIR = EDATLAS_DIR / "data"

# Model specification — must match the PyMC training code
AGE_KNOTS = [22.0, 43.0, 65.0, 88.0]
AGE_RANGE = (18, 109)  # inclusive
PATSY_FORMULA = "sex * cr(age, knots=k, lower_bound=lb, upper_bound=ub) * symptoms"


def get_symptom_columns(dummies_path: Path) -> list[str]:
    """Return symptom column names from training dummies, in order.

    Column order determines the beta-to-symptom mapping.
    """
    cols = pl.read_parquet(dummies_path, n_rows=0).columns
    return [c for c in cols if c.startswith("symptoms_")]


def symptom_col_to_sctid(col: str) -> str:
    """'symptoms_[1023001]' → '1023001', 'symptoms_nosct:other' → 'nosct:other'."""
    return col.removeprefix("symptoms_").strip("[]")


def symptom_dims(si: int, n_symptoms: int) -> list[int]:
    """Return the 10 beta dimensions for symptom index si.

    Matches the patsy column layout for the interaction formula.
    """
    return [
        10 + si,                             # main effect
        10 + n_symptoms + si,                # sex:symptom
        10 + 2 * n_symptoms + 4 * si + 0,   # cr[0]:symptom
        10 + 2 * n_symptoms + 4 * si + 1,   # cr[1]:symptom
        10 + 2 * n_symptoms + 4 * si + 2,   # cr[2]:symptom
        10 + 2 * n_symptoms + 4 * si + 3,   # cr[3]:symptom
        10 + 6 * n_symptoms + 4 * si + 0,   # sex:cr[0]:symptom
        10 + 6 * n_symptoms + 4 * si + 1,   # sex:cr[1]:symptom
        10 + 6 * n_symptoms + 4 * si + 2,   # sex:cr[2]:symptom
        10 + 6 * n_symptoms + 4 * si + 3,   # sex:cr[3]:symptom
    ]


BASE_DIMS = list(range(10))


def compute_cr_basis() -> np.ndarray:
    """Precompute the natural cubic spline basis for integer ages 18-109.

    Returns (92, 4) float32 array. Saved into posterior npz files so that
    the ed_atlas module can reconstruct design vectors without patsy.
    """
    ages = np.arange(AGE_RANGE[0], AGE_RANGE[1] + 1, dtype=np.float64)
    dm = patsy.dmatrix(
        "cr(age, knots=k, lower_bound=lb, upper_bound=ub) - 1",
        {"age": ages, "k": AGE_KNOTS[1:-1], "lb": AGE_KNOTS[0], "ub": AGE_KNOTS[-1]},
    )
    return np.asarray(dm, dtype=np.float32)


def extract_structured_covariance(
    betas: np.ndarray,
    n_symptoms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the structured exact covariance from posterior samples.

    Returns (beta_mean, base_cov, sym_covs, cross_covs).
    """
    beta_mean = betas.mean(axis=0).astype(np.float32)
    centered = betas - betas.mean(axis=0)

    base_block = centered[:, BASE_DIMS]
    base_cov = (base_block.T @ base_block / (len(centered) - 1)).astype(np.float32)

    sym_covs = np.zeros((n_symptoms, 10, 10), dtype=np.float32)
    cross_covs = np.zeros((n_symptoms, 10, 10), dtype=np.float32)

    for si in range(n_symptoms):
        sd = symptom_dims(si, n_symptoms)
        sym_block = centered[:, sd]
        sym_covs[si] = (sym_block.T @ sym_block / (len(centered) - 1)).astype(np.float32)
        cross_covs[si] = (base_block.T @ sym_block / (len(centered) - 1)).astype(np.float32)

    return beta_mean, base_cov, sym_covs, cross_covs


def build_prediction_grid(
    symptom_cols: list[str],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Build design matrix for the full (symptom, age, sex) prediction grid.

    Returns (D, sctids, grid_age, grid_sex).
    """
    n_symptoms = len(symptom_cols)
    sctids = [symptom_col_to_sctid(c) for c in symptom_cols]
    ages = np.arange(AGE_RANGE[0], AGE_RANGE[1] + 1)
    n_ages = len(ages)
    n_rows = n_symptoms * n_ages * 2

    grid_age = np.tile(np.repeat(ages, 2), n_symptoms)
    grid_sex = np.tile([0, 1], n_symptoms * n_ages)
    grid_symptom_idx = np.repeat(np.arange(n_symptoms), n_ages * 2)

    symptom_dummies = np.zeros((n_rows, n_symptoms), dtype=np.float32)
    symptom_dummies[np.arange(n_rows), grid_symptom_idx] = 1.0

    D = patsy.dmatrix(
        PATSY_FORMULA,
        {
            "sex": grid_sex.astype(np.float64),
            "age": grid_age.astype(np.float64),
            "k": AGE_KNOTS[1:-1],
            "lb": AGE_KNOTS[0],
            "ub": AGE_KNOTS[-1],
            "symptoms": symptom_dummies,
        },
    )
    D = np.asarray(D, dtype=np.float32)

    expected_cols = 10 * (1 + n_symptoms)
    assert D.shape == (n_rows, expected_cols), (
        f"Design matrix {D.shape} != ({n_rows}, {expected_cols})"
    )

    return D, sctids, grid_age, grid_sex


def convert_mortality(symptom_cols: list[str]) -> None:
    """Convert trace.nc → prediction grid + structured posterior."""
    trace_path = EDATLAS_DIR / "trace.nc"
    if not trace_path.exists():
        print(f"SKIP mortality: {trace_path} not found")
        return

    print(f"Loading {trace_path} ...")
    idata = az.from_netcdf(str(trace_path))
    betas = idata.posterior.betas.values.reshape(-1, idata.posterior.betas.shape[-1])
    n_samples, n_dims = betas.shape
    n_symptoms = len(symptom_cols)
    print(f"  Posterior: {n_samples} samples × {n_dims} dims")

    expected_dims = 10 * (1 + n_symptoms)
    if n_dims != expected_dims:
        print(f"  ERROR: expected {expected_dims} dims, got {n_dims}")
        sys.exit(1)

    # --- Prediction grid ---
    print("  Building prediction grid ...")
    D, sctids, grid_age, grid_sex = build_prediction_grid(symptom_cols)
    n_ages = AGE_RANGE[1] - AGE_RANGE[0] + 1
    rows_per_symptom = n_ages * 2

    print("  Computing posterior predictive ...")
    betas_f32 = betas.astype(np.float32)
    mu_all = np.empty(D.shape[0], dtype=np.float32)
    sigma_all = np.empty(D.shape[0], dtype=np.float32)

    for si in range(n_symptoms):
        start = si * rows_per_symptom
        end = start + rows_per_symptom
        logits = D[start:end] @ betas_f32.T
        mu_all[start:end] = logits.mean(axis=1)
        sigma_all[start:end] = logits.std(axis=1)
        if (si + 1) % 100 == 0:
            print(f"    {si+1}/{n_symptoms}")

    sctid_col = [sctids[i] for i in np.repeat(np.arange(n_symptoms), rows_per_symptom)]
    out_df = pl.DataFrame({
        "sctid": sctid_col,
        "age": grid_age.astype(np.int64),
        "is_male": grid_sex.astype(np.int64),
        "mean_prediction": mu_all,
        "std_prediction": sigma_all,
    }).sort(["sctid", "age", "is_male"])

    grid_path = RESOURCES_DIR / "atlas_mortality_logitnormals.parquet"
    out_df.write_parquet(grid_path)
    print(f"  Saved grid: {grid_path} ({out_df.shape})")

    # --- Structured posterior ---
    print("  Extracting structured covariance ...")
    beta_mean, base_cov, sym_covs, cross_covs = extract_structured_covariance(
        betas, n_symptoms
    )
    cr_basis = compute_cr_basis()
    post_path = DATA_DIR / "atlas_mortality_posterior.npz"
    np.savez_compressed(
        post_path,
        beta_mean=beta_mean,
        base_cov=base_cov,
        sym_covs=sym_covs,
        cross_covs=cross_covs,
        sctids=np.array(sctids),
        cr_basis=cr_basis,
    )
    print(f"  Saved posterior: {post_path} ({post_path.stat().st_size / 1024:.0f} KB)")


def convert_all_diagnoses(symptom_cols: list[str]) -> None:
    """Convert all trace_diag_*.nc → single atlas_diagnosis_posterior.npz.

    Stacks all ICD models into one file with arrays indexed by ICD position.
    """
    nc_paths = sorted(EDATLAS_DIR.glob("trace_diag_*.nc"))
    if not nc_paths:
        print("SKIP diagnosis: no trace_diag_*.nc files found")
        return

    # Determine diagnosis symptom set
    diag_parquet = RESOURCES_DIR / "atlas_logitnormals.parquet"
    if diag_parquet.exists():
        diag_sctids_set = set(
            pl.read_parquet(diag_parquet, columns=["sctid"])["sctid"]
            .unique().to_list()
        )
        diag_symptom_cols = [c for c in symptom_cols
                             if symptom_col_to_sctid(c) in diag_sctids_set]
    else:
        diag_symptom_cols = symptom_cols

    n_symptoms = len(diag_symptom_cols)
    expected_dims = 10 * (1 + n_symptoms)
    sctids = [symptom_col_to_sctid(c) for c in diag_symptom_cols]

    icd_codes = []
    beta_means_list = []
    base_covs_list = []
    sym_covs_list = []
    cross_covs_list = []

    for nc_path in nc_paths:
        icd_code = nc_path.stem.removeprefix("trace_diag_")
        print(f"Loading {nc_path} (ICD: {icd_code}) ...")

        idata = az.from_netcdf(str(nc_path))
        betas = idata.posterior.betas.values.reshape(-1, idata.posterior.betas.shape[-1])
        n_samples, n_dims = betas.shape
        print(f"  Posterior: {n_samples} samples × {n_dims} dims")

        if n_dims != expected_dims:
            print(f"  ERROR: expected {expected_dims} dims ({n_symptoms} symptoms), "
                  f"got {n_dims}. Skipping.")
            continue

        print("  Extracting structured covariance ...")
        beta_mean, base_cov, sym_covs_i, cross_covs_i = extract_structured_covariance(
            betas, n_symptoms
        )

        icd_codes.append(icd_code)
        beta_means_list.append(beta_mean)
        base_covs_list.append(base_cov)
        sym_covs_list.append(sym_covs_i)
        cross_covs_list.append(cross_covs_i)

    if not icd_codes:
        print("  No valid diagnosis traces processed.")
        return

    cr_basis = compute_cr_basis()
    post_path = DATA_DIR / "atlas_diagnosis_posterior.npz"
    np.savez_compressed(
        post_path,
        icd_codes=np.array(icd_codes),
        beta_means=np.stack(beta_means_list),
        base_covs=np.stack(base_covs_list),
        sym_covs=np.stack(sym_covs_list),
        cross_covs=np.stack(cross_covs_list),
        sctids=np.array(sctids),
        cr_basis=cr_basis,
    )
    print(f"  Saved {len(icd_codes)} ICD posteriors: {post_path} "
          f"({post_path.stat().st_size / 1024:.0f} KB)")


def main():
    dummies_path = EDATLAS_DIR / "mortality_dummies.parquet"
    if not dummies_path.exists():
        print(f"ERROR: {dummies_path} not found")
        sys.exit(1)

    print("Reading symptom columns from training data ...")
    symptom_cols = get_symptom_columns(dummies_path)
    print(f"  {len(symptom_cols)} symptoms")

    convert_mortality(symptom_cols)
    convert_all_diagnoses(symptom_cols)

    print("\nDone.")


if __name__ == "__main__":
    main()
