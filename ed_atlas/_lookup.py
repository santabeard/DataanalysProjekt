"""ED Atlas lookup and posterior access.

Ships with posterior npz files (small). On first use, computes prediction
grids from the posteriors and caches them in ~/.cache/ed_atlas/.

Shipped data (in ed_atlas/data/):
    atlas_mortality_posterior.npz    — mortality parameter posterior (~300KB)
    atlas_diagnosis_posterior.npz    — all diagnosis posteriors stacked (~300KB per ICD)

Cached grids (in ~/.cache/ed_atlas/, auto-generated):
    atlas_logitnormals.parquet           — diagnosis prediction grid
    atlas_mortality_logitnormals.parquet  — mortality prediction grid

See convert_traces.py for how the npz files are produced from PyMC traces.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

_DATA_DIR = Path(__file__).parent / "data"
_CACHE_DIR = Path.home() / ".cache" / "ed_atlas"

# Age range matching the Bayesian model training grid
AGE_MIN, AGE_MAX = 18, 109
N_AGES = AGE_MAX - AGE_MIN + 1  # 92


@dataclass
class ParameterPosterior:
    """Structured exact posterior for a Bayesian logistic regression model.

    The model formula is:
        logit(P) = beta . x
    where x is built from sex * cr(age) * symptoms (patsy interaction).

    Each prediction uses exactly 20 of the ~4000+ beta dimensions:
    10 "base" dims (intercept, sex, cr[0..3], sex:cr[0..3]) plus
    10 per-symptom dims. The covariance decomposes exactly into:
        base_cov (10x10) + sym_covs[i] (10x10) + cross_covs[i] (10x10)

    Attributes:
        beta_mean: (n_dims,) posterior mean of all betas
        base_cov: (10, 10) covariance of the 10 base dimensions
        sym_covs: (n_symptoms, 10, 10) per-symptom covariance blocks
        cross_covs: (n_symptoms, 10, 10) cross-covariance Cov(base, sym_i)
        sctids: symptom SCTID labels, length n_symptoms
        cr_basis: (92, 4) precomputed natural cubic spline basis for ages 18-109
    """

    beta_mean: np.ndarray
    base_cov: np.ndarray
    sym_covs: np.ndarray
    cross_covs: np.ndarray
    sctids: list[str]
    cr_basis: np.ndarray

    @staticmethod
    def load(path: Path) -> ParameterPosterior:
        """Load from an npz file produced by convert_traces.py."""
        d = np.load(path)
        return ParameterPosterior(
            beta_mean=d["beta_mean"],
            base_cov=d["base_cov"],
            sym_covs=d["sym_covs"],
            cross_covs=d["cross_covs"],
            sctids=d["sctids"].tolist(),
            cr_basis=d["cr_basis"],
        )

    @property
    def n_symptoms(self) -> int:
        return len(self.sctids)

    def _sctid_index(self, sctid: str) -> int:
        try:
            return self.sctids.index(sctid)
        except ValueError:
            raise KeyError(f"Unknown sctid {sctid!r}. Available: {len(self.sctids)}")

    def _symptom_dims(self, si: int) -> list[int]:
        """Return the 10 beta dimensions for symptom index si.

        Matches the patsy column layout for the interaction formula.
        """
        S = self.n_symptoms
        return [
            10 + si,                       # main effect
            10 + S + si,                   # sex:symptom
            10 + 2 * S + 4 * si + 0,      # cr[0]:symptom
            10 + 2 * S + 4 * si + 1,      # cr[1]:symptom
            10 + 2 * S + 4 * si + 2,      # cr[2]:symptom
            10 + 2 * S + 4 * si + 3,      # cr[3]:symptom
            10 + 6 * S + 4 * si + 0,      # sex:cr[0]:symptom
            10 + 6 * S + 4 * si + 1,      # sex:cr[1]:symptom
            10 + 6 * S + 4 * si + 2,      # sex:cr[2]:symptom
            10 + 6 * S + 4 * si + 3,      # sex:cr[3]:symptom
        ]

    def _design_vector(self, age: int, is_male: int) -> tuple[np.ndarray, np.ndarray]:
        """Build the 20-dim design vector for a single prediction.

        Returns (x_base, x_sym) each length 10.
        """
        cr = self.cr_basis[age - AGE_MIN]  # (4,) from precomputed basis
        sex = float(is_male)

        # Base: [intercept, sex, cr[0..3], sex*cr[0..3]]
        x_base = np.array(
            [1.0, sex, cr[0], cr[1], cr[2], cr[3],
             sex * cr[0], sex * cr[1], sex * cr[2], sex * cr[3]],
            dtype=np.float32,
        )

        # Symptom: [1, sex, cr[0..3], sex*cr[0..3]] (same pattern, symptom=1)
        x_sym = np.array(
            [1.0, sex, cr[0], cr[1], cr[2], cr[3],
             sex * cr[0], sex * cr[1], sex * cr[2], sex * cr[3]],
            dtype=np.float32,
        )

        return x_base, x_sym

    def predict(self, age: int, is_male: int, sctid: str) -> tuple[float, float]:
        """Compute posterior predictive LogitNormal(mu, sigma) from parameters.

        Returns (mu, sigma) in logit space. Apply sigmoid(mu) for point estimate.
        """
        si = self._sctid_index(sctid)
        x_base, x_sym = self._design_vector(age, is_male)

        # Mean: beta_mean[base_dims] . x_base + beta_mean[sym_dims] . x_sym
        base_dims = list(range(10))
        sym_dims = self._symptom_dims(si)
        mu = (
            float(self.beta_mean[base_dims] @ x_base)
            + float(self.beta_mean[sym_dims] @ x_sym)
        )

        # Variance: exact decomposition
        var = (
            float(x_base @ self.base_cov @ x_base)
            + float(x_sym @ self.sym_covs[si] @ x_sym)
            + 2.0 * float(x_base @ self.cross_covs[si] @ x_sym)
        )
        sigma = float(np.sqrt(max(var, 0.0)))

        return mu, sigma

    def beta_summary(self, sctid: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) of the 20 active betas for a symptom.

        First 10 are base dims, last 10 are symptom dims.
        """
        si = self._sctid_index(sctid)
        base_dims = list(range(10))
        sym_dims = self._symptom_dims(si)
        dims = base_dims + sym_dims

        mean = self.beta_mean[dims]

        # Diagonal of the 20x20 joint covariance
        base_var = np.diag(self.base_cov)
        sym_var = np.diag(self.sym_covs[si])
        std = np.sqrt(np.concatenate([base_var, sym_var]))

        return mean, std


# ── Grid generation from posteriors ──────────────────────────────────


def _build_design_matrices(cr_basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build design matrices for all (age, sex) combinations.

    Returns (X_base, X_sym) each shape (184, 10) for 92 ages × 2 sexes.
    Row order: age=18 sex=0, age=18 sex=1, age=19 sex=0, ...
    """
    n_rows = N_AGES * 2
    X_base = np.empty((n_rows, 10), dtype=np.float32)
    X_sym = np.empty((n_rows, 10), dtype=np.float32)

    for ai in range(N_AGES):
        cr = cr_basis[ai]
        for sx in range(2):
            row = ai * 2 + sx
            s = float(sx)
            v = [1.0, s, cr[0], cr[1], cr[2], cr[3],
                 s * cr[0], s * cr[1], s * cr[2], s * cr[3]]
            X_base[row] = v
            X_sym[row] = v

    return X_base, X_sym


def _generate_mortality_grid(post: ParameterPosterior) -> pl.DataFrame:
    """Compute mortality prediction grid from posterior parameters."""
    X_base, X_sym = _build_design_matrices(post.cr_basis)
    n_points = X_base.shape[0]  # 184

    # Precompute base quadratic form (shared across symptoms)
    base_quad = (X_base @ post.base_cov * X_base).sum(axis=1)  # (184,)

    sctid_col = []
    age_col = []
    sex_col = []
    mu_col = []
    sigma_col = []

    beta_base = post.beta_mean[:10]

    for si in range(post.n_symptoms):
        sym_dims = post._symptom_dims(si)
        beta_sym = post.beta_mean[sym_dims]

        mu = X_base @ beta_base + X_sym @ beta_sym  # (184,)
        sym_quad = (X_sym @ post.sym_covs[si] * X_sym).sum(axis=1)
        cross_quad = (X_base @ post.cross_covs[si] * X_sym).sum(axis=1)
        var = base_quad + sym_quad + 2.0 * cross_quad
        sigma = np.sqrt(np.maximum(var, 0.0))

        for row in range(n_points):
            ai, sx = divmod(row, 2)
            sctid_col.append(post.sctids[si])
            age_col.append(AGE_MIN + ai)
            sex_col.append(sx)
            mu_col.append(float(mu[row]))
            sigma_col.append(float(sigma[row]))

    return pl.DataFrame({
        "sctid": sctid_col,
        "age": age_col,
        "is_male": sex_col,
        "mean_prediction": np.array(mu_col, dtype=np.float32),
        "std_prediction": np.array(sigma_col, dtype=np.float32),
    }).sort(["sctid", "age", "is_male"])


def _generate_diagnosis_grid(
    diag_npz_path: Path,
) -> pl.DataFrame:
    """Compute diagnosis prediction grid from stacked posterior npz."""
    d = np.load(diag_npz_path)
    icd_codes = d["icd_codes"].tolist()
    sctids = d["sctids"].tolist()
    cr_basis = d["cr_basis"]
    n_icds = len(icd_codes)
    n_symptoms = len(sctids)

    X_base, X_sym = _build_design_matrices(cr_basis)
    n_points = X_base.shape[0]  # 184

    total_rows = n_icds * n_symptoms * n_points
    all_sctid = []
    all_age = np.empty(total_rows, dtype=np.int64)
    all_sex = np.empty(total_rows, dtype=np.int64)
    all_icd = []
    all_mu = np.empty(total_rows, dtype=np.float32)
    all_sigma = np.empty(total_rows, dtype=np.float32)

    # Precompute age/sex pattern (repeats for every symptom and ICD)
    ages_184 = np.array([AGE_MIN + ai for ai in range(N_AGES) for _ in range(2)], dtype=np.int64)
    sexes_184 = np.array([sx for _ in range(N_AGES) for sx in range(2)], dtype=np.int64)

    row_offset = 0
    for ki in range(n_icds):
        icd = icd_codes[ki]
        beta_mean = d["beta_means"][ki]
        base_cov = d["base_covs"][ki]
        sym_covs_k = d["sym_covs"][ki]
        cross_covs_k = d["cross_covs"][ki]

        beta_base = beta_mean[:10]
        base_quad = (X_base @ base_cov * X_base).sum(axis=1)

        for si in range(n_symptoms):
            S = n_symptoms
            sym_dims = [
                10 + si, 10 + S + si,
                10 + 2*S + 4*si, 10 + 2*S + 4*si+1, 10 + 2*S + 4*si+2, 10 + 2*S + 4*si+3,
                10 + 6*S + 4*si, 10 + 6*S + 4*si+1, 10 + 6*S + 4*si+2, 10 + 6*S + 4*si+3,
            ]
            beta_sym = beta_mean[sym_dims]

            mu = X_base @ beta_base + X_sym @ beta_sym
            sym_quad = (X_sym @ sym_covs_k[si] * X_sym).sum(axis=1)
            cross_quad = (X_base @ cross_covs_k[si] * X_sym).sum(axis=1)
            var = base_quad + sym_quad + 2.0 * cross_quad
            sigma = np.sqrt(np.maximum(var, 0.0))

            end = row_offset + n_points
            all_sctid.extend([sctids[si]] * n_points)
            all_icd.extend([icd] * n_points)
            all_age[row_offset:end] = ages_184
            all_sex[row_offset:end] = sexes_184
            all_mu[row_offset:end] = mu
            all_sigma[row_offset:end] = sigma
            row_offset = end

        if (ki + 1) % 50 == 0 or ki == n_icds - 1:
            print(f"    {ki+1}/{n_icds} ICD codes", file=sys.stderr)

    return pl.DataFrame({
        "age": all_age[:row_offset],
        "is_male": all_sex[:row_offset],
        "sctid": all_sctid,
        "mean_prediction": all_mu[:row_offset],
        "std_prediction": all_sigma[:row_offset],
        "ICD code": all_icd,
    })


def _ensure_cache() -> Path:
    """Ensure cached grids exist. Generate from posteriors if needed.

    Returns the cache directory path.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    mort_grid = _CACHE_DIR / "atlas_mortality_logitnormals.parquet"
    diag_grid = _CACHE_DIR / "atlas_logitnormals.parquet"

    mort_npz = _DATA_DIR / "atlas_mortality_posterior.npz"
    diag_npz = _DATA_DIR / "atlas_diagnosis_posterior.npz"

    if mort_grid.exists() and diag_grid.exists():
        return _CACHE_DIR

    if not mort_npz.exists():
        raise FileNotFoundError(
            f"Posterior data not found: {mort_npz}\n"
            f"Place atlas_mortality_posterior.npz in {_DATA_DIR}/"
        )

    print("╭─ ED Atlas ─────────────────────────────────────╮", file=sys.stderr)
    print("│ Populating prediction cache from posteriors...  │", file=sys.stderr)
    print("│ This only happens once.                         │", file=sys.stderr)
    print("╰─────────────────────────────────────────────────╯", file=sys.stderr)

    if not mort_grid.exists():
        print("  Generating mortality grid...", file=sys.stderr)
        post = ParameterPosterior.load(mort_npz)
        df = _generate_mortality_grid(post)
        df.write_parquet(mort_grid)
        print(f"  Saved {mort_grid} ({df.shape[0]} rows)", file=sys.stderr)

    if not diag_grid.exists() and diag_npz.exists():
        print("  Generating diagnosis grid...", file=sys.stderr)
        df = _generate_diagnosis_grid(diag_npz)
        df.write_parquet(diag_grid)
        print(f"  Saved {diag_grid} ({df.shape[0]} rows)", file=sys.stderr)

    print("  Cache ready.", file=sys.stderr)
    return _CACHE_DIR


class EdAtlas:
    """ED Atlas: Bayesian logistic regression priors for ED triage.

    Provides precomputed LogitNormal predictive distributions for patients
    presenting to the emergency department, conditioned on age, sex, and
    presenting complaint (SNOMED CT concept ID).

    Two outcome types:
      - Diagnosis (ICD-10 codes within 30 days)
      - Death (30-day mortality)

    Three levels of access:
      1. predict() / predict_death() — posterior predictive mean (fast lookup)
      2. predict_distribution() / predict_death_distribution() — LogitNormal params
      3. mortality_posterior / diagnosis_posterior() — full parameter posterior

    When a patient has two complaints, predictions mix in probability space:
        P = 0.5 * sigmoid(mu_1) + 0.5 * sigmoid(mu_2)
    This is the correct E[P] under complaint uncertainty.
    """

    def __init__(self, resources_dir: Path | None = None) -> None:
        res = resources_dir or _ensure_cache()

        # --- Diagnosis grid ---
        diag_path = res / "atlas_logitnormals.parquet"
        if not diag_path.exists():
            raise FileNotFoundError(f"Diagnosis grid not found: {diag_path}")

        diag_df = pl.read_parquet(diag_path)
        self._icd_codes = sorted(diag_df["ICD code"].unique().to_list())
        self._diag_sctids = sorted(diag_df["sctid"].unique().to_list())
        n_sctids_d = len(self._diag_sctids)
        n_icds = len(self._icd_codes)

        # Build 4D lookup: (sctid_idx, age_idx, sex, icd_idx)
        sctid_to_idx_d = {s: i for i, s in enumerate(self._diag_sctids)}
        icd_to_idx = {c: i for i, c in enumerate(self._icd_codes)}

        self._diag_mus = np.full((n_sctids_d, N_AGES, 2, n_icds), np.nan, dtype=np.float32)
        self._diag_sigmas = np.full((n_sctids_d, N_AGES, 2, n_icds), np.nan, dtype=np.float32)

        # Vectorized fill
        si = diag_df["sctid"].replace_strict(sctid_to_idx_d, return_dtype=pl.Int64).to_numpy()
        ai = (diag_df["age"].to_numpy() - AGE_MIN)
        xi = diag_df["is_male"].to_numpy()
        ii = diag_df["ICD code"].replace_strict(icd_to_idx, return_dtype=pl.Int64).to_numpy()
        self._diag_mus[si, ai, xi, ii] = diag_df["mean_prediction"].to_numpy()
        self._diag_sigmas[si, ai, xi, ii] = diag_df["std_prediction"].to_numpy()

        assert not np.any(np.isnan(self._diag_mus)), "Missing cells in diagnosis grid"

        self._diag_sctid_to_idx = sctid_to_idx_d
        self._icd_to_idx = icd_to_idx

        # --- Mortality grid ---
        mort_path = res / "atlas_mortality_logitnormals.parquet"
        if not mort_path.exists():
            raise FileNotFoundError(f"Mortality grid not found: {mort_path}")

        mort_df = pl.read_parquet(mort_path)
        self._mort_sctids = sorted(mort_df["sctid"].unique().to_list())
        n_sctids_m = len(self._mort_sctids)
        sctid_to_idx_m = {s: i for i, s in enumerate(self._mort_sctids)}

        self._mort_mus = np.full((n_sctids_m, N_AGES, 2), np.nan, dtype=np.float32)
        self._mort_sigmas = np.full((n_sctids_m, N_AGES, 2), np.nan, dtype=np.float32)

        si_m = mort_df["sctid"].replace_strict(sctid_to_idx_m, return_dtype=pl.Int64).to_numpy()
        ai_m = (mort_df["age"].to_numpy() - AGE_MIN)
        xi_m = mort_df["is_male"].to_numpy()
        self._mort_mus[si_m, ai_m, xi_m] = mort_df["mean_prediction"].to_numpy()
        self._mort_sigmas[si_m, ai_m, xi_m] = mort_df["std_prediction"].to_numpy()

        assert not np.any(np.isnan(self._mort_mus)), "Missing cells in mortality grid"

        self._mort_sctid_to_idx = sctid_to_idx_m

        # --- Posteriors (loaded lazily from shipped data) ---
        self._mort_posterior: ParameterPosterior | None = None
        self._diag_posteriors: dict[str, ParameterPosterior] = {}

    def __repr__(self) -> str:
        return (
            f"EdAtlas(diag_sctids={len(self._diag_sctids)}, "
            f"mort_sctids={len(self._mort_sctids)}, "
            f"icd_codes={len(self._icd_codes)}, "
            f"ages={AGE_MIN}-{AGE_MAX})"
        )

    @property
    def icd_codes(self) -> list[str]:
        return self._icd_codes

    @property
    def diag_sctids(self) -> list[str]:
        return self._diag_sctids

    @property
    def mort_sctids(self) -> list[str]:
        return self._mort_sctids

    # ── Diagnosis prediction ──────────────────────────────────────────

    def _validate_age(self, age: int) -> int:
        if not (AGE_MIN <= age <= AGE_MAX):
            raise ValueError(f"Age {age} outside range [{AGE_MIN}, {AGE_MAX}]")
        return age - AGE_MIN

    def predict(
        self, age: int, is_male: int, sctids: list[str]
    ) -> np.ndarray:
        """Posterior predictive mean P(diagnosis) for each ICD code.

        Args:
            age: Patient age (18-109).
            is_male: 0 or 1.
            sctids: 1 or 2 presenting complaint SCTID strings.

        Returns:
            (n_icds,) float32 array of probabilities.
        """
        ai = self._validate_age(age)
        if len(sctids) == 1:
            si = self._diag_sctid_to_idx[sctids[0]]
            mus = self._diag_mus[si, ai, is_male]
            return _sigmoid(mus)
        elif len(sctids) == 2:
            si0 = self._diag_sctid_to_idx[sctids[0]]
            si1 = self._diag_sctid_to_idx[sctids[1]]
            p0 = _sigmoid(self._diag_mus[si0, ai, is_male])
            p1 = _sigmoid(self._diag_mus[si1, ai, is_male])
            return 0.5 * p0 + 0.5 * p1
        else:
            raise ValueError(f"Expected 1 or 2 sctids, got {len(sctids)}")

    def predict_batch(
        self,
        ages: np.ndarray,
        is_males: np.ndarray,
        sctids_list: list[list[str]],
    ) -> np.ndarray:
        """Batch diagnosis prediction. Returns (N, n_icds) float32."""
        N = len(ages)
        out = np.empty((N, len(self._icd_codes)), dtype=np.float32)
        for i in range(N):
            out[i] = self.predict(int(ages[i]), int(is_males[i]), sctids_list[i])
        return out

    def predict_distribution(
        self, age: int, is_male: int, sctids: list[str]
    ) -> tuple[np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """LogitNormal(mu, sigma) parameters per ICD code.

        For 1 complaint: returns (mu, sigma) each (n_icds,).
        For 2 complaints: returns ((mu1, sigma1), (mu2, sigma2)).
        """
        ai = self._validate_age(age)
        if len(sctids) == 1:
            si = self._diag_sctid_to_idx[sctids[0]]
            return self._diag_mus[si, ai, is_male].copy(), self._diag_sigmas[si, ai, is_male].copy()
        elif len(sctids) == 2:
            si0 = self._diag_sctid_to_idx[sctids[0]]
            si1 = self._diag_sctid_to_idx[sctids[1]]
            return (
                (self._diag_mus[si0, ai, is_male].copy(), self._diag_sigmas[si0, ai, is_male].copy()),
                (self._diag_mus[si1, ai, is_male].copy(), self._diag_sigmas[si1, ai, is_male].copy()),
            )
        else:
            raise ValueError(f"Expected 1 or 2 sctids, got {len(sctids)}")

    # ── Mortality prediction ──────────────────────────────────────────

    def predict_death(
        self, age: int, is_male: int, sctids: list[str]
    ) -> float:
        """Posterior predictive mean P(death within 30 days).

        Args:
            age: Patient age (18-109).
            is_male: 0 or 1.
            sctids: 1 or 2 presenting complaint SCTID strings.

        Returns:
            Probability as float.
        """
        ai = self._validate_age(age)
        if len(sctids) == 1:
            si = self._mort_sctid_to_idx[sctids[0]]
            return float(_sigmoid(self._mort_mus[si, ai, is_male]))
        elif len(sctids) == 2:
            si0 = self._mort_sctid_to_idx[sctids[0]]
            si1 = self._mort_sctid_to_idx[sctids[1]]
            p0 = _sigmoid(self._mort_mus[si0, ai, is_male])
            p1 = _sigmoid(self._mort_mus[si1, ai, is_male])
            return float(0.5 * p0 + 0.5 * p1)
        else:
            raise ValueError(f"Expected 1 or 2 sctids, got {len(sctids)}")

    def predict_death_batch(
        self,
        ages: np.ndarray,
        is_males: np.ndarray,
        sctids_list: list[list[str]],
    ) -> np.ndarray:
        """Batch mortality prediction. Returns (N,) float32."""
        N = len(ages)
        out = np.empty(N, dtype=np.float32)
        for i in range(N):
            out[i] = self.predict_death(int(ages[i]), int(is_males[i]), sctids_list[i])
        return out

    def predict_death_distribution(
        self, age: int, is_male: int, sctids: list[str]
    ) -> tuple[float, float] | tuple[tuple[float, float], tuple[float, float]]:
        """LogitNormal(mu, sigma) for mortality.

        For 1 complaint: returns (mu, sigma) scalars.
        For 2 complaints: returns ((mu1, sigma1), (mu2, sigma2)).
        """
        ai = self._validate_age(age)
        if len(sctids) == 1:
            si = self._mort_sctid_to_idx[sctids[0]]
            return float(self._mort_mus[si, ai, is_male]), float(self._mort_sigmas[si, ai, is_male])
        elif len(sctids) == 2:
            si0 = self._mort_sctid_to_idx[sctids[0]]
            si1 = self._mort_sctid_to_idx[sctids[1]]
            return (
                (float(self._mort_mus[si0, ai, is_male]), float(self._mort_sigmas[si0, ai, is_male])),
                (float(self._mort_mus[si1, ai, is_male]), float(self._mort_sigmas[si1, ai, is_male])),
            )
        else:
            raise ValueError(f"Expected 1 or 2 sctids, got {len(sctids)}")

    # ── Parameter posteriors ──────────────────────────────────────────

    @property
    def mortality_posterior(self) -> ParameterPosterior:
        """Structured exact posterior for the mortality model."""
        if self._mort_posterior is None:
            path = _DATA_DIR / "atlas_mortality_posterior.npz"
            if not path.exists():
                raise FileNotFoundError(
                    f"Mortality posterior not found: {path}. "
                    f"Place atlas_mortality_posterior.npz in {_DATA_DIR}/"
                )
            self._mort_posterior = ParameterPosterior.load(path)
        return self._mort_posterior

    def diagnosis_posterior(self, icd_code: str) -> ParameterPosterior:
        """Structured exact posterior for a diagnosis model.

        Args:
            icd_code: ICD-10 code (e.g. "X80").
        """
        if icd_code not in self._diag_posteriors:
            self._load_diagnosis_posteriors()
            if icd_code not in self._diag_posteriors:
                raise KeyError(
                    f"No posterior for ICD code {icd_code!r}. "
                    f"Available: {sorted(self._diag_posteriors.keys())}"
                )
        return self._diag_posteriors[icd_code]

    def _load_diagnosis_posteriors(self) -> None:
        """Load all diagnosis posteriors from single stacked npz."""
        if self._diag_posteriors:
            return
        path = _DATA_DIR / "atlas_diagnosis_posterior.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Diagnosis posteriors not found: {path}. "
                f"Place atlas_diagnosis_posterior.npz in {_DATA_DIR}/"
            )
        d = np.load(path)
        icd_codes = d["icd_codes"].tolist()
        sctids = d["sctids"].tolist()
        cr_basis = d["cr_basis"]
        for k, icd in enumerate(icd_codes):
            self._diag_posteriors[icd] = ParameterPosterior(
                beta_mean=d["beta_means"][k],
                base_cov=d["base_covs"][k],
                sym_covs=d["sym_covs"][k],
                cross_covs=d["cross_covs"][k],
                sctids=sctids,
                cr_basis=cr_basis,
            )

    def available_posteriors(self) -> dict[str, list[str]]:
        """List available posterior files."""
        result: dict[str, list[str]] = {}
        mort_path = _DATA_DIR / "atlas_mortality_posterior.npz"
        if mort_path.exists():
            result["mortality"] = ["mortality"]
        diag_path = _DATA_DIR / "atlas_diagnosis_posterior.npz"
        if diag_path.exists():
            result["diagnosis"] = np.load(diag_path)["icd_codes"].tolist()
        return result


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))
