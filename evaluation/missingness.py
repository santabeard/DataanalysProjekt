#!/usr/bin/env python3
"""Missingness diagnostics, imputation strategies and the kappa sensitivity sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.cohort import Cohort, Split, with_raw
from evaluation.models import LaplaceModel, fit_map_logistic, fit_m_ref, sigmoid, vital_features

try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, KNNImputer
    from sklearn.linear_model import BayesianRidge
except Exception:  # pragma: no cover
    IterativeImputer = KNNImputer = BayesianRidge = None


# =============================================================================
# Diagnostics
# =============================================================================

def hospital_vital_diagnostics(cohort: Cohort) -> pd.DataFrame:
    rows = []
    for name, split in cohort.splits.items():
        hospitals = split.visits["ed_hospital"].to_numpy()
        for h in pd.unique(hospitals):
            m = hospitals == h
            for j, vital in enumerate(cohort.vital_names):
                observed = split.R[m, j]
                rows.append(
                    {
                        "split": name,
                        "hospital": h,
                        "vital": vital,
                        "visits": int(m.sum()),
                        "observed": int(observed.sum()),
                        "p_observed": round(float(observed.mean()), 4) if len(observed) else None,
                    }
                )
    return pd.DataFrame(rows)


def coobservation_matrix(cohort: Cohort) -> pd.DataFrame:
    R = np.vstack([s.R for s in cohort.splits.values()])
    n = np.maximum(R.T @ R, 1)
    P = R.T @ R / n
    return pd.DataFrame(P, index=cohort.vital_names, columns=cohort.vital_names).round(4)


# =============================================================================
# Imputation
# =============================================================================

def _filled_cohort(cohort: Cohort, fill: np.ndarray) -> Cohort:
    raws = {
        name: np.where(np.isnan(split.raw), fill, split.raw)
        for name, split in cohort.splits.items()
    }
    return with_raw(cohort, raws, preserve_mask=True)


def mean_cohort(cohort: Cohort) -> Cohort:
    fill = np.nanmean(cohort["new"].raw, axis=0)
    return _filled_cohort(cohort, fill)


def median_cohort(cohort: Cohort) -> Cohort:
    fill = np.nanmedian(cohort["new"].raw, axis=0)
    return _filled_cohort(cohort, fill)


def _transform_cohort(cohort: Cohort, imputer) -> Cohort:
    raws = {name: imputer.transform(split.raw) for name, split in cohort.splits.items()}
    return with_raw(cohort, raws, preserve_mask=True)


def knn_cohort(cohort: Cohort, n_neighbors: int = 5, max_fit_rows: int = 5000) -> Cohort:
    if KNNImputer is None:
        raise ImportError("sklearn is required for KNN imputation")
    rng = np.random.default_rng(42)
    raw_new = cohort["new"].raw
    if len(raw_new) > max_fit_rows:
        raw_new = raw_new[rng.choice(len(raw_new), size=max_fit_rows, replace=False)]
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputer.fit(raw_new)
    return _transform_cohort(cohort, imputer)


def mice_cohorts(cohort: Cohort, m: int = 5, seed: int = 42) -> list[Cohort]:
    if IterativeImputer is None:
        raise ImportError("sklearn is required for MICE imputation")
    out = []
    for i in range(m):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=seed + i,
        )
        imputer.fit(cohort["new"].raw)
        out.append(_transform_cohort(cohort, imputer))
    return out


def imputed_cohorts(
    cohort: Cohort,
    method: str,
    m: int = 5,
    seed: int = 42,
) -> list[Cohort]:
    if method == "mean":
        return [mean_cohort(cohort)]
    if method == "median":
        return [median_cohort(cohort)]
    if method == "knn":
        return [knn_cohort(cohort)]
    if method == "mice":
        return mice_cohorts(cohort, m=m, seed=seed)
    raise ValueError(f"unknown imputation method: {method}")


# =============================================================================
# Missing-mask interaction model
# =============================================================================

def _mask_design(split: Split, ref: LaplaceModel) -> np.ndarray:
    s = ref.linear_predictor(split)
    missing = 1.0 - split.R
    interactions = np.column_stack(
        [missing[:, j : j + 1] * split.X_base for j in range(split.R.shape[1])]
    )
    return np.column_stack([np.ones(len(s)), s, vital_features(split), missing, interactions])


def fit_mask_model(cohort: Cohort, ref: LaplaceModel | None = None) -> LaplaceModel:
    """Strategy B plus the observation mask and ``(1-R) * X_base`` interactions."""
    ref = ref or fit_m_ref(cohort)
    new = cohort["new"]
    Z = _mask_design(new, ref)
    mu = np.zeros(Z.shape[1])
    mu[1] = 1.0
    P = np.eye(Z.shape[1])
    theta, cov = fit_map_logistic(Z, new.y, mu, P)
    return LaplaceModel("Mask", theta, cov, lambda split: (_mask_design(split, ref), None))


# =============================================================================
# Kappa sweep
# =============================================================================

def _hospital_dummies(split: Split, hospitals: list[str]) -> np.ndarray:
    h = split.visits["ed_hospital"].to_numpy()
    return np.column_stack([(h == x).astype(float) for x in hospitals])


def fit_observation_model(cohort: Cohort) -> tuple[LaplaceModel, list[str]]:
    """Logistic model for ``P(all selected vitals observed | X_base, hospital)``."""
    hospitals = sorted(cohort.all_visits["ed_hospital"].dropna().unique().tolist())
    new = cohort["new"]
    y = new.R.all(axis=1).astype(float)
    if y.min() == y.max():
        raise ValueError("observation indicator has only one class; cannot fit kappa model")
    Z = np.column_stack([np.ones(len(y)), new.X_base, _hospital_dummies(new, hospitals)])
    mu = np.zeros(Z.shape[1])
    P = np.eye(Z.shape[1])
    theta, cov = fit_map_logistic(Z, y, mu, P)

    def design(split: Split) -> tuple[np.ndarray, None]:
        return (
            np.column_stack(
                [np.ones(len(split.y)), split.X_base, _hospital_dummies(split, hospitals)]
            ),
            None,
        )

    return LaplaceModel("Observation", theta, cov, design), hospitals


def kappa_shift(p: np.ndarray, kappa: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return sigmoid(np.log(p / (1 - p)) + kappa)


def kappa_sweep(
    cohort: Cohort,
    predictions: dict[str, np.ndarray],
    kappas: np.ndarray,
    n_boot: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Shift each model's logit by kappa and rescore without refitting."""
    from evaluation.metrics import bootstrap_ci

    test = cohort["test"]
    groups = test.visits["complaint"].to_numpy()
    patients = test.visits["lopnr_swed"].to_numpy()
    rows = []
    for name, p in predictions.items():
        for kappa in kappas:
            p_shift = kappa_shift(p, float(kappa))
            if n_boot > 0:
                res = bootstrap_ci(
                    test.y, p_shift, patients, groups, B=n_boot, seed=seed
                )
                row = {"model": name, "kappa": float(kappa), "n_boot": n_boot}
                for metric in ("micro_auc", "oe_ratio", "cal_intercept", "cal_slope", "bss"):
                    row[metric] = res[metric]["value"]
                    row[f"{metric}_lo"] = res[metric]["lo"]
                    row[f"{metric}_hi"] = res[metric]["hi"]
            else:
                from evaluation.metrics import all_metrics

                m = all_metrics(test.y, p_shift, groups)
                row = {"model": name, "kappa": float(kappa), "n_boot": 0}
                row.update({k: m[k] for k in ("micro_auc", "oe_ratio", "cal_intercept", "cal_slope", "bss")})
            rows.append(row)
    return pd.DataFrame(rows)
