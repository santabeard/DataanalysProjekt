#!/usr/bin/env python3
"""MAP/Laplace implementations of the mortality models.

All models share one logistic engine. A model is a design builder plus a
Gaussian prior on the coefficients, so fitting is a single penalised
Newton-Raphson solve and the Laplace covariance is ``H^{-1}`` at the MAP.

Models
------
``M_ref``      baseline on ``D_old``; retains ``(mu_old, Sigma_old)``
``M_base``     baseline refit on ``D_new``
``A``          fixed offset ``s_i`` from ``M_ref``; fit ``gamma`` on vitals
``B``          ``a + b*s_i + v_i @ gamma`` with ``a~N(0,1)``, ``b~N(1,1)``, ``gamma~N(0,I)``
``C``          posterior-as-prior: ``N([mu_old, 0], blkdiag(Sigma_old, I))``
``M_direct``   de novo joint fit on ``D_new`` with ``N(0,I)``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from evaluation.cohort import Cohort, Split


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(np.log(p / (1 - p)))


def _with_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def _safe_inv(H: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(H, hermitian=True)


def fit_map_logistic(
    Z: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    prior_prec: np.ndarray,
    offset: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Penalised Newton-Raphson for a logistic model with a Gaussian prior.

    Maximises ``log p(y|theta) - 0.5 (theta-mu)' P (theta-mu)``. Returns the
    MAP coefficients and the Laplace covariance ``H^{-1}``.
    """
    y = np.asarray(y, dtype=float)
    theta = prior_mean.astype(float).copy()
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)

    for _ in range(max_iter):
        p = sigmoid(off + Z @ theta)
        W = p * (1.0 - p)
        grad = Z.T @ (y - p) - prior_prec @ (theta - prior_mean)
        H = Z.T @ (Z * W[:, None]) + prior_prec
        step = np.linalg.solve(H, grad)
        theta += step
        if np.max(np.abs(step)) < tol:
            break

    p = sigmoid(off + Z @ theta)
    W = p * (1.0 - p)
    H = Z.T @ (Z * W[:, None]) + prior_prec
    return theta, _safe_inv(H)


@dataclass
class LaplaceModel:
    name: str
    theta: np.ndarray
    cov: np.ndarray
    design: Callable[[Split], tuple[np.ndarray, np.ndarray | None]]

    def linear_predictor(self, split: Split) -> np.ndarray:
        Z, off = self.design(split)
        eta = Z @ self.theta
        return eta if off is None else eta + off

    def predict(self, split: Split) -> np.ndarray:
        """Posterior-mode probability for each row."""
        return sigmoid(self.linear_predictor(split))

    def predict_draws(
        self,
        split: Split,
        n_draws: int = 1000,
        seed: int = 42,
    ) -> np.ndarray:
        """Probability draws from the Laplace posterior, shape (n_rows, n_draws)."""
        Z, off = self.design(split)
        rng = np.random.default_rng(seed)
        jitter = 1e-10 * np.eye(len(self.theta))
        L = np.linalg.cholesky(self.cov + jitter)
        draws = self.theta + rng.standard_normal((n_draws, len(self.theta))) @ L.T
        eta = Z @ draws.T
        if off is not None:
            eta = eta + np.asarray(off)[:, None]
        return sigmoid(eta)

    @property
    def n_params(self) -> int:
        return len(self.theta)


def _base_design(split: Split) -> tuple[np.ndarray, None]:
    return _with_intercept(split.X_base), None


def vital_features(split: Split) -> np.ndarray:
    """Vital design ``(z, z**2)`` plus the observation mask (median + mask)."""
    return np.column_stack([split.V, split.R])


def _isotropic(k: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(k), np.eye(k)


def fit_m_ref(cohort: Cohort) -> LaplaceModel:
    old = cohort["old"]
    Z = _with_intercept(old.X_base)
    mu, P = _isotropic(Z.shape[1])
    theta, cov = fit_map_logistic(Z, old.y, mu, P)
    return LaplaceModel("M_ref", theta, cov, _base_design)


def fit_m_base(cohort: Cohort) -> LaplaceModel:
    new = cohort["new"]
    Z = _with_intercept(new.X_base)
    mu, P = _isotropic(Z.shape[1])
    theta, cov = fit_map_logistic(Z, new.y, mu, P)
    return LaplaceModel("M_base", theta, cov, _base_design)


def _ref_offset(ref: LaplaceModel, split: Split) -> np.ndarray:
    return ref.linear_predictor(split)


def fit_strategy_a(cohort: Cohort, ref: LaplaceModel | None = None) -> LaplaceModel:
    ref = ref or fit_m_ref(cohort)
    new = cohort["new"]
    s_new = _ref_offset(ref, new)
    V_new = vital_features(new)
    mu, P = _isotropic(V_new.shape[1])
    theta, cov = fit_map_logistic(V_new, new.y, mu, P, offset=s_new)

    def design(split: Split) -> tuple[np.ndarray, np.ndarray]:
        return vital_features(split), _ref_offset(ref, split)

    return LaplaceModel("A", theta, cov, design)


def fit_strategy_b(cohort: Cohort, ref: LaplaceModel | None = None) -> LaplaceModel:
    ref = ref or fit_m_ref(cohort)
    new = cohort["new"]
    s_new = _ref_offset(ref, new)
    V_new = vital_features(new)
    Z = np.column_stack([np.ones(len(new.y)), s_new, V_new])
    mu = np.r_[0.0, 1.0, np.zeros(V_new.shape[1])]
    P = np.eye(Z.shape[1])
    theta, cov = fit_map_logistic(Z, new.y, mu, P)

    def design(split: Split) -> tuple[np.ndarray, None]:
        s = _ref_offset(ref, split)
        return np.column_stack([np.ones(len(s)), s, vital_features(split)]), None

    return LaplaceModel("B", theta, cov, design)


def fit_strategy_c(cohort: Cohort, ref: LaplaceModel | None = None) -> LaplaceModel:
    ref = ref or fit_m_ref(cohort)
    new = cohort["new"]
    V_new = vital_features(new)
    Z = np.column_stack([np.ones(len(new.y)), new.X_base, V_new])
    k_base = ref.n_params
    k_vit = V_new.shape[1]
    mu = np.r_[ref.theta, np.zeros(k_vit)]
    P = np.zeros((k_base + k_vit, k_base + k_vit))
    P[:k_base, :k_base] = np.linalg.pinv(ref.cov, hermitian=True)
    P[k_base:, k_base:] = np.eye(k_vit)
    theta, cov = fit_map_logistic(Z, new.y, mu, P)

    def design(split: Split) -> tuple[np.ndarray, None]:
        return np.column_stack([np.ones(len(split.y)), split.X_base, vital_features(split)]), None

    return LaplaceModel("C", theta, cov, design)


def fit_m_direct(cohort: Cohort) -> LaplaceModel:
    new = cohort["new"]
    Z = np.column_stack([np.ones(len(new.y)), new.X_base, vital_features(new)])
    mu, P = _isotropic(Z.shape[1])
    theta, cov = fit_map_logistic(Z, new.y, mu, P)

    def design(split: Split) -> tuple[np.ndarray, None]:
        return np.column_stack([np.ones(len(split.y)), split.X_base, vital_features(split)]), None

    return LaplaceModel("M_direct", theta, cov, design)


def fit_all(cohort: Cohort) -> dict[str, LaplaceModel]:
    ref = fit_m_ref(cohort)
    return {
        "M_ref": ref,
        "M_base": fit_m_base(cohort),
        "A": fit_strategy_a(cohort, ref),
        "B": fit_strategy_b(cohort, ref),
        "C": fit_strategy_c(cohort, ref),
        "M_direct": fit_m_direct(cohort),
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from evaluation.cohort import load_cohort

    p = argparse.ArgumentParser(description="Fit all models on the built cohort")
    p.add_argument("--built", type=Path, default=Path("data_pipeline/synthetic/built_synth_subset_50k"))
    args = p.parse_args()

    cohort = load_cohort(args.built)
    models = fit_all(cohort)
    test = cohort["test"]
    for name, model in models.items():
        pred = model.predict(test)
        print(
            f"{name:9s} params={model.n_params:3d} "
            f"mean_pred={pred.mean():.4f} observed={test.y.mean():.4f}"
        )
