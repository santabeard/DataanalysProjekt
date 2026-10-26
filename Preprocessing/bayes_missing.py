#!/usr/bin/env python3
"""
bayes_missing.py -- joint Bayesian logistic regression with missing covariates
and per-variable selection models for (possibly non-ignorable) missingness.

Pure numpy/scipy Gibbs sampler (no PyMC/Stan needed).

Model
-----
Base variables z_i (q, never missing), vitals x_i (p, partly missing), outcome y_i.

 1. Vitals  : x_i | z_i ~ MVN(B' [1, z_i], Sigma)              B ~ matrix-normal, Sigma ~ IW
 2. Outcome : y_i | z_i, x_i ~ Bernoulli(logit^-1(a + z_i b_z + x_i b_x))
              (Polya-gamma augmentation -> all conditionals conjugate)
 3. Missingness (only for vitals you list in `selection`; others are treated as ignorable)
              R_ij = 1[observed],  P(R_ij = 1) = Phi(g0 + z_i g_z + y_i g_y + g_x * x_ij)
              (Albert-Chib probit augmentation)

    * g_x is the NON-IGNORABLE part: missingness depends on the vital's own value.
      selection[name] = None   -> g_x free, prior N(0, tau_gx^2)
      selection[name] = delta  -> g_x FIXED at delta (pattern-mixture-style sensitivity analysis)
    * vitals not in `selection` are assumed MCAR/MAR given (z, y, other observed values):
      the likelihood factorises and no missingness model is needed.

Missing x_ij are latent parameters: each Gibbs sweep draws them from their full
conditional, which combines (i) the MVN vitals model, (ii) the outcome likelihood and
(iii) the selection model.  So imputation uncertainty propagates into every credible
interval, and the outcome informs the imputations (congenial imputation).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.special import log_ndtr, ndtri, ndtri_exp
from scipy.stats import invwishart


# --------------------------------------------------------------------------- samplers
def sample_pg1(c, rng, K=200):
    """Draws from Polya-gamma PG(1, c): truncated series + mean-corrected remainder."""
    c = np.abs(c)
    k = np.arange(1, K + 1) - 0.5
    denom = k[None, :] ** 2 + (c[:, None] ** 2) / (4 * np.pi ** 2)
    g = rng.standard_gamma(1.0, size=denom.shape)
    s = (g / denom).sum(1) / (2 * np.pi ** 2)
    full_mean = np.where(c > 1e-6, np.tanh(c / 2) / (2 * np.maximum(c, 1e-6)), 0.25)
    trunc_mean = (1.0 / denom).sum(1) / (2 * np.pi ** 2)
    return s + (full_mean - trunc_mean)


def draw_latent_probit(mean, observed, rng):
    """w ~ N(mean, 1) truncated to w>0 where observed else w<0 (numerically stable)."""
    u = np.clip(rng.random(mean.shape), 1e-12, 1.0)
    # R=0: w = m + Phi^-1(u Phi(-m)) ; R=1: w = m - Phi^-1(u Phi(m))
    z0 = ndtri_exp(np.log(u) + log_ndtr(-mean))
    z1 = ndtri_exp(np.log(u) + log_ndtr(mean))
    return np.where(observed, mean - z1, mean + z0)


def _chol_sample(prec, rhs, rng):
    """Draw from N(prec^-1 rhs, prec^-1)."""
    L = np.linalg.cholesky(prec)
    mean = linalg.cho_solve((L, True), rhs)
    eps = rng.standard_normal(len(rhs))
    return mean + linalg.solve_triangular(L, eps, lower=True, trans="T")


# --------------------------------------------------------------------------- data prep
def prepare(df, base_cols, vital_cols, outcome_col, log_vars=()):
    """Standardise base variables (fully observed) and vitals (on observed values).

    Returns a dict; coefficients come back per 1 'model unit' and are mapped to
    original units with `x_scale`.
    """
    base = df[base_cols].to_numpy(float)
    z_loc, z_scale = base.mean(0), base.std(0)
    Z = (base - z_loc) / z_scale
    X = df[vital_cols].to_numpy(float).copy()
    for j, name in enumerate(vital_cols):
        if name in log_vars:
            X[:, j] = np.log(X[:, j])
    x_loc, x_scale = np.nanmean(X, 0), np.nanstd(X, 0)
    X = (X - x_loc) / x_scale
    return dict(Z=Z, X=X, y=df[outcome_col].to_numpy(float), z_scale=z_scale,
                x_loc=x_loc, x_scale=x_scale, base_cols=list(base_cols), vital_cols=list(vital_cols))


# --------------------------------------------------------------------------- one chain
def run_chain(Z, X, y, selection, names, n_iter=3000, n_burn=1000, seed=0,
              tau_beta=2.5, tau_g=2.0, tau_gx=1.0, tau_B=np.sqrt(10.0),
              prior_obs_rate=None, tau_g0_informative=0.5, verbose=False):
    """Gibbs sampler for one chain.  `selection`: {vital_name: None | fixed g_x}."""
    rng = np.random.default_rng(seed)
    N, p = X.shape
    q = Z.shape[1]
    obs = ~np.isnan(X)
    miss = ~obs
    Xc = np.where(obs, X, 0.0)                    # start imputations at the (standardised) mean
    kb = 1 + q + p
    kappa = y - 0.5

    # --- vitals model
    Zc = np.column_stack([np.ones(N), Z])
    K0 = np.eye(q + 1) / tau_B ** 2
    V_B = np.linalg.inv(Zc.T @ Zc + K0)
    L_VB = np.linalg.cholesky(V_B)
    B = V_B @ Zc.T @ Xc
    Sigma = np.cov((Xc - Zc @ B).T) + 1e-3 * np.eye(p)
    nu0, Psi0 = p + 2, np.eye(p)

    # --- outcome model
    beta = np.zeros(kb)
    P0 = np.eye(kb) / tau_beta ** 2

    # --- selection models
    sel = [j for j, nm in enumerate(names) if nm in selection]
    M = np.column_stack([np.ones(N), Z, y])        # [1, z, y]
    km = M.shape[1]
    fixed = {j: selection[names[j]] for j in sel}
    gam = {j: np.zeros(km + 1) for j in sel}
    for j in sel:
        if fixed[j] is not None:
            gam[j][km] = fixed[j]
    w = np.zeros((N, p))
    prior_m = {}
    prior_sd = {}
    for j in sel:
        m0 = np.zeros(km + 1)
        sd = np.full(km + 1, tau_g)
        sd[km] = tau_gx
        if prior_obs_rate is not None and names[j] in prior_obs_rate:       # informative prior on the
            m0[0] = ndtri(np.clip(prior_obs_rate[names[j]], 1e-3, 1 - 1e-3))  # observation rate
            sd[0] = tau_g0_informative
        prior_m[j], prior_sd[j] = m0, sd
        w[:, j] = np.where(obs[:, j], 0.5, -0.5)

    n_keep = n_iter - n_burn
    out = dict(
        beta=np.zeros((n_keep, kb)),
        gamma=np.zeros((n_keep, len(sel), km + 1)),
        imp_mean=np.zeros((n_keep, p)),
        Sigma=np.zeros((n_keep, p, p)),
        B=np.zeros((n_keep, q + 1, p)),
        X_post_mean=np.zeros((N, p)),
        sel=sel,
    )
    t0 = time.time()
    for it in range(n_iter):
        # (1) outcome: Polya-gamma + Gaussian update of (a, b_z, b_x)
        D = np.column_stack([np.ones(N), Z, Xc])
        eta = D @ beta
        om = sample_pg1(eta, rng)
        beta = _chol_sample((D * om[:, None]).T @ D + P0, D.T @ kappa, rng)
        eta = D @ beta

        # (2) selection models: latent probit scores, then coefficients
        for j in sel:
            g = gam[j]
            w[:, j] = draw_latent_probit(M @ g[:km] + g[km] * Xc[:, j], obs[:, j], rng)
            if fixed[j] is None:
                G, target = np.column_stack([M, Xc[:, j]]), w[:, j]
                prec0 = np.diag(1 / prior_sd[j] ** 2)
                m0 = prior_m[j]
            else:
                G, target = M, w[:, j] - fixed[j] * Xc[:, j]
                prec0 = np.diag(1 / prior_sd[j][:km] ** 2)
                m0 = prior_m[j][:km]
            draw = _chol_sample(G.T @ G + prec0, G.T @ target + prec0 @ m0, rng)
            gam[j] = np.append(draw, g[km]) if fixed[j] is not None else draw

        # (3) missing vitals: full conditionals (Gaussian)
        mu = Zc @ B
        Om = np.linalg.inv(Sigma)
        R = Xc - mu
        for j in range(p):
            idx = miss[:, j]
            if not idx.any():
                continue
            Rj = R[idx]
            cond_mean = mu[idx, j] - (Rj @ Om[:, j] - Rj[:, j] * Om[j, j]) / Om[j, j]
            prec = np.full(idx.sum(), Om[j, j])
            lin = Om[j, j] * cond_mean
            bj = beta[1 + q + j]
            eta_minus = eta[idx] - bj * Xc[idx, j]
            prec = prec + om[idx] * bj ** 2
            lin = lin + bj * (kappa[idx] - om[idx] * eta_minus)
            if j in gam:
                gx = gam[j][km]
                prec = prec + gx ** 2
                lin = lin + gx * (w[idx, j] - M[idx] @ gam[j][:km])
            var = 1.0 / prec
            xnew = lin * var + np.sqrt(var) * rng.standard_normal(idx.sum())
            eta[idx] += bj * (xnew - Xc[idx, j])
            Xc[idx, j] = xnew
            R[idx, j] = xnew - mu[idx, j]

        # (4) vitals regression and covariance
        Mn = V_B @ Zc.T @ Xc
        E = Xc - Zc @ Mn
        S = Psi0 + E.T @ E + Mn.T @ K0 @ Mn
        Sigma = np.atleast_2d(invwishart.rvs(df=nu0 + N, scale=S, random_state=rng))
        B = Mn + L_VB @ rng.standard_normal((q + 1, p)) @ np.linalg.cholesky(Sigma).T

        if it >= n_burn:
            k = it - n_burn
            out["beta"][k] = beta
            for a, j in enumerate(sel):
                out["gamma"][k, a] = gam[j]
            out["imp_mean"][k] = [Xc[miss[:, j], j].mean() if miss[:, j].any() else np.nan for j in range(p)]
            out["Sigma"][k] = Sigma
            out["B"][k] = B
            out["X_post_mean"] += Xc / n_keep
        if verbose and (it + 1) % 500 == 0:
            print(f"  [seed {seed}] iter {it + 1}/{n_iter}  {time.time() - t0:.0f}s", flush=True)
    return out


def _run(args):
    return run_chain(**args)


def fit(data, selection=None, chains=2, n_iter=3000, n_burn=1000, seed=1, verbose=True, **kw):
    """Run several chains in parallel. Returns dict with arrays shaped (chains, draws, ...)."""
    selection = selection or {}
    jobs = [dict(Z=data["Z"], X=data["X"], y=data["y"], selection=selection, names=data["vital_cols"],
                 n_iter=n_iter, n_burn=n_burn, seed=seed * 1000 + c, verbose=verbose and c == 0, **kw)
            for c in range(chains)]
    if chains == 1:
        res = [_run(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=min(chains, os.cpu_count() or 1)) as ex:
            res = list(ex.map(_run, jobs))
    stack = lambda k: np.stack([r[k] for r in res])
    return dict(beta=stack("beta"), gamma=stack("gamma"), imp_mean=stack("imp_mean"),
                Sigma=stack("Sigma"), B=stack("B"), X_post_mean=np.mean([r["X_post_mean"] for r in res], 0),
                sel=res[0]["sel"], data=data, selection=selection)


# --------------------------------------------------------------------------- diagnostics
def rhat(x):
    """Split R-hat. x: (chains, n)."""
    m, n = x.shape
    h = n // 2
    s = np.concatenate([x[:, :h], x[:, h:2 * h]], 0)
    n_, m_ = s.shape[1], s.shape[0]
    W = s.var(1, ddof=1).mean()
    Bn = s.mean(1).var(ddof=1)
    return float(np.sqrt(((n_ - 1) / n_ * W + Bn) / W)) if W > 0 else np.nan


def ess(x):
    """Effective sample size (Geyer initial positive sequence). x: (chains, n)."""
    m, n = x.shape
    xc = x - x.mean(1, keepdims=True)
    f = np.fft.rfft(xc, 2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n * (n / (n - 1))
    W = acov[:, 0].mean()
    var_plus = (n - 1) / n * W + (x.mean(1).var(ddof=1) if m > 1 else 0.0)
    if var_plus <= 0:
        return np.nan
    rho = 1 - (W - acov.mean(0)) / var_plus
    tau, t = 0.0, 0
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        tau += pair
        t += 2
    tau = max(-1 + 2 * tau, 1e-3)
    return float(m * n / tau)


def summarize(draws, labels, scale=None):
    """draws: (chains, n, k). Optional `scale` (k,) multiplies draws (unit conversion)."""
    d = draws if scale is None else draws * np.asarray(scale)
    flat = d.reshape(-1, d.shape[-1])
    return pd.DataFrame({
        "mean": flat.mean(0), "sd": flat.std(0),
        "q2.5": np.quantile(flat, 0.025, 0), "q97.5": np.quantile(flat, 0.975, 0),
        "rhat": [rhat(draws[:, :, k]) for k in range(d.shape[-1])],
        "ess": [ess(draws[:, :, k]) for k in range(d.shape[-1])],
    }, index=labels)


def coef_labels(res):
    return ["intercept"] + res["data"]["base_cols"] + res["data"]["vital_cols"]


def coef_scale(res):
    """Multiplier converting model-scale slopes to 'per original SD' units (1 for base vars)."""
    q = len(res["data"]["base_cols"])
    return np.concatenate([np.ones(1 + q), res["data"]["x_scale"]])
