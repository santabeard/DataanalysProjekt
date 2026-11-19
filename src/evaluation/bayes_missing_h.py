#!/usr/bin/env python3
"""
bayes_missing_h.py -- multi-label Bayesian model with hierarchical options for missingness.

Builds on bayes_missing_ml.py (K logistic diagnoses with half-Cauchy partial pooling, shared vitals
model, held-out patients scored inside the sampler with latent labels).  Four switches in `cfg`:

  selection  : model the recording of each vital:  R_ij ~ Probit( g0 + z g_z + g_s * s_i + g_x * x_ij )
               s_i = standardised number of diagnoses ("sickness load"); g_x = own-value (MNAR) effect.
               False  -> missingness ignored (MAR).
  ward_sel   : WARD random effects in the selection model, per vital:
                   g0_w ~ N(0, sigma0^2)   (how much this ward measures the vital at all)
                   gs_w ~ N(0, sigmas^2)   (how strongly this ward measures it only when patients are sick)
               sigma0, sigmas ~ half-Cauchy, learned.  Wards with few patients shrink to the global rule.
  pattern    : PATTERN-GROUP random effects in the outcome model:  eta_ik += u_k,group(i),
               u ~ N(0, sigma_u^2), sigma_u ~ half-Cauchy(0, 0.5).  `pgroup` (an integer per patient)
               is computed from the OBSERVED missingness, so it is available at prediction time.
               This is a missing-indicator style add-on (not a coherent joint model on its own: with
               `selection` also on, missingness informs the labels twice).  Judge it empirically.
  gx_shrink  : the own-value effects g_x,j share a learned scale: g_x,j ~ N(0, omega^2),
               omega ~ half-Cauchy(0,1) -> vitals with weak evidence of MNAR shrink to ignorable.

Held-out rows (train=False) never update any parameter.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.special import expit, log_ndtr
from scipy.stats import invwishart

from bayes_missing import _chol_sample, draw_latent_probit, rhat, ess, sample_pg1, prepare  # noqa: F401


def pattern_groups(miss, blocks):
    """Integer group per patient from the observed missingness.
    miss: (N,p) bool; blocks: list of lists of column indices.  For each block the tier is
    0 = nothing missing, 1 = some missing, 2 = all missing; group = base-3 number over blocks."""
    g = np.zeros(miss.shape[0], dtype=int)
    for cols in blocks:
        m = miss[:, cols].sum(1)
        tier = np.where(m == 0, 0, np.where(m == len(cols), 2, 1))
        g = g * 3 + tier
    return g


def _ig(shape, rate, rng):
    return 1.0 / rng.gamma(shape, 1.0 / rate)


def run_chain_h(Z, X, Y, train, ward, pgroup, names, cfg, n_iter=1800, n_burn=700, seed=0,
                tau_g=2.0, tau_gx=1.0, A_slope=1.0, A_pat=0.5, A_ward=0.5, mu_sd=1.0, int_sd=5.0,
                tau_B=np.sqrt(10.0), pg_terms=20, thin_pred=10, verbose=False):
    rng = np.random.default_rng(seed)
    N, p = X.shape
    q = Z.shape[1]
    K = Y.shape[1]
    tr, te = np.where(train)[0], np.where(~train)[0]
    Ntr, Nte = len(tr), len(te)
    obs = ~np.isnan(X)
    miss = ~obs
    Xc = np.where(obs, X, 0.0)

    use_sel, use_ward, use_pat, use_shr = cfg["selection"], cfg["ward_sel"], cfg["pattern"], cfg["gx_shrink"]
    W = int(ward.max()) + 1
    Wd = np.eye(W)[ward]                                           # (N, W) one-hot
    G = int(pgroup.max()) + 1 if use_pat else 0
    Pd = np.eye(G)[pgroup] if use_pat else np.zeros((N, 0))
    pb = q + p                                                     # pooled slopes
    pbt = pb + G                                                   # + pattern block
    kb = 1 + pbt
    Avec = np.concatenate([np.full(pb, A_slope), np.full(G, A_pat)])

    # ---- labels
    Yl = Y.astype(float).copy()
    Yl[te] = (rng.random((Nte, K)) < Y[tr].mean(0)).astype(float)
    cnt_tr = Y[tr].sum(1)
    c0, c1 = cnt_tr.mean(), cnt_tr.std()
    s = (Yl.sum(1) - c0) / c1

    # ---- vitals model (training rows)
    Zc = np.column_stack([np.ones(N), Z])
    K0 = np.eye(q + 1) / tau_B ** 2
    V_B = np.linalg.inv(Zc[tr].T @ Zc[tr] + K0)
    L_VB = np.linalg.cholesky(V_B)
    B = V_B @ Zc[tr].T @ Xc[tr]
    Sigma = np.cov((Xc[tr] - Zc[tr] @ B).T) + 1e-3 * np.eye(p)
    Psi0, nu0 = np.eye(p), p + 2

    # ---- outcome coefficients and pooling hyper-parameters
    Beta = np.zeros((K, kb))
    mu, tau2, nu = np.zeros(pbt), np.ones(pbt), np.ones(pbt)
    pool = cfg.get("pool", True)         # False -> single/independent outcomes: fixed N(0, 2.5^2) slopes
    nu_u = 1.0
    if not pool:
        tau2[:pb] = 2.5 ** 2
        tau2[pb:] = A_pat ** 2

    # ---- selection models
    sel = list(range(p)) if use_sel else []
    km = q + 2                                                     # [1, z, s]
    gam = {j: np.zeros(km + 1) for j in sel}                       # [1, z, s, x]
    d0 = {j: np.zeros(W) for j in sel}
    ds = {j: np.zeros(W) for j in sel}
    v0 = {j: 0.25 for j in sel}                                    # ward variances
    vs = {j: 0.25 for j in sel}
    n0 = {j: 1.0 for j in sel}
    ns = {j: 1.0 for j in sel}
    omega2, nuo = tau_gx ** 2, 1.0
    w = np.zeros((N, p))
    for j in sel:
        w[:, j] = np.where(obs[:, j], 0.5, -0.5)
    Wd_tr = Wd[tr]
    rest = np.zeros((N, p))

    n_keep = n_iter - n_burn
    n_pred = len(range(0, n_keep, thin_pred))
    out = dict(beta=np.zeros((n_keep, K, kb), np.float32), mu=np.zeros((n_keep, pbt)), tau=np.zeros((n_keep, pbt)),
               gamma=np.zeros((n_keep, len(sel), km + 1)), d0=np.zeros((n_keep, len(sel), W), np.float32),
               ds=np.zeros((n_keep, len(sel), W), np.float32), omega=np.zeros(n_keep),
               Q=np.zeros((n_pred, Nte, K), np.float32), sel=sel, c0=c0, c1=c1)
    ip = 0
    t0 = time.time()
    for it in range(n_iter):
        D = np.column_stack([np.ones(N), Z, Xc, Pd])
        Eta = D @ Beta.T
        Om = sample_pg1(Eta.ravel(), rng, K=pg_terms).reshape(N, K)
        kappa = Yl - 0.5

        # (1) diagnosis coefficients, training rows only
        Dt, Omt = D[tr], Om[tr]
        DD = (Dt[:, :, None] * Dt[:, None, :]).reshape(Ntr, kb * kb)
        PREC = (DD.T @ Omt).T.reshape(K, kb, kb)
        RHS = (Dt.T @ kappa[tr]).T
        pp = np.concatenate([[1.0 / int_sd ** 2], 1.0 / tau2])
        pm = np.concatenate([[0.0], mu])
        for k in range(K):
            Beta[k] = _chol_sample(PREC[k] + np.diag(pp), RHS[k] + pp * pm, rng)

        # (2) pooling hyper-parameters (slopes pooled around mu; pattern block centred at 0)
        S = Beta[:, 1:]
        if pool:
            pmu = K / tau2 + 1.0 / mu_sd ** 2
            mu = (S.sum(0) / tau2) / pmu + rng.standard_normal(pbt) / np.sqrt(pmu)
            mu[pb:] = 0.0
            ss = ((S - mu) ** 2).sum(0)
            tau2 = _ig((K + 1) / 2.0, 1.0 / nu + ss / 2.0, rng)
            nu = _ig(1.0, 1.0 / Avec ** 2 + 1.0 / tau2, rng)
        elif G:
            # one outcome: slopes keep their fixed prior; the G pattern-group effects share ONE learned scale
            ss_u = (S[:, pb:] ** 2).sum()
            tu = _ig((K * G + 1) / 2.0, 1.0 / nu_u + ss_u / 2.0, rng)
            nu_u = _ig(1.0, 1.0 / A_pat ** 2 + 1.0 / tu, rng)
            tau2[pb:] = tu

        # (3) selection models
        M = np.column_stack([np.ones(N), Z, s])
        Ws_tr = Wd_tr * s[tr][:, None]
        gx_list = []
        for j in sel:
            g = gam[j]
            eta_w = M @ g[:km] + g[km] * Xc[:, j]
            if use_ward:
                eta_w = eta_w + d0[j][ward] + ds[j][ward] * s
            w[:, j] = draw_latent_probit(eta_w, obs[:, j], rng)
            cols = [M[tr], Xc[tr, j][:, None]]
            prec_d = [np.full(km, 1.0 / tau_g ** 2), [1.0 / omega2]]
            if use_ward:
                cols += [Wd_tr, Ws_tr]
                prec_d += [np.full(W, 1.0 / v0[j]), np.full(W, 1.0 / vs[j])]
            Gt = np.column_stack(cols)
            draw = _chol_sample(Gt.T @ Gt + np.diag(np.concatenate(prec_d)), Gt.T @ w[tr, j], rng)
            gam[j] = draw[:km + 1]
            if use_ward:
                d0[j], ds[j] = draw[km + 1:km + 1 + W], draw[km + 1 + W:]
                v0[j] = _ig((W + 1) / 2.0, 1.0 / n0[j] + (d0[j] ** 2).sum() / 2.0, rng)
                n0[j] = _ig(1.0, 1.0 / A_ward ** 2 + 1.0 / v0[j], rng)
                vs[j] = _ig((W + 1) / 2.0, 1.0 / ns[j] + (ds[j] ** 2).sum() / 2.0, rng)
                ns[j] = _ig(1.0, 1.0 / A_ward ** 2 + 1.0 / vs[j], rng)
            rest[:, j] = M @ gam[j][:km] + ((d0[j][ward] + ds[j][ward] * s) if use_ward else 0.0)
            gx_list.append(gam[j][km])
        if use_sel and use_shr:
            gx2 = np.sum(np.square(gx_list))
            omega2 = _ig((len(sel) + 1) / 2.0, 1.0 / nuo + gx2 / 2.0, rng)
            nuo = _ig(1.0, 1.0 / 1.0 ** 2 + 1.0 / omega2, rng)

        # (4) missing vitals, all rows
        Eta = D @ Beta.T
        mu_x = Zc @ B
        Omg = np.linalg.inv(Sigma)
        R = Xc - mu_x
        for j in range(p):
            idx = miss[:, j]
            if not idx.any():
                continue
            Rj = R[idx]
            cm = mu_x[idx, j] - (Rj @ Omg[:, j] - Rj[:, j] * Omg[j, j]) / Omg[j, j]
            bj = Beta[:, 1 + q + j]
            Oi, Ei = Om[idx], Eta[idx] - Xc[idx, j][:, None] * bj[None, :]
            prec = Omg[j, j] + Oi @ (bj ** 2)
            lin = Omg[j, j] * cm + (kappa[idx] - Oi * Ei) @ bj
            if j in gam:
                gx = gam[j][km]
                prec = prec + gx ** 2
                lin = lin + gx * (w[idx, j] - rest[idx, j])
            var = 1.0 / prec
            xnew = lin * var + np.sqrt(var) * rng.standard_normal(idx.sum())
            Eta[idx] += (xnew - Xc[idx, j])[:, None] * bj[None, :]
            Xc[idx, j] = xnew
            R[idx, j] = xnew - mu_x[idx, j]

        # (5) held-out labels
        Q = np.empty((Nte, K))
        if Nte:
            lp = Eta[te]
            cnt = Yl[te].sum(1)
            wte = ward[te]
            base_ll = {}
            for j in sel:
                g = gam[j]
                m0 = np.column_stack([np.ones(Nte), Z[te]]) @ g[:q + 1] + g[km] * Xc[te, j]
                gs = g[q + 1] + 0.0
                if use_ward:
                    m0 = m0 + d0[j][wte]
                    gs = g[q + 1] + ds[j][wte]
                base_ll[j] = (m0, np.where(obs[te, j], 1.0, -1.0), gs)
            for k in rng.permutation(K):
                c_wo = cnt - Yl[te, k]
                dl = np.zeros(Nte)
                for j in sel:
                    m0, sg, gs = base_ll[j]
                    dl += log_ndtr(sg * (m0 + gs * (c_wo + 1 - c0) / c1)) - log_ndtr(sg * (m0 + gs * (c_wo - c0) / c1))
                qk = expit(lp[:, k] + dl)
                ynew = rng.random(Nte) < qk
                Yl[te, k] = ynew
                cnt = c_wo + ynew
                Q[:, k] = qk
            s = (Yl.sum(1) - c0) / c1

        # (6) vitals regression and covariance
        Mn = V_B @ Zc[tr].T @ Xc[tr]
        E = Xc[tr] - Zc[tr] @ Mn
        Sn = Psi0 + E.T @ E + Mn.T @ K0 @ Mn
        Sigma = np.atleast_2d(invwishart.rvs(df=nu0 + Ntr, scale=Sn, random_state=rng))
        B = Mn + L_VB @ rng.standard_normal((q + 1, p)) @ np.linalg.cholesky(Sigma).T

        if it >= n_burn:
            k_ = it - n_burn
            out["beta"][k_] = Beta
            out["mu"][k_], out["tau"][k_] = mu, np.sqrt(tau2)
            out["omega"][k_] = np.sqrt(omega2)
            for a_, j in enumerate(sel):
                out["gamma"][k_, a_] = gam[j]
                out["d0"][k_, a_], out["ds"][k_, a_] = d0[j], ds[j]
            if k_ % thin_pred == 0 and Nte:
                out["Q"][ip] = Q
                ip += 1
        if verbose and (it + 1) % 200 == 0:
            print(f"  [seed {seed}] iter {it + 1}/{n_iter}  {time.time() - t0:.0f}s", flush=True)
    return out


def _run(args):
    return run_chain_h(**args)


def fit_h(data, Y, train, ward, pgroup, cfg, chains=2, n_iter=1800, n_burn=700, seed=1, verbose=True, **kw):
    jobs = [dict(Z=data["Z"], X=data["X"], Y=Y, train=train, ward=ward, pgroup=pgroup, names=data["vital_cols"],
                 cfg=cfg, n_iter=n_iter, n_burn=n_burn, seed=seed * 1000 + c, verbose=verbose and c == 0, **kw)
            for c in range(chains)]
    if chains == 1:
        res = [_run(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=min(chains, os.cpu_count() or 1)) as ex:
            res = list(ex.map(_run, jobs))
    st = lambda k: np.stack([r[k] for r in res])
    return dict(beta=st("beta"), mu=st("mu"), tau=st("tau"), omega=st("omega"), gamma=st("gamma"), d0=st("d0"),
                ds=st("ds"), Q=np.concatenate([r["Q"] for r in res], 0), sel=res[0]["sel"],
                c0=res[0]["c0"], c1=res[0]["c1"], cfg=cfg)
