#!/usr/bin/env python3
"""
simulate_hospital.py -- multi-diagnosis hospital data with WARD-level measurement practice.

Builds on simulate_multilabel.py (3 base variables, 11 vitals, K diagnoses) and adds two things
that real hospital data have and the earlier simulations did not:

1. WARDS (W of them).  Each ward has its own recording behaviour:
     * an offset on the probit scale for every vital (some wards measure more of everything);
     * for lactate and WBC, its own dependence on sickness load (a ward that measures routinely has a
       small slope; a ward that only measures when patients look ill has a large one).
2. LAB PANELS.  glucose, lactate, creatinine, WBC and sodium are ordered together: a latent
   "panel ordered" decision (more likely for sicker patients, ward-dependent) switches ALL five off.
   Given that a panel was ordered, each lab can still be individually missing (MNAR as before).
   This makes missingness across labs correlated beyond what independent per-vital selection models
   can represent -- the situation where pattern-level effects are meant to help.

Outputs in data_h/: synthetic_hospital.csv (analyst view, with `ward`), synthetic_hospital_complete.csv,
true_risk.npy, truth_h.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, ndtr

import simulate_data as sd

LABS = ["glucose", "lactate", "creatinine", "wbc", "sodium"]


def simulate(n, K, W, seed):
    rng = np.random.default_rng(seed)
    P = sd.P
    age = np.clip(rng.normal(62, 16, n), 18, 95)
    sex = rng.binomial(1, 0.5, n).astype(float)
    cci = np.clip(rng.poisson(2.2, n), 0, 10).astype(float)
    base = np.column_stack([age, sex, cci])
    z = (base - base.mean(0)) / base.std(0)
    C = sd.make_corr()
    resid_sd = np.sqrt(1 - (sd.BZ ** 2).sum(0))
    e = rng.standard_normal((n, P)) @ np.linalg.cholesky(C).T * resid_sd
    v = z @ sd.BZ + e
    v = (v - v.mean(0)) / v.std(0)
    cov = np.column_stack([z, v])

    # ---- diagnoses (as in simulate_multilabel)
    mu = np.concatenate([sd.BETA_Z, sd.BETA_X])
    prev = np.exp(np.linspace(np.log(0.02), np.log(0.25), K)) if K > 1 else np.array([0.18])
    rng.shuffle(prev)
    mask = rng.random((K, len(mu))) < 0.5
    Bk = 0.7 * mu[None, :] + 0.35 * rng.standard_normal((K, len(mu))) * mask
    a = np.empty(K)
    Y = np.empty((n, K), dtype=int)
    Ptrue = np.empty((n, K))
    for k in range(K):
        lin = cov @ Bk[k]
        a[k] = brentq(lambda x: expit(x + lin).mean() - prev[k], -20, 10)
        Ptrue[:, k] = expit(a[k] + lin)
        Y[:, k] = rng.random(n) < Ptrue[:, k]
    cnt = Y.sum(1).astype(float)
    load = (cnt - cnt.mean()) / cnt.std()

    # ---- wards
    sizes = rng.dirichlet(np.full(W, 4.0))
    ward = rng.choice(W, n, p=sizes)
    off0 = rng.normal(0, 0.45, (P, W))                       # ward offsets, every vital (probit scale)
    gs_base = {"lactate": 0.7, "wbc": 0.5} if K > 1 else {"lactate": 0.4, "wbc": 0.3}   # one binary outcome: SD of load is small
    slope = np.zeros((P, W))
    for name, g in gs_base.items():
        slope[sd.NAMES.index(name)] = g * np.exp(rng.normal(0, 0.6, W))
    off_panel = rng.normal(0, 0.4, W)

    # ---- lab panel decision
    lp = 0.8 * load + 0.3 * z[:, 2] + off_panel[ward]
    c_panel = brentq(lambda c: ndtr(c + lp).mean() - 0.93, -10, 10)
    ordered = rng.random(n) < ndtr(c_panel + lp)

    # ---- per-vital recording
    obs = np.ones((n, P), dtype=bool)
    spec_out = {}
    for j, name in enumerate(sd.NAMES):
        s = sd.MISS[name]
        gz = np.array(s.get("gz", [0, 0, 0]), dtype=float)
        gx = s.get("gx", 0.0)
        lj = z @ gz + off0[j, ward] + slope[j, ward] * load + gx * v[:, j]
        gate = ordered if name in LABS else np.ones(n, dtype=bool)
        c = brentq(lambda c: (gate * ndtr(c + lj)).mean() - (1 - s["miss"]), -12, 12)
        obs[:, j] = gate & (rng.random(n) < ndtr(c + lj))
        spec_out[name] = dict(mechanism=s["mech"], gz=gz.tolist(), gx=gx, panel=name in LABS, intercept=c,
                              target_missing=s["miss"], realised_missing=float(1 - obs[:, j].mean()))

    nat = np.empty((n, P))
    for j, (_, centre, spread, is_log) in enumerate(sd.VITALS):
        t = centre + spread * v[:, j]
        nat[:, j] = np.exp(t) if is_log else t
    full = pd.DataFrame(base, columns=sd.BASE)
    for j, name in enumerate(sd.NAMES):
        full[name] = nat[:, j]
    full.insert(0, "ward", ward)
    dx_cols = [f"dx_{k + 1:02d}" for k in range(K)]
    for k, c in enumerate(dx_cols):
        full[c] = Y[:, k]
    full.insert(0, "patient_id", np.arange(1, n + 1))
    observed = full.copy()
    for j, name in enumerate(sd.NAMES):
        observed.loc[~obs[:, j], name] = np.nan
    truth = dict(n=n, K=K, W=W, seed=seed, base=sd.BASE, vitals=sd.NAMES, dx_cols=dx_cols, labs=LABS,
                 vital_centre=[x[1] for x in sd.VITALS], vital_spread=[x[2] for x in sd.VITALS],
                 prevalence_target=prev.tolist(), intercepts=a.tolist(), beta=Bk.tolist(),
                 ward_sizes=sizes.tolist(), ward_offset=off0.tolist(), ward_slope=slope.tolist(),
                 ward_panel_offset=off_panel.tolist(), panel_ordered_rate=float(ordered.mean()),
                 missingness=spec_out)
    return observed, full, truth, Ptrue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--W", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data_h"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    observed, full, truth, Ptrue = simulate(a.n, a.K, a.W, a.seed)
    observed.to_csv(out / "synthetic_hospital.csv", index=False)
    full.to_csv(out / "synthetic_hospital_complete.csv", index=False)
    np.save(out / "true_risk.npy", Ptrue)
    (out / "truth_h.json").write_text(json.dumps(truth, indent=2))
    Y = full[truth["dx_cols"]].to_numpy()
    vit = truth["vitals"]
    labs_all = observed[LABS].isna().all(1).mean()
    print(f"n={a.n} K={a.K} wards={a.W}  dx/patient {Y.sum(1).mean():.2f}  rarest dx events {Y.sum(0).min()}")
    print(f"lab panel not ordered (all 5 labs missing): {labs_all:.1%};  complete cases {observed.dropna().shape[0] / a.n:.1%}")
    print("missing by vital:", {v: f"{truth['missingness'][v]['realised_missing']:.0%}" for v in vit})
    lm = observed.groupby("ward")["lactate"].apply(lambda s: s.isna().mean())
    print(f"lactate missing by ward: min {lm.min():.0%}  max {lm.max():.0%}")
    print("written to", out)


if __name__ == "__main__":
    main()
