#!/usr/bin/env python3
"""
simulate_data.py -- synthetic EHR-style data with a KNOWN ground truth and a
different missingness mechanism for each vital sign.

Structure
---------
* 3 base variables, never missing:  age, sex, comorbidity (count)
* 11 vital / lab parameters, correlated with each other and with the base variables
* binary outcome y (e.g. in-hospital deterioration), logistic in base vars + vitals
* missingness indicator for every vital, generated from a probit selection model

      P(vital j observed) = Phi( c_j + g_z . z_i + g_y * y_i + g_x * v_ij )

  where v_ij is the (standardised) TRUE value of the vital.  Mechanisms:

      g_z = g_y = g_x = 0  -> MCAR
      g_z != 0 only        -> MAR on the base variables
      g_y != 0 (g_x = 0)   -> MAR on the (observed) outcome  ("informative presence")
      g_x != 0             -> MNAR (missingness depends on the value itself)

Outputs (in --out directory)
----------------------------
synthetic_ehr.csv           what the analyst sees (NaN = missing)
synthetic_ehr_complete.csv  same data BEFORE deletion (oracle, for validation only)
truth.json                  true parameters, vital scaling, missingness spec
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, ndtr

BASE = ["age", "sex", "comorbidity"]

# name, centre, spread (on the analysis scale), is_log (stored on natural scale as exp(.))
VITALS = [
    ("heart_rate", 85.0, 15.0, False),
    ("resp_rate", 18.0, 4.0, False),
    ("spo2", 96.0, 2.5, False),
    ("temperature", 37.0, 0.6, False),
    ("sbp", 125.0, 20.0, False),
    ("dbp", 75.0, 12.0, False),
    ("glucose", np.log(6.5), 0.30, True),
    ("lactate", np.log(1.5), 0.45, True),
    ("creatinine", np.log(85.0), 0.30, True),
    ("wbc", np.log(9.0), 0.35, True),
    ("sodium", 139.0, 3.5, False),
]
NAMES = [v[0] for v in VITALS]
P = len(VITALS)

# effect of (age, sex, comorbidity) on each vital, in SD units, columns follow VITALS
BZ = np.array([
    # hr    rr    spo2  temp  sbp   dbp   gluc  lact  creat wbc   sodium
    [-0.15, 0.10, -0.20, -0.10, 0.30, -0.10, 0.10, 0.10, 0.30, 0.00, -0.10],  # age
    [0.00, 0.00, 0.00, 0.00, -0.10, -0.05, 0.00, 0.00, 0.20, 0.00, 0.00],     # sex
    [0.10, 0.20, -0.25, 0.00, 0.10, 0.00, 0.30, 0.20, 0.30, 0.10, -0.15],     # comorbidity
])

# true outcome model (per 1 SD of base variable / vital)
BETA_Z = np.array([0.45, 0.10, 0.35])
BETA_X = np.array([0.25, 0.35, -0.40, 0.10, -0.35, 0.00, 0.20, 0.60, 0.30, 0.20, -0.10])
PREVALENCE = 0.18

# missingness spec (probit scale, per SD of the true value). g_z order = BASE
MISS = {
    "heart_rate": dict(mech="MCAR", miss=0.05),
    "resp_rate": dict(mech="MCAR", miss=0.10),
    "spo2": dict(mech="MAR (base vars)", miss=0.10, gz=[0.3, 0.0, 0.4]),
    "temperature": dict(mech="MAR (base vars)", miss=0.15, gz=[-0.2, 0.3, 0.0]),
    "sbp": dict(mech="MAR (base vars)", miss=0.08, gz=[0.3, 0.0, 0.2]),
    "dbp": dict(mech="MAR (base vars)", miss=0.08, gz=[0.3, 0.0, 0.2]),
    "glucose": dict(mech="MNAR", miss=0.25, gz=[0.0, 0.0, 0.2], gx=0.7),
    "lactate": dict(mech="MNAR + outcome-dependent", miss=0.40, gy=1.0, gx=0.6),
    "creatinine": dict(mech="MNAR", miss=0.20, gz=[0.2, 0.0, 0.3], gx=0.5),
    "wbc": dict(mech="MAR (outcome)", miss=0.15, gy=0.7),
    "sodium": dict(mech="MNAR", miss=0.12, gx=-0.5),
}


def make_corr():
    """Residual correlation between vitals (physiologically plausible blocks)."""
    cardio = {"heart_rate", "resp_rate", "spo2", "temperature", "sbp", "dbp"}
    C = np.eye(P)
    for a in range(P):
        for b in range(a + 1, P):
            same = (NAMES[a] in cardio) == (NAMES[b] in cardio)
            C[a, b] = C[b, a] = 0.15 if same else 0.05
    over = {
        ("sbp", "dbp"): 0.65, ("heart_rate", "resp_rate"): 0.40, ("heart_rate", "temperature"): 0.35,
        ("resp_rate", "spo2"): -0.40, ("heart_rate", "sbp"): -0.20, ("lactate", "glucose"): 0.35,
        ("lactate", "creatinine"): 0.30, ("creatinine", "wbc"): 0.15, ("creatinine", "sodium"): -0.10,
        ("lactate", "heart_rate"): 0.30, ("lactate", "sbp"): -0.30, ("lactate", "resp_rate"): 0.30,
        ("wbc", "temperature"): 0.30, ("wbc", "heart_rate"): 0.20,
    }
    for (a, b), r in over.items():
        i, j = NAMES.index(a), NAMES.index(b)
        C[i, j] = C[j, i] = r
    lo = np.linalg.eigvalsh(C).min()
    if lo < 0.05:  # make safely positive definite, keep unit diagonal
        C = (C + (0.05 - lo) * np.eye(P)) / (1 + 0.05 - lo)
    return C


def simulate(n, seed):
    rng = np.random.default_rng(seed)

    # base variables
    age = np.clip(rng.normal(62, 16, n), 18, 95)
    sex = rng.binomial(1, 0.5, n).astype(float)
    cci = np.clip(rng.poisson(2.2, n), 0, 10).astype(float)
    base = np.column_stack([age, sex, cci])
    z = (base - base.mean(0)) / base.std(0)

    # vitals: v = z BZ + e, standardised exactly so that the truth is "per 1 SD"
    C = make_corr()
    resid_sd = np.sqrt(1 - (BZ ** 2).sum(0))
    e = rng.standard_normal((n, P)) @ np.linalg.cholesky(C).T * resid_sd
    v = z @ BZ + e
    v = (v - v.mean(0)) / v.std(0)

    # outcome
    lin = z @ BETA_Z + v @ BETA_X
    alpha = brentq(lambda a: expit(a + lin).mean() - PREVALENCE, -15, 15)
    y = rng.binomial(1, expit(alpha + lin))

    # missingness
    obs = np.ones((n, P), dtype=bool)
    spec_out = {}
    for j, name in enumerate(NAMES):
        s = MISS[name]
        gz = np.array(s.get("gz", [0, 0, 0]), dtype=float)
        gy, gx = s.get("gy", 0.0), s.get("gx", 0.0)
        lj = z @ gz + gy * y + gx * v[:, j]
        c = brentq(lambda c: ndtr(c + lj).mean() - (1 - s["miss"]), -10, 10)
        obs[:, j] = rng.random(n) < ndtr(c + lj)
        spec_out[name] = dict(mechanism=s["mech"], intercept=c, gz=gz.tolist(), gy=gy, gx=gx,
                              target_missing=s["miss"], realised_missing=float(1 - obs[:, j].mean()))

    # natural units
    nat = np.empty((n, P))
    for j, (_, centre, spread, is_log) in enumerate(VITALS):
        t = centre + spread * v[:, j]
        nat[:, j] = np.exp(t) if is_log else t

    full = pd.DataFrame(base, columns=BASE)
    full[BASE[0]] = full[BASE[0]].round(0)
    for j, name in enumerate(NAMES):
        full[name] = nat[:, j]
    full["outcome"] = y
    full.insert(0, "patient_id", np.arange(1, n + 1))
    observed = full.copy()
    for j, name in enumerate(NAMES):
        observed.loc[~obs[:, j], name] = np.nan

    truth = dict(
        n=n, seed=seed, alpha_standardised=alpha, base=BASE, vitals=NAMES,
        vital_centre=[v_[1] for v_ in VITALS], vital_spread=[v_[2] for v_ in VITALS],
        vital_is_log=[v_[3] for v_ in VITALS],
        beta_base=BETA_Z.tolist(), beta_vitals=BETA_X.tolist(),
        vital_on_base=BZ.tolist(), residual_corr=C.tolist(),
        missingness=spec_out,
    )
    return observed, full, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    observed, full, truth = simulate(a.n, a.seed)
    observed.to_csv(out / "synthetic_ehr.csv", index=False)
    full.to_csv(out / "synthetic_ehr_complete.csv", index=False)
    (out / "truth.json").write_text(json.dumps(truth, indent=2))

    print(f"n={a.n}  event rate={observed.outcome.mean():.3f}")
    print(f"complete cases (all 11 vitals observed): {observed.dropna().shape[0]} ({observed.dropna().shape[0] / a.n:.1%})\n")
    print(f"{'vital':12s} {'missing':>8s}  mechanism")
    for name in NAMES:
        m = truth["missingness"][name]
        print(f"{name:12s} {m['realised_missing']:8.1%}  {m['mechanism']}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
