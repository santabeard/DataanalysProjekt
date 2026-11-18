#!/usr/bin/env python3
"""SWED-calibrated missingness simulations for the preprocessing experiments.

The built synthetic SWED cohort has a very specific recording structure:
vital signs exist only at three hospitals, and even there roughly 40% of
per-vital measurements are absent.  This module generates data with the same
shape and a KNOWN missingness mechanism so ``run_analysis.py`` can check
whether the selection model recovers it and how the observed distribution
shifts away from the complete one.

Scenarios
---------
structural    hospital availability + random within-hospital recording only
mar_base      recording depends on age/sex within the three vital hospitals
mar_outcome   recording depends on the 30-day outcome
mnar          recording depends on the vital's own value
composite     all of the above

Outputs per scenario: synthetic_ehr.csv, synthetic_ehr_complete.csv, truth.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, ndtr

BASE = ["age", "sex"]
VITALS = [
    # name, centre, spread
    ("sbp", 134.7, 22.0),
    ("pulse", 84.2, 17.1),
    ("rr", 17.1, 4.0),
    ("spo2", 96.3, 2.4),
    ("temp", 36.9, 0.6),
]
NAMES = [v[0] for v in VITALS]
P = len(VITALS)

# Approximate hospital shares in the 50k synthetic subset (12 hospitals).
HOSPITALS = {
    "danderyd": 0.145,
    "karolinska": 0.228,
    "södersjukhuset": 0.177,
    "sahlgrenska": 0.160,
    "st_göran": 0.118,
    "västerås": 0.039,
    "hosp_07": 0.022,
    "hosp_08": 0.022,
    "hosp_09": 0.022,
    "hosp_10": 0.022,
    "hosp_11": 0.022,
    "hosp_12": 0.022,
}
VITAL_HOSPITALS = ["danderyd", "karolinska", "södersjukhuset"]
SELECTION_COLS = ["sel_karolinska", "sel_sodersjukhuset"]

# At a vital hospital, about 41% of visits have any vital recorded; conditional
# on that, the per-vital recording shares are close to the built cohort.
VISIT_OBS = 0.414
WITHIN_GIVEN_VISIT = np.array([1.00, 0.75, 0.89, 0.95, 0.89])

# Standardised effect of (age, sex) on each vital.
BZ = np.array([
    [0.15, -0.10, 0.10, -0.15, -0.05],
    [-0.10, 0.00, 0.00, 0.00, 0.00],
])
BETA_BASE = np.array([0.45, 0.10])
BETA_VITALS = np.array([-0.15, 0.25, 0.35, -0.30, 0.10])
PREVALENCE = 0.015

# Own-value / outcome / base mechanisms per scenario.  Signs are chosen so the
# observed marginal moves in a visible direction when the mechanism is active.
GZ = np.array([
    [-0.35, 0.00],   # sbp: younger patients more likely recorded
    [0.30, 0.00],    # pulse: older patients more likely recorded
    [0.20, -0.15],   # rr
    [-0.25, 0.00],   # spo2
    [0.00, 0.20],    # temp
])
GY = np.array([0.30, 0.80, 0.60, 0.50, 0.40])
GX = np.array([-0.60, 0.50, 0.35, -0.70, 0.40])

SCENARIOS = ("structural", "mar_base", "mar_outcome", "mnar", "composite")


def _corr() -> np.ndarray:
    C = np.full((P, P), 0.15)
    np.fill_diagonal(C, 1.0)
    C[0, 1] = C[1, 0] = -0.15
    C[2, 3] = C[3, 2] = -0.25
    C[0, 4] = C[4, 0] = -0.10
    return C


def _mechanism(scenario: str, j: int) -> tuple[np.ndarray, float, float]:
    gz = np.zeros(len(BASE))
    gy = gx = 0.0
    if scenario in ("mar_base", "composite"):
        gz = GZ[j].copy()
    if scenario in ("mar_outcome", "composite"):
        gy = float(GY[j])
    if scenario in ("mnar", "composite"):
        gx = float(GX[j])
    return gz, gy, gx


def simulate(n: int, scenario: str, seed: int = 20261112):
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(58, 20, n), 18, 109)
    sex = rng.binomial(1, 0.5, n).astype(float)
    base = np.column_stack([age, sex])
    z = (base - base.mean(0)) / base.std(0)

    C = _corr()
    resid_sd = np.sqrt(np.clip(1 - (BZ ** 2).sum(0), 1e-6, None))
    e = rng.standard_normal((n, P)) @ np.linalg.cholesky(C).T * resid_sd
    v = z @ BZ + e
    v = (v - v.mean(0)) / v.std(0)

    lin = z @ BETA_BASE + v @ BETA_VITALS
    alpha = brentq(lambda a: expit(a + lin).mean() - PREVALENCE, -20, 10)
    y = rng.binomial(1, expit(alpha + lin))

    names = list(HOSPITALS)
    probs = np.array([HOSPITALS[h] for h in names], float)
    hospital = rng.choice(names, n, p=probs / probs.sum())
    available = np.isin(hospital, VITAL_HOSPITALS)
    visit_has_vitals = available & (rng.random(n) < VISIT_OBS)

    obs = np.zeros((n, P), dtype=bool)
    spec = {}
    for j, name in enumerate(NAMES):
        gz, gy, gx = _mechanism(scenario, j)
        lj = z @ gz + gy * y + gx * v[:, j]
        target = WITHIN_GIVEN_VISIT[j]
        c = brentq(lambda cc: ndtr(cc + lj[visit_has_vitals]).mean() - target, -12, 12)
        obs[:, j] = visit_has_vitals & (rng.random(n) < ndtr(c + lj))
        spec[name] = dict(
            mechanism=scenario,
            intercept=float(c),
            gz=gz.tolist(),
            gy=float(gy),
            gx=float(gx),
            target_obs_given_visit=float(target),
            visit_observation_rate=float(VISIT_OBS),
            realised_obs_within=float(obs[available, j].mean()),
            realised_missing=float(1 - obs[:, j].mean()),
        )

    natural = np.empty((n, P))
    for j, (_, centre, spread) in enumerate(VITALS):
        natural[:, j] = centre + spread * v[:, j]

    full = pd.DataFrame({"patient_id": np.arange(1, n + 1), "age": age, "sex": sex})
    for j, name in enumerate(NAMES):
        full[name] = natural[:, j]
    full["outcome"] = y
    full["hospital"] = hospital
    full["sel_karolinska"] = (hospital == "karolinska").astype(float)
    full["sel_sodersjukhuset"] = (hospital == "södersjukhuset").astype(float)

    observed = full.copy()
    for j, name in enumerate(NAMES):
        observed.loc[~obs[:, j], name] = np.nan

    truth = dict(
        n=n,
        scenario=scenario,
        seed=seed,
        prevalence=float(y.mean()),
        base=BASE,
        vitals=NAMES,
        vital_centre=[x[1] for x in VITALS],
        vital_spread=[x[2] for x in VITALS],
        vital_is_log=[False] * P,
        beta_base=BETA_BASE.tolist(),
        beta_vitals=BETA_VITALS.tolist(),
        vital_on_base=BZ.tolist(),
        residual_corr=C.tolist(),
        selection_covariates=SELECTION_COLS,
        vital_hospitals=VITAL_HOSPITALS,
        hospital_shares=HOSPITALS,
        missingness=spec,
    )
    return observed, full, truth


def write(out: Path, observed: pd.DataFrame, full: pd.DataFrame, truth: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    observed.to_csv(out / "synthetic_ehr.csv", index=False)
    full.to_csv(out / "synthetic_ehr_complete.csv", index=False)
    (out / "truth.json").write_text(json.dumps(truth, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--scenario", choices=SCENARIOS, default="structural")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).parent / "swed_sim"))
    a = ap.parse_args()

    scenarios = SCENARIOS if a.all else (a.scenario,)
    for scenario in scenarios:
        out = Path(a.out) / scenario
        observed, full, truth = simulate(a.n, scenario, a.seed)
        write(out, observed, full, truth)
        print(f"{scenario:12s} n={a.n:,}  outcome={truth['prevalence']:.3f}  "
              f"obs(any vital)={observed[NAMES].notna().any(axis=1).mean():.3f}  -> {out}")
        for name in NAMES:
            m = truth["missingness"][name]
            print(f"  {name:6s} within-hospital obs {m['realised_obs_within']:.3f}  "
                  f"overall missing {m['realised_missing']:.3f}")


if __name__ == "__main__":
    main()
