#!/usr/bin/env python3
"""Model 3.4 ("new data only", demo_model_update.ipynb section 3.4) fit on a
subset of <N>K_train_clean.parquet (scripts_v2/phase1).

Full Bayesian logistic regression, b ~ N(mu, sigma):
    logit P(died_30d) = intercept + age + sex + top complaints + vitals (linear + squared)
Same maths as 07_compare_models.py's new_only model, applied to train_clean
instead of the v1 built_subset. One row per visit (primary complaint only --
train_clean's secondary-complaint duplicate rows are 3.6/joint-model
machinery that 3.4 has no use for). Each vital is standardised, missing ->
0 with its own flag column, since real vitals (unlike the demo's simulated
ones) are not always recorded. Intercept is centred on the subset's death
rate (logit(rate), sd 2), not N(0, 1) -- real mortality is far from 50%, see
07_compare_models.py's prior().

    SWED_OUT=/home/theon/Desktop/data_std python3 01_fit_model_3_4.py
    python3 01_fit_model_3_4.py --label 500K --n-sample 20000
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

CHAINS = 6
# must be set before pymc/jax are imported
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CHAINS}"

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

VITALS =["systolic_bp", "diastolic_bp", "spo2", "temperature", "pulse", "resp_rate"]
TOP_COMPLAINTS = 8
DRAWS, TUNE = 667, 1000
SAMPLER = "numpyro"  # use "pymc" if numpyro is not installed
SEED = 42


def logit(p):
    return np.log(p / (1 - p))


def find_train_clean(subset_dir: Path, label: str | None) -> Path:
    if label:
        path = subset_dir / f"{label}_train_clean.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        return path
    candidates = sorted(subset_dir.glob("*_train_clean.parquet"))
    if len(candidates) != 1:
        raise SystemExit(f"expected exactly one *_train_clean.parquet in {subset_dir}, found {candidates}")
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset-dir", type=Path, default=Path(__file__).resolve().parent,
                    help="folder holding <N>K_train_clean.parquet (default: this script's own folder)")
    ap.add_argument("--label", default=None, help="the <N>K size label (default: auto-detect)")
    ap.add_argument("--n-sample", type=int, default=20_000, help="visits sampled for the fit")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    path = find_train_clean(args.subset_dir, args.label)
    print(f"reading {path}", flush=True)
    df = pd.read_parquet(path, columns=["ed_visit_id", "ed_age", "is_male", "complaint_sctid",
                                         "complaint_role", "died_30d"] + [f"vital_{v}" for v in VITALS])

    # one row per visit: primary complaint only (secondary rows are 3.5/3.6 machinery, unused here)
    df = df[df["complaint_role"] == "primary"].drop_duplicates("ed_visit_id")
    df = df[df["ed_age"].between(18, 109) & df["is_male"].notna()]

    rng = np.random.default_rng(args.seed)
    if len(df) > args.n_sample:
        df = df.iloc[rng.choice(len(df), args.n_sample, replace=False)]
    print(f"fitting on {len(df):,} visits, {int(df['died_30d'].sum())} deaths", flush=True)
    if df["died_30d"].sum() < 10:
        raise SystemExit("fewer than 10 deaths in the subset, too few to fit -- raise --n-sample")

    # features: age + sex + top complaints (base), each vital linear + squared + missing flag (vitals)
    age_mean, age_sd = df["ed_age"].mean(), df["ed_age"].std()
    top = df["complaint_sctid"].value_counts().index[:TOP_COMPLAINTS]

    X = pd.DataFrame({"age": (df["ed_age"] - age_mean) / age_sd, "male": df["is_male"].astype(int)})
    for c in top:
        X[f"sct_{c}"] = (df["complaint_sctid"] == c).astype(int)
    for v in VITALS:
        col = f"vital_{v}"
        mean, sd = df[col].mean(), df[col].std()
        z = ((df[col] - mean) / sd).fillna(0)
        X[f"{v}_z"] = z
        X[f"{v}_sq"] = z ** 2
        X[f"{v}_missing"] = df[col].isna().astype(int)

    y = df["died_30d"].astype(int).values
    Xv = X.values.astype(float)

    mu = np.r_[logit(y.mean()), np.zeros(Xv.shape[1])]
    sigma = np.r_[2.0, np.ones(Xv.shape[1])]

    print("fitting model_3_4 (new data only)", flush=True)
    with pm.Model():
        b = pm.Normal("b", mu=mu, sigma=sigma, shape=Xv.shape[1] + 1)
        pm.Bernoulli("y", logit_p=b[0] + pm.math.dot(Xv, b[1:]), observed=y)
        idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, nuts_sampler=SAMPLER,
                           random_seed=args.seed, progressbar=False)

    summary = az.summary(idata, var_names=["b"])
    summary.index = ["intercept"] + list(X.columns)

    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "model_3_4_summary.csv")
    idata.to_netcdf(args.out / "model_3_4_idata.nc")
    print(summary.round(3))
    print(f"wrote {args.out / 'model_3_4_summary.csv'} and model_3_4_idata.nc", flush=True)


if __name__ == "__main__":
    main()
