#!/usr/bin/env python3
"""Five 30-day mortality models on one old/new/test patient split of the built subset (step 07).

    base      age, sex, complaint; fitted on the old patients           (demo §2.1)
    recal     a + b * base score + vitals; new patients                 (demo §3.2)
    new_only  age, sex, complaint, vitals; new patients                 (demo §3.4)
    joint_v2  old and new models fitted together, linked through tau    (demo §3.6)
    ed_atlas  already trained, predictions only

Patients are split 40/40/20 into old, new and test, with all their visits. The old model
never sees vitals; the new models use a vital when the visit has it and a missing flag when not.
Writes scores.csv (aggregated, counts rounded to 5).

    python 07_compare_models.py
"""

import os
from pathlib import Path

CHAINS = 6
# must be set before pymc/jax are imported
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CHAINS}"

import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(os.environ.get("SWED_OUT", "/home/theon/Desktop/data_std"))
BUILT = ROOT / "built_subset"
ATLAS = ROOT / "out" / "atlas_mortality_posterior.npz"
OUT = ROOT / "compare_models"

VITALS = {"temp": "body_temp", "sbp": "systolic_bp", "pulse": "pulse_rate"}  # name -> concept in vitals.parquet
TOP_COMPLAINTS = 8
SPLIT = {"old": 0.4, "new": 0.4, "test": 0.2}  # share of patients
DRAWS, TUNE = 667, 1000
SAMPLER = "numpyro"  # use "pymc" if numpyro is not installed
SEED = 42


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


# =============================================================================
# 1. Data: one row per visit
# =============================================================================

ed = pd.read_parquet(BUILT / "ed_visits.parquet", columns=["ed_visit_id", "lopnr_swed", "ed_arrival_datetime",
                                                           "ed_age", "is_male", "ed_presented_complaint_std"])
ed = ed[ed["ed_age"].between(18, 109) & ed["is_male"].notna()].copy()  # the atlas covers ages 18-109
ed["age"] = ed["ed_age"].astype(int)
ed["male"] = ed["is_male"].astype(int)
ed["complaint"] = ed["ed_presented_complaint_std"].str.extract(r"\[(\d+)\]", expand=False)  # first SNOMED code

# died within 30 days of arrival
deaths = pd.read_parquet(BUILT / "deaths.parquet", columns=["lopnr_swed", "date_death_minimp"])
death = pd.to_datetime(ed["lopnr_swed"].map(deaths.set_index("lopnr_swed")["date_death_minimp"]))
arrival = ed["ed_arrival_datetime"].dt.tz_convert("Europe/Stockholm").dt.tz_localize(None).dt.normalize()
days = (death - arrival).dt.days
ed = ed[~(days < 0)].copy()  # death date before the visit: a data error
ed["dead"] = days.between(0, 30).astype(int)

# first plausible value of each vital per visit
v = pd.read_parquet(BUILT / "vitals.parquet", columns=["ed_visit_id", "datetime_analysis", "concept", "value", "implausible"])
v["concept"] = v["concept"].replace({"heart_rate": "pulse_rate"})
v = v[v["ed_visit_id"].notna() & v["value"].notna() & ~v["implausible"] & v["concept"].isin(VITALS.values())]
v["ed_visit_id"] = v["ed_visit_id"].astype("int64")
first = v.sort_values("datetime_analysis").groupby(["ed_visit_id", "concept"])["value"].first().unstack()
for name, concept in VITALS.items():
    ed[name] = ed["ed_visit_id"].map(first[concept]) if concept in first else np.nan

# split the patients 40/40/20 into old, new and test; each patient keeps all their visits
rng = np.random.default_rng(SEED)
patients = rng.permutation(ed["lopnr_swed"].unique())
cuts = np.cumsum([int(share * len(patients)) for share in SPLIT.values()])[:-1]
role = {p: name for name, part in zip(SPLIT, np.split(patients, cuts)) for p in part}
ed["role"] = ed["lopnr_swed"].map(role)

old, new, test = (ed[ed["role"] == name] for name in SPLIT)
for name, d in [("old", old), ("new", new), ("test", test)]:
    print(f"{name:5s} {len(d):7,} visits {d['lopnr_swed'].nunique():6,} patients {d['dead'].sum():5} deaths", flush=True)
    if d["dead"].sum() < 10:
        raise SystemExit(f"{name}: fewer than 10 deaths, too few to fit or export")
print("share of new visits with each vital:", new[list(VITALS)].notna().mean().round(3).to_dict(), flush=True)


# =============================================================================
# 2. Features, scaled on training data only
# =============================================================================

top = pd.concat([old, new])["complaint"].value_counts().index[:TOP_COMPLAINTS]
age_mean, age_sd = old["age"].mean(), old["age"].std()
vit_mean, vit_sd = new[list(VITALS)].mean(), new[list(VITALS)].std()


def base_features(d):
    X = pd.DataFrame({"age": (d["age"] - age_mean) / age_sd, "male": d["male"]})
    for c in top:
        X[f"sct_{c}"] = (d["complaint"] == c).astype(int)
    return X.values.astype(float)


def vital_features(d):
    cols = []
    for name in VITALS:
        z = ((d[name] - vit_mean[name]) / vit_sd[name]).fillna(0).values  # missing -> the average value
        missing = d[name].isna().values.astype(float)
        cols += [z, z ** 2, missing]  # linear + square (low and high can both raise risk) + missing flag
    return np.column_stack(cols)


Xb_old, Xb_new, Xb_test = base_features(old), base_features(new), base_features(test)
Xv_new, Xv_test = vital_features(new), vital_features(test)
y_old, y_new, y_test = old["dead"].values, new["dead"].values, test["dead"].values


# =============================================================================
# 3. Models
# =============================================================================

def prior(k, rate):
    """Intercept centred on the death rate (N(0, 1) would pull a ~2% mortality towards 50%), N(0, 1) for the rest."""
    return np.r_[logit(rate), np.zeros(k)], np.r_[2.0, np.ones(k)]


def sample(**kw):
    return pm.sample(DRAWS, tune=TUNE, chains=CHAINS, nuts_sampler=SAMPLER, random_seed=SEED, progressbar=False, **kw)


def fit_logistic(X, y, mu, sigma):
    """Bayesian logistic regression; returns draws x (intercept + one coefficient per column of X)."""
    with pm.Model():
        b = pm.Normal("b", mu=mu, sigma=sigma, shape=X.shape[1] + 1)
        pm.Bernoulli("y", logit_p=b[0] + pm.math.dot(X, b[1:]), observed=y)
        idata = sample()
    return idata.posterior["b"].values.reshape(-1, X.shape[1] + 1)


def predict(draws, X):
    """Posterior mean P(death) per row."""
    return sigmoid(draws[:, :1].T + X @ draws[:, 1:].T).mean(axis=1)


print("fitting base", flush=True)
base = fit_logistic(Xb_old, y_old, *prior(Xb_old.shape[1], y_old.mean()))


def base_score(Xb):
    """The base model as one fixed number per visit: log-odds with the posterior-mean coefficients."""
    return base[:, 0].mean() + Xb @ base[:, 1:].mean(axis=0)


print("fitting recal", flush=True)
k_v = Xv_new.shape[1]
recal = fit_logistic(np.column_stack([base_score(Xb_new), Xv_new]), y_new,
                     mu=np.r_[0.0, 1.0, np.zeros(k_v)], sigma=1.0)  # slope on the base score centred on 1

print("fitting new_only", flush=True)
new_only = fit_logistic(np.hstack([Xb_new, Xv_new]), y_new, *prior(Xb_new.shape[1] + k_v, y_new.mean()))

print("fitting joint_v2", flush=True)
k_b = Xb_old.shape[1]
mu0, s0 = prior(k_b, y_old.mean())
with pm.Model():
    beta = pm.Normal("beta", mu=mu0, sigma=s0, shape=k_b + 1)        # old model: intercept + base features
    tau = pm.HalfNormal("tau", sigma=10)                              # how far the new model may drift from beta
    shared = pm.Normal("shared", mu=beta, sigma=tau, shape=k_b + 1)   # new model: same coefficients, centred on beta
    vit = pm.Normal("vit", mu=0, sigma=1, shape=k_v)                  # new model: vitals
    pm.Bernoulli("y_old", logit_p=beta[0] + pm.math.dot(Xb_old, beta[1:]), observed=y_old)
    pm.Bernoulli("y_new", logit_p=shared[0] + pm.math.dot(Xb_new, shared[1:]) + pm.math.dot(Xv_new, vit), observed=y_new)
    idata = sample(target_accept=0.99)  # tau near 0 makes a funnel; small steps avoid divergences
post = idata.posterior
joint = np.hstack([post["shared"].values.reshape(-1, k_b + 1), post["vit"].values.reshape(-1, k_v)])
print(f"joint_v2: tau mean {float(post['tau'].mean()):.3f}, divergences {int(idata.sample_stats['diverging'].sum())}")


# =============================================================================
# 4. ed_atlas: predictions only (same maths as ParameterPosterior.predict in ed_atlas.zip)
# =============================================================================

a = np.load(ATLAS)
sctids = a["sctids"].tolist()
S = len(sctids)
si = test["complaint"].map({s: i for i, s in enumerate(sctids)}).fillna(sctids.index("nosct:other")).astype(int).values
cr = a["cr_basis"][test["age"].values - 18]                       # age spline
male = test["male"].values[:, None]
x = np.hstack([np.ones((len(test), 1)), male, cr, male * cr])     # intercept, sex, age, sex x age
dims = np.column_stack([10 + si, 10 + S + si] + [10 + 2 * S + 4 * si + k for k in range(4)]
                       + [10 + 6 * S + 4 * si + k for k in range(4)])
p_atlas = sigmoid(x @ a["beta_mean"][:10] + (x * a["beta_mean"][dims]).sum(axis=1))  # first complaint only


# =============================================================================
# 5. Scores on the test visits
# =============================================================================

def auc(y, p):
    """Probability that a random death gets a higher prediction than a random survivor."""
    r = pd.Series(p).rank().values
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


preds = {
    "base": predict(base, Xb_test),
    "recal": predict(recal, np.column_stack([base_score(Xb_test), Xv_test])),
    "new_only": predict(new_only, np.hstack([Xb_test, Xv_test])),
    "joint_v2": predict(joint, np.hstack([Xb_test, Xv_test])),
    "ed_atlas": p_atlas,
}
rows = []
for name, p in preds.items():
    p = np.clip(p, 1e-9, 1 - 1e-9)
    rows.append({"model": name, "auc": auc(y_test, p), "brier": np.mean((p - y_test) ** 2),
                 "log_loss": -np.mean(y_test * np.log(p) + (1 - y_test) * np.log(1 - p)),
                 "mean_predicted": p.mean(), "observed_rate": y_test.mean()})
scores = pd.DataFrame(rows).round(4)
scores["test_visits"] = int(5 * round(len(y_test) / 5))
scores["test_deaths"] = int(5 * round(y_test.sum() / 5))

OUT.mkdir(parents=True, exist_ok=True)
scores.to_csv(OUT / "scores.csv", index=False)
print(scores.to_string(index=False))
print(f"wrote {OUT / 'scores.csv'}")
