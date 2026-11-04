"""Joint hierarchical model (demo_model_update.ipynb §3.6) on the built data (step 06).

Death within 30 days of an ED arrival, from two groups of visits:

    old  visits with no vital signs     logit p = x~ᵀβ              x~ = (1, age, male, complaints)
    new  visits with all chosen vitals  logit p = w~ᵀθ              w~ = (1, age, male, complaints, vitals, vitals²)

    β ~ N(m0, I),   θ | β, τ ~ N(Aβ, diag(τ² for shared coefficients, 1 for vitals)),   τ ~ Half-Normal(10)

θ's shared coefficients are centred on β and τ learns how far apart they are, as in the notebook.
Visits with only some of the chosen vitals are left out of both groups. 20% of the new
patients are held out as a test set, scored with θ (with vitals) and with β (without).

Adaptations from the notebook:
  - Complaints: the TOP_COMPLAINTS most common first SNOMED codes as dummies; all others are the reference.
  - Intercept prior: N(logit(death rate in the old group), 2) instead of N(0, 1), because
    30-day mortality is ~1% (logit ≈ -4.6), which N(0, 1) would pull towards 50%.
  - Vitals: first plausible value per visit, standardised on the new group; linear + square term.

Needs pymc (+ numpyro for speed); check on Bianca with  python -c "import pymc, numpyro".
Writes aggregated CSVs (rules as in 03: >= 10 patients per group, counts rounded to 5)
plus the posterior draws (idata.nc, parameters only, no patient data) under $SWED_OUT.

    python 06_joint_hierarchical.py                          # built_subset, temp/sbp/pulse
    python 06_joint_hierarchical.py --vitals temp,sbp,pulse,spo2,resp_rate --draws 1000
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--built", type=Path, help="output folder of 02_build.py (default $SWED_OUT/built_subset)")
ap.add_argument("--out", type=Path, help="default: $SWED_OUT/joint_hierarchical_<built folder name>")
ap.add_argument("--vitals", default="temp,sbp,pulse", help="comma list from: sbp,pulse,resp_rate,spo2,temp,rls")
ap.add_argument("--top-complaints", type=int, default=8)
ap.add_argument("--draws", type=int, default=667)
ap.add_argument("--tune", type=int, default=1000)
ap.add_argument("--chains", type=int, default=6)
ap.add_argument("--sampler", default="numpyro", choices=["numpyro", "pymc"])
ap.add_argument("--test-share", type=float, default=0.2)
args = ap.parse_args()

# must be set before pymc/jax are imported
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={args.chains}")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("atlas_vitals", HERE / "05_atlas_vitals.py")
av = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(av)  # reuse VITALS, auc(), scores(), log(), round5()

SEED = 42
MIN_CELL = av.MIN_CELL


def load(built: Path) -> pd.DataFrame:
    ed = pd.read_parquet(built / "ed_visits.parquet", columns=[
        "ed_visit_id", "lopnr_swed", "ed_hospital", "ed_arrival_datetime", "ed_age", "is_male",
        "ed_presented_complaint_std"])
    ed = ed[ed["ed_age"].between(18, 109) & ed["is_male"].notna()].copy()
    ed["complaint"] = ed["ed_presented_complaint_std"].str.extract(r"\[(\d+)\]", expand=False).fillna("none")

    deaths = pd.read_parquet(built / "deaths.parquet", columns=["lopnr_swed", "date_death_minimp"])
    ed = ed.merge(deaths, on="lopnr_swed", how="left")
    day = ed["ed_arrival_datetime"].dt.tz_convert("Europe/Stockholm").dt.tz_localize(None).dt.normalize()
    days = (pd.to_datetime(ed["date_death_minimp"]) - day).dt.days
    ed = ed[~(days < 0).to_numpy()].copy()
    ed["dead"] = (days[~(days < 0)] <= 30).fillna(False).astype(int).to_numpy()

    v = pd.read_parquet(built / "vitals.parquet", columns=["ed_visit_id", "datetime_analysis", "concept", "value", "implausible"])
    v = v[v["ed_visit_id"].notna() & v["value"].notna() & ~v["implausible"]]
    first = v.sort_values("datetime_analysis").groupby(["ed_visit_id", "concept"])["value"].first().unstack("concept")
    for name, (concepts, *_rest) in av.VITALS.items():
        have = [c for c in concepts if c in first]
        val = first[have].bfill(axis=1).iloc[:, 0] if have else pd.Series(dtype=float)
        ed[name] = ed["ed_visit_id"].map(val)
    return ed


def main() -> None:
    out_root = av.OUT_ROOT
    built = args.built or out_root / "built_subset"
    out = args.out or out_root / f"joint_hierarchical_{built.name}"
    out.mkdir(parents=True, exist_ok=True)
    vitals = [s.strip() for s in args.vitals.split(",")]
    unknown = set(vitals) - set(av.VITALS)
    if unknown:
        raise SystemExit(f"unknown vitals {sorted(unknown)}; choose from {sorted(av.VITALS)}")
    av.log(f"built: {built}  out: {out}  vitals: {vitals}")

    ed = load(built)
    n_vitals = ed[list(av.VITALS)].notna().sum(axis=1)
    ed["group"] = np.select([n_vitals == 0, ed[vitals].notna().all(axis=1)], ["old", "new"], "partial")

    # hold out a share of the new *patients* as test
    rng = np.random.default_rng(SEED)
    new_pts = ed.loc[ed["group"] == "new", "lopnr_swed"].unique()
    test_pts = set(rng.choice(new_pts, int(len(new_pts) * args.test_share), replace=False))
    is_test = (ed["group"] == "new") & ed["lopnr_swed"].isin(test_pts)
    ed.loc[is_test, "group"] = "test"
    # a patient may have old and new visits; keep test patients out of training entirely
    ed = ed[~((ed["group"] != "test") & ed["lopnr_swed"].isin(test_pts))]

    groups = (ed.groupby("group").agg(visits=("dead", "size"), patients=("lopnr_swed", "nunique"), deaths=("dead", "sum"))
              .reindex(["old", "new", "test", "partial"]).fillna(0).astype(int))
    av.log("groups:\n" + groups.to_string())
    for g in ["old", "new", "test"]:
        if groups.loc[g, "deaths"] < MIN_CELL:
            raise SystemExit(f"group {g!r} has fewer than {MIN_CELL} deaths within 30 days: too few to fit or export")

    # --- features, scaled as in the notebook: base on the old group, vitals on the new group ---
    old, new, test = (ed["group"] == g for g in ["old", "new", "test"])
    top = ed.loc[old | new, "complaint"].value_counts().index.drop("none", errors="ignore")[:args.top_complaints].tolist()
    age_mean, age_sd = ed.loc[old, "ed_age"].mean(), ed.loc[old, "ed_age"].std()
    X = pd.DataFrame({"age": (ed["ed_age"] - age_mean) / age_sd, "male": ed["is_male"].astype(float)}, index=ed.index)
    for c in top:
        X[f"sct_{c}"] = (ed["complaint"] == c).astype(float)
    BASE_COLS = list(X.columns)
    for v in vitals:
        m, s = ed.loc[new, v].mean(), ed.loc[new, v].std()
        z = ((ed[v] - m) / s).fillna(0)
        X[v], X[v + "_sq"] = z, z ** 2
    VITAL_COLS = [c for v in vitals for c in (v, v + "_sq")]
    ALL_COLS = BASE_COLS + VITAL_COLS
    OLD_COLS, NEW_COLS = ["intercept"] + BASE_COLS, ["intercept"] + ALL_COLS
    A = np.vstack([np.eye(len(OLD_COLS)), np.zeros((len(VITAL_COLS), len(OLD_COLS)))])
    shared = A.sum(axis=1)
    m0 = np.zeros(len(OLD_COLS))
    m0[0] = av.logit(np.array(ed.loc[old, "dead"].mean()))  # intercept prior centred on the old death rate
    s0 = np.ones(len(OLD_COLS))
    s0[0] = 2.0

    # --- model (notebook §3.6) ---
    with pm.Model(coords={"old_feature": OLD_COLS, "new_feature": NEW_COLS}) as model:
        beta_old = pm.Normal("beta_old", mu=m0, sigma=s0, dims="old_feature")
        tau = pm.HalfNormal("tau", sigma=10)
        theta = pm.Normal("theta", mu=pm.math.dot(A, beta_old), sigma=tau * shared + (1 - shared), dims="new_feature")
        pm.Bernoulli("dead_old", logit_p=beta_old[0] + pm.math.dot(X.loc[old, BASE_COLS].values, beta_old[1:]),
                     observed=ed.loc[old, "dead"].values)
        pm.Bernoulli("dead_new", logit_p=theta[0] + pm.math.dot(X.loc[new, ALL_COLS].values, theta[1:]),
                     observed=ed.loc[new, "dead"].values)
        av.log(f"sampling: {args.chains} chains x {args.draws} draws ({args.tune} tune), sampler={args.sampler}")
        t0 = time.time()
        idata = pm.sample(args.draws, tune=args.tune, chains=args.chains, nuts_sampler=args.sampler,
                          random_seed=SEED, target_accept=0.99, progressbar=False)
    av.log(f"sampled in {time.time() - t0:.0f}s")
    try:
        idata.to_netcdf(str(out / "idata.nc"))
    except Exception as e:  # needs netCDF4 or h5netcdf; the CSVs below don't
        av.log(f"idata.nc not saved ({e.__class__.__name__}: needs netCDF4 or h5netcdf)")

    # --- posterior summary ---
    post = idata.posterior
    rows = []
    for var, names in [("beta_old", OLD_COLS), ("theta", NEW_COLS), ("tau", ["tau"])]:
        d = post[var].values.reshape(post[var].shape[0] * post[var].shape[1], -1)
        for k, name in enumerate(names):
            x = d[:, k]
            rows.append({"param": var, "term": name, "mean": x.mean(), "sd": x.std(),
                         "q03": np.quantile(x, .03), "q50": np.quantile(x, .5), "q97": np.quantile(x, .97)})
    summ = pd.DataFrame(rows)
    try:
        import arviz as az
        rhat = az.rhat(idata, var_names=["beta_old", "theta", "tau"])
        summ["r_hat"] = np.concatenate([np.atleast_1d(rhat[v].values) for v in ["beta_old", "theta", "tau"]])
    except Exception as e:  # arviz API differs between versions; the summary is still valid without r_hat
        av.log(f"r_hat skipped ({e.__class__.__name__})")
    summ.round(4).to_csv(out / "posterior_summary.csv", index=False)
    divergences = int(idata.sample_stats["diverging"].sum())

    # --- test set: θ (with vitals) vs β (without), posterior-mean risk over draws ---
    th = post["theta"].values.reshape(-1, len(NEW_COLS))
    bo = post["beta_old"].values.reshape(-1, len(OLD_COLS))
    Xt = np.column_stack([np.ones(test.sum()), X.loc[test, ALL_COLS].values])
    Xb = np.column_stack([np.ones(test.sum()), X.loc[test, BASE_COLS].values])
    y = ed.loc[test, "dead"].to_numpy().astype(float)
    preds = {"theta_with_vitals": av.sigmoid(Xt @ th.T).mean(1), "beta_without_vitals": av.sigmoid(Xb @ bo.T).mean(1)}
    sc = [{"model": k, "n": av.round5(len(y)), "persons": av.round5(ed.loc[test, "lopnr_swed"].nunique()), **av.scores(y, p)}
          for k, p in preds.items()]
    pd.DataFrame(sc).to_csv(out / "test_scores.csv", index=False)

    g = groups.reset_index().rename(columns={"index": "group"})
    for c in ["visits", "patients", "deaths"]:
        g[c] = g[c].map(av.round5)
    g.to_csv(out / "groups.csv", index=False)
    pd.DataFrame({"item": ["divergences", "draws", "chains", "vitals", "top_complaints", "built_from"],
                  "value": [divergences, args.draws, args.chains, ",".join(vitals), args.top_complaints, built.name]}
                 ).to_csv(out / "run_info.csv", index=False)

    tau_row = summ[summ["param"] == "tau"].iloc[0]
    av.log(f"tau: mean {tau_row['mean']:.3f} (94% interval {tau_row['q03']:.3f}-{tau_row['q97']:.3f}); divergences: {divergences}")
    av.log("test scores:\n" + pd.DataFrame(sc)[["model", "auc", "brier", "calibration_slope"]].to_string(index=False))
    av.log(f"done. Read every CSV in {out} before export (idata.nc holds parameter draws only).")


if __name__ == "__main__":
    main()
