#!/usr/bin/env python3
"""Step 2: do the update methods in 08_update_atlas.py recover a known truth? Local only, no real data.

Simulated ED visits where we know the answer:

    age, sex, complaint   an ED-like population over the atlas's complaints; a complaint's frequency is
                          set to 1 / (its log-odds variance), since an atlas complaint is uncertain when it
                          is rare (the real frequencies are only in the real data)
    true model            logit P = x . (atlas mean_c + drift_c) + vitals . GAMMA_TRUE
                          drift_c ~ N(0, TAU_TRUE^2 I): the data differs from the atlas by a known tau
    vitals                SBP, pulse, resp. rate, SpO2, temp, RLS. Every patient has them, but they are
                          recorded in only ~30% of visits, at random (as in SWED, ~1 in 4)
    deaths                drawn from the true model with the true vitals, also when they weren't recorded

Checks, printed as tables:

  1. recovery   on the whole update pool, joint_v2's vital coefficients vs GAMMA_TRUE (|z| < 2 for
                most), and the missing flags vs the average effect of the vitals they stand in for
  2. tau        the tau the validation grid picks, vs TAU_TRUE
  3. curve      test log-loss of atlas, recal, new_only, 3.3 (tau = 0), joint_v2 and the true model,
                for update sets of 500 to 50,000 patients: a preview of the learning curve (Q14)

    python synthetic/sim_update_atlas.py                       # run from data_pipeline/, real atlas
    python synthetic/sim_update_atlas.py --atlas <npz> --cov-scale 0.01

Uses the real mortality npz (data/ed_atlas/, from ed_atlas.zip; not in git) when present, else the
5-complaint toy atlas. --cov-scale < 1 shrinks the atlas's uncertainty, to see how the methods behave
with a tighter atlas than the real one.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("update_atlas", HERE.parent / "08_update_atlas.py")
u = importlib.util.module_from_spec(_spec)
sys.modules["update_atlas"] = u  # dataclasses need their module registered
_spec.loader.exec_module(u)

REAL_ATLAS = HERE.parents[1] / "data" / "ed_atlas" / "atlas_mortality_posterior.npz"
TOY_ATLAS = HERE.parents[1] / "bayesianUpdating" / "synth_ed_atlas" / "data" / "atlas_mortality_posterior.npz"
SEED = 7
N_PATIENTS = 120_000           # whole simulated population; 20% become the test set
SIZES = [500, 2_000, 10_000, 50_000]  # update-set sizes (patients) for the curve
REPEATS = 3                    # random update sets per size
TAU_TRUE = 0.2
RECORDED = 0.30                # share of visits with a round of vitals
DROPPED = 0.10                 # within a recorded round, share of single vitals not recorded

# per standard deviation (z) and squared standard deviation (z2); the flags are not in the truth
GAMMA_TRUE = {
    "sbp_z": -0.30, "sbp_z2": 0.10,
    "pulse_z": 0.20, "pulse_z2": 0.10,
    "resp_rate_z": 0.30, "resp_rate_z2": 0.05,
    "spo2_z": -0.30, "spo2_z2": 0.05,
    "temp_z": 0.10, "temp_z2": 0.04,
    "rls_2": 0.70, "rls_3": 1.20, "rls_4plus": 1.80,
}


def complaint_shares(atlas: "u.Atlas") -> np.ndarray:
    """Share of visits per complaint, proportional to 1 / Var(log-odds) for a 70-year-old man."""
    S = len(atlas.sctids)
    x = atlas.design(np.full(S, 70), np.ones(S))
    w = 1 / np.einsum("ni,nij,nj->n", x, atlas.cov, x)
    return w / w.sum()


def simulate(atlas: "u.Atlas", rng, recorded_share: float, dropped: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Visits, true vitals and recorded vitals (NaN where not recorded)."""
    shares = complaint_shares(atlas)
    n_visits = 1 + rng.poisson(2.5, N_PATIENTS)
    pid = np.repeat(np.arange(N_PATIENTS), n_visits)
    n = len(pid)
    age0 = rng.normal(55, 20, N_PATIENTS).clip(18, 100)
    ed = pd.DataFrame({
        "lopnr_swed": pid.astype(float),
        "age": (age0[pid] + rng.integers(0, 3, n)).astype(int).clip(18, 109),
        "male": rng.integers(0, 2, N_PATIENTS)[pid],
        "g": rng.choice(len(shares), n, p=shares),
    })
    a = ed["age"].to_numpy()
    true = pd.DataFrame({
        "sbp": rng.normal(132 + 0.3 * (a - 55), 20).clip(60, 250),
        "pulse": rng.normal(84, 16, n).clip(30, 200),
        "resp_rate": rng.normal(17, 4, n).clip(5, 50),
        "spo2": (98 - np.abs(rng.normal(0, 2.5, n))).clip(70, 100),
        "temp": rng.normal(36.9, 0.6, n).clip(33, 42),
        "rls": rng.choice([1, 2, 3, 5], n, p=[0.90, 0.06, 0.03, 0.01]).astype(float),
    })
    recorded = (rng.random(n) < recorded_share)[:, None] & (rng.random((n, true.shape[1])) >= dropped)
    return ed, true, true.where(recorded)


def report_flags(rec, flags, j, low, high) -> None:
    """Only the sum of the flags is well determined; check it against its bounds."""
    sd_sum = float(np.sqrt(j.S_vv[np.ix_(flags, flags)].sum()))
    lo, hi = sum(low.values()), sum(high.values())
    est = rec.loc[flags, "estimate"].sum()
    print(f"   sum of the 6 flags: {est:.3f} (sd {sd_sum:.3f}); should lie between {lo:.3f} and {hi:.3f}: "
          f"{'yes' if lo - 2 * sd_sum <= est <= hi + 2 * sd_sum else 'NO'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas", type=Path, default=REAL_ATLAS if REAL_ATLAS.exists() else TOY_ATLAS)
    ap.add_argument("--cov-scale", type=float, default=1.0, help="multiply the atlas's covariance by this (< 1: tighter)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--recorded", type=float, default=RECORDED,
                    help="share of visits with vitals; 1 also turns off single dropped vitals (no missing data)")
    ap.add_argument("--no-curve", action="store_true", help="skip check 3, the learning curve")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    atlas = u.Atlas(args.atlas)
    print(f"atlas: {args.atlas} ({len(atlas.sctids)} complaints), covariance x {args.cov_scale:g}")
    atlas.cov *= args.cov_scale
    ed, true, observed = simulate(atlas, rng, args.recorded, 0.0 if args.recorded >= 1 else DROPPED)

    # features exactly as 08 builds them: scale from the update set, observed vitals with flags
    ed["role"] = u.split_patients(ed, args.seed)
    update = ed["role"].isin(["train", "val"]).to_numpy()
    scale = u.vital_scale(observed, update)
    V_obs = u.vital_features(observed, scale)
    V_true = u.vital_features(true, scale)  # nothing missing, so every flag is 0
    gamma = pd.Series(GAMMA_TRUE).reindex(V_obs.columns).fillna(0.0)

    x = atlas.design(ed["age"].to_numpy(), ed["male"].to_numpy())
    g = ed["g"].to_numpy()
    drift = rng.normal(0, TAU_TRUE, atlas.mean.shape)
    p_true = u.sigmoid((x * (atlas.mean + drift)[g]).sum(1) + V_true.to_numpy() @ gamma.to_numpy())
    y = rng.binomial(1, p_true).astype(float)
    s = u.logit(atlas.p_death(x, g))
    data = u.Visits(g, x, s, V_obs.to_numpy(), y)
    print(f"{len(ed):,} visits, {N_PATIENTS:,} patients, death rate {y.mean():.4f}, "
          f"vitals recorded in {1 - V_obs['sbp_missing'].mean():.0%} of visits")

    test = (ed["role"] == "test").to_numpy()
    T = data[test]
    pool = ed.loc[update, "lopnr_swed"].unique()

    def fit_all(patients) -> dict:
        """Every method on one update set (train + val of the given patients)."""
        m = ed["lopnr_swed"].isin(patients).to_numpy() & update
        tr, va = m & (ed["role"] == "train").to_numpy(), m & (ed["role"] == "val").to_numpy()
        U = data[m]
        tau, tau_table = u.pick_tau(data[tr], data[va], atlas, u.TAU_GRID)
        fits = {
            "recal": u.predict_recal(u.fit_recal(U), T),
            "new_only": u.predict_grouped(u.fit_new_only(U, len(atlas.sctids)), T),
            "3.3 (tau 0)": u.predict_grouped(u.fit_joint(U, atlas, 0.0), T),
            "joint_v2": u.predict_grouped(joint := u.fit_joint(U, atlas, tau), T),
        }
        return {"fits": fits, "tau": tau, "tau_table": tau_table, "joint": joint, "visits": int(m.sum())}

    # --- 1 + 2: recovery and tau on the whole update pool ---
    full = fit_all(pool)
    j = full["joint"]
    sd = np.sqrt(np.diag(j.S_vv))
    miss_v = observed.loc[update].isna()
    # A flag stands in for the effect of a vital nobody recorded. Its exact target has no closed form,
    # but it lies between two bounds: the plain average effect E[effect] (low), and log E[exp(effect)],
    # the average on the risk scale when death is rare (high).
    effects = {f"{name}_missing": V_true.loc[update, f"{name}_z"] * gamma[f"{name}_z"]
               + V_true.loc[update, f"{name}_z2"] * gamma[f"{name}_z2"] for name in u.CONTINUOUS}
    effects["rls_missing"] = V_true.loc[update, list(u.RLS_LEVELS)] @ gamma[list(u.RLS_LEVELS)]
    vital_of = {f"{name}_missing": name for name in u.VITALS}
    low = {f: e[miss_v[vital_of[f]].to_numpy()].mean() for f, e in effects.items()}
    high = {f: np.log(np.exp(e[miss_v[vital_of[f]].to_numpy()]).mean()) for f, e in effects.items()}
    rec = pd.DataFrame({"true": gamma, "estimate": j.gamma, "sd": sd}, index=V_obs.columns)
    flags = rec.index.str.endswith("_missing")
    rec.loc[flags, "true"] = np.nan  # not in the true model; see the bounds below
    rec["z"] = (rec["estimate"] - rec["true"]) / rec["sd"]
    print(f"\n1. joint_v2 on the whole update pool ({full['visits']:,} visits): vital coefficients vs the truth")
    print(rec.round(3).to_string())
    ok = rec["z"].abs() < 2
    print(f"   |z| < 2 for {int(ok.sum())} of {int(rec['z'].notna().sum())} vital coefficients")
    # vitals go missing together in rounds, so single flags are nearly collinear: only their sum is well determined
    if not miss_v.to_numpy().any():
        print("   no missing vitals: the flags are all 0 and stay at their prior")
    else:
        report_flags(rec, flags, j, low, high)
    print(f"\n2. tau picked on validation: {full['tau']} (true {TAU_TRUE}); validation log-loss x 1000 per tau:")
    print("   " + ", ".join(f"{t:g}: {1000 * v:.3f}" for t, v in zip(full["tau_table"]["tau"], full["tau_table"]["val_log_loss"])))

    # --- 3: learning-curve preview ---
    if args.no_curve:
        return
    ll_true = u.log_loss(T.y, p_true[test])
    ll_atlas = u.log_loss(T.y, u.sigmoid(T.s))  # s = logit(P_atlas), so this is the atlas's own risk
    rows, taus = [], {}
    for n_pat in SIZES:
        for r in range(REPEATS):
            res = fit_all(rng.choice(pool, n_pat, replace=False))
            rows += [{"patients": n_pat, "method": k, "log_loss": u.log_loss(T.y, p)} for k, p in res["fits"].items()]
            taus.setdefault(n_pat, []).append(f"{res['tau']:g}")
    curve = pd.DataFrame(rows)
    table = (curve.groupby(["patients", "method"])["log_loss"].mean().unstack() - ll_true) * 1000
    table["atlas"] = (ll_atlas - ll_true) * 1000
    table["tau_picked"] = pd.Series({n: ", ".join(t) for n, t in taus.items()})
    print(f"\n3. test log-loss above the true model's, x 1000 (lower is better; mean of {REPEATS} update sets)")
    print(f"   true model's own test log-loss: {ll_true:.4f}")
    print(table[["atlas", "recal", "new_only", "3.3 (tau 0)", "joint_v2", "tau_picked"]].round(2).to_string())


if __name__ == "__main__":
    main()
