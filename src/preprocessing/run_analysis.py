#!/usr/bin/env python3
"""
run_analysis.py -- fit the joint Bayesian missing-data model to the synthetic EHR data,
compare it with standard approaches, check convergence, and run a sensitivity analysis.

Methods compared (all coefficients reported per 1 SD of the TRUE vital / base variable,
i.e. the scale the truth was generated on):

  oracle        logistic regression on the complete data (before deletion; unattainable)
  complete-case logistic regression on patients with all 11 vitals observed
  mean-imput.   mean imputation + logistic regression
  MICE          chained equations (20 imputations, Rubin's rules), outcome in imputation model
  Bayes-MAR     joint Bayesian model, missingness ignorable (MAR)
  Bayes-MNAR    joint Bayesian model + selection model for every vital (g_x free)

Then a delta-shift sensitivity analysis for the suspected-MNAR vitals.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

import bayes_missing as bm

warnings.filterwarnings("ignore")

LOG_VARS = ("glucose", "lactate", "creatinine", "wbc")
# Vitals a clinician would flag as plausibly MNAR, with the assumed DIRECTION of the effect:
# +1 = higher values more likely to be recorded, -1 = lower values more likely to be recorded.
SUSPECT = {"glucose": +1, "lactate": +1, "creatinine": +1, "sodium": -1}
SENS_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]      # |g_x| per SD, probit scale


# ------------------------------------------------------------------ frequentist helpers
def logit_fit(D, y, iters=100):
    b = np.zeros(D.shape[1])
    for _ in range(iters):
        pr = expit(D @ b)
        H = (D * (pr * (1 - pr))[:, None]).T @ D + 1e-8 * np.eye(D.shape[1])
        step = np.linalg.solve(H, D.T @ (y - pr))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b, np.linalg.inv(H)


def freq_table(b, cov, to_true):
    est, se = b[1:] * to_true, np.sqrt(np.diag(cov))[1:] * to_true
    return pd.DataFrame({"est": est, "lo": est - 1.96 * se, "hi": est + 1.96 * se})


def design(Z, X):
    return np.column_stack([np.ones(len(Z)), Z, X])


def mice_fit(Z, X, y, to_true, m=20):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge
    A = np.column_stack([Z, X, y])
    bs, vs = [], []
    for k in range(m):
        imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=10, random_state=k)
        Ai = imp.fit_transform(A)
        b, cov = logit_fit(design(Ai[:, :Z.shape[1]], Ai[:, Z.shape[1]:-1]), y)
        bs.append(b)
        vs.append(np.diag(cov))
    bs, vs = np.array(bs), np.array(vs)
    qbar = bs.mean(0)
    T = vs.mean(0) + (1 + 1 / m) * bs.var(0, ddof=1)          # Rubin's rules
    est, se = qbar[1:] * to_true, np.sqrt(T)[1:] * to_true
    return pd.DataFrame({"est": est, "lo": est - 1.96 * se, "hi": est + 1.96 * se})


def bayes_table(res, to_true):
    labels = bm.coef_labels(res)
    s = bm.summarize(res["beta"], labels, scale=np.concatenate([[1.0], to_true]))
    s = s.iloc[1:]
    return s, pd.DataFrame({"est": s["mean"].values, "lo": s["q2.5"].values, "hi": s["q97.5"].values})


def metrics(tab, truth, mnar_idx):
    err = tab["est"].values - truth
    return dict(RMSE_all=float(np.sqrt((err ** 2).mean())),
                RMSE_mnar_vitals=float(np.sqrt((err[mnar_idx] ** 2).mean())),
                coverage=float(((tab["lo"].values <= truth) & (truth <= tab["hi"].values)).mean()),
                mean_width=float((tab["hi"] - tab["lo"]).mean()))


# ------------------------------------------------------------------ plotting
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def _style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def make_recovery_figure(coef, imp, gam, labels, vit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.2), facecolor=SURF,
                             gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    # A: estimation error
    ax = axes[0]
    _style(ax, "A  Error in outcome coefficients (estimate - truth)")
    meth = [("Complete-case", C_BLUE), ("MICE", C_ORANGE), ("Bayes, MAR", C_AQUA), ("Bayes, MNAR selection", C_YELLOW)]
    off = np.linspace(-0.27, 0.27, 4)
    truth = coef["truth"].values
    for (name, col), o in zip(meth, off):
        t = coef[name]
        y = np.arange(len(labels)) + o
        ax.hlines(y, t["lo"] - truth, t["hi"] - truth, color=col, lw=1.6, alpha=0.9)
        ax.plot(t["est"] - truth, y, "o", color=col, ms=6, mec=SURF, mew=1.5, label=name)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("log-odds per 1 SD (0 = truth)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, labelcolor=INK2)
    # B: imputation check
    ax = axes[1]
    _style(ax, "B  Mean of the missing values: imputed - true")
    y = np.arange(len(vit))
    ax.plot(imp["observed_mean"] - imp["true_missing_mean"], y, "o", mfc=SURF, mec=MUTED, mew=1.6, ms=7,
            label="Observed mean (no adjustment)")
    ax.plot(imp["bayes_mar"] - imp["true_missing_mean"], y, "o", color=C_AQUA, ms=7, mec=SURF, mew=1.5, label="Bayes, MAR")
    ax.plot(imp["bayes_mnar"] - imp["true_missing_mean"], y, "o", color=C_YELLOW, ms=7, mec=SURF, mew=1.5,
            label="Bayes, MNAR selection")
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{v}  ({m})" for v, m in zip(vit, imp["mech"])], color=INK2, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("difference, in SDs of the observed values (0 = unbiased)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, labelcolor=INK2)
    # C: g_x
    ax = axes[2]
    _style(ax, "C  Own-value effect on being observed (g_x)")
    ax.hlines(y, gam["lo"], gam["hi"], color=C_YELLOW, lw=1.8)
    ax.plot(gam["est"], y, "o", color=C_YELLOW, ms=7, mec=SURF, mew=1.5, label="Posterior mean, 95% CrI")
    ax.plot(gam["true"], y, "D", color=INK, ms=6, mec=SURF, mew=1.2, label="Truth")
    ax.axvline(0, color=MUTED, lw=1.0, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("probit coefficient per 1 SD of the vital (0 = ignorable)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURF)
    plt.close(fig)


def make_sensitivity_figure(sens, truth_by_vital, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURF)
    _style(ax, "Sensitivity of outcome coefficients to the assumed MNAR strength")
    ax.grid(axis="y", color=GRID, lw=0.8)
    cols = dict(zip(SUSPECT, [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW]))
    for v, c in cols.items():
        est, lo, hi = sens[f"{v}__est"], sens[f"{v}__lo"], sens[f"{v}__hi"]
        ax.fill_between(sens["s"], lo, hi, color=c, alpha=0.12, lw=0)
        ax.plot(sens["s"], est, "-o", color=c, lw=2, ms=6, mec=SURF, mew=1.5, label=v)
        ax.axhline(truth_by_vital[v], color=c, lw=1, ls="--")
        ax.text(sens["s"].iloc[-1] + 0.02, est.iloc[-1], v, color=INK2, fontsize=9, va="center")
    ax.set_xlabel("assumed |g_x| (probit, per SD) -- higher values more likely recorded (sodium: lower)", color=INK2, fontsize=9)
    ax.set_ylabel("log-odds per 1 SD (dashed = truth)", color=INK2, fontsize=9)
    ax.set_xlim(-0.03, 1.2)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK2, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURF)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--data-dir", default=str(here / "data"))
    ap.add_argument("--out-dir", default=str(here / "results"))
    ap.add_argument("--iter", type=int, default=6000)
    ap.add_argument("--burn", type=int, default=2500)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--skip-sensitivity", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(a.data_dir) / "synthetic_ehr.csv")
    full = pd.read_csv(Path(a.data_dir) / "synthetic_ehr_complete.csv")
    truth = json.loads((Path(a.data_dir) / "truth.json").read_text())
    base, vit = truth["base"], truth["vitals"]
    q, p = len(base), len(vit)
    spread = np.array(truth["vital_spread"])
    beta_true = np.array(truth["beta_base"] + truth["beta_vitals"])
    labels = base + vit
    mnar_idx = np.array([q + vit.index(v) for v in ("glucose", "lactate", "creatinine", "sodium")])

    data = bm.prepare(df, base, vit, "outcome", LOG_VARS)
    factor = data["x_scale"] / spread                      # model slope -> per true SD
    to_true = np.concatenate([np.ones(q), factor])
    Z, X, y = data["Z"], data["X"], data["y"]
    coef = {"truth": pd.Series(beta_true)}

    print("== baselines", flush=True)
    dfull = bm.prepare(full, base, vit, "outcome", LOG_VARS)
    b, cov = logit_fit(design(dfull["Z"], dfull["X"]), dfull["y"])
    coef["Oracle (complete data)"] = freq_table(b, cov, np.concatenate([np.ones(q), dfull["x_scale"] / spread]))
    cc = ~np.isnan(X).any(1)
    b, cov = logit_fit(design(Z[cc], X[cc]), y[cc])
    coef["Complete-case"] = freq_table(b, cov, to_true)
    print(f"   complete cases: {cc.sum()} of {len(cc)}", flush=True)
    b, cov = logit_fit(design(Z, np.nan_to_num(X)), y)
    coef["Mean imputation"] = freq_table(b, cov, to_true)
    coef["MICE"] = mice_fit(Z, X, y, to_true)

    print("== Bayesian model, MAR (ignorable)", flush=True)
    t0 = time.time()
    res_mar = bm.fit(data, {}, chains=a.chains, n_iter=a.iter, n_burn=a.burn, seed=1)
    diag_mar, coef["Bayes, MAR"] = bayes_table(res_mar, to_true)
    print(f"   {time.time() - t0:.0f}s", flush=True)

    print("== Bayesian model, selection model for every vital (g_x free)", flush=True)
    t0 = time.time()
    res_sel = bm.fit(data, {v: None for v in vit}, chains=a.chains, n_iter=a.iter, n_burn=a.burn, seed=2)
    diag_sel, coef["Bayes, MNAR selection"] = bayes_table(res_sel, to_true)
    print(f"   {time.time() - t0:.0f}s", flush=True)

    # ---- coefficient table
    wide = pd.DataFrame({"truth": beta_true}, index=labels)
    for k, v in coef.items():
        if k == "truth":
            continue
        v = v.copy()
        v.index = labels
        wide[k] = v["est"]
        wide[k + " lo"] = v["lo"]
        wide[k + " hi"] = v["hi"]
    wide.round(3).to_csv(out / "coefficients.csv")
    est_cols = ["truth", "Oracle (complete data)", "Complete-case", "Mean imputation", "MICE",
                "Bayes, MAR", "Bayes, MNAR selection"]
    print("\nOutcome coefficients (log-odds per 1 SD)")
    print(wide[est_cols].round(2).to_string())

    rows = {}
    for k in est_cols[1:]:
        rows[k] = metrics(coef[k], beta_true, mnar_idx)
    mt = pd.DataFrame(rows).T
    mt.round(3).to_csv(out / "metrics.csv")
    print("\nAccuracy against the truth (single simulated dataset)")
    print(mt.round(3).to_string())

    # ---- convergence
    dg = pd.concat([diag_mar.add_prefix("MAR_"), diag_sel.add_prefix("MNAR_")], axis=1)
    dg.round(3).to_csv(out / "convergence_beta.csv")
    print(f"\nConvergence (outcome coefficients): max R-hat  MAR={diag_mar['rhat'].max():.3f}  "
          f"MNAR={diag_sel['rhat'].max():.3f};  min ESS  MAR={diag_mar['ess'].min():.0f}  MNAR={diag_sel['ess'].min():.0f}")

    # ---- selection-model coefficients
    sel = res_sel["sel"]
    km = q + 2
    g = res_sel["gamma"]                                    # (chains, n, nsel, km+1)
    rows = []
    for ai, j in enumerate(sel):
        name = vit[j]
        m = truth["missingness"][name]
        dx = g[:, :, ai, km] / factor[j]                    # per true SD
        dy = g[:, :, ai, q + 1]
        flat = dx.reshape(-1)
        rows.append(dict(vital=name, mechanism=m["mechanism"], true_gx=m["gx"], gx_mean=flat.mean(),
                         gx_lo=np.quantile(flat, .025), gx_hi=np.quantile(flat, .975),
                         gx_rhat=bm.rhat(dx), true_gy=m["gy"], gy_mean=dy.mean(),
                         gy_lo=np.quantile(dy, .025), gy_hi=np.quantile(dy, .975)))
    gtab = pd.DataFrame(rows)
    gtab.round(3).to_csv(out / "missingness_model.csv", index=False)
    print("\nSelection-model coefficients (probit, per SD of the vital; g_y = effect of the outcome)")
    print(gtab.round(2).to_string(index=False))

    # ---- imputation check
    full_t = full[vit].to_numpy(float).copy()
    for j, nm in enumerate(vit):
        if nm in LOG_VARS:
            full_t[:, j] = np.log(full_t[:, j])
    full_m = (full_t - data["x_loc"]) / data["x_scale"]
    miss = np.isnan(X)
    true_miss_mean = np.array([full_m[miss[:, j], j].mean() for j in range(p)])
    imp = pd.DataFrame({
        "vital": vit, "mech": [truth["missingness"][v]["mechanism"].split(" (")[0].split(" +")[0] for v in vit],
        "observed_mean": 0.0, "true_missing_mean": true_miss_mean,
        "bayes_mar": res_mar["imp_mean"].mean((0, 1)), "bayes_mnar": res_sel["imp_mean"].mean((0, 1)),
    })
    imp.round(3).to_csv(out / "imputation_check.csv", index=False)
    print("\nMean of missing values (observed-data SD units)")
    print(imp.round(2).to_string(index=False))

    make_recovery_figure(
        dict(truth=pd.Series(beta_true), **{k: coef[k] for k in ("Complete-case", "MICE", "Bayes, MAR", "Bayes, MNAR selection")}),
        imp, dict(est=gtab["gx_mean"].values, lo=gtab["gx_lo"].values, hi=gtab["gx_hi"].values, true=gtab["true_gx"].values),
        labels, vit, out / "fig_recovery.png")
    np.savez_compressed(out / "posterior_draws.npz", beta_mar=res_mar["beta"], beta_mnar=res_sel["beta"],
                        gamma_mnar=res_sel["gamma"], labels=np.array(bm.coef_labels(res_sel)))

    # ---- sensitivity analysis
    if not a.skip_sensitivity:
        print("\n== sensitivity analysis (g_x fixed for suspect vitals)", flush=True)
        recs = []
        for s in SENS_GRID:
            t0 = time.time()
            selection = {v: d * s * factor[vit.index(v)] for v, d in SUSPECT.items()}
            r = bm.fit(data, selection, chains=a.chains, n_iter=3000, n_burn=1000, seed=10 + int(s * 100), verbose=False)
            _, tab = bayes_table(r, to_true)
            rec = {"s": s}
            for v in vit:
                rec[f"{v}__est"], rec[f"{v}__lo"], rec[f"{v}__hi"] = tab.loc[labels.index(v), ["est", "lo", "hi"]]
            recs.append(rec)
            print(f"   s={s:.2f}  {time.time() - t0:.0f}s", flush=True)
        sens = pd.DataFrame(recs)
        sens.round(3).to_csv(out / "sensitivity.csv", index=False)
        for v in SUSPECT:
            print(f"   {v:11s} beta by s:", ", ".join(f"{x:.2f}" for x in sens[f"{v}__est"]),
                  f"  (truth {beta_true[labels.index(v)]:.2f})")
        make_sensitivity_figure(sens, {v: beta_true[labels.index(v)] for v in SUSPECT}, out / "fig_sensitivity.png")
    print("\ndone ->", out)


if __name__ == "__main__":
    main()
