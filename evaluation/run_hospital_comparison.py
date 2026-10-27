#!/usr/bin/env python3
"""
run_hospital_comparison.py -- which prediction model + missing-data handling works best?

Every row of the result is one COMBINATION of a prediction model and a way of handling missing values.

Bayesian multi-diagnosis models (missing vitals are imputed JOINTLY inside the model, and held-out
patients are scored by the same sampler)
  B0  ignore missingness            imputation uses the other vitals only
  B1  hospital-wide rule            + one rule for "who gets measured"; "not measured" is a clue about health
  B2  per-ward rule                 + a separate rule per ward (shrunk toward the hospital-wide one)
  B3  missing-flags                 + effects for "which block of measurements is missing" (no measurement rule)
  B4  per-ward rule + flags         B2 + B3
  B5  per-ward rule, shrunk         B2 + shared shrinkage of the "own value" effects (vitals without evidence -> ignorable)
Standard (non-Bayesian) benchmarks, one model per diagnosis
  L1-L4  ridge logistic regression  with mean imputation / mean imputation + missing flags / MICE / MICE + flags
  G1-G2  gradient boosting          with native missing-value handling / MICE imputation

Evaluation: 1,500 held-out patients are scored three times -- as recorded, and after randomly deleting
25% / 50% of their recorded vitals (a ward changing its measuring habits after deployment).
Paired bootstrap over patients decides which models are statistically tied with the best.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression

import bayes_missing as bm
import bayes_missing_h as bh

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
LOG_VARS = ("glucose", "lactate", "creatinine", "wbc")
LEVELS = [0.0, 0.25, 0.5]
SIT = ["As recorded", "25% more deleted", "50% more deleted"]

def _b(cfg): return dict(selection=cfg[0], ward_sel=cfg[1], pattern=cfg[2], gx_shrink=cfg[3])
MODELS = {   # key: (family, short name, prediction model, missing-data handling, bayes cfg)
    "B0": ("Bayesian", "Bayes · ignore missingness", "Bayesian logistic regression", "Joint imputation; missingness ignored", _b((0, 0, 0, 0))),
    "B1": ("Bayesian", "Bayes · hospital-wide rule", "Bayesian logistic regression", "Joint imputation + one hospital-wide measurement rule", _b((1, 0, 0, 0))),
    "B2": ("Bayesian", "Bayes · per-ward rule", "Bayesian logistic regression", "Joint imputation + separate measurement rule per ward", _b((1, 1, 0, 0))),
    "B3": ("Bayesian", "Bayes · missing-flags", "Bayesian logistic regression", "Joint imputation + flags for which measurements are missing", _b((0, 0, 1, 0))),
    "B4": ("Bayesian", "Bayes · per-ward rule + flags", "Bayesian logistic regression", "Joint imputation + per-ward rule + flags", _b((1, 1, 1, 0))),
    "B5": ("Bayesian", "Bayes · per-ward rule, shrunk", "Bayesian logistic regression", "Joint imputation + per-ward rule, own-value effect shrunk", _b((1, 1, 0, 1))),
    "L1": ("Standard", "Ridge · mean imputation", "Ridge logistic regression", "Mean imputation", None),
    "L2": ("Standard", "Ridge · mean imp. + flags", "Ridge logistic regression", "Mean imputation + missing flags", None),
    "L3": ("Standard", "Ridge · MICE", "Ridge logistic regression", "MICE imputation", None),
    "L4": ("Standard", "Ridge · MICE + flags", "Ridge logistic regression", "MICE imputation + missing flags", None),
    "G1": ("Standard", "Boosting · native missing", "Gradient boosting", "Handles missing values natively", None),
    "G2": ("Standard", "Boosting · MICE", "Gradient boosting", "MICE imputation", None),
}
FAM_COLOR = {"Bayesian": "#2a78d6", "Standard": "#eb6834"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


# ----------------------------------------------------------------------------- metrics
def auc_cols(R, Y):
    rk = rankdata(R, axis=0)
    n1 = Y.sum(0)
    return ((rk * Y).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Y) - n1))


def logloss(R, Y):
    R = np.clip(R, 1e-6, 1 - 1e-6)
    return -(Y * np.log(R) + (1 - Y) * np.log(1 - R)).mean()


def recalibrate(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    D = np.column_stack([np.ones_like(x), x])
    b = np.array([0.0, 1.0])
    for _ in range(30):
        pr = 1 / (1 + np.exp(-D @ b))
        H = (D * (pr * (1 - pr))[:, None]).T @ D + 1e-8 * np.eye(2)
        step = np.linalg.solve(H, D.T @ (y - pr))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b[1], b[0]


def evaluate(R, Y, lac_miss, nolabs):
    r = dict(AUC=auc_cols(R, Y).mean(), LogLoss=logloss(R, Y), Brier=((R - Y) ** 2).mean())
    r["CalSlope"], r["CalIntercept"] = recalibrate(R.ravel(), Y.ravel())
    r["CITL"] = R.mean() - Y.mean()
    r["Gap_lactate_missing"] = R[lac_miss].sum(1).mean() - Y[lac_miss].sum(1).mean()
    r["Gap_no_labs"] = R[nolabs].sum(1).mean() - Y[nolabs].sum(1).mean() if nolabs.any() else np.nan
    return r


def bootstrap(Rd, Y, B=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(Y)
    out = {k: dict(AUC=np.zeros(B), LogLoss=np.zeros(B)) for k in Rd}
    for b in range(B):
        i = rng.integers(0, n, n)
        for k, R in Rd.items():
            out[k]["AUC"][b] = auc_cols(R[i], Y[i]).mean()
            out[k]["LogLoss"][b] = logloss(R[i], Y[i])
    return out


# ----------------------------------------------------------------------------- standard models
def standard_models(Ztr, Xtr, wtr, Ytr, copies, K):
    W = int(wtr.max()) + 1
    mean = np.nanmean(Xtr, 0)
    imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0).fit(np.column_stack([Ztr, Xtr]))
    q = Ztr.shape[1]
    mice = lambda Z, X: imp.transform(np.column_stack([Z, X]))[:, q:]

    def feats(key, Z, X, w):
        flag, oh = np.isnan(X).astype(float), np.eye(W)[w]
        if key == "L1": return np.column_stack([Z, np.where(np.isnan(X), mean, X), oh])
        if key == "L2": return np.column_stack([Z, np.where(np.isnan(X), mean, X), flag, oh])
        if key == "L3": return np.column_stack([Z, mice(Z, X), oh])
        if key == "L4": return np.column_stack([Z, mice(Z, X), flag, oh])
        if key == "G1": return np.column_stack([Z, X, w])
        if key == "G2": return np.column_stack([Z, mice(Z, X), w])

    keys = ["L1", "L2", "L3", "L4", "G1", "G2"]
    Ftr = {k: feats(k, Ztr, Xtr, wtr) for k in keys}
    Fc = {k: [feats(k, *c) for c in copies] for k in keys}
    out = {k: [np.zeros((len(c[0]), K)) for c in copies] for k in keys}
    for k_ in range(K):
        for key in keys:
            if key.startswith("L"):
                m = LogisticRegression(C=0.3, max_iter=3000).fit(Ftr[key], Ytr[:, k_])
            else:
                m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=150, l2_regularization=1.0,
                                                   min_samples_leaf=40, early_stopping=False, random_state=0,
                                                   categorical_features=[Ftr[key].shape[1] - 1]).fit(Ftr[key], Ytr[:, k_])
            for c in range(len(copies)):
                out[key][c][:, k_] = m.predict_proba(Fc[key][c])[:, 1]
    return out


# ----------------------------------------------------------------------------- plotting
def style(ax, title=None):
    ax.set_facecolor(SURF)
    if title:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.tick_params(colors=INK2, labelsize=9.5, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=1600)
    ap.add_argument("--burn", type=int, default=600)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--ntrain", type=int, default=4500)
    ap.add_argument("--reuse", action="store_true", help="reuse saved Bayesian predictions if present")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results_h"))
    ap.add_argument("--data-dir", default=str(HERE / "data_h"))
    ap.add_argument("--with-standard", action="store_true", help="also run ridge/boosting benchmarks")
    a = ap.parse_args()
    if a.quick:
        a.iter, a.burn, a.chains = 60, 30, 1
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    df = pd.read_csv(Path(a.data_dir) / "synthetic_hospital.csv")
    truth = json.loads((Path(a.data_dir) / "truth_h.json").read_text())
    base, vit, dx, labs = truth["base"], truth["vitals"], truth["dx_cols"], truth["labs"]
    K, q, p, W = len(dx), len(base), len(vit), truth["W"]
    data = bm.prepare(df, base, vit, dx[0], LOG_VARS)
    Y = df[dx].to_numpy(float)
    ward = df["ward"].to_numpy(int)
    n = len(df)
    tr, te = np.arange(n) < a.ntrain, np.arange(n) >= a.ntrain
    Z, X = data["Z"], data["X"]
    Yte, Nte, ntr = Y[te], int(te.sum()), int(tr.sum())

    rng = np.random.default_rng(123)
    Xte_c = []
    for d in LEVELS:
        Xc_ = X[te].copy()
        if d > 0:
            Xc_[(~np.isnan(Xc_)) & (rng.random(Xc_.shape) < d)] = np.nan
        Xte_c.append(Xc_)
    X_aug = np.vstack([X[tr]] + Xte_c)
    Z_aug = np.vstack([Z[tr]] + [Z[te]] * len(LEVELS))
    ward_aug = np.concatenate([ward[tr]] + [ward[te]] * len(LEVELS))
    Y_aug = np.vstack([Y[tr]] + [Yte] * len(LEVELS))
    train_aug = np.concatenate([np.ones(ntr, bool), np.zeros(Nte * len(LEVELS), bool)])
    lab_idx = [vit.index(v) for v in labs]
    bed_idx = [j for j in range(p) if j not in lab_idx]
    pgroup = bh.pattern_groups(np.isnan(X_aug), [lab_idx, bed_idx])
    data_aug = dict(data, Z=Z_aug, X=X_aug)
    sl = [slice(ntr + c * Nte, ntr + (c + 1) * Nte) for c in range(len(LEVELS))]
    lac = vit.index("lactate")
    groups = [(np.isnan(Xte_c[c][:, lac]), np.isnan(Xte_c[c][:, lab_idx]).all(1)) for c in range(len(LEVELS))]
    print(f"train {ntr}  held-out {Nte} x {len(LEVELS)} copies  wards {W}", flush=True)

    R, width, extra = {}, {}, {}
    if a.with_standard:
        print("== standard models (ridge / boosting, mean / MICE / native missing)", flush=True)
        t0 = time.time()
        copies = [(Z[te], Xte_c[c], ward[te]) for c in range(len(LEVELS))]
        for key, val in standard_models(Z[tr], X[tr], ward[tr], Y[tr], copies, K).items():
            R[key], width[key], extra[key] = val, [np.nan] * 3, dict(max_rhat=np.nan)
        print(f"   {time.time() - t0:.0f}s", flush=True)

    for key, (fam, short, _, _, cfg) in MODELS.items():
        if cfg is None:
            continue
        cfg = dict(cfg, pool=K > 1)
        pf = out / f"pred_{key}.npz"
        if a.reuse and pf.exists():
            z_ = np.load(pf, allow_pickle=True)
            R[key], width[key] = list(z_["R"]), list(z_["width"])
            extra[key] = {k: z_[k] for k in z_.files if k in ("d0", "ds", "gs")}
            extra[key]["max_rhat"] = float(z_["max_rhat"])
            print(f"== {key} loaded from disk", flush=True)
            continue
        t0 = time.time()
        print(f"== fitting {key} {short}", flush=True)
        f = bh.fit_h(data_aug, Y_aug, train_aug, ward_aug, pgroup, cfg, chains=a.chains, n_iter=a.iter,
                     n_burn=a.burn, seed=int(key[1]) + 1, verbose=True)
        Q = f["Q"]
        mean = Q.mean(0)
        lo, hi = np.quantile(Q, [.025, .975], axis=0)
        R[key] = [mean[s.start - ntr:s.stop - ntr] for s in sl]
        width[key] = [float((hi - lo)[s.start - ntr:s.stop - ntr].mean()) for s in sl]
        rh = [bm.rhat(f["beta"][:, :, k, j]) for k in range(K) for j in range(1, 1 + q + p)] if a.chains > 1 else [np.nan]
        extra[key] = dict(max_rhat=float(np.max(rh)))
        save = dict(R=np.array(R[key]), width=np.array(width[key]), max_rhat=extra[key]["max_rhat"])
        if cfg["ward_sel"]:
            extra[key]["d0"] = f["d0"].reshape(-1, p, W).mean(0)
            extra[key]["ds"] = f["ds"].reshape(-1, p, W).mean(0)
            extra[key]["gs"] = f["gamma"][..., q + 1].reshape(-1, p).mean(0)
            save.update({k: extra[key][k] for k in ("d0", "ds", "gs")})
        np.savez_compressed(pf, **save)
        print(f"   done in {(time.time() - t0) / 60:.1f} min   max R-hat {extra[key]['max_rhat']:.3f}", flush=True)
        del f, Q

    keys = [k for k in MODELS if k in R]
    # ---------------------------------------------------------------- metrics
    rows = []
    for key in keys:
        for c, d in enumerate(LEVELS):
            m = evaluate(R[key][c], Yte, *groups[c])
            m.update(model=key, deletion=d, interval_width=width[key][c])
            rows.append(m)
    M = pd.DataFrame(rows)
    M.round(4).to_csv(out / "metrics_long.csv", index=False)
    get = lambda k, c, col: float(M[(M.model == k) & (M.deletion == LEVELS[c])][col].iloc[0])

    boot = {c: bootstrap({k: R[k][c] for k in keys}, Yte, seed=c) for c in range(3)}
    ll = np.array([[get(k, c, "LogLoss") for c in range(3)] for k in keys])
    rank = np.apply_along_axis(lambda v: rankdata(v, method="min"), 0, ll)
    overall = rank.mean(1) + 1e-6 * ll[:, 0]
    best = ll.argmin(0)
    tied = np.zeros((len(keys), 3), bool)
    for c in range(3):
        for i, k in enumerate(keys):
            d = boot[c][k]["LogLoss"] - boot[c][keys[best[c]]]["LogLoss"]
            tied[i, c] = (i == best[c]) or (np.quantile(d, 0.025) <= 0)
    order = np.argsort(overall)

    # ---------------------------------------------------------------- scoreboard
    rows = []
    for pos, i in enumerate(order, 1):
        k = keys[i]
        fam, short, model, handling, _ = MODELS[k]
        rows.append({
            "Rank": pos, "Prediction model": model, "Missing-data handling": handling,
            "AUC as recorded": f"{get(k, 0, 'AUC'):.3f}", "AUC, 50% more deleted": f"{get(k, 2, 'AUC'):.3f}",
            "AUC drop": f"{get(k, 2, 'AUC') - get(k, 0, 'AUC'):+.3f}",
            "Log-loss as recorded": f"{ll[i, 0]:.4f}", "Log-loss, 50% more deleted": f"{ll[i, 2]:.4f}",
            "Best or tied-with-best: as recorded": "yes" if tied[i, 0] else "no",
            "... 25% more deleted": "yes" if tied[i, 1] else "no",
            "... 50% more deleted": "yes" if tied[i, 2] else "no",
            "Calibration slope (1 = ideal)": f"{get(k, 0, 'CalSlope'):.2f}",
            "Key": k})
    T = pd.DataFrame(rows)
    T.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.md").write_text(T.drop(columns="Key").to_markdown(index=False))
    print("\n" + T.drop(columns="Key").to_markdown(index=False))

    # ---------------------------------------------------------------- verdict
    def name(i): return MODELS[keys[i]][1]
    fam = np.array([MODELS[k][0] for k in keys])
    lines = ["VERDICT (lower log-loss = better predictions; ties judged by paired bootstrap over 1,500 held-out patients)", ""]
    for c, s in enumerate(SIT):
        t = [name(i) for i in range(len(keys)) if tied[i, c] and i != best[c]]
        lines.append(f"{s}: best = {name(best[c])} (log-loss {ll[best[c], c]:.4f}, AUC {get(keys[best[c]], c, 'AUC'):.3f})."
                     + (f" Statistically tied: {'; '.join(t)}." if t else " No other model is tied with it."))
    lines.append("")
    lines.append(f"Best overall (average rank across the three situations): {name(order[0])}.")
    for f_ in ("Bayesian", "Standard"):
        idx = [i for i in range(len(keys)) if fam[i] == f_]
        if not idx:
            continue
        bi = min(idx, key=lambda i: overall[i])
        lines.append(f"Best {f_} combination: {name(bi)} (overall rank {list(order).index(bi) + 1} of {len(keys)}).")
    open(out / "verdict.txt", "w").write("\n".join(lines))
    print("\n" + "\n".join(lines))

    # ---------------------------------------------------------------- charts
    # (1) dumbbell: as recorded vs 50% more deleted
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.6), facecolor=SURF, sharey=True)
    ypos = {i: r for r, i in enumerate(order)}
    for ax, col, ttl, xl, hi_good in (
            (axs[0], "AUC", "How well does it rank patients?  (AUC, higher = better)", "AUC   (0.5 = coin flip, 1.0 = perfect)", True),
            (axs[1], "LogLoss", "How trustworthy are the risk numbers?  (log-loss, lower = better)", "log-loss", False)):
        style(ax, ttl)
        for i in order:
            k = keys[i]
            a0, a1 = get(k, 0, col), get(k, 2, col)
            c_ = FAM_COLOR[MODELS[k][0]]
            ax.plot([a0, a1], [ypos[i]] * 2, color=c_, lw=2.2, alpha=0.55)
            ax.plot(a0, ypos[i], "o", color=c_, ms=9, mec=SURF, mew=1.4)
            ax.plot(a1, ypos[i], "o", mfc=SURF, mec=c_, mew=2, ms=9)
            ax.text(max(a0, a1) + (0.002 if hi_good else 0.0007), ypos[i], f"{a1 - a0:+.3f}", va="center", fontsize=8.5, color=INK2)
        ax.set_xlabel(xl, color=INK2, fontsize=9.5)
    axs[0].set_yticks(range(len(keys)))
    axs[0].set_yticklabels([("★ " if r == 0 else "") + MODELS[keys[i]][1] for r, i in enumerate(order)], color=INK)
    axs[0].invert_yaxis()
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color="none", mfc=INK2, mec=SURF, ms=9, label="as recorded"),
         Line2D([], [], marker="o", color="none", mfc=SURF, mec=INK2, mew=2, ms=9, label="after 50% more values deleted"),
         ]
    if a.with_standard:
        h += [Line2D([], [], marker="s", color=FAM_COLOR["Bayesian"], ls="none", ms=9, label="Bayesian model"),
              Line2D([], [], marker="s", color=FAM_COLOR["Standard"], ls="none", ms=9, label="standard model")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=len(h), frameon=False, fontsize=9.5, labelcolor=INK2)
    fig.text(0.01, 0.005, "Models are sorted best-to-worst overall (★). A short line means the model holds up when measurement practice shifts; "
             "the number is the change.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "fig_scoreboard.png", dpi=130, facecolor=SURF)
    plt.close(fig)

    # (2) rank heat-map by situation
    fig, ax = plt.subplots(figsize=(8.4, 6.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    mat = np.column_stack([rank[order], rank.mean(1)[order]])
    cmap = plt.get_cmap("Blues_r")
    ax.imshow(mat, cmap=cmap, vmin=1, vmax=len(keys) + 3, aspect="auto")
    for r in range(len(order)):
        for c in range(4):
            v = mat[r, c]
            ax.text(c, r, f"{v:.0f}" if c < 3 else f"{v:.1f}", ha="center", va="center", fontsize=10,
                    color="white" if v < len(keys) * 0.45 else INK, fontweight="bold" if (c < 3 and tied[order[r], c]) else None)
    ax.set_xticks(range(4))
    ax.set_xticklabels(SIT + ["Average rank"], color=INK2, fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([MODELS[keys[i]][1] for i in order], color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Rank by log-loss in each situation (1 = best of {len(keys)})\nbold = best or statistically tied with the best",
                 loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=28)
    fig.tight_layout()
    fig.savefig(out / "fig_ranks.png", dpi=130, facecolor=SURF)
    plt.close(fig)

    # ward-effect recovery (check that the per-ward rule learned something real)
    if "d0" in extra.get("B2", {}):
        tw_off, tw_sl = np.array(truth["ward_offset"]), np.array(truth["ward_slope"])
        rec = {v: float(np.corrcoef(extra["B2"]["d0"][j] - extra["B2"]["d0"][j].mean(),
                                    tw_off[j] - tw_off[j].mean())[0, 1]) for j, v in enumerate(vit)}
        sl_est = extra["B2"]["gs"][lac] + extra["B2"]["ds"][lac]
        pd.Series(rec).round(3).to_csv(out / "ward_intercept_recovery.csv", header=["correlation"])
        print(f"\nPer-ward rule recovery (B2): correlation of estimated vs true ward measuring offsets "
              f"{np.round(list(rec.values()), 2).tolist()}; lactate sickness slope {np.corrcoef(sl_est, tw_sl[lac])[0, 1]:.2f}")
    print("\ndone ->", out)


if __name__ == "__main__":
    main()
