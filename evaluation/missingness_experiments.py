#!/usr/bin/env python3
"""Missingness experiments on the built synthetic SWED cohort.

This is the SWED counterpart of preprocessing/run_analysis.py: describe the
recording structure, fit the Bayesian selection model, and report (i) estimated
missingness attributes per vital and (ii) observed vs imputed distribution
shifts.  It deliberately uses the SWED recording structure: only a few
hospitals contribute vital signs at all, so the model is fitted within the
hospitals that actually have vitals, with hospital indicators in the
selection-only design.

Example
-------
python -m evaluation.missingness_experiments --sample 20000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "preprocessing") not in sys.path:
    sys.path.insert(0, str(ROOT / "preprocessing"))

import bayes_missing as bm  # noqa: E402
from evaluation.cohort import load_cohort  # noqa: E402


def _visits_with_vitals(cohort, splits=("old", "new", "test")) -> pd.DataFrame:
    parts = []
    for name in splits:
        split = cohort[name]
        d = split.visits.copy()
        for j, vital in enumerate(cohort.vital_names):
            d[vital] = split.raw[:, j]
        d["role"] = name
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _vital_hospitals(df: pd.DataFrame, vitals: list[str]) -> list[str]:
    has_vital = df[vitals].notna().any(axis=1)
    rates = df.loc[has_vital, "ed_hospital"].value_counts()
    return sorted(rates.index.tolist())


def _one_visit_per_patient(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    return df.sample(frac=1.0, random_state=seed).drop_duplicates("lopnr_swed", keep="first")


def add_base_columns(df: pd.DataFrame, complaints: list[str]) -> tuple[pd.DataFrame, list[str]]:
    complaint_cols = [f"sct_{c}" for c in complaints]
    for c, col in zip(complaints, complaint_cols):
        df[col] = (df["complaint"].to_numpy() == c).astype(float)
    return df, ["age", "male"] + complaint_cols


def sample_fit_rows(df: pd.DataFrame, sample: int | None, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    fit = df.iloc[order].drop_duplicates("lopnr_swed", keep="first")
    if sample is not None and len(fit) > sample:
        fit = fit.iloc[:sample]
    return fit.reset_index(drop=True)


def hospital_selection_columns(df: pd.DataFrame, hospitals: list[str]) -> list[str]:
    cols = [f"sel_hosp_{h.replace('-', '_')}" for h in hospitals[1:]]
    for h, col in zip(hospitals[1:], cols):
        df[col] = (df["ed_hospital"].to_numpy() == h).astype(float)
    return cols


def build_frame(
    built: str | Path,
    sample: int | None,
    seed: int,
    vital_hospitals: list[str] | None = None,
    cohorts=("old", "new"),
    condition_any_vital: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    """Return (all visits, fit rows, vital hospitals, base columns, selection columns)."""
    cohort = load_cohort(built, seed=seed)
    all_visits = _visits_with_vitals(cohort, ("old", "new", "test"))
    vitals = list(cohort.vital_names)
    vh = _vital_hospitals(all_visits, vitals) if vital_hospitals is None else list(vital_hospitals)
    if not vh:
        raise ValueError("no hospital has any observed vital signs")

    development = all_visits[all_visits["role"].isin(cohorts)].copy()
    fit_pool = development[development["ed_hospital"].isin(vh)].copy()
    if condition_any_vital:
        fit_pool = fit_pool[fit_pool[vitals].notna().any(axis=1)].copy()
    if fit_pool.empty:
        raise ValueError("no development visits at the vital-recording hospitals")

    fit = sample_fit_rows(fit_pool, sample, seed)
    fit, base_cols = add_base_columns(fit, cohort.complaints)
    sel_cols = hospital_selection_columns(fit, vh)
    fit.attrs["reference_hospital"] = vh[0]
    fit.attrs["vital_hospitals"] = vh
    fit.attrs["condition_any_vital"] = condition_any_vital
    return all_visits, fit, vh, base_cols, sel_cols


def lab_visit_values(built: str | Path, concepts: list[str], sources: list[str] | None = None) -> pd.DataFrame:
    """First observed value per hospital-source visit and lab concept."""
    lab = pd.read_parquet(
        Path(built) / "lab_meas.parquet",
        columns=["ed_visit_id", "source", "term_name", "value_num", "datetime_analysis"],
    )
    lab = lab[lab["ed_visit_id"].notna() & lab["value_num"].notna() & lab["term_name"].isin(concepts)]
    if sources:
        lab = lab[lab["source"].isin(sources)]
    first = (
        lab.sort_values("datetime_analysis")
        .groupby(["source", "ed_visit_id", "term_name"])["value_num"]
        .first()
        .unstack("term_name")
    )
    first.columns = [f"lab_{c}" for c in first.columns]
    return first.reset_index()


def hospital_observation_table(all_visits: pd.DataFrame, vitals: list[str]) -> pd.DataFrame:
    rows = []
    for hospital, g in all_visits.groupby("ed_hospital", dropna=False):
        row = {"hospital": hospital, "n_visits": len(g)}
        for v in vitals:
            row[f"{v}_observed"] = float(g[v].notna().mean())
        row["any_vital_observed"] = float(g[vitals].notna().any(axis=1).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def coobservation(all_visits: pd.DataFrame, vitals: list[str]) -> pd.DataFrame:
    R = all_visits[vitals].notna().astype(float)
    return R.T @ R / len(R)


def _summ(draws: np.ndarray) -> dict:
    flat = np.asarray(draws).reshape(-1)
    return dict(mean=float(flat.mean()), lo=float(np.quantile(flat, 0.025)),
                hi=float(np.quantile(flat, 0.975)))


def selection_table(res: dict, data: dict, model: str) -> pd.DataFrame:
    g = res["gamma"]
    sel = res["sel"]
    vit = data["vital_cols"]
    q = len(data["base_cols"])
    s = 0 if data.get("S") is None else data["S"].shape[1]
    labels = (["intercept"] + data.get("selection_cols", []) + data["base_cols"]
              + ["outcome", "own_value"])
    km = len(labels) - 1
    gy_idx = labels.index("outcome")
    gx_idx = labels.index("own_value")
    rows = []
    for a, j in enumerate(sel):
        name = vit[j]
        rec = {"model": model, "vital": name, "observed_rate": float(np.mean(~np.isnan(data["X"][:, j])))}
        gxj = g[:, :, a, gx_idx]
        gyj = g[:, :, a, gy_idx]
        rec.update({f"gx_{k}": v for k, v in _summ(gxj).items()})
        rec.update({f"gy_{k}": v for k, v in _summ(gyj).items()})
        rec["gx_rhat"] = bm.rhat(gxj)
        rec["gy_rhat"] = bm.rhat(gyj)
        params = g[:, :, a, :].reshape(-1, g.shape[-1])
        means = params.mean(0)
        for lab in data["base_cols"]:
            vals = params[:, labels.index(lab)]
            rec[f"gz_{lab}"] = float(means[labels.index(lab)])
            rec[f"gz_{lab}_lo"] = float(np.quantile(vals, 0.025))
            rec[f"gz_{lab}_hi"] = float(np.quantile(vals, 0.975))
        base_offsets = [labels.index(lab) for lab in data["base_cols"]]
        rec["gz_max_abs"] = float(np.abs(means[base_offsets]).max())
        rec["gz_max_name"] = labels[base_offsets[int(np.abs(means[base_offsets]).argmax())]]
        if data.get("selection_cols"):
            offs = [labels.index(lab) for lab in data["selection_cols"]]
            best = int(np.abs(means[offs]).argmax())
            rec["hospital_max_abs"] = float(np.abs(means[offs][best]))
            rec["hospital_max_name"] = labels[offs[best]]
            vals = params[:, offs[best]]
            rec["hospital_max_lo"] = float(np.quantile(vals, 0.025))
            rec["hospital_max_hi"] = float(np.quantile(vals, 0.975))
        else:
            rec["hospital_max_abs"] = 0.0
            rec["hospital_max_name"] = ""
            rec["hospital_max_lo"] = rec["hospital_max_hi"] = 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def _classify(row: pd.Series) -> str:
    def excludes_zero(lo, hi):
        return lo > 0 or hi < 0

    if excludes_zero(row["gx_lo"], row["gx_hi"]):
        return "MNAR (own value) - prior-sensitive"
    if excludes_zero(row["gy_lo"], row["gy_hi"]):
        return "MAR (outcome)"
    base_ci = any(excludes_zero(row.get(f"gz_{lab}_lo", 0.0), row.get(f"gz_{lab}_hi", 0.0))
                  for lab in row.index if lab.startswith("gz_") and lab.endswith("_lo"))
    if base_ci:
        return "MAR (base)"
    if excludes_zero(row.get("hospital_max_lo", 0.0), row.get("hospital_max_hi", 0.0)):
        return "hospital-structured"
    return "MCAR/MAR within hospital"


def distribution_table(res: dict, data: dict, model: str, vitals: list[str] | None = None) -> pd.DataFrame:
    X = data["X"]
    loc, scale = data["x_loc"], data["x_scale"]
    miss = np.isnan(X)
    raw = X * scale + loc
    stats = res["imp_stats"]  # chains, draws, p, (mean, sd, q10, q50, q90)
    rows = []
    keep = set(data["vital_cols"] if vitals is None else vitals)
    for j, name in enumerate(data["vital_cols"]):
        if name not in keep:
            continue
        obs = raw[~miss[:, j], j]
        complete = raw[~miss.any(axis=1), j]
        im = bm.summarize_imputations(stats)[j]
        im_raw = im * scale[j]
        im_raw[0] += loc[j]
        im_raw[2:] += loc[j]
        row = {
            "model": model, "vital": name,
            "observed_n": int((~miss[:, j]).sum()),
            "observed_mean": float(np.mean(obs)), "observed_sd": float(np.std(obs)),
            "complete_case_n": int((~miss.any(axis=1)).sum()),
            "complete_case_mean": float(np.mean(complete)), "complete_case_sd": float(np.std(complete)),
            "imputed_mean": float(im_raw[0]), "imputed_sd": float(im_raw[1]),
            "imputed_q10": float(im_raw[2]), "imputed_q50": float(im_raw[3]), "imputed_q90": float(im_raw[4]),
            "imputed_iqr": float(im_raw[4] - im_raw[2]),
        }
        row["shift_imputed_vs_observed"] = row["imputed_mean"] - row["observed_mean"]
        row["shift_standardised"] = row["shift_imputed_vs_observed"] / row["observed_sd"]
        row["shift_complete_vs_observed"] = row["complete_case_mean"] - row["observed_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def observation_table(
    df: pd.DataFrame,
    columns: list[str],
    group_col: str = "ed_hospital",
    source_col: str | None = None,
    domain: str = "",
) -> pd.DataFrame:
    """Long-format observed-rate table for arbitrary visit-level variables."""
    keys = [group_col] + ([source_col] if source_col else [])
    rows = []
    for key, g in df.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        base = {"domain": domain, "group": key[0], "source": key[1] if source_col else "",
                "n_visits": len(g), "n_patients": g["lopnr_swed"].nunique(),
                "any_observed": float(g[columns].notna().any(axis=1).mean())}
        for c in columns:
            rows.append(dict(base, variable=c, observed_rate=float(g[c].notna().mean())))
    return pd.DataFrame(rows)


def pattern_table(
    df: pd.DataFrame,
    columns: list[str],
    group_col: str = "ed_hospital",
    source_col: str | None = None,
    domain: str = "",
) -> pd.DataFrame:
    """Joint observation patterns, one row per group and pattern."""
    keys = [group_col] + ([source_col] if source_col else [])
    rows = []
    for key, g in df.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        pattern = g[columns].notna().astype(int).astype(str).agg("".join, axis=1)
        counts = pattern.value_counts()
        persons = g.assign(_pattern=pattern).groupby("_pattern")["lopnr_swed"].nunique()
        for p, n in counts.items():
            rows.append({
                "domain": domain, "group": key[0], "source": key[1] if source_col else "",
                "pattern": p, "n": int(n), "persons": int(persons.loc[p]),
            })
    return pd.DataFrame(rows)


def selection_long_table(
    res: dict,
    data: dict,
    model: str,
    domain: str = "vital",
    group: str = "",
    source: str = "",
    log_cols=(),
) -> pd.DataFrame:
    """All selection-model coefficients in long format, including hospital offsets."""
    labels = (["intercept"] + list(data.get("selection_cols", [])) + list(data["base_cols"])
              + ["outcome", "own_value"])
    gamma = np.asarray(res["gamma"])
    x_loc, x_scale = np.asarray(data["x_loc"]), np.asarray(data["x_scale"])
    z_loc, z_scale = np.asarray(data["z_loc"]), np.asarray(data["z_scale"])
    base_cols = list(data["base_cols"])
    rows = []
    for a, j in enumerate(res["sel"]):
        name = data["vital_cols"][j]
        params = gamma[:, :, a, :].reshape(-1, len(labels))
        meta = {
            "model": model, "domain": domain, "group": group, "source": source, "variable": name,
            "observed_rate": float(np.mean(~np.isnan(data["X"][:, j]))),
            "x_loc": float(x_loc[j]), "x_scale": float(x_scale[j]),
            "log": int(name in set(log_cols)),
        }
        for k, label in enumerate(labels):
            vals = params[:, k]
            rec = dict(meta, coef=label, mean=float(vals.mean()),
                       lo=float(np.quantile(vals, 0.025)), hi=float(np.quantile(vals, 0.975)),
                       z_loc=np.nan, z_scale=np.nan)
            if label in base_cols:
                b = base_cols.index(label)
                rec["z_loc"], rec["z_scale"] = float(z_loc[b]), float(z_scale[b])
            rows.append(rec)
    return pd.DataFrame(rows)


def imputation_long_table(
    res: dict,
    data: dict,
    model: str,
    domain: str = "vital",
    group: str = "",
    source: str = "",
    log_cols=(),
) -> pd.DataFrame:
    """Posterior imputation summaries converted to the original value scale."""
    X = np.asarray(data["X"])
    loc, scale = np.asarray(data["x_loc"]), np.asarray(data["x_scale"])
    imputed = bm.summarize_imputations(res["imp_stats"])
    logs = set(log_cols)
    rows = []
    for j, name in enumerate(data["vital_cols"]):
        observed = X[~np.isnan(X[:, j]), j]

        def back(z):
            z = np.asarray(z, dtype=float)
            return np.exp(z * scale[j] + loc[j]) if name in logs else z * scale[j] + loc[j]

        obs_raw = np.asarray(back(observed), dtype=float)
        q10, q50, q90 = (float(x) for x in back(imputed[j, 2:5]))
        im_mean = float(back(imputed[j, 0]))
        im_sd = (q90 - q10) / bm.NORMAL_Q10_Q90 if name in logs else float(imputed[j, 1] * scale[j])
        rows.append({
            "model": model, "domain": domain, "group": group, "source": source, "variable": name,
            "observed_n": int(len(obs_raw)), "observed_mean": float(np.mean(obs_raw)),
            "observed_sd": float(np.std(obs_raw)), "imputed_mean": im_mean, "imputed_sd": im_sd,
            "imputed_q10": q10, "imputed_q50": q50, "imputed_q90": q90,
            "shift_imputed_vs_observed": im_mean - float(np.mean(obs_raw)),
            "shift_standardised": (im_mean - float(np.mean(obs_raw))) / max(float(np.std(obs_raw)), 1e-9),
        })
    return pd.DataFrame(rows)


def prior_sensitivity_table(
    res: dict,
    data: dict,
    tau: float,
    domain: str = "vital",
    group: str = "",
    source: str = "",
) -> pd.DataFrame:
    rows = []
    for a, j in enumerate(res["sel"]):
        gx = res["gamma"][:, :, a, -1].reshape(-1)
        rows.append({
            "tau_gx": float(tau), "domain": domain, "group": group, "source": source,
            "variable": data["vital_cols"][j], "gx_mean": float(gx.mean()),
            "gx_lo": float(np.quantile(gx, 0.025)), "gx_hi": float(np.quantile(gx, 0.975)),
        })
    return pd.DataFrame(rows)


def _plot_hospital_heatmap(hosp: pd.DataFrame, vitals: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = hosp.set_index("hospital")[[f"{v}_observed" for v in vitals]]
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.42 * len(mat))))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(vitals)))
    ax.set_xticklabels(vitals)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if val > 0.55 else "black")
    ax.set_title("Observed vital rate by hospital (built SWED cohort)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_distribution(dist: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(dist))
    ax.hlines(y, dist["observed_mean"], dist["imputed_mean"], color="#b0b0b0", lw=2,
              label="shift to imputed missing")
    ax.plot(dist["observed_mean"], y, "o", color="#2a78d6", ms=7, label="observed")
    ax.plot(dist["complete_case_mean"], y, "s", mfc="white", mec="#eb6834", mew=1.8, ms=7,
            label="complete case")
    ax.plot(dist["imputed_mean"], y, "D", color="#1baf7a", ms=6, label="imputed missing")
    ax.set_yticks(y)
    ax.set_yticklabels(dist["vital"])
    ax.invert_yaxis()
    ax.set_xlabel("mean in original units")
    ax.set_title("Observed vs imputed distribution of missing vitals (MNAR fit)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_selection(tab: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    y = np.arange(len(tab))
    for ax, (col, title) in zip(axes, [("gx", "Own-value effect (g_x)"), ("gy", "Outcome effect (g_y)")]):
        ax.hlines(y, tab[f"{col}_lo"], tab[f"{col}_hi"], color="#eb6834", lw=2)
        ax.plot(tab[f"{col}_mean"], y, "o", color="#eb6834", ms=7)
        ax.axvline(0, color="grey", lw=1, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels(tab["vital"])
        ax.set_title(title)
        ax.set_xlabel("probit coefficient (95% CrI)")
        ax.invert_yaxis()
    fig.suptitle("Estimated SWED missingness attributes (MNAR selection fit)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(
    built: str | Path,
    out: Path,
    sample: int | None,
    chains: int,
    n_iter: int,
    n_burn: int,
    seed: int,
    skip_plots: bool = False,
    tau_gx: float = 0.1,
    tau_gx_grid: tuple[float, ...] = (0.1, 0.5),
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    all_visits, fit, vh, base_cols, sel_cols = build_frame(built, sample, seed)
    vitals = [v for v in ["sbp", "pulse", "rr", "spo2", "temp"]]

    hosp = hospital_observation_table(all_visits, vitals)
    hosp.to_csv(out / "missingness_by_hospital.csv", index=False)
    coobservation(all_visits, vitals).round(4).to_csv(out / "coobservation.csv")
    if not skip_plots:
        _plot_hospital_heatmap(hosp, vitals, out / "figures" / "hospital_observation.png")

    data = bm.prepare(fit, base_cols, vitals, "died30", selection_cols=sel_cols)
    obs_rates = {v: float(np.clip(np.mean(~np.isnan(data["X"][:, j])), 1e-3, 1 - 1e-3))
                 for j, v in enumerate(vitals)}
    fit_vitals = [v for j, v in enumerate(vitals)
                  if 0.0 < np.mean(np.isnan(data["X"][:, j])) < 0.999]
    if not fit_vitals:
        raise ValueError("no vital has any missing values after conditioning")
    print(f"fit rows={len(fit):,}  vital hospitals={vh}  reference={fit.attrs['reference_hospital']}")
    print("observed rates:", {v: round(obs_rates[v], 3) for v in vitals})
    print(f"selection-model vitals: {fit_vitals}  (tau_gx={tau_gx})")

    res_mar = bm.fit(data, {}, chains=chains, n_iter=n_iter, n_burn=n_burn, seed=seed + 1,
                     prior_obs_rate=obs_rates, verbose=True)
    res_mnar = bm.fit(data, {v: None for v in fit_vitals}, chains=chains, n_iter=n_iter,
                      n_burn=n_burn, seed=seed + 2, prior_obs_rate=obs_rates,
                      tau_gx=tau_gx, verbose=True)

    tabs = [selection_table(res_mar, data, "MAR"), selection_table(res_mnar, data, "MNAR")]
    attrs = pd.concat(tabs, ignore_index=True)
    attrs["evidence"] = attrs.apply(_classify, axis=1)
    attrs.to_csv(out / "missingness_attributes.csv", index=False)

    dist = pd.concat([distribution_table(res_mar, data, "MAR", fit_vitals),
                      distribution_table(res_mnar, data, "MNAR", fit_vitals)], ignore_index=True)
    dist.to_csv(out / "distribution_shift.csv", index=False)

    prior_rows = []
    for tau in tau_gx_grid:
        r = bm.fit(data, {v: None for v in fit_vitals}, chains=chains,
                   n_iter=min(n_iter, 2000), n_burn=min(n_burn, 800), seed=seed + 20 + int(tau * 100),
                   prior_obs_rate=obs_rates, tau_gx=tau, verbose=False)
        for a, j in enumerate(r["sel"]):
            gx = r["gamma"][:, :, a, -1].reshape(-1)
            prior_rows.append({"tau_gx": tau, "vital": data["vital_cols"][j],
                               "gx_mean": float(gx.mean()),
                               "gx_lo": float(np.quantile(gx, 0.025)),
                               "gx_hi": float(np.quantile(gx, 0.975))})
    pd.DataFrame(prior_rows).to_csv(out / "prior_sensitivity.csv", index=False)

    diag = pd.DataFrame({
        "model": ["MAR", "MNAR"],
        "max_rhat_beta": [float(np.nanmax([bm.rhat(res_mar["beta"][:, :, k]) for k in range(res_mar["beta"].shape[-1])])),
                          float(np.nanmax([bm.rhat(res_mnar["beta"][:, :, k]) for k in range(res_mnar["beta"].shape[-1])]))],
        "max_rhat_gx": [np.nan, float(attrs.query("model == 'MNAR'")["gx_rhat"].max())],
    })
    diag.to_csv(out / "missingness_fit_diagnostics.csv", index=False)

    if not skip_plots:
        _plot_distribution(dist.query("model == 'MNAR'"), out / "figures" / "distribution_shift.png")
        _plot_selection(attrs.query("model == 'MNAR'"), out / "figures" / "selection_effects.png")

    (out / "run_config.json").write_text(json.dumps({
        "built": str(built), "sample": sample, "chains": chains,
        "n_iter": n_iter, "n_burn": n_burn, "seed": seed,
        "vital_hospitals": vh, "reference_hospital": fit.attrs["reference_hospital"],
        "condition_any_vital": bool(fit.attrs.get("condition_any_vital", False)),
        "base_cols": base_cols, "selection_cols": sel_cols, "vitals": vitals,
        "selection_model_vitals": fit_vitals, "tau_gx": tau_gx,
        "tau_gx_grid": list(tau_gx_grid),
    }, indent=2))

    lines = [
        "# SWED missingness experiment",
        "",
        f"- Fitted rows: {len(fit):,} (one visit per patient, development splits).",
        f"- Vital-recording hospitals: {', '.join(vh)} (reference: {fit.attrs['reference_hospital']}).",
        f"- Fit conditioned on a visit having at least one vital: {fit.attrs.get('condition_any_vital', False)}.",
        f"- Vitals: {', '.join(vitals)}.",
        "",
        "## What to read",
        "",
        "- `missingness_by_hospital.csv`: the structural pattern. Hospitals without any vital source cannot be",
        "  explained by patient-level MAR/MNAR.",
        "- `missingness_attributes.csv`: selection-model estimates in the vital hospitals.",
        "  `gx` = own-value (MNAR) effect, `gy` = outcome effect, `gz_*` = base-variable effects.",
        "  The primary MNAR fit uses a skeptical prior on gx (see `run_config.json`);",
        "  `prior_sensitivity.csv` shows how much the data move gx under weaker priors.",
        "- `distribution_shift.csv`: observed vs complete-case vs posterior-imputed missing distributions.",
        "  `shift_standardised` is in observed SD units.",
        "- `figures/hospital_observation.png`, `figures/distribution_shift.png`, `figures/selection_effects.png`.",
        "",
        "## Caveat",
        "",
        "The synthetic SWED generator records vitals only at three hospitals and otherwise at random.",
        "No clinical MNAR mechanism was built in. The fit is therefore a pipeline/validation check;",
        "g_x is prior-sensitive and should not be read as a clinical finding.",
        "Fitting is also restricted to visits with at least one observed vital; the 9 hospitals with",
        "no vital source are structural missingness, not a patient-level mechanism.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines))
    print("wrote", out)
    return {"attributes": attrs, "distribution": dist, "diagnostics": diag}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--built", default=str(ROOT / "data_pipeline" / "synthetic" / "built_synth_subset_50k"))
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results" / "missingness_analysis" / "swed"))
    ap.add_argument("--sample", type=int, default=20_000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--iter", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--tau-gx", type=float, default=0.1)
    ap.add_argument("--tau-gx-grid", default="0.1,0.5")
    a = ap.parse_args(argv)
    grid = tuple(float(x) for x in a.tau_gx_grid.split(",") if x.strip())
    run(Path(a.built), Path(a.out), a.sample, a.chains, a.iter, a.burn, a.seed,
        a.skip_plots, a.tau_gx, grid)


if __name__ == "__main__":
    main()
