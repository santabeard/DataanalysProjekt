#!/usr/bin/env python3
"""Disclosure-safe missingness parameters for the synthetic SWED pipeline.

Runs next to 03_export_stats.py on Bianca. It fits the MAR/MNAR selection
model from preprocessing/bayes_missing.py to the real built cohort and writes
only aggregated, rounded parameters that 04_synthesize.py can use as
assumptions about unobserved vitals and labs.

    python 03b_export_missingness.py --built built_clean_parquet
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(ROOT / "preprocessing")):
    if _path not in __import__("sys").path:
        __import__("sys").path.insert(0, _path)

import bayes_missing as bm
from evaluation.cohort import load_cohort
from evaluation.missingness_experiments import (
    add_base_columns,
    build_frame,
    hospital_selection_columns,
    imputation_long_table,
    lab_visit_values,
    observation_table,
    pattern_table,
    prior_sensitivity_table,
    sample_fit_rows,
    selection_long_table,
)

_spec = importlib.util.spec_from_file_location("export_stats", HERE / "03_export_stats.py")
export_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_stats)

MIN_CELL = export_stats.MIN_CELL
ROUND_TO = export_stats.ROUND_TO
MIN_N_QUANTILE = export_stats.MIN_N_QUANTILE
ORIGINAL = export_stats.ORIGINAL
WRITTEN: list[str] = []


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / f"{name}.csv", index=False)
    WRITTEN.append(name)
    log(f"  {name}.csv: {len(df)} rows")


def rounded_counts(df: pd.DataFrame, cols=("n", "persons", "n_visits", "n_patients")) -> pd.DataFrame:
    df = df.copy()
    if "persons" in df:
        df = df[df["persons"] >= MIN_CELL]
    if "n_patients" in df:
        df = df[df["n_patients"] >= MIN_CELL]
    for c in cols:
        if c in df:
            df[c] = (np.round(df[c].to_numpy(dtype=float) / ROUND_TO) * ROUND_TO).astype(int)
    if "persons" in df:
        df = df[df["persons"] >= MIN_CELL]
    if "n_patients" in df:
        df = df[df["n_patients"] >= MIN_CELL]
    return df.reset_index(drop=True)


def round_floats(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(decimals)
    return df


def fit_selection(
    fit: pd.DataFrame,
    base_cols: list[str],
    cols: list[str],
    sel_cols: list[str],
    args: argparse.Namespace,
    seed: int,
    log_cols=(),
) -> dict:
    log_cols = list(log_cols)
    data = bm.prepare(fit, base_cols, cols, "died30", log_vars=log_cols, selection_cols=sel_cols)
    rates = {v: float(np.clip(np.mean(~np.isnan(data["X"][:, j])), 1e-3, 1 - 1e-3))
             for j, v in enumerate(cols)}
    useful = [v for j, v in enumerate(cols) if 0.0 < np.mean(np.isnan(data["X"][:, j])) < 0.999]
    mar = {v: 0.0 for v in cols}
    mnar = {v: (None if v in useful else 0.0) for v in cols}
    common = dict(chains=args.chains, n_iter=args.iter, n_burn=args.burn, seed=seed,
                  prior_obs_rate=rates, tau_gx=args.tau_gx, verbose=False)
    res_mar = bm.fit(data, mar, **common)
    res_mnar = bm.fit(data, mnar, **common)
    prior = []
    for tau in args.tau_gx_grid:
        r = bm.fit(data, mnar, chains=args.chains, n_iter=min(args.iter, 2000),
                   n_burn=min(args.burn, 800), seed=seed + 20 + int(tau * 100),
                   prior_obs_rate=rates, tau_gx=tau, verbose=False)
        prior.append(r)
    return {"data": data, "mar": res_mar, "mnar": res_mnar, "prior": prior,
            "prior_taus": args.tau_gx_grid, "log_cols": log_cols, "useful": useful}


def export_fit(result: dict, args: argparse.Namespace, domain: str, group: str, source: str,
               tables: dict[str, list[pd.DataFrame]]) -> None:
    data, log_cols = result["data"], result["log_cols"]
    for model, res in [("MAR", result["mar"]), ("MNAR", result["mnar"])]:
        tables["selection"].append(selection_long_table(
            res, data, model, domain=domain, group=group, source=source, log_cols=log_cols))
        tables["imputation"].append(imputation_long_table(
            res, data, model, domain=domain, group=group, source=source, log_cols=log_cols))
    for tau, res in zip(result["prior_taus"], result["prior"]):
        tables["prior"].append(prior_sensitivity_table(
            res, data, tau, domain=domain, group=group, source=source))


def choose_labs(built: Path, top_labs: int) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    lab = pd.read_parquet(built / "lab_meas.parquet",
                          columns=["ed_visit_id", "source", "term_name", "value_num"])
    ed = pd.read_parquet(built / "ed_visits.parquet", columns=["ed_visit_id", "ed_hospital"])
    lab = lab[lab["ed_visit_id"].notna() & lab["value_num"].notna()]
    counts = lab.groupby(["source", "term_name"])["ed_visit_id"].nunique()
    with_hosp = lab.merge(ed, on="ed_visit_id", how="left")
    concepts: dict[str, list[str]] = {}
    hospitals: dict[str, list[str]] = {}
    for source, group in counts.groupby(level=0):
        top = group.droplevel(0).sort_values(ascending=False)
        top = top[top >= MIN_N_QUANTILE].head(top_labs).index.tolist()
        if not top:
            continue
        h = with_hosp[with_hosp["source"] == source].groupby("ed_hospital")["ed_visit_id"].nunique()
        h = sorted(h[h >= MIN_N_QUANTILE].index.tolist())
        if h:
            concepts[source], hospitals[source] = top, h
    return concepts, hospitals


def skewed_log_columns(fit: pd.DataFrame, cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        v = fit[c].dropna()
        if len(v) >= MIN_N_QUANTILE and v.min() > 0 and v.quantile(0.99) / max(v.quantile(0.5), 1e-9) > 10:
            out.append(c)
    return out


def self_check(out: Path) -> list[str]:
    problems = []
    forbidden = {"lopnr_swed", "ed_visit_id", "ed_presented_complaint_nonstd", "source_row"}
    for name in WRITTEN:
        df = pd.read_csv(out / f"{name}.csv")
        bad = forbidden & set(df.columns)
        if bad:
            problems.append(f"{name}: forbidden columns {sorted(bad)}")
        for c in ["n", "persons", "n_visits", "n_patients"]:
            if c in df and len(df) and ((df[c] < MIN_CELL) | (df[c] % ROUND_TO != 0)).any():
                problems.append(f"{name}: {c} below {MIN_CELL} or not rounded to {ROUND_TO}")
        if any(c in df.columns for c in ["min", "max"]):
            problems.append(f"{name}: has min/max")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--built", type=Path, default=export_stats.OUT_ROOT / "built_subset")
    ap.add_argument("--out", type=Path, help="default: $SWED_OUT/export_stats_<built folder name>")
    ap.add_argument("--sample", type=int, default=20_000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--iter", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--tau-gx", type=float, default=0.1)
    ap.add_argument("--tau-gx-grid", default="0.1,0.5")
    ap.add_argument("--top-labs", type=int, default=8)
    args = ap.parse_args()
    args.tau_gx_grid = tuple(float(x) for x in args.tau_gx_grid.split(",") if x.strip())
    out = args.out or export_stats.OUT_ROOT / f"export_stats_{args.built.name}"
    export_stats.refuse_to_write_into(out, args.built, ORIGINAL)
    out.mkdir(parents=True, exist_ok=True)
    log(f"built: {args.built}")
    log(f"out:   {out}")

    tables: dict[str, list[pd.DataFrame]] = {"observed": [], "patterns": [], "selection": [],
                                             "imputation": [], "prior": []}
    all_visits, fit, vital_hospitals, base_cols, sel_cols = build_frame(args.built, args.sample, args.seed)
    vitals = [v for v in ["sbp", "pulse", "rr", "spo2", "temp"] if v in all_visits]
    if not vitals:
        raise SystemExit("no modelled vitals in the built cohort")

    tables["observed"].append(observation_table(all_visits, vitals, domain="vital"))
    tables["patterns"].append(pattern_table(all_visits, vitals, domain="vital"))
    log("fitting vital selection model")
    result = fit_selection(fit, base_cols, vitals, sel_cols, args, args.seed + 1)
    export_fit(result, args, "vital", "", "", tables)

    cohort = load_cohort(args.built, seed=args.seed)
    lab_concepts, lab_hospitals = choose_labs(args.built, args.top_labs)
    for i, (source, concepts) in enumerate(sorted(lab_concepts.items())):
        hospitals = [h for h in lab_hospitals[source] if h in set(cohort.all_visits["ed_hospital"])]
        if not hospitals:
            continue
        lab_cols = [f"lab_{c}" for c in concepts]
        values = lab_visit_values(args.built, concepts, [source])
        frame = cohort.all_visits[cohort.all_visits["ed_hospital"].isin(hospitals)].merge(
            values, on="ed_visit_id", how="left")
        frame = frame[frame["source"].notna()].reset_index(drop=True)
        for c in lab_cols:
            if c not in frame:
                frame[c] = np.nan
        log(f"fitting lab selection model: {source} ({len(concepts)} concepts)")
        tables["observed"].append(observation_table(frame, lab_cols, source_col="source", domain="lab"))
        tables["patterns"].append(pattern_table(frame, lab_cols, source_col="source", domain="lab"))
        pool = frame[frame[lab_cols].notna().any(axis=1)]
        if pool.empty:
            continue
        lab_fit = sample_fit_rows(pool, args.sample, args.seed + 10 + i)
        lab_fit, lab_base = add_base_columns(lab_fit, cohort.complaints)
        lab_sel = hospital_selection_columns(lab_fit, hospitals)
        log_cols = skewed_log_columns(lab_fit, lab_cols)
        result = fit_selection(lab_fit, lab_base, lab_cols, lab_sel, args, args.seed + 100 + i, log_cols)
        export_fit(result, args, "lab", "", source, tables)

    observed = pd.concat(tables["observed"], ignore_index=True)
    vital_patterns = rounded_counts(pd.concat([t for t in tables["patterns"] if t["domain"].iloc[0] == "vital"],
                                              ignore_index=True))
    lab_patterns = rounded_counts(pd.concat([t for t in tables["patterns"] if t["domain"].iloc[0] == "lab"],
                                            ignore_index=True)) if any(
        t["domain"].iloc[0] == "lab" for t in tables["patterns"]) else pd.DataFrame()
    save(rounded_counts(observed), out, "missingness_by_hospital")
    save(vital_patterns, out, "missingness_patterns")
    if not lab_patterns.empty:
        save(lab_patterns, out, "lab_patterns")
    save(round_floats(pd.concat(tables["selection"], ignore_index=True)), out, "missingness_selection")
    save(round_floats(pd.concat(tables["imputation"], ignore_index=True)), out, "missingness_imputation")
    save(round_floats(pd.concat(tables["prior"], ignore_index=True)), out, "missingness_prior_sensitivity")
    save(pd.DataFrame({
        "rule": ["MIN_CELL", "ROUND_TO", "MIN_N_QUANTILE", "models", "gx_prior", "built_from"],
        "value": [MIN_CELL, ROUND_TO, MIN_N_QUANTILE, "MAR(gx=0),MNAR(gx free)", args.tau_gx, args.built.name],
    }), out, "missingness_rules")

    (out / "missingness_config.json").write_text(json.dumps({
        "built": str(args.built), "sample": args.sample, "chains": args.chains,
        "iter": args.iter, "burn": args.burn, "seed": args.seed,
        "tau_gx": args.tau_gx, "tau_gx_grid": list(args.tau_gx_grid),
        "vitals": vitals, "vital_hospitals": vital_hospitals,
        "lab_concepts": {s: list(cs) for s, cs in lab_concepts.items()},
        "lab_hospitals": lab_hospitals,
    }, indent=2))

    problems = self_check(out)
    if problems:
        for p in problems:
            log(f"PROBLEM {p}")
        raise SystemExit(f"self-check failed: do NOT export {out}")
    log(f"self-check passed for {len(WRITTEN)} files. Read every CSV before export.")


if __name__ == "__main__":
    main()
