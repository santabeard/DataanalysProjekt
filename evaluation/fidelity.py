#!/usr/bin/env python3
"""Compare a synthetic build with the real build on fidelity, utility and privacy."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.cohort import SEED, Cohort, Split, design_from_raw, load_cohort, round5, vital_mean_sd
from evaluation.metrics import bootstrap_ci, metrics_frame
from evaluation.models import fit_all

TABLES = ["patients", "ed_visits", "vitals", "lab_meas", "diagnoses", "deaths"]
MODEL_ORDER = ["M_ref", "M_base", "A", "B", "C", "M_direct"]


def _read(directory: Path, name: str) -> pd.DataFrame | None:
    path = directory / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def _ed(directory: Path) -> pd.DataFrame:
    df = _read(directory, "ed_visits")
    if df is None:
        return pd.DataFrame()
    cols = ["ed_visit_id", "lopnr_swed", "ed_hospital", "ed_arrival_datetime", "ed_age", "is_male",
            "ed_presented_complaint_std", "ed_priority_in", "ed_arrival_type", "ed_exit_to"]
    df = df[[c for c in cols if c in df]].copy()
    if "ed_age" in df:
        df["age"] = df["ed_age"].astype(int)
    if "is_male" in df:
        df["sex"] = df["is_male"].astype(int)
    if "ed_presented_complaint_std" in df:
        df["complaint"] = df["ed_presented_complaint_std"].str.extract(r"\[(\d+)\]", expand=False).fillna("unknown")
    if "ed_arrival_datetime" in df:
        local = pd.to_datetime(df["ed_arrival_datetime"], utc=True).dt.tz_convert("Europe/Stockholm")
        df["month"] = local.dt.month
    return df


def _distribution(name: str, group: str, real: pd.Series, synth: pd.Series, top: int = 20) -> tuple[dict, pd.DataFrame]:
    a, b = real.dropna().astype(str), synth.dropna().astype(str)
    ra, rb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    keys = list(dict.fromkeys([*ra.nlargest(top).index, *rb.nlargest(top).index]))
    table = pd.DataFrame({"real_share": ra.reindex(keys).fillna(0), "synth_share": rb.reindex(keys).fillna(0)})
    table["abs_diff"] = (table["real_share"] - table["synth_share"]).abs()
    return {"check": name, "group": group, "tvd": round(float(table["abs_diff"].sum() / 2), 4),
            "real_n": int(len(a)), "synth_n": int(len(b))}, table.reset_index(names="value")


def _vital_wide(directory: Path) -> pd.DataFrame:
    v = _read(directory, "vitals")
    if v is None or v.empty:
        return pd.DataFrame()
    v = v[v["ed_visit_id"].notna() & v["value"].notna() & ~v["implausible"].astype("boolean").fillna(False)]
    return v.sort_values("datetime_analysis").groupby(["ed_visit_id", "concept"])["value"].first().unstack("concept")


def fidelity_tables(real_dir: Path, synth_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, detail = [], {}
    for name in TABLES:
        a, b = _read(real_dir, name), _read(synth_dir, name)
        rows.append({"check": "row_count", "group": name,
                     "real": round5(0 if a is None else len(a)), "synth": round5(0 if b is None else len(b))})

    re, se = _ed(real_dir), _ed(synth_dir)
    for col in ["ed_hospital", "complaint", "month", "ed_priority_in", "ed_arrival_type", "ed_exit_to"]:
        if col in re and col in se:
            summary, table = _distribution("category", col, re[col], se[col])
            rows.append(summary)
            detail[f"dist_{col}"] = table
    if "age" in re and "age" in se:
        rows.append({"check": "age_mean", "group": "ed", "real": round(re["age"].mean(), 2),
                     "synth": round(se["age"].mean(), 2)})
        rows.append({"check": "age_sd", "group": "ed", "real": round(re["age"].std(), 2),
                     "synth": round(se["age"].std(), 2)})

    rv, sv = _vital_wide(real_dir), _vital_wide(synth_dir)
    concepts = sorted(set(rv.columns) & set(sv.columns))
    if concepts:
        qrows = []
        for c in concepts:
            for q in (0.25, 0.5, 0.75):
                qrows.append({"check": "vital_quantile", "group": f"{c}_q{int(q*100)}",
                              "real": round(float(rv[c].quantile(q)), 2), "synth": round(float(sv[c].quantile(q)), 2)})
        rows.extend(qrows)
        detail["vital_observed"] = pd.DataFrame([
            {"concept": c, "real_share": round(float(rv[c].notna().mean()), 3),
             "synth_share": round(float(sv[c].notna().mean()), 3)} for c in concepts
        ])

        common = rv[concepts].dropna().shape[0]
        cors = []
        for i, a in enumerate(concepts):
            for b in concepts[i + 1:]:
                x, y = rv[[a, b]].dropna(), sv[[a, b]].dropna()
                if len(x) < 20 or len(y) < 20:
                    continue
                rx, sy = x.corr(method="spearman").iloc[0, 1], y.corr(method="spearman").iloc[0, 1]
                if np.isfinite(rx) and np.isfinite(sy):
                    cors.append({"pair": f"{a}-{b}", "real_rho": round(float(rx), 3),
                                 "synth_rho": round(float(sy), 3), "abs_diff": round(abs(rx - sy), 3)})
        if cors:
            detail["vital_correlations"] = pd.DataFrame(cors)
            rows.append({"check": "vitals_mean_abs_correlation_diff", "group": "all",
                         "real": round(float(np.mean([c["real_rho"] for c in cors])), 3),
                         "synth": round(float(np.mean([c["synth_rho"] for c in cors])), 3)})

    for name, col in [("lab_meas", "term_name"), ("diagnoses", "icd10_3")]:
        a, b = _read(real_dir, name), _read(synth_dir, name)
        if a is not None and b is not None and col in a and col in b:
            summary, table = _distribution("category", name, a[col], b[col])
            rows.append(summary)
            detail[f"dist_{name}"] = table
    return pd.DataFrame(rows), detail


def privacy_table(real_dir: Path, synth_dir: Path) -> pd.DataFrame:
    re, se = _ed(real_dir), _ed(synth_dir)
    keys = [c for c in ["ed_hospital", "age", "sex", "complaint", "month"] if c in re and c in se]
    if not keys:
        return pd.DataFrame(columns=["check", "group", "real", "synth"])
    real_keys = set(map(tuple, re[keys].astype(str).to_numpy()))
    synth_keys = list(map(tuple, se[keys].astype(str).to_numpy()))
    matches = sum(k in real_keys for k in synth_keys)
    return pd.DataFrame([
        {"check": "exact_key_match", "group": "+".join(keys), "real": round5(len(re)), "synth": round5(len(se))},
        {"check": "exact_key_match_share", "group": "+".join(keys), "real": 0.0,
         "synth": round(matches / len(synth_keys), 4) if synth_keys else 0.0},
    ])


def _rebase_test(test: Split, reference: Cohort, complaints: list[str]) -> Split:
    mean, sd = vital_mean_sd(reference)
    V, R = design_from_raw(test.raw, mean, sd)
    age = reference.scaling["age"]
    X = np.empty((len(test.visits), 2 + len(complaints)), dtype=float)
    X[:, 0] = (test.visits["age"].to_numpy() - age["mean"]) / age["sd"]
    X[:, 1] = test.visits["male"].to_numpy()
    complaint = test.visits["complaint"].to_numpy()
    for j, code in enumerate(complaints, start=2):
        X[:, j] = (complaint == code).astype(float)
    return Split(visits=test.visits, X_base=X, V=V, R=R, y=test.y, raw=test.raw)


def utility_table(real_dir: Path, synth_dir: Path, n_boot: int, seed: int) -> pd.DataFrame:
    real = load_cohort(real_dir, seed=seed)
    synth = load_cohort(synth_dir, seed=seed)
    cross = Cohort(
        splits={"old": synth["old"], "new": synth["new"],
                "test": _rebase_test(real["test"], synth, synth.complaints)},
        base_feature_names=synth.base_feature_names,
        vital_feature_names=synth.vital_feature_names,
        vital_names=synth.vital_names,
        complaints=synth.complaints,
        scaling=synth.scaling,
    )
    rows = []
    for source, cohort in (("train_synthetic_test_real", cross), ("train_real_test_real", real)):
        fitted = fit_all(cohort)
        split = cohort["test"]
        for name in MODEL_ORDER:
            p = fitted[name].predict(split)
            res = bootstrap_ci(split.y, p, split.visits["lopnr_swed"].to_numpy(),
                               groups=split.visits["ed_hospital"].to_numpy(), B=n_boot, seed=seed)
            frame = metrics_frame(res, name, len(split.y), int(split.y.sum()))
            frame.insert(0, "experiment", source)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def write_report(out: Path, summary: pd.DataFrame, utility: pd.DataFrame, privacy: pd.DataFrame) -> None:
    piv = utility.pivot_table(index=["model", "metric"], columns="experiment", values="value")
    lines = ["# Synthetic fidelity report", "", "## Fidelity", "", summary.to_markdown(index=False), "",
             "## Utility (train-synthetic → test-real is the arbiter)", "", piv.round(4).to_markdown(), "",
             "## Privacy", "", privacy.to_markdown(index=False), ""]
    (out / "report.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--synthetic", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-utility", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary, detail = fidelity_tables(args.real, args.synthetic)
    summary.to_csv(args.out / "fidelity_summary.csv", index=False)
    for name, df in detail.items():
        df.to_csv(args.out / f"{name}.csv", index=False)
    privacy = privacy_table(args.real, args.synthetic)
    privacy.to_csv(args.out / "privacy.csv", index=False)
    utility = pd.DataFrame()
    if not args.skip_utility:
        utility = utility_table(args.real, args.synthetic, args.n_boot, args.seed)
        utility.to_csv(args.out / "utility.csv", index=False)
    write_report(args.out, summary, utility, privacy)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "real": str(args.real), "synthetic": str(args.synthetic),
        "python": sys.version, "platform": platform.platform(),
        "n_boot": args.n_boot, "seed": args.seed, "skip_utility": args.skip_utility,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))
    if not utility.empty:
        print(utility[utility["metric"] == "log_loss"].to_string(index=False))


if __name__ == "__main__":
    main()
