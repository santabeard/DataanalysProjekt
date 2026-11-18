#!/usr/bin/env python3
"""Disclosure-safe joint and conditional statistics for the synthetic pipeline.

Runs on Bianca after 03_export_stats.py and 03b_export_missingness.py. It writes
only aggregated, rounded CSVs that 04_synthesize.py can use to generate linked
variables instead of independent marginal draws.

    python 03c_export_joint_stats.py --built built_clean_parquet --out export_stats_built
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(ROOT / "Preprocessing"), str(ROOT / "preprocessing")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

_spec = importlib.util.spec_from_file_location("export_stats", HERE / "03_export_stats.py")
export_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_stats)

MIN_CELL = export_stats.MIN_CELL
ROUND_TO = export_stats.ROUND_TO
MIN_N_QUANTILE = export_stats.MIN_N_QUANTILE
MIN_N_TAIL = export_stats.MIN_N_TAIL
ORIGINAL = export_stats.ORIGINAL
P = export_stats.P
NEVER_EXPORT = export_stats.NEVER_EXPORT
WRITTEN: list[str] = []
VITAL_VARIABLES = ["sbp", "pulse", "rr", "spo2", "temp"]
VITAL_CONCEPTS = {"sbp": "systolic_bp", "pulse": "pulse_rate", "rr": "resp_rate", "spo2": "spo2", "temp": "body_temp"}
CONDITION_LEVELS = [
    ["domain", "group", "source", "variable", "age_band", "sex", "complaint", "outcome"],
    ["domain", "group", "source", "variable", "age_band", "sex", "outcome"],
    ["domain", "group", "source", "variable", "age_band", "sex"],
    ["domain", "group", "source", "variable", "age_band"],
    ["domain", "group", "source", "variable"],
    ["domain", "source", "variable", "age_band", "sex", "outcome"],
    ["domain", "source", "variable", "age_band", "sex"],
    ["domain", "source", "variable", "age_band"],
    ["domain", "source", "variable"],
]
RATE_LEVELS = [
    ["group", "age_band", "sex", "complaint"],
    ["group", "age_band", "sex"],
    ["group", "age_band"],
    ["group", "sex", "complaint"],
    ["group"],
    ["age_band", "sex", "complaint"],
    ["age_band", "sex"],
    ["age_band"],
    ["all"],
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / f"{name}.csv", index=False)
    WRITTEN.append(name)
    log(f"  {name}.csv: {len(df)} rows")


def round_floats(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes(include=[np.number]).columns:
        if c not in ("n", "persons", "year", "hour", "weekday", "month", "is_male", "sex", "outcome"):
            df[c] = df[c].round(decimals)
    return df


def _bands(age: pd.Series) -> pd.Series:
    v = pd.to_numeric(age, errors="coerce")
    out = pd.Series("all", index=age.index, dtype=object)
    out[v < 40] = "18_39"
    out[(v >= 40) & (v < 60)] = "40_59"
    out[(v >= 60) & (v < 80)] = "60_79"
    out[v >= 80] = "80+"
    return out


def _string(s: pd.Series, default: str = "all") -> pd.Series:
    return s.astype("string").fillna(default).astype(str).replace({"": default, "<NA>": default})


def _visit_context(built: Path) -> pd.DataFrame:
    ed = pd.read_parquet(
        built / "ed_visits.parquet",
        columns=["ed_visit_id", P, "ed_hospital", "ed_arrival_datetime", "ed_age", "is_male",
                 "ed_presented_complaint_std", "ed_priority_in", "ed_arrival_type", "ed_exit_to"],
    )
    ed["complaint"] = _string(ed["ed_presented_complaint_std"].astype("string").str.extract(r"\[(\d+)\]", expand=False))
    ed["age_band"] = _bands(ed["ed_age"])
    ed["sex"] = _string(ed["is_male"].astype("Int64").astype("string"))
    deaths = pd.read_parquet(built / "deaths.parquet", columns=[P, "date_death_minimp"])
    death = pd.to_datetime(ed[P].map(deaths.set_index(P)["date_death_minimp"]))
    arrival = ed["ed_arrival_datetime"].dt.tz_convert("Europe/Stockholm").dt.tz_localize(None).dt.normalize()
    days = (death - arrival).dt.days
    ed["died30"] = days.between(0, 30).astype(int)
    ed["outcome"] = _string(ed["died30"].astype("Int64").astype("string"))
    return ed


def _first_values(df: pd.DataFrame, key: list[str], value: str) -> pd.DataFrame:
    return df.sort_values("datetime_analysis").drop_duplicates(key).copy()


def _value_frames(built: Path, ed: pd.DataFrame, lab_concepts: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    v = pd.read_parquet(
        built / "vitals.parquet",
        columns=[P, "ed_visit_id", "concept", "value", "implausible", "datetime_analysis"],
    )
    v = v[v["ed_visit_id"].notna() & v["value"].notna() & ~v["implausible"] & v["concept"].isin(VITAL_CONCEPTS.values())]
    v["variable"] = v["concept"].map({c: k for k, c in VITAL_CONCEPTS.items()})
    v = _first_values(v, ["ed_visit_id", "variable"], "value")
    frame = v.merge(ed[["ed_visit_id", "ed_hospital", "age_band", "sex", "complaint", "outcome"]], on="ed_visit_id")
    frames["vital"] = pd.DataFrame({
        "domain": "vital", "group": frame["ed_hospital"].astype(str), "source": "",
        "variable": frame["variable"].astype(str), "age_band": frame["age_band"].astype(str),
        "sex": frame["sex"], "complaint": frame["complaint"], "outcome": frame["outcome"],
        "value": pd.to_numeric(frame["value"], errors="coerce"), P: frame[P],
    }).dropna(subset=["value"])

    if lab_concepts:
        lab = pd.read_parquet(
            built / "lab_meas.parquet",
            columns=[P, "ed_visit_id", "source", "term_name", "value_num", "datetime_analysis"],
        )
        pairs = {(s, c) for s, concepts in lab_concepts.items() for c in concepts}
        lab = lab[lab["ed_visit_id"].notna() & lab["value_num"].notna()]
        lab = lab[[(s, c) in pairs for s, c in zip(lab["source"], lab["term_name"])]]
        if len(lab):
            lab["variable"] = "lab_" + lab["term_name"].astype(str)
            lab = _first_values(lab, ["ed_visit_id", "source", "variable"], "value_num")
            lf = lab.merge(ed[["ed_visit_id", "ed_hospital", "age_band", "sex", "complaint", "outcome"]], on="ed_visit_id")
            frames["lab"] = pd.DataFrame({
                "domain": "lab", "group": lf["ed_hospital"].astype(str), "source": lf["source"].astype(str),
                "variable": lf["variable"].astype(str), "age_band": lf["age_band"].astype(str),
                "sex": lf["sex"], "complaint": lf["complaint"], "outcome": lf["outcome"],
                "value": pd.to_numeric(lf["value_num"], errors="coerce"), P: lf[P],
            }).dropna(subset=["value"])
    return frames


def export_joint_categorical(ed: pd.DataFrame, out: Path) -> None:
    pairs = [
        ("ed_presented_complaint_std", "ed_priority_in"),
        ("ed_presented_complaint_std", "ed_arrival_type"),
        ("ed_presented_complaint_std", "ed_exit_to"),
        ("ed_age_band", "ed_priority_in"),
        ("is_male", "ed_priority_in"),
        ("ed_priority_in", "ed_exit_to"),
    ]
    ed = ed.copy()
    ed["ed_age_band"] = _bands(ed["ed_age"])
    rows = []
    for a, b in pairs:
        for hospital in [c for c in ed["ed_hospital"].astype(str).unique()] + ["all"]:
            d = ed if hospital == "all" else ed[ed["ed_hospital"].astype(str) == hospital]
            d = d[d[a].notna() & d[b].notna()]
            d = pd.DataFrame({
                "ed_hospital": hospital if hospital == "all" else d["ed_hospital"].astype(str),
                "column_a": a, "value_a": d[a].astype(str), "column_b": b, "value_b": d[b].astype(str), P: d[P],
            })
            rows.append(export_stats.safe_counts(d, ["ed_hospital", "column_a", "value_a", "column_b", "value_b"], P))
    save(round_floats(pd.concat(rows, ignore_index=True).drop_duplicates(
        ["ed_hospital", "column_a", "value_a", "column_b", "value_b"])), out, "joint_categorical")


def _normal_scores(v: pd.Series) -> np.ndarray:
    r = pd.Series(v).rank(method="average").to_numpy(dtype=float)
    p = np.clip((r - 0.5) / len(r), 1e-6, 1 - 1e-6)
    return norm.ppf(p)


def export_correlations(frames: dict[str, pd.DataFrame], out: Path) -> None:
    rows = []
    for domain, base in frames.items():
        for hospital in [c for c in base["group"].unique()] + ["all"]:
            d = base if hospital == "all" else base[base["group"] == hospital]
            d = d.assign(group=hospital)
            for source, s in d.groupby("source", dropna=False):
                variables = sorted(s["variable"].astype(str).unique())
                for a, b in itertools.combinations(variables, 2):
                    pair = s[s["variable"].isin([a, b])].pivot_table(index=P, columns="variable", values="value", aggfunc="first")
                    if a not in pair or b not in pair:
                        continue
                    pair = pair[[a, b]].dropna()
                    persons = len(pair)
                    if persons < MIN_CELL or len(pair) < MIN_N_QUANTILE:
                        continue
                    za, zb = _normal_scores(pair[a]), _normal_scores(pair[b])
                    rho = float(np.corrcoef(za, zb)[0, 1])
                    if not np.isfinite(rho):
                        continue
                    rows.append({"domain": domain, "group": hospital, "source": str(source),
                                 "var_a": a, "var_b": b, "n": export_stats._round(len(pair)),
                                 "persons": export_stats._round(persons), "rho": round(rho, 4)})
    df = pd.DataFrame(rows, columns=["domain", "group", "source", "var_a", "var_b", "n", "persons", "rho"])
    save(df, out, "joint_correlation")


def export_conditional_quantiles(frames: dict[str, pd.DataFrame], out: Path) -> None:
    rows = []
    for base in frames.values():
        for by in CONDITION_LEVELS:
            d = base.copy()
            for key in ("group", "source", "variable", "age_band", "sex", "complaint", "outcome"):
                if key not in by:
                    d[key] = "all"
            d = d.rename(columns={"group": "group", "value": "value"})
            q = export_stats.safe_quantiles(d, by, "value", P, decimals=3)
            if not q.empty:
                rows.append(q)
    cols = ["domain", "group", "source", "variable", "age_band", "sex", "complaint", "outcome",
            "n", "null_rate"] + [f"q{int(q * 100):02d}" for q in sorted(export_stats.QS + export_stats.QS_TAIL)]
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)
    df = round_floats(df.reindex(columns=cols)).drop_duplicates(
        ["domain", "group", "source", "variable", "age_band", "sex", "complaint", "outcome"])
    save(df, out, "conditional_quantiles")


def export_temporal(ed: pd.DataFrame, out: Path) -> None:
    ed = ed.copy()
    local = ed["ed_arrival_datetime"].dt.tz_convert("Europe/Stockholm")
    ed["month"] = local.dt.month
    save(export_stats.safe_counts(ed, ["ed_hospital", "month"], P), out, "ed_visits_hospital_month")
    ordered = ed.sort_values([P, "ed_arrival_datetime"])
    ordered["interval_days"] = ordered.groupby(P)["ed_arrival_datetime"].diff().dt.total_seconds() / 86400
    ordered = ordered[ordered["interval_days"].notna() & (ordered["interval_days"] > 0)]
    rows = []
    for hospital in [c for c in ordered["ed_hospital"].astype(str).unique()] + ["all"]:
        d = ordered if hospital == "all" else ordered[ordered["ed_hospital"].astype(str) == hospital]
        d = d.assign(ed_hospital=hospital, all="all")
        q = export_stats.safe_quantiles(d, ["ed_hospital", "all"], "interval_days", P, decimals=1)
        if not q.empty:
            rows.append(q)
    save(round_floats(pd.concat(rows, ignore_index=True)), out, "ed_visit_intervals")


def export_mortality(ed: pd.DataFrame, out: Path) -> None:
    first = ed.sort_values("ed_arrival_datetime").drop_duplicates(P).copy()
    first["age_band"] = _bands(first["ed_age"])
    first["sex"] = _string(first["is_male"].astype("Int64").astype("string"))
    first["complaint"] = _string(first["ed_presented_complaint_std"].astype("string").str.extract(r"\[(\d+)\]", expand=False))
    first["group"] = first["ed_hospital"].astype(str)
    first["all"] = "all"
    rows = []
    for by in RATE_LEVELS:
        d = first.copy()
        for key in ("group", "age_band", "sex", "complaint"):
            if key not in by:
                d[key] = "all"
        r = export_stats.safe_rate(d, by, "died30", "died30_rate", P)
        if not r.empty:
            rows.append(r)
    cols = ["group", "age_band", "sex", "complaint", "all", "n", "died30_rate"]
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)
    df = round_floats(df.reindex(columns=cols)).drop_duplicates(
        [c for c in cols if c in ("group", "age_band", "sex", "complaint", "all")])
    save(df, out, "mortality_rate")


def export_diagnoses(built: Path, ed: pd.DataFrame, out: Path) -> None:
    if not (built / "diagnoses.parquet").exists():
        return
    d = pd.read_parquet(built / "diagnoses.parquet", columns=[P, "care_level", "diag_pos", "icd10_3"])
    context = ed.sort_values("ed_arrival_datetime").drop_duplicates(P)[
        [P, "age_band", "sex", "complaint", "outcome"]
    ]
    d = d.merge(context, on=P, how="left")
    d["age_band"] = _string(d["age_band"])
    d["sex"] = _string(d["sex"])
    d["complaint"] = _string(d["complaint"])
    d["outcome"] = _string(d["outcome"])
    d["diag_main"] = _string(d["diag_pos"].astype("string").str.lower().eq("hdia").astype("Int64").astype("string"))
    rows = []
    levels = [
        ["age_band", "sex", "complaint", "outcome", "care_level", "diag_main", "icd10_3"],
        ["age_band", "sex", "complaint", "care_level", "diag_main", "icd10_3"],
        ["age_band", "sex", "care_level", "diag_main", "icd10_3"],
        ["age_band", "care_level", "diag_main", "icd10_3"],
        ["care_level", "diag_main", "icd10_3"],
    ]
    for by in levels:
        x = d.copy()
        for key in ("age_band", "sex", "complaint", "outcome"):
            if key not in by:
                x[key] = "all"
        rows.append(export_stats.safe_counts(x, by, P))
    cols = ["age_band", "sex", "complaint", "outcome", "care_level", "diag_main", "icd10_3", "n", "persons"]
    df = pd.concat(rows, ignore_index=True).reindex(columns=cols)
    save(round_floats(df.drop_duplicates(cols[:-2])), out, "diagnoses_conditional")


def self_check(out: Path) -> list[str]:
    problems = []
    for name in WRITTEN:
        df = pd.read_csv(out / f"{name}.csv")
        bad = NEVER_EXPORT & set(df.columns)
        if bad:
            problems.append(f"{name}: forbidden columns {sorted(bad)}")
        for c in ("n", "persons"):
            if c in df and len(df) and ((df[c] < MIN_CELL) | (df[c] % ROUND_TO != 0)).any():
                problems.append(f"{name}: {c} below {MIN_CELL} or not rounded to {ROUND_TO}")
        if {"min", "max"} & set(df.columns):
            problems.append(f"{name}: has min/max")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--built", type=Path, default=export_stats.OUT_ROOT / "built_subset")
    ap.add_argument("--out", type=Path, help="default: $SWED_OUT/export_stats_<built folder name>")
    args = ap.parse_args()
    out = args.out or export_stats.OUT_ROOT / f"export_stats_{args.built.name}"
    export_stats.refuse_to_write_into(out, args.built, ORIGINAL)
    out.mkdir(parents=True, exist_ok=True)
    log(f"built: {args.built}")
    log(f"out:   {out}")

    config_path = out / "missingness_config.json"
    lab_concepts: dict[str, list[str]] = {}
    if config_path.exists():
        lab_concepts = json.loads(config_path.read_text()).get("lab_concepts", {})
    ed = _visit_context(args.built)
    frames = _value_frames(args.built, ed, lab_concepts)
    log("joint categorical")
    export_joint_categorical(ed, out)
    log("correlations")
    export_correlations(frames, out)
    log("conditional quantiles")
    export_conditional_quantiles(frames, out)
    log("temporal")
    export_temporal(ed, out)
    log("mortality")
    export_mortality(ed, out)
    log("diagnoses")
    export_diagnoses(args.built, ed, out)
    (out / "joint_config.json").write_text(json.dumps({
        "built": str(args.built), "domain_tables": sorted(frames), "lab_concepts": lab_concepts,
        "condition_levels": CONDITION_LEVELS,
    }, indent=2))
    problems = self_check(out)
    if problems:
        for p in problems:
            log(f"PROBLEM {p}")
        raise SystemExit(f"self-check failed: do NOT export {out}")
    log(f"self-check passed for {len(WRITTEN)} files. Read every CSV before export.")


if __name__ == "__main__":
    main()
