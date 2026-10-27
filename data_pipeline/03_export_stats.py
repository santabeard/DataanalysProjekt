#!/usr/bin/env python3
"""Aggregated statistics from the built dataset, for synthesising data off Bianca.

Reads the output of 02_build.py and writes only aggregated tables (CSV) that
04_synthesize.py can turn into synthetic data on your own machine. Nothing in
the output is a row about a person.

Disclosure rules (change only with your supervisor's agreement):
  - Every group must hold at least MIN_CELL distinct PERSONS (not just rows), else it is
    removed. Counts that remain are rounded to the nearest ROUND_TO.
    Rounding makes it hard to recover a removed cell by subtracting from a total.
  - Percentiles only for groups with at least MIN_N_QUANTILE rows, and the 1st/99th
    only from MIN_N_TAIL rows. Min and max are never written: they are one real value.
  - Rates (null rate, share of visits ...) only for groups of at least MIN_N_QUANTILE.
  - Dates appear only as year, hour of day or weekday.
  - No IDs and no free text (ed_presented_complaint_nonstd is left out).
  - Distributions per person (visits per patient ...) are top-coded at TOP_CODE.
At the end every CSV is checked against these rules; the script fails if one breaks.

The automatic check does not replace reading the files. Look through every CSV
before you move the folder off Bianca, and follow your project's export routine.

Reads from and writes under $SWED_OUT (default /home/theon/Desktop/data_std).

    python 03_export_stats.py                         # reads built_subset/
    python 03_export_stats.py --built /home/theon/Desktop/data_std/built_clean_parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ORIGINAL = Path("/cygnus/proj/nobackup/swed_db/clean_parquet")  # never written to
OUT_ROOT = Path(os.environ.get("SWED_OUT", "/home/theon/Desktop/data_std"))

MIN_CELL = 10
ROUND_TO = 5
MIN_N_QUANTILE = 50
MIN_N_TAIL = 1000
TOP_CODE = 10
QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
QS_TAIL = [0.01, 0.99]

ED_CATEGORICAL = ["ed_arrival_type", "care_unit", "ed_exit_to", "ed_exit_to_care_unit", "ed_priority_in",
                  "ed_priority_out", "ed_presented_complaint_std", "ed_inexact_time", "is_male"]
ED_NUMERIC = ["ed_age", "ed_time_to_doctor_minutes", "ed_time_total_minutes"]
P = "lopnr_swed"  # person column
NEVER_EXPORT = {"lopnr_swed", "ed_visit_id", "ed_presented_complaint_nonstd", "source_row"}

WRITTEN: list[str] = []


def refuse_to_write_into(out: Path, *protected: Path) -> None:
    """Outputs must never land in, or overwrite anything in, the source data folders."""
    o = out.expanduser().resolve()
    for p in protected:
        p = p.expanduser().resolve()
        if o == p or p in o.parents:
            sys.exit(f"refusing to write to {out}: it is inside the source data folder {p}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# Disclosure-safe building blocks
# =============================================================================

def _round(n) -> int:
    return int(round(n / ROUND_TO) * ROUND_TO)


def safe_counts(df: pd.DataFrame, by: list[str], person: str | None = None) -> pd.DataFrame:
    """Rows (and persons) per group. A group is kept only if it holds at least MIN_CELL
    distinct persons; without a person column every row is taken to be a different person."""
    g = df.groupby(by, dropna=False, observed=True)
    c = g.size().rename("n").to_frame()
    c["persons"] = g[person].nunique() if person else c["n"]
    c = c[c["persons"] >= MIN_CELL].reset_index()
    c["n"], c["persons"] = c["n"].map(_round), c["persons"].map(_round)
    c = c[c["persons"] >= MIN_CELL]
    if not person:
        c = c.drop(columns="persons")
    return c.sort_values(by).reset_index(drop=True)


def safe_quantiles(df: pd.DataFrame, by: list[str], col: str, person: str | None = None,
                   decimals: int = 1) -> pd.DataFrame:
    """Percentiles and null rate of col per group. Needs MIN_CELL persons and MIN_N_QUANTILE
    values; the 1st/99th also need MIN_N_TAIL values from MIN_N_QUANTILE persons."""
    rows = []
    for key, g in df.groupby(by, dropna=False, observed=True):
        persons = g[person].nunique() if person else len(g)
        if persons < MIN_CELL or len(g) < MIN_N_QUANTILE:
            continue
        v = g[col].dropna().astype("float64")
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        r["n"] = _round(len(g))
        r["null_rate"] = round(1 - len(v) / len(g), 3)
        if len(v) >= MIN_N_QUANTILE:
            tails = len(v) >= MIN_N_TAIL and persons >= MIN_N_QUANTILE
            qs = QS + (QS_TAIL if tails else [])
            for q, x in zip(qs, np.quantile(v, qs)):
                r[f"q{int(round(q * 100)):02d}"] = round(float(x), decimals)
        rows.append(r)
    cols = by + ["n", "null_rate"] + [f"q{int(round(q * 100)):02d}" for q in sorted(QS + QS_TAIL)]
    return pd.DataFrame(rows).reindex(columns=cols)


def safe_rate(df: pd.DataFrame, by: list[str], flag: str, name: str, person: str | None = None) -> pd.DataFrame:
    g = df.groupby(by, dropna=False, observed=True)
    r = g[flag].agg(["size", "mean"])
    r["persons"] = g[person].nunique() if person else r["size"]
    r = r[(r["size"] >= MIN_N_QUANTILE) & (r["persons"] >= MIN_CELL)].reset_index()
    r["n"] = r["size"].map(_round)
    return r[by + ["n", "mean"]].rename(columns={"mean": name}).round({name: 3})


def top_coded(s: pd.Series) -> pd.Series:
    return s.clip(upper=TOP_CODE).astype("int64").astype(str).replace({str(TOP_CODE): f"{TOP_CODE}+"})


def save(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / f"{name}.csv", index=False)
    WRITTEN.append(name)
    log(f"  {name}.csv: {len(df)} rows")


# =============================================================================
# Per table
# =============================================================================

def patients_stats(b: Path, out: Path) -> None:
    p = pd.read_parquet(b / "patients.parquet")
    ed = pd.read_parquet(b / "ed_visits.parquet", columns=["lopnr_swed", "ed_arrival_datetime", "ed_age"])
    first = ed.sort_values("ed_arrival_datetime").drop_duplicates("lopnr_swed")
    p = p.merge(first[["lopnr_swed", "ed_age"]].rename(columns={"ed_age": "age_first_visit"}), on="lopnr_swed", how="left")
    p["n_ed_visits"] = top_coded(p["n_ed_visits"])
    p["died"] = p["date_death_minimp"].notna()
    p["all"] = "all"
    save(safe_counts(p, ["is_male"]), out, "patients_sex")
    save(safe_counts(p, ["n_ed_visits"]), out, "patients_n_ed_visits")
    save(safe_quantiles(p, ["is_male"], "age_first_visit"), out, "patients_age_first_visit")
    save(safe_rate(p, ["all"], "died", "share_died"), out, "patients_died")


def ed_stats(b: Path, out: Path) -> None:
    ed = pd.read_parquet(b / "ed_visits.parquet")
    ed = ed.drop(columns=[c for c in NEVER_EXPORT if c in ed and c != P])
    t = ed["ed_arrival_datetime"].dt.tz_convert("Europe/Stockholm")
    ed["year"], ed["hour"], ed["weekday"] = t.dt.year, t.dt.hour, t.dt.day_of_week
    save(safe_counts(ed, ["ed_hospital", "year"], P), out, "ed_visits_hospital_year")
    save(safe_counts(ed, ["ed_hospital", "hour"], P), out, "ed_visits_hospital_hour")
    save(safe_counts(ed, ["ed_hospital", "weekday"], P), out, "ed_visits_hospital_weekday")

    cat = []
    for col in [c for c in ED_CATEGORICAL if c in ed]:
        c = safe_counts(ed.assign(value=ed[col].astype("string")), ["ed_hospital", "value"], P)
        c.insert(1, "column", col)
        cat.append(c)
    save(pd.concat(cat, ignore_index=True), out, "ed_visits_categorical")

    num = []
    for col in [c for c in ED_NUMERIC if c in ed]:
        q = safe_quantiles(ed, ["ed_hospital"], col, P)
        q.insert(1, "column", col)
        num.append(q)
    save(pd.concat(num, ignore_index=True), out, "ed_visits_numeric")


def vitals_stats(b: Path, out: Path) -> None:
    v = pd.read_parquet(b / "vitals.parquet", columns=["lopnr_swed", "ed_visit_id", "source", "from", "datetime_analysis", "term_name",
                                                       "concept", "value", "unit_original", "unit", "unit_status", "implausible"])
    ed = pd.read_parquet(b / "ed_visits.parquet", columns=["ed_visit_id", "ed_hospital", "ed_arrival_datetime"])

    # Units per hospital (the supervisor's question) and how each unit was decided.
    save(safe_counts(v.assign(unit_original=v["unit_original"].fillna("<missing>")),
                     ["source", "from", "concept", "term_name", "unit_original", "unit", "unit_status"], P), out, "vitals_units")
    ok = v[~v["implausible"]]
    save(safe_quantiles(ok, ["source", "from", "concept"], "value", P), out, "vitals_values")
    save(safe_rate(v, ["source", "from", "concept"], "implausible", "share_implausible", P), out, "vitals_implausible")

    linked = v.dropna(subset=["ed_visit_id"]).merge(ed, on="ed_visit_id", how="left")
    linked["minutes_from_arrival"] = (linked["datetime_analysis"] - linked["ed_arrival_datetime"]).dt.total_seconds() / 60
    save(safe_quantiles(linked, ["ed_hospital", "concept"], "minutes_from_arrival", P, decimals=0), out, "vitals_timing")
    save(safe_counts(linked, ["ed_hospital", "source", "from", "concept"], P), out, "vitals_source_by_hospital")
    save(safe_rate(v.assign(linked=v["ed_visit_id"].notna()), ["source", "from"], "linked", "share_linked_to_visit", P),
         out, "vitals_linked")

    # Per visit: does the visit have this vital at all, and how many measurements?
    per = linked.groupby(["ed_visit_id", "ed_hospital", "concept"]).agg(k=(P, "size"), **{P: (P, "first")}).reset_index()
    visits = ed.groupby("ed_hospital")["ed_visit_id"].nunique().rename("visits")
    share = (per.groupby(["ed_hospital", "concept"]).agg(visits_with=("ed_visit_id", "size"), persons_with=(P, "nunique"))
             .reset_index().merge(visits, on="ed_hospital"))
    share = share[(share["persons_with"] >= MIN_CELL) & (share["visits"] >= MIN_N_QUANTILE)]
    share["share_visits_with"] = (share["visits_with"] / share["visits"]).round(3)
    save(share[["ed_hospital", "concept", "share_visits_with"]], out, "vitals_share_of_visits")
    save(safe_counts(per.assign(per_visit=top_coded(per["k"])), ["ed_hospital", "concept", "per_visit"], P), out,
         "vitals_count_per_visit")


def labs_stats(b: Path, out: Path) -> None:
    lab = pd.read_parquet(b / "lab_meas.parquet", columns=[P, "ed_visit_id", "source", "from", "datetime_analysis", "term_name", "snomed_id",
                                                           "unit", "unit_status", "value_num", "lt_flag", "gt_flag"])
    lab["unit"] = lab["unit"].fillna("<missing>")
    lab["snomed_id"] = lab["snomed_id"].fillna("<unmapped>")
    key = ["source", "from", "term_name", "snomed_id", "unit"]
    save(safe_counts(lab, key + ["unit_status"], P), out, "labs_terms")
    save(safe_quantiles(lab, key, "value_num", P, decimals=2), out, "labs_values")
    lab["linked"] = lab["ed_visit_id"].notna()
    save(safe_rate(lab, ["source", "from"], "linked", "share_linked_to_visit", P), out, "labs_linked")

    ed = pd.read_parquet(b / "ed_visits.parquet", columns=["ed_visit_id", "ed_hospital", "ed_arrival_datetime"])
    linked = lab[lab["linked"]].merge(ed, on="ed_visit_id", how="left")
    linked["minutes_from_arrival"] = (linked["datetime_analysis"] - linked["ed_arrival_datetime"]).dt.total_seconds() / 60
    save(safe_counts(linked, ["ed_hospital", "source", "from"], P), out, "labs_source_by_hospital")
    save(safe_quantiles(linked, ["ed_hospital", "source"], "minutes_from_arrival", P, decimals=0), out, "labs_timing")
    per = linked.groupby(["ed_visit_id", "ed_hospital", "source"]).agg(k=(P, "size"), **{P: (P, "first")}).reset_index()
    save(safe_quantiles(per, ["ed_hospital", "source"], "k", P, decimals=0), out, "labs_rows_per_visit")
    visits = ed.groupby("ed_hospital")["ed_visit_id"].nunique().rename("visits")
    share = (per.groupby(["ed_hospital", "source"]).agg(visits_with=("ed_visit_id", "size"), persons_with=(P, "nunique"))
             .reset_index().merge(visits, on="ed_hospital"))
    share = share[(share["persons_with"] >= MIN_CELL) & (share["visits"] >= MIN_N_QUANTILE)]
    share["share_visits_with"] = (share["visits_with"] / share["visits"]).round(3)
    save(share[["ed_hospital", "source", "share_visits_with"]], out, "labs_share_of_visits")


def diagnoses_stats(b: Path, out: Path) -> None:
    if not (b / "diagnoses.parquet").exists():
        log("  skip diagnoses: not built")
        return
    d = pd.read_parquet(b / "diagnoses.parquet", columns=["lopnr_swed", "INDATUM", "care_level", "diag_pos", "icd10_3"])
    d["diag_main"] = d["diag_pos"].astype("string").str.lower().eq("hdia")
    d["year"] = pd.to_datetime(d["INDATUM"]).dt.year
    save(safe_counts(d, ["care_level", "diag_main", "icd10_3"], P), out, "diagnoses_icd10_3")
    save(safe_counts(d, ["care_level", "year"], P), out, "diagnoses_year")
    per = d.groupby("lopnr_swed").size()
    save(safe_counts(pd.DataFrame({"per_patient": top_coded(per)}), ["per_patient"]), out, "diagnoses_per_patient")
    pts = pd.read_parquet(b / "patients.parquet", columns=[P])
    pts["has_diagnosis"] = pts[P].isin(per.index)
    pts["all"] = "all"
    save(safe_rate(pts, ["all"], "has_diagnosis", "share_with_diagnosis"), out, "diagnoses_share_patients")


def deaths_stats(b: Path, out: Path) -> None:
    if not (b / "deaths.parquet").exists():
        log("  skip deaths: not built")
        return
    dr = pd.read_parquet(b / "deaths.parquet")
    dr["year"] = pd.to_datetime(dr["date_death_minimp"]).dt.year
    dr["ulorsak_3"] = dr["ULORSAK"].astype("string").str.upper().str[:3]
    for col, name in [("ulorsak_3", "deaths_underlying_cause_3"), ("year", "deaths_year"),
                      ("DODSPL", "deaths_place"), ("DBGRUND1", "deaths_autopsy"), ("inexact_death_date", "deaths_inexact_date")]:
        if col in dr:
            save(safe_counts(dr.assign(**{col: dr[col].astype("string")}), [col]), out, name)
    ed = pd.read_parquet(b / "ed_visits.parquet", columns=["lopnr_swed", "ed_arrival_datetime"])
    last = ed.groupby("lopnr_swed")["ed_arrival_datetime"].max().dt.tz_localize(None).dt.normalize()
    dr = dr.merge(last.rename("last_visit").reset_index(), on="lopnr_swed", how="left")
    dr["days_after_last_visit"] = (pd.to_datetime(dr["date_death_minimp"]) - dr["last_visit"]).dt.days
    dr["all"] = "all"
    save(safe_quantiles(dr, ["all"], "days_after_last_visit", decimals=0), out, "deaths_days_after_last_visit")


# =============================================================================
# Self-check
# =============================================================================

def self_check(out: Path) -> list[str]:
    problems = []
    for name in WRITTEN:
        df = pd.read_csv(out / f"{name}.csv")
        bad_cols = NEVER_EXPORT & set(df.columns)
        if bad_cols:
            problems.append(f"{name}: forbidden column(s) {sorted(bad_cols)}")
        for col in [c for c in ["n", "persons"] if c in df]:
            if len(df) and ((df[col] < MIN_CELL) | (df[col] % ROUND_TO != 0)).any():
                problems.append(f"{name}: {col} below {MIN_CELL} or not rounded to {ROUND_TO}")
        if any(c in df.columns for c in ["min", "max"]):
            problems.append(f"{name}: has min/max")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--built", type=Path, default=OUT_ROOT / "built_subset", help="output folder of 02_build.py")
    ap.add_argument("--out", type=Path, help="default: $SWED_OUT/export_stats_<built folder name>")
    args = ap.parse_args()
    out = args.out or OUT_ROOT / f"export_stats_{args.built.name}"
    refuse_to_write_into(out, args.built, ORIGINAL)
    out.mkdir(parents=True, exist_ok=True)
    log(f"built: {args.built}")
    log(f"out:   {out}")

    for step in [patients_stats, ed_stats, vitals_stats, labs_stats, diagnoses_stats, deaths_stats]:
        log(step.__name__)
        step(args.built, out)

    rules = pd.DataFrame({"rule": ["MIN_CELL", "ROUND_TO", "MIN_N_QUANTILE", "MIN_N_TAIL", "TOP_CODE", "built_from"],
                          "value": [MIN_CELL, ROUND_TO, MIN_N_QUANTILE, MIN_N_TAIL, TOP_CODE, args.built.name]})
    save(rules, out, "rules")

    problems = self_check(out)
    if problems:
        for p in problems:
            log(f"PROBLEM {p}")
        raise SystemExit(f"self-check failed: do NOT export {out}")
    log(f"self-check passed for {len(WRITTEN)} files. Read every CSV yourself before export.")


if __name__ == "__main__":
    main()
