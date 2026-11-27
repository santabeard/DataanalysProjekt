#!/usr/bin/env python3
"""Build the model-ready train_clean table from the <N>K_SUB_* subset files.

Implements the decisions in notes/ed_atlas_vitals_pipeline_discussion.tex
(Chapter 3 and Section 3.1), in this order:

  visits     swed_ed_core. Visits with no end time or with ed_inexact_time
             are dropped.
  vitals     The 6 vitals, by the vitals_snomed column subset.py added to
             both measurement tables.
             The first reading between 60 min before arrival and the visit
             end is kept. No hospital filter.
  diagnosis  The first Hdia code from 0 to 30 days after arrival (local
             Stockholm date, day 30 included).
  death      date_death_minimp from 0 to 30 days after arrival.
  diag_class NODIAG if no diagnosis. The Hdia code itself if it occurs at
             least --hdia-min-count times. Otherwise other_<ICD-10 chapter>.
  complaints One row per complaint. A visit with two complaints becomes two
             rows weighted 0.5 each; a visit with none gets complaint "none".
             Codes rarer than --complaint-min-count become "other_complaint"
             in the one-hot complaint_bucket_* columns.

    python3 build_train_clean.py
    python3 build_train_clean.py --label 50K --hdia-min-count 50
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from common import INTERIM, TRAIN, VITAL_SNOMED, guard_output, log

WINDOW_DAYS = 30
VITALS = sorted(VITAL_SNOMED.values())

ICD10_CHAPTERS = [  # WHO ICD-10 chapter ranges, by first three characters
    ("A00", "B99", "infectious"), ("C00", "D48", "neoplasm"), ("D50", "D89", "blood_immune"),
    ("E00", "E90", "endocrine_metabolic"), ("F00", "F99", "mental_behavioural"),
    ("G00", "G99", "nervous_system"), ("H00", "H59", "eye"), ("H60", "H95", "ear"),
    ("I00", "I99", "circulatory"), ("J00", "J99", "respiratory"), ("K00", "K93", "digestive"),
    ("L00", "L99", "skin"), ("M00", "M99", "musculoskeletal"), ("N00", "N99", "genitourinary"),
    ("O00", "O99", "pregnancy_childbirth"), ("P00", "P96", "perinatal"), ("Q00", "Q99", "congenital"),
    ("R00", "R99", "symptoms_signs"), ("S00", "T98", "injury_poisoning"),
    ("U00", "U99", "special_purposes"), ("V01", "Y98", "external_causes"),
    ("Z00", "Z99", "health_status_factors"),
]


def icd10_chapter(code):
    """'I219' -> 'circulatory'. Block ranges like 'C15-C26' use their start code."""
    head = code.split("-")[0].strip().upper()[:3]
    if len(head) == 3 and head[0].isalpha() and head[1:].isdigit():
        for start, end, name in ICD10_CHAPTERS:
            if start <= head <= end:
                return name
    return "unknown"


def complaint_codes(raw):
    """'x [123]+y [456]' -> ['123', '456']: the first [sctid] in each '+'-separated part."""
    return [m.group(1) for part in raw.split("+") if (m := re.search(r"\[(\d+)\]", part))]


def first_vitals(visits, meas, key, arrival, minutes):
    """Add vital_* columns: per visit and vital, the first reading in [arrival - 60 min, arrival + minutes].

    visits: ed_visit_id, key, arrival, minutes. meas: key, datetime_analysis, vital, value_num.
    Used by both the SWED and the UU build, so both apply the same window rule.
    """
    j = visits[["ed_visit_id", key, arrival, minutes]].merge(meas, on=key)
    start = j[arrival] - pd.Timedelta(minutes=60)
    end = j[arrival] + pd.to_timedelta(j[minutes], unit="m")
    j = j[(j["datetime_analysis"] >= start) & (j["datetime_analysis"] <= end)]
    first = j.sort_values("datetime_analysis").drop_duplicates(["ed_visit_id", "vital"])
    vitals = first.pivot(index="ed_visit_id", columns="vital", values="value_num").reindex(columns=VITALS)
    visits = visits.join(vitals.add_prefix("vital_"), on="ed_visit_id")
    for v in VITALS:
        log(f"  vital_{v}: {visits[f'vital_{v}'].notna().mean():.1%} of visits")
    return visits


def complaint_rows(visits, raw_col, min_count):
    """One row per complaint, plus one-hot complaint_bucket_* columns.

    raw_col holds "text [sctid]" or "a [x]+b [y]". A visit with two complaints becomes two
    rows weighted 0.5 each; a visit with none gets complaint "none". Codes rarer than
    min_count become "other_complaint" in the buckets. Shared by the SWED and UU builds.
    """
    raw = visits[raw_col]
    codes = {s: complaint_codes(s) for s in raw.dropna().unique()}
    sct1 = raw.map({s: c[0] for s, c in codes.items() if c})
    sct2 = raw.map({s: c[1] for s, c in codes.items() if len(c) > 1})
    two = sct2.notna()

    primary = visits.assign(complaint_sctid=sct1.fillna("none"), complaint_role="primary",
                            complaint_weight=1.0 - 0.5 * two)
    secondary = visits[two].assign(complaint_sctid=sct2[two], complaint_role="secondary", complaint_weight=0.5)
    rows = pd.concat([primary, secondary], ignore_index=True).sort_values(["ed_visit_id", "complaint_role"])
    log(f"complaint rows: {len(visits):,} visits, {int(two.sum()):,} with two complaints -> {len(rows):,} rows")

    counts = rows["complaint_sctid"].value_counts()
    vocab = set(counts.index[counts >= min_count]) | {"none"}
    bucket = rows["complaint_sctid"].where(rows["complaint_sctid"].isin(vocab), "other_complaint")
    rows = pd.concat([rows, pd.get_dummies(bucket, prefix="complaint_bucket")], axis=1)
    log(f"complaint vocabulary: {len(vocab) - 1:,} codes with >= {min_count} occurrences")
    return rows


def assign_diag_class(visits, min_count):
    """diag_class: NODIAG if no diagnosis, the Hdia code if it occurs >= min_count times,
    otherwise other_<ICD-10 chapter>. Needs diag_hdia_first / diag_hdia_any. Shared with UU."""
    hdia_counts = visits["diag_hdia_first"].value_counts()
    own_class = hdia_counts.index[hdia_counts >= min_count]
    chapter = visits["diag_hdia_first"].map({c: icd10_chapter(c) for c in hdia_counts.index})
    diag_class = ("other_" + chapter).where(~visits["diag_hdia_first"].isin(own_class), visits["diag_hdia_first"])
    visits["diag_class"] = diag_class.where(visits["diag_hdia_any"], "NODIAG")
    log(f"diag_class: {len(own_class):,} Hdia codes with >= {min_count} visits get their own class")
    return visits


def first_within_30d(visits, registry, date_col, key="lopnr_swed"):
    """Per visit, the earliest registry row dated 0-30 days after the arrival date (day 0 included)."""
    registry = registry.dropna(subset=[date_col]).copy()
    registry[date_col] = pd.to_datetime(registry[date_col])
    j = visits[["ed_visit_id", key, "arrival_date"]].merge(registry, on=key)
    j = j[(j[date_col] >= j["arrival_date"]) & (j[date_col] <= j["arrival_date"] + pd.Timedelta(days=WINDOW_DAYS))]
    j = j.assign(days=(j[date_col] - j["arrival_date"]).dt.days)
    return j.sort_values(date_col).drop_duplicates("ed_visit_id").set_index("ed_visit_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-dir", type=Path, default=INTERIM, help="where subset.py wrote <N>K_SUB_*")
    ap.add_argument("--out-dir", type=Path, default=TRAIN, help="where <N>K_train_clean.parquet goes")
    ap.add_argument("--label", default="500K", help="the <N>K prefix of the subset files")
    ap.add_argument("--hdia-min-count", type=int, default=30)
    ap.add_argument("--complaint-min-count", type=int, default=30)
    args = ap.parse_args()

    def read(table, columns):
        return pd.read_parquet(args.subset_dir / f"{args.label}_SUB_{table}.parquet", columns=columns)

    dst = args.out_dir / f"{args.label}_train_clean.parquet"
    guard_output(dst)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # -- visits ----------------------------------------------------------------
    visits = read("swed_ed_core", ["lopnr_swed", "ed_arrival_datetime", "ed_time_total_minutes", "ed_age",
                                   "is_male", "ed_inexact_time", "ed_presented_complaint_std"])
    visits.insert(0, "ed_visit_id", range(len(visits)))
    n_total = len(visits)
    visits = visits[visits["ed_time_total_minutes"].notna() & ~visits["ed_inexact_time"].fillna(False).astype(bool)]
    log(f"visits: {n_total:,} read, {len(visits):,} kept after dropping null end time / inexact time")

    # -- vitals: first reading per vital in [arrival - 60 min, visit end] -------
    meas = []
    for table in ["ehr_karolinska_meas", "ehr_danderyd_lab_meas"]:
        m = read(table, ["lopnr_swed", "datetime_analysis", "vitals_snomed", "value_num"])
        m["vital"] = m["vitals_snomed"].map(VITAL_SNOMED)  # labs have vitals_snomed null -> dropped
        meas.append(m.dropna(subset=["vital", "value_num"]).drop(columns="vitals_snomed"))
    meas = pd.concat(meas, ignore_index=True)
    visits = first_vitals(visits, meas, "lopnr_swed", "ed_arrival_datetime", "ed_time_total_minutes")
    del meas

    # -- diagnosis and death within 30 days -----------------------------------
    visits["arrival_date"] = (visits["ed_arrival_datetime"].dt.tz_convert("Europe/Stockholm")
                              .dt.tz_localize(None).dt.normalize())

    diag = first_within_30d(visits, read("sos_patreg_wide", ["lopnr_swed", "INDATUM", "Hdia"]).dropna(subset=["Hdia"]),
                            "INDATUM")
    visits["diag_hdia_first"] = visits["ed_visit_id"].map(diag["Hdia"])
    visits["diag_hdia_any"] = visits["diag_hdia_first"].notna()

    death = first_within_30d(visits, read("sos_deathreg_wide", ["lopnr_swed", "date_death_minimp", "inexact_death_date"]),
                             "date_death_minimp")
    visits["death_days_to_death"] = visits["ed_visit_id"].map(death["days"])
    visits["death_date_inexact"] = visits["ed_visit_id"].map(death["inexact_death_date"])
    visits["died_30d"] = visits["death_days_to_death"].notna()
    log(f"diagnosed within 30d: {visits['diag_hdia_any'].mean():.1%}, died within 30d: {visits['died_30d'].mean():.2%}")

    # -- diag_class: NODIAG / frequent Hdia code / other_<chapter> -------------
    visits = assign_diag_class(visits, args.hdia_min_count)

    # -- complaints: one row per complaint ------------------------------------
    rows = complaint_rows(visits, "ed_presented_complaint_std", args.complaint_min_count)

    # -- write ----------------------------------------------------------------
    columns = (["ed_visit_id", "lopnr_swed", "ed_age", "is_male", "complaint_sctid", "complaint_role", "complaint_weight"]
               + [c for c in rows.columns if c.startswith("complaint_bucket_")]
               + [f"vital_{v}" for v in VITALS]
               + ["died_30d", "death_days_to_death", "death_date_inexact",
                  "diag_hdia_first", "diag_hdia_any", "diag_class"])
    rows[columns].to_parquet(dst, index=False)
    log(f"wrote {dst} ({len(rows):,} rows x {len(columns)} columns)")


if __name__ == "__main__":
    main()
