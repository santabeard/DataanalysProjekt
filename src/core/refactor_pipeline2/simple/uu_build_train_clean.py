#!/usr/bin/env python3
"""Build the model-ready UU train_clean table from the <label>_UU_SUB_* subset files.

Same rules as build_train_clean.py (SWED), and the same code for the shared steps,
so the two tables have the same columns wherever the data allows:

  visits     copy_edvisit. Visits with no ED_VisitDuration are dropped, and so are
             visits with Sex other than F/M (U: 15 visits in the whole table).
  vitals     The first reading per vital between 60 min before arrival and the
             visit end, from the vitals_snomed readings uu_subset.py wrote.
             Measurements start 2017-11, so earlier visits have no vitals; they
             are kept with empty vital_* columns, as missing vitals are in SWED.
  diagnosis  The first MainDiagnosis from a specialist care contact (Type OV or SV,
             what SWED's patient register covers) dated 0-30 days after the arrival
             date, day 0 and day 30 included. Codes are mapped to SWED's level of
             detail (see to_swed_level), then diag_class is built as for SWED.
  death      DateOfDeath from 0 to 30 days after the arrival date (day 30 included).
  complaints complaint_snomed -> one row per complaint, weights, buckets.

UU times are Swedish local time (no timezone), so the arrival date is taken as is.
Not in UU, so not in the output: death_date_inexact. The patient and visit IDs keep
their UU names (PersonId, ContactId): a different ID system from SWED's lopnr_swed.

    python3 uu_build_train_clean.py --label 100K
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from build_train_clean import VITALS, WINDOW_DAYS, assign_diag_class, complaint_rows, first_vitals, first_within_30d
from common import INTERIM, TRAIN, VITAL_SNOMED, guard_output, log, raw_table

SPECIALIST_CARE = ["OV", "SV"]  # outpatient specialist + inpatient, as in SWED's patient register


def swed_hdia_vocab(cache):
    """Every distinct Hdia value in SWED's raw sos_patreg_wide, cached as a one-column CSV.

    Only ICD codes are stored, no counts and nothing about patients."""
    if cache.exists():
        vocab = set(pd.read_csv(cache, dtype=str)["Hdia"].dropna())
        log(f"SWED Hdia vocabulary: {len(vocab):,} codes from {cache.name}")
        return vocab
    pf = pq.ParquetFile(raw_table("sos_patreg_wide"))
    vocab = set()
    for batch in pf.iter_batches(batch_size=5_000_000, columns=["Hdia"]):
        vocab.update(pc.unique(batch.column("Hdia")).to_pylist())
    vocab.discard(None)
    pd.DataFrame({"Hdia": sorted(vocab)}).to_csv(cache, index=False)
    log(f"SWED Hdia vocabulary: {len(vocab):,} codes read from {raw_table('sos_patreg_wide').name} -> {cache.name}")
    return vocab


def to_swed_level(code, vocab, blocks):
    """Map a full ICD-10-SE code to the form SWED stores it in.

    The Swedish extra letter is dropped first ('R104X' -> 'R104'). Then, using the codes
    that actually occur in SWED's Hdia: the 4-character code if SWED has it ('I489'),
    else the 3-character code if SWED has it ('F32'), else the narrowest SWED block
    containing it ('C189' -> 'C15-C26'). Returns (code, how) with how one of
    '4char', '3char', 'block', 'fallback' (none matched: the 3-character code) or 'invalid'."""
    m = re.match(r"^([A-Z])(\d{2})(\d?)", str(code).strip().upper())
    if not m:
        return None, "invalid"
    base3 = m.group(1) + m.group(2)
    base4 = base3 + m.group(3) if m.group(3) else None
    if base4 and base4 in vocab:
        return base4, "4char"
    if base3 in vocab:
        return base3, "3char"
    containing = [(end_key - start_key, block) for start_key, end_key, block in blocks
                  if block[0] == base3[0] and start_key <= int(base3[1:]) <= end_key]
    if containing:
        return min(containing)[1], "block"
    return base3, "fallback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-dir", type=Path, default=INTERIM, help="where uu_subset.py wrote <label>_UU_SUB_*")
    ap.add_argument("--out-dir", type=Path, default=TRAIN, help="where <label>_uu_train_clean.parquet goes")
    ap.add_argument("--label", default="100K", help="the label prefix of the subset files")
    ap.add_argument("--hdia-min-count", type=int, default=30)
    ap.add_argument("--complaint-min-count", type=int, default=30)
    args = ap.parse_args()

    def read(table, columns):
        return pd.read_parquet(args.subset_dir / f"{args.label}_UU_SUB_{table}.parquet", columns=columns)

    dst = args.out_dir / f"{args.label}_uu_train_clean.parquet"
    guard_output(dst)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # -- visits ----------------------------------------------------------------
    visits = read("edvisit", ["PersonId", "ContactId", "ED_VisitDateTime", "ED_VisitDuration",
                              "AgeAtED_Visit", "Sex", "DateOfDeath", "complaint_snomed"])
    visits.insert(0, "ed_visit_id", range(len(visits)))  # ContactId is not quite unique in the source
    n_total = len(visits)
    visits = visits[visits["ED_VisitDuration"].notna() & visits["Sex"].isin(["F", "M"])].copy()
    visits["is_male"] = visits["Sex"] == "M"
    visits["arrival_date"] = visits["ED_VisitDateTime"].dt.normalize()
    log(f"visits: {n_total:,} read, {len(visits):,} kept after dropping null duration / sex not F or M")

    # -- vitals: first reading per vital in [arrival - 60 min, visit end] -------
    meas = read("measurements", ["PersonId", "DateTimeOfMeasurement", "vitals_snomed", "value_num"])
    meas = meas.rename(columns={"DateTimeOfMeasurement": "datetime_analysis"})
    meas["vital"] = meas["vitals_snomed"].map(VITAL_SNOMED)
    meas = meas.dropna(subset=["vital", "value_num"]).drop(columns="vitals_snomed")
    visits = first_vitals(visits, meas, "PersonId", "ED_VisitDateTime", "ED_VisitDuration")
    del meas
    has_vital = visits[[f"vital_{v}" for v in VITALS]].notna().any(axis=1)
    by_year = has_vital.groupby(visits["ED_VisitDateTime"].dt.year).mean()
    log("visits with any vital, by year: " + ", ".join(f"{y}: {p:.0%}" for y, p in by_year.items()))

    # -- diagnosis: first specialist-care MainDiagnosis within 30 days, at SWED's level --
    care = read("healthcarevisits", ["PersonId", "AdmissionDate", "MainDiagnosis", "Type"])
    care = care[care["Type"].isin(SPECIALIST_CARE) & care["MainDiagnosis"].notna()]
    vocab = swed_hdia_vocab(args.subset_dir / "swed_hdia_vocab.csv")
    blocks = []
    for b in vocab:
        bm = re.match(r"^([A-Z])(\d{2})-([A-Z])(\d{2})$", b)
        if bm and bm.group(1) == bm.group(3):
            blocks.append((int(bm.group(2)), int(bm.group(4)), b))
    mapped = {c: to_swed_level(c, vocab, blocks) for c in care["MainDiagnosis"].unique()}
    care["Hdia"] = care["MainDiagnosis"].map({c: m[0] for c, m in mapped.items()})
    how = care["MainDiagnosis"].map({c: m[1] for c, m in mapped.items()}).value_counts(normalize=True)
    log(f"specialist care contacts with a diagnosis: {len(care):,}; mapped to SWED level: "
        + ", ".join(f"{k} {v:.1%}" for k, v in how.items()))
    fallback = care.loc[care["MainDiagnosis"].map({c: m[1] for c, m in mapped.items()}) == "fallback", "Hdia"]
    if len(fallback):
        log("  most common codes with no SWED match (kept as 3 characters): "
            + ", ".join(f"{c} ({n:,})" for c, n in fallback.value_counts().head(10).items() if n >= 10))

    diag = first_within_30d(visits, care[["PersonId", "AdmissionDate", "Hdia"]].dropna(subset=["Hdia"]),
                            "AdmissionDate", key="PersonId")
    visits["diag_hdia_first"] = visits["ed_visit_id"].map(diag["Hdia"])
    visits["diag_hdia_any"] = visits["diag_hdia_first"].notna()
    visits = assign_diag_class(visits, args.hdia_min_count)
    del care

    # -- death within 30 days ---------------------------------------------------
    days = (pd.to_datetime(visits["DateOfDeath"]) - visits["arrival_date"]).dt.days
    died = days.between(0, WINDOW_DAYS)
    visits["died_30d"] = died
    visits["death_days_to_death"] = days.where(died)
    n_before = int((days < 0).sum())
    log(f"diagnosed within {WINDOW_DAYS}d: {visits['diag_hdia_any'].mean():.1%}, died within {WINDOW_DAYS}d: {died.mean():.2%}"
        + (f"; NOTE {n_before:,} visits have DateOfDeath before the visit (kept, not counted as died)"
           if n_before else ""))

    # -- complaints: one row per complaint ------------------------------------
    rows = complaint_rows(visits, "complaint_snomed", args.complaint_min_count)

    # -- write ----------------------------------------------------------------
    rows = rows.rename(columns={"AgeAtED_Visit": "ed_age"})
    columns = (["ed_visit_id", "PersonId", "ContactId", "ed_age", "is_male",
                "complaint_sctid", "complaint_role", "complaint_weight"]
               + [c for c in rows.columns if c.startswith("complaint_bucket_")]
               + [f"vital_{v}" for v in VITALS]
               + ["died_30d", "death_days_to_death", "diag_hdia_first", "diag_hdia_any", "diag_class"])
    rows[columns].to_parquet(dst, index=False)
    log(f"wrote {dst} ({len(rows):,} rows x {len(columns)} columns)")


if __name__ == "__main__":
    main()
