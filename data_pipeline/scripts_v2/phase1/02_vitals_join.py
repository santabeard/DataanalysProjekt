#!/usr/bin/env python3
"""Vitals join: adds first-in-window vital signs to the training table.

Implements notes/ed_atlas_vitals_pipeline_discussion.tex, Chapter 3, "Vitals join
(lopnr_swed + time window, -60 min before visit start to end of visit)" -- the
five Q&A decisions there, exactly as settled:

  1. Site/hospital match: NOT a hard filter. Agreed in principle ("can't be in
     two places at once") but treated as a bug/validity check, not a filter --
     every matched vital row is flagged `vital_site_mismatch` instead of being
     dropped, and the join report counts how often it happens.
  2. `ed_time_total_minutes` (visit end): 5.2% null and capped at 2880 min (48h)
     in the raw data. Round-1 decision: drop visits with a null end time
     entirely (counted in the report); the capped-value question is deferred.
  3. Overlapping/adjacent visits for the same patient: same answer as (1) --
     no window-clipping logic here, just a `vital_overlap_window` diagnostic
     flag per visit so a later bug test can look at it.
  4. Multiple readings of the same vital within one visit's window: take the
     first reading by `datetime_analysis` (not mean/last/all).
  5. Tooling: pandas (not polars -- Bianca's python_ML_packages/3.9.5-cpu ships
     pandas 1.2.4 + pyarrow, but no polars, and installing polars there needs a
     wheel transferred in via wharf; pandas avoids that entirely). Every API
     used below (merge, groupby().first()/.shift(), .dt.tz_convert,
     get_dummies, NamedAgg) has been stable since well before pandas 1.2.4
     (Dec 2020), so this should run as-is.

Also implements the two lookup joins Chapter 3 said must happen before the
window join:
  - SNOMED mapping (data_info/swed_lab_meas_to_snomed.csv) to isolate vitals
    term_names out of ehr_danderyd_lab_meas, which mixes vitals with labs under
    the same term_name/term_id columns. Restricting to the 6 vitals SNOMED IDs
    IS this isolation: every lab term_name simply has no match and is dropped
    by the inner join, so no separate "is this a lab" step is needed.
  - Unit normalization (data_info/lab_unit_normalization_07sep2026.csv): checked,
    and it has no rows for any of the 6 vitals term_names (it is lab-chemistry
    only -- B-/P-/S-/U- prefixed analytes). So it is a no-op for this join and is
    deliberately NOT applied here; Karolinska and Danderyd both already report
    these vitals in the same units per data_info/swed_profile.txt (mmHg,
    slag/min, andetag/min., degC, %). This is flagged, not assumed silently.

In: the patient subset written by 01_subset_v2.py: "<N>K_train_raw.parquet"
(the growing training table) and "<N>K_SUB_ehr_karolinska_meas.parquet" /
"<N>K_SUB_ehr_danderyd_lab_meas.parquet" (the vitals sources, read-only -- the
"_SUB_" files are a fixed snapshot and are never written to by this script).
Out: one column per vital (value, minutes-from-arrival, reading count in
window) added to "<N>K_train_raw.parquet" IN PLACE, by default. The "<N>K" is
auto-detected from whichever single "*_train_raw.parquet" file is in
--subset-dir; pass --label if more than one size is present.

    SWED_OUT=/home/theon/Desktop/data_std python3 02_vitals_join.py
    python3 02_vitals_join.py --label 500K       # when more than one size exists
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

OUT_ROOT = Path(os.environ.get("SWED_OUT", "/home/theon/Desktop/data_std"))  # everything that stays on Bianca
ORIGINAL = Path("/cygnus/proj/nobackup/swed_db/clean_parquet")  # never written to
REF = Path(os.environ.get("SWED_REF", Path(__file__).resolve().parent.parent.parent / "data_info"))

SNOMED_MAP_CSV = REF / "swed_lab_meas_to_snomed.csv"
UNIT_NORM_CSV = REF / "lab_unit_normalization_07sep2026.csv"  # checked: no vitals rows, not applied (see docstring)

WINDOW_BEFORE_MIN = 60  # arrival - 60 min

# The 6 vitals, by SNOMED ID, per Chapter 3 ("82 distinct term_name ... only 6
# ... all 6 map cleanly to SNOMED"). Any term_name mapping to one of these IDs
# counts as that vital, regardless of which of the two source tables it's in or
# which exact spelling it uses (e.g. "Blodtryck systoliskt" and "Blodtryck
# systoliskt - övre" are both 271649006 in swed_lab_meas_to_snomed.csv).
VITAL_SNOMED = {
    "271649006": "systolic_bp",
    "271650006": "diastolic_bp",
    "284032008": "spo2",
    "56342008": "temperature",
    "65653002": "pulse",
    "86290005": "resp_rate",
}
VITALS = sorted(VITAL_SNOMED.values())

# Which ED hospitals each vitals source table can plausibly belong to, used
# only for the Q1 bug/validity flag -- never to filter a row out.
SOURCE_HOSPITALS = {
    "ehr_karolinska_meas": {"karolinska"},
    "ehr_danderyd_lab_meas": {"danderyd", "södersjukhuset"},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def refuse_to_write_into(out: Path, *protected: Path) -> None:
    """Outputs must never land in, or overwrite anything in, the source data folders."""
    o = out.expanduser().resolve()
    for p in protected:
        p = p.expanduser().resolve()
        if o == p or p in o.parents:
            sys.exit(f"refusing to write to {out}: it is inside the source data folder {p}")


def load_vital_snomed_map() -> pd.DataFrame:
    """term_name -> vital, restricted to the 6 vitals SNOMED IDs. This is the
    isolation join: inner-joining measurement rows against this map is what
    drops every lab term_name out of the mixed Danderyd table."""
    if not SNOMED_MAP_CSV.exists():
        sys.exit(f"missing {SNOMED_MAP_CSV}")
    vital_ids = pd.DataFrame({"snomed_id": list(VITAL_SNOMED), "vital": list(VITAL_SNOMED.values())})
    snomed = pd.read_csv(SNOMED_MAP_CSV, dtype={"snomed_id": str})
    m = (
        snomed[["term_name", "snomed_id"]]
        .merge(vital_ids, on="snomed_id", how="inner")[["term_name", "vital"]]
        .drop_duplicates()
    )
    log(f"vital SNOMED map: {len(m)} term_name variants -> {m['vital'].nunique()} vitals")
    return m


def find_train_raw(subset_dir: Path, label: Optional[str]) -> "tuple[Path, str]":
    """Locate "<N>K_train_raw.parquet" in subset_dir -- the file 01_subset_v2.py
    wrote and this script enriches in place. If --label wasn't given, there
    must be exactly one "*_train_raw.parquet" to pick."""
    if label:
        path = subset_dir / f"{label}_train_raw.parquet"
        if not path.exists():
            sys.exit(f"missing {path}")
        return path, label

    candidates = sorted(subset_dir.glob("*_train_raw.parquet"))
    if not candidates:
        sys.exit(f"no *_train_raw.parquet in {subset_dir}; run 01_subset_v2.py first")
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        sys.exit(f"more than one *_train_raw.parquet in {subset_dir} ({names}); pass --label to pick one")
    path = candidates[0]
    found_label = path.name[: -len("_train_raw.parquet")]
    return path, found_label


def load_visits(train_raw_path: Path) -> "tuple[pd.DataFrame, dict]":
    """The train_raw table, with ed_visit_id (added once, kept on rerun) and
    the triage window. Drops visits with a null ed_time_total_minutes (Q2,
    round 1). Any vital_* / window_* columns from a previous run of this
    script are dropped first, so reruns don't collide with their own output."""
    visits = pd.read_parquet(train_raw_path)
    stale = [c for c in visits.columns if c.startswith("vital_") or c in ("window_start", "window_end")]
    if stale:
        log(f"dropping {len(stale)} column(s) from a previous run of this script before rejoining: {stale}")
        visits = visits.drop(columns=stale)
    if "ed_visit_id" not in visits.columns:
        visits.insert(0, "ed_visit_id", np.arange(len(visits), dtype="int64"))

    n_total = len(visits)
    visits = visits[visits["ed_time_total_minutes"].notna()].copy()
    n_dropped_null_end = n_total - len(visits)

    visits["window_start"] = visits["ed_arrival_datetime"] - pd.Timedelta(minutes=WINDOW_BEFORE_MIN)
    visits["window_end"] = visits["ed_arrival_datetime"] + pd.to_timedelta(
        visits["ed_time_total_minutes"], unit="m"
    )
    log(f"visits: {n_total:,} total, {n_dropped_null_end:,} dropped for null ed_time_total_minutes, "
        f"{len(visits):,} kept")
    return visits, {"visits_total": n_total, "visits_dropped_null_end_time": n_dropped_null_end,
                     "visits_kept": len(visits)}


def load_vital_rows(subset_dir: Path, label: str, vital_map: pd.DataFrame) -> pd.DataFrame:
    """Vitals rows from both SUB source tables, concept-mapped to the 6 vitals
    and tagged with the source table they came from. The inner join against
    vital_map both isolates vitals-out-of-labs (Danderyd) and restricts
    Karolinska's 82 term_names down to the 6 that matter."""
    frames = []
    for src_name in ["ehr_karolinska_meas", "ehr_danderyd_lab_meas"]:
        path = subset_dir / f"{label}_SUB_{src_name}.parquet"
        if not path.exists():
            sys.exit(f"missing {path}; run 01_subset_v2.py first")
        df = pd.read_parquet(path, columns=["lopnr_swed", "datetime_analysis", "term_name", "value_num"])
        df["source"] = src_name
        frames.append(df)
    rows = pd.concat(frames, ignore_index=True)

    before = len(rows)
    rows = rows.merge(vital_map, on="term_name", how="inner")
    log(f"vital rows: {before:,} read from both sources, {len(rows):,} matched a vitals SNOMED code "
        f"({before - len(rows):,} dropped -- unmapped term_names, including every Danderyd lab value)")
    return rows


def join_window(visits: pd.DataFrame, vital_rows: pd.DataFrame) -> pd.DataFrame:
    """Q1/Q3: equi-join on lopnr_swed + a time filter, no site filter and no
    overlap clipping -- just diagnostic flags for both. A patient's vitals are
    compared against every one of their visits, so a reading can legitimately
    match more than one visit if those visits' windows overlap."""
    j = visits.merge(vital_rows, on="lopnr_swed", how="inner")
    j = j[(j["datetime_analysis"] >= j["window_start"]) & (j["datetime_analysis"] <= j["window_end"])].copy()

    # Vectorized rather than a per-row python callback: OR together one
    # "this source but wrong hospital" condition per entry in SOURCE_HOSPITALS.
    mismatch = pd.Series(False, index=j.index)
    for source, hospitals in SOURCE_HOSPITALS.items():
        mismatch = mismatch | ((j["source"] == source) & (~j["ed_hospital"].isin(hospitals)))

    delta = j["datetime_analysis"] - j["ed_arrival_datetime"]
    j["minutes_from_arrival"] = (delta.dt.total_seconds() // 60).astype("int64")
    j["vital_site_mismatch"] = mismatch
    return j


def flag_overlaps(visits: pd.DataFrame) -> pd.DataFrame:
    """Q3 diagnostic: for each patient, does this visit's window overlap the
    next visit's window (sorted by arrival)? Reported, never clipped."""
    v = visits.sort_values(["lopnr_swed", "ed_arrival_datetime"]).copy()
    next_start = v.groupby("lopnr_swed")["window_start"].shift(-1)
    v["vital_overlap_window"] = (v["window_end"] > next_start).fillna(False)
    return v


def first_reading_wide(joined: pd.DataFrame, visit_ids: pd.Series) -> pd.DataFrame:
    """Q4: first reading per (visit, vital) by datetime_analysis; pivot to one
    row per visit with per-vital value / minutes_from_arrival / reading count.

    `visit_ids` must be the *actual* surviving ed_visit_id values, not a
    range(): ids are assigned before the Q2 null-end-time filter runs, so they
    are a subset of 0..n_total-1, not contiguous from 0."""
    ranked = (
        joined.sort_values("datetime_analysis")
        .groupby(["ed_visit_id", "vital"])
        .agg(
            value=("value_num", "first"),
            minutes_from_arrival=("minutes_from_arrival", "first"),
            site_mismatch=("vital_site_mismatch", "first"),
            n_in_window=("vital_site_mismatch", "size"),
        )
        .reset_index()
    )

    wide = pd.DataFrame({"ed_visit_id": np.sort(visit_ids.unique())})
    for v in VITALS:
        sub = ranked.loc[ranked["vital"] == v, ["ed_visit_id", "value", "minutes_from_arrival",
                                                 "n_in_window", "site_mismatch"]].rename(columns={
            "value": f"vital_{v}",
            "minutes_from_arrival": f"vital_{v}_minutes_from_arrival",
            "n_in_window": f"vital_{v}_n_in_window",
            "site_mismatch": f"vital_{v}_site_mismatch",
        })
        wide = wide.merge(sub, on="ed_visit_id", how="left")
    return wide


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset-dir", type=Path, default=OUT_ROOT / "subset",
                    help="output of 01_subset_v2.py (default $SWED_OUT/subset)")
    ap.add_argument("--label", default=None,
                    help="the <N>K size label (default: auto-detect from the one *_train_raw.parquet present)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the enriched table here instead of overwriting <N>K_train_raw.parquet in place")
    args = ap.parse_args()
    refuse_to_write_into(args.subset_dir, ORIGINAL)

    train_raw_path, label = find_train_raw(args.subset_dir, args.label)
    log(f"train_raw: {train_raw_path} (label {label})")
    dst = args.out if args.out else train_raw_path
    if args.out:
        refuse_to_write_into(args.out, ORIGINAL)
        args.out.parent.mkdir(parents=True, exist_ok=True)

    vital_map = load_vital_snomed_map()
    visits, report = load_visits(train_raw_path)
    visits = flag_overlaps(visits)
    n_overlap = int(visits["vital_overlap_window"].sum())
    log(f"visits with an overlapping next-visit window: {n_overlap:,} (Q3 diagnostic, not clipped)")

    vital_rows = load_vital_rows(args.subset_dir, label, vital_map)
    joined = join_window(visits, vital_rows)
    n_mismatch = int(joined["vital_site_mismatch"].sum())
    log(f"matched vital readings from a non-visit-hospital source: {n_mismatch:,} of {len(joined):,} "
        f"(Q1 diagnostic, not filtered)")

    wide = first_reading_wide(joined, visits["ed_visit_id"])
    train = (
        visits.drop(columns=["window_start", "window_end"])
        .merge(wide, on="ed_visit_id", how="left")
        .sort_values("ed_visit_id")
        .reset_index(drop=True)
    )

    tmp = dst.with_suffix(".parquet.tmp")
    train.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.rename(dst)
    log(f"wrote {dst} ({len(train):,} visits x {len(train.columns)} columns)"
        + (" [in place]" if dst == train_raw_path else ""))

    for v in VITALS:
        n_present = int(train[f"vital_{v}"].notna().sum())
        log(f"  vital_{v}: {n_present:,} / {len(train):,} visits ({n_present / len(train):.1%})")

    report.update({
        "vital_rows_matched": len(vital_rows),
        "visits_with_overlapping_window": n_overlap,
        "matched_readings_site_mismatch": n_mismatch,
    })
    report_path = args.subset_dir / f"{label}_02_vitals_join_report.csv"
    pd.DataFrame([report]).to_csv(report_path, index=False)
    log(f"report written to {report_path}")


if __name__ == "__main__":
    main()
