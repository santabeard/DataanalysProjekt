#!/usr/bin/env python3
"""Diagnosis / death join: adds 30-day diagnosis and mortality outcomes.

Implements notes/ed_atlas_vitals_pipeline_discussion.tex, Chapter 3, "Diagnosis
/ death joins (lopnr_swed + date range vs. ed_arrival_datetime)" -- both Q&A
decisions there, plus the two open items flagged in "Still open / unanswered"
that this script has to pick a default for:

  6. Death-date imputation column: date_death_minimp, not date_death_meanimp --
     more conservative/sensitive, catches every death that could possibly fall
     within 30 days (minimp never places a death later than it actually was).
  7. Timezone: ed_arrival_datetime is a tz-aware UTC timestamp; INDATUM and
     date_death_minimp are plain dates (no tz, Swedish civil dates already).
     Convert ed_arrival_datetime to an Europe/Stockholm local date before
     comparing, so the 30-day window isn't shifted by a midnight-UTC artifact.
  Open (not resolved by Chapter 3, defaulted here and reported):
     - Day-30 boundary inclusivity: this script treats day 30 as IN the
       window ([arrival_date, arrival_date + 30d], both ends inclusive). The
       report counts matches that land exactly on day 30 so the sensitivity
       of this default is visible, not hidden.
     - ed_inexact_time / inexact_death_date rows are NOT filtered out here
       (Chapter 3 raised this, did not decide it); inexact_death_date is kept
       as a column instead so a later stage/model can decide.

Diagnosis is Hdia only -- the main diagnosis column of sos_patreg_wide. This
is a deliberate simplification of Chapter 3's original "existence/any-match
across the 930 atlas categories" framing, which needed the (not-yet-available,
externally licensed) icd-10-se.tsv parent-code table to even define. That
ICD-10 rollup and the R*/Z* exclusion are explicitly OUT of scope here.
DIAGNOS2..30 (other positions) are also not used -- Hdia only, as instructed.

Semantics, for both joins: lopnr_swed match + INDATUM / date_death_minimp in
[arrival_local_date, arrival_local_date + 30 days]. Direction is INDATUM/death
date >= arrival date (day 0 included -- same-day admission diagnoses are the
central case). A visit can match more than one registry row in the window;
diagnosis keeps first-by-date Hdia plus a count (existence semantics), death
is boolean + the matched date (sos_deathreg_wide is one row per patient, so at
most one match is expected; the join does not assume this and would average
over several if it ever happened, which the report would make visible via a
higher match count than patients).

Tooling: pandas, not polars (see 02_vitals_join.py's docstring for why --
Bianca has pandas 1.2.4 + pyarrow already; no polars without a wheel transfer).

In: "<N>K_train_raw.parquet" and the two read-only SUB tables
"<N>K_SUB_sos_patreg_wide.parquet" / "<N>K_SUB_sos_deathreg_wide.parquet".
Out: diag_hdia_* and death-outcome columns added to "<N>K_train_raw.parquet"
IN PLACE, by default (same convention as 02_vitals_join.py).

    SWED_OUT=/home/theon/Desktop/data_std python3 03_diagnosis_death_join.py
    python3 03_diagnosis_death_join.py --label 500K
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

WINDOW_DAYS = 30  # arrival day (day 0) through day 30, both ends inclusive -- see docstring
LOCAL_TZ = "Europe/Stockholm"

# Columns this script owns; dropped and recomputed on rerun so reruns don't
# collide with their own previous output (same convention as 02).
OWNED_PREFIXES = ("diag_hdia_", "death_", "died_")
OWNED_EXTRA = ("arrival_local_date",)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def refuse_to_write_into(out: Path, *protected: Path) -> None:
    """Outputs must never land in, or overwrite anything in, the source data folders."""
    o = out.expanduser().resolve()
    for p in protected:
        p = p.expanduser().resolve()
        if o == p or p in o.parents:
            sys.exit(f"refusing to write to {out}: it is inside the source data folder {p}")


def find_train_raw(subset_dir: Path, label: Optional[str]) -> "tuple[Path, str]":
    """Same lookup as 02_vitals_join.py: locate "<N>K_train_raw.parquet"."""
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


def load_visits(train_raw_path: Path) -> pd.DataFrame:
    """train_raw with ed_visit_id (reused if already present, e.g. from
    02_vitals_join.py) and the UTC arrival timestamp converted to an
    Europe/Stockholm local date (Q7), kept as a tz-naive midnight Timestamp
    so it compares directly against INDATUM/date_death_minimp (plain dates,
    no tz, loaded by pandas/pyarrow as tz-naive datetime64[ns] at midnight).
    Columns this script owns from a previous run are dropped first, so
    reruns are idempotent."""
    visits = pd.read_parquet(train_raw_path)
    stale = [c for c in visits.columns if c.startswith(OWNED_PREFIXES) or c in OWNED_EXTRA]
    if stale:
        log(f"dropping {len(stale)} column(s) from a previous run of this script before rejoining: {stale}")
        visits = visits.drop(columns=stale)
    if "ed_visit_id" not in visits.columns:
        visits.insert(0, "ed_visit_id", np.arange(len(visits), dtype="int64"))

    visits["arrival_local_date"] = (
        visits["ed_arrival_datetime"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    )
    log(f"visits: {len(visits):,}, arrival converted to {LOCAL_TZ} local date")
    return visits


def window_match(visits: pd.DataFrame, registry: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """lopnr_swed equi-join + the [arrival_local_date, arrival_local_date + 30d]
    filter shared by both the diagnosis and death joins. Adds days_from_arrival."""
    j = visits.merge(registry, on="lopnr_swed", how="inner")
    window_end = j["arrival_local_date"] + pd.Timedelta(days=WINDOW_DAYS)
    j = j[(j[date_col] >= j["arrival_local_date"]) & (j[date_col] <= window_end)].copy()
    j["days_from_arrival"] = (j[date_col] - j["arrival_local_date"]).dt.days
    return j


def diagnosis_join(visits: pd.DataFrame, subset_dir: Path, label: str) -> "tuple[pd.DataFrame, dict]":
    """Hdia only (see docstring). Existence semantics: any match in the
    window, the earliest matching Hdia code, how many registry rows matched,
    and days to that first match."""
    path = subset_dir / f"{label}_SUB_sos_patreg_wide.parquet"
    if not path.exists():
        sys.exit(f"missing {path}; run 01_subset_v2.py first")
    diag = pd.read_parquet(path, columns=["lopnr_swed", "INDATUM", "Hdia"])
    diag = diag[diag["Hdia"].notna()].copy()
    diag["INDATUM"] = pd.to_datetime(diag["INDATUM"])
    log(f"diagnosis rows: {len(diag):,} with non-null Hdia")

    # Only the 3 columns the merge actually needs -- window_match's intermediate
    # frame is ~tens of millions of rows before the date filter runs, and
    # visits here is train_raw-after-02 (44 columns incl. every vital_*); not
    # slimming this down first is what caused the OOM.
    slim = visits[["ed_visit_id", "lopnr_swed", "arrival_local_date"]]
    j = window_match(slim, diag, "INDATUM")
    n_on_day30 = int((j["days_from_arrival"] == WINDOW_DAYS).sum())
    log(f"diagnosis matches landing exactly on day {WINDOW_DAYS} (boundary-inclusivity open question): {n_on_day30:,}")

    first = (
        j.sort_values("INDATUM")
        .groupby("ed_visit_id")
        .agg(
            diag_hdia_first=("Hdia", "first"),
            diag_hdia_days_to_first=("days_from_arrival", "first"),
            diag_hdia_n_in_window=("Hdia", "size"),
        )
        .reset_index()
    )
    first["diag_hdia_any"] = True

    out = visits[["ed_visit_id"]].merge(first, on="ed_visit_id", how="left")
    out["diag_hdia_any"] = out["diag_hdia_any"].fillna(False).astype(bool)
    out["diag_hdia_n_in_window"] = out["diag_hdia_n_in_window"].fillna(0).astype("int64")
    n_any = int(out["diag_hdia_any"].sum())
    log(f"visits with a 30-day Hdia diagnosis: {n_any:,} / {len(visits):,}")
    return out, {"diag_rows_with_hdia": len(diag), "diag_visits_any": n_any,
                 "diag_matches_on_day30": n_on_day30}


def death_join(visits: pd.DataFrame, subset_dir: Path, label: str) -> "tuple[pd.DataFrame, dict]":
    """date_death_minimp only (Q6). Boolean + matched date + the
    inexact_death_date flag, carried through rather than filtered (open item)."""
    path = subset_dir / f"{label}_SUB_sos_deathreg_wide.parquet"
    if not path.exists():
        sys.exit(f"missing {path}; run 01_subset_v2.py first")
    death = pd.read_parquet(path, columns=["lopnr_swed", "date_death_minimp", "inexact_death_date"])
    death = death[death["date_death_minimp"].notna()].copy()
    death["date_death_minimp"] = pd.to_datetime(death["date_death_minimp"])
    log(f"death rows: {len(death):,} with non-null date_death_minimp")

    n_dup_patients = len(death) - death["lopnr_swed"].nunique()
    if n_dup_patients:
        log(f"NOTE: {n_dup_patients:,} patient(s) have more than one row in sos_deathreg_wide -- "
            f"expected at most one per patient; taking the earliest matching date per visit regardless")

    slim = visits[["ed_visit_id", "lopnr_swed", "arrival_local_date"]]
    j = window_match(slim, death, "date_death_minimp")
    n_on_day30 = int((j["days_from_arrival"] == WINDOW_DAYS).sum())
    log(f"death matches landing exactly on day {WINDOW_DAYS} (boundary-inclusivity open question): {n_on_day30:,}")

    first = (
        j.sort_values("date_death_minimp")
        .groupby("ed_visit_id")
        .agg(
            death_days_to_death=("days_from_arrival", "first"),
            death_date_inexact=("inexact_death_date", "first"),
        )
        .reset_index()
    )
    first["died_30d"] = True

    out = visits[["ed_visit_id"]].merge(first, on="ed_visit_id", how="left")
    out["died_30d"] = out["died_30d"].fillna(False).astype(bool)
    n_died = int(out["died_30d"].sum())
    log(f"visits with a 30-day death: {n_died:,} / {len(visits):,}")
    return out, {"death_rows": len(death), "death_rows_duplicate_patients": n_dup_patients,
                 "death_visits_died_30d": n_died, "death_matches_on_day30": n_on_day30}


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

    visits = load_visits(train_raw_path)

    diag_cols, diag_report = diagnosis_join(visits, args.subset_dir, label)
    death_cols, death_report = death_join(visits, args.subset_dir, label)

    train = (
        visits.drop(columns=["arrival_local_date"])
        .merge(diag_cols, on="ed_visit_id", how="left")
        .merge(death_cols, on="ed_visit_id", how="left")
        .sort_values("ed_visit_id")
        .reset_index(drop=True)
    )

    tmp = dst.with_suffix(".parquet.tmp")
    train.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.rename(dst)
    log(f"wrote {dst} ({len(train):,} visits x {len(train.columns)} columns)"
        + (" [in place]" if dst == train_raw_path else ""))

    report = {"visits": len(visits), **diag_report, **death_report}
    report_path = args.subset_dir / f"{label}_03_diagnosis_death_join_report.csv"
    pd.DataFrame([report]).to_csv(report_path, index=False)
    log(f"report written to {report_path}")


if __name__ == "__main__":
    main()
