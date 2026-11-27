#!/usr/bin/env python3
"""train_clean: the model-ready table for the Bayesian LR models.

Implements notes/ed_atlas_vitals_pipeline_discussion.tex, Section 3.1,
"train_clean design (Stage 04)" -- every decision made there:

  8.  Diagnosis target: pruned multinomial. Top-K most frequent Hdia codes
      each get their own class; everything else falls into an
      other_<chapter> bucket by ICD-10 chapter (cancer, injury, circulatory,
      ...). TODO(BLOCKED): the top-K *list* needs real Hdia frequency counts
      from Bianca -- see --hdia-top-k-csv below. The chapter lookup itself is
      NOT blocked (ICD-10 chapter ranges are public) and is fully
      implemented (icd10_chapter()).
  9.  NODIAG is its own class, for visits with no 30-day Hdia match
      (diag_hdia_any == False). Fully implemented.
  10. Complaint encoding: one-hot + probability-averaging, atlas-style (not
      multi-hot). This script produces the long-format rows that encoding
      needs (one row per active complaint); it does not itself build the
      final one-hot columns -- see item 13.
  11. Two-complaint visits: the training row is DUPLICATED, not sct1-only.
      Each visit with two complaints becomes two output rows sharing
      ed_visit_id, each with complaint_weight=0.5. Single/no-complaint
      visits get one row, complaint_weight=1.0. Fully implemented.
  12. Null-complaint visits: kept, complaint_sctid = "none". Fully
      implemented.
  13. Complaint vocabulary: built from this data, not ed_atlas's list.
      TODO(BLOCKED): the actual vocabulary/pruning threshold needs real
      per-code frequency counts from Bianca -- see --complaint-vocab-csv
      below. Without it, only the raw complaint_sctid/_role/_weight columns
      are written (no complaint_<sctid> one-hot columns yet).
  14. Visits with ed_inexact_time == True are dropped entirely. Fully
      implemented.
  15. Missing vitals: passed through raw and nullable. No imputation here.
  16. Vitals and age: left raw. No scaling here.
  17. Train/val/test split: NOT done here -- that is Stage 05.

TODO(BLOCKED) summary -- both need data this script's author cannot get
without you running something on Bianca (never done by the assistant):
  --hdia-top-k-csv FILE         columns: Hdia,n -- the real Hdia frequency
                                 table (or just the codes you want their own
                                 class). Without it every Hdia code falls
                                 through to other_<chapter> and a warning is
                                 logged once. 03b_pruning_thresholds.py
                                 produces this file for you.
  --complaint-vocab-csv FILE    columns: sctid,n -- the real complaint-code
                                 frequency table. Without it no complaint_*
                                 one-hot columns are produced at all, only
                                 the raw complaint_sctid/_role/_weight -- a
                                 warning is logged once. Also produced by
                                 03b_pruning_thresholds.py.

Tooling: pandas, not polars (see 02_vitals_join.py's docstring for why).

In: "<N>K_train_raw.parquet", READ-ONLY. This script never writes to it, so
re-running 04 after changing a decision above always starts from the same
untouched train_raw.
Out: "<N>K_train_clean.parquet", a new file (not in-place, unlike 02/03).

    SWED_OUT=/home/theon/Desktop/data_std python3 04_train_clean.py
    python3 04_train_clean.py --hdia-top-k-csv hdia_top_k.csv --complaint-vocab-csv complaint_vocab.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

OUT_ROOT = Path(os.environ.get("SWED_OUT", "/home/theon/Desktop/data_std"))  # everything that stays on Bianca
ORIGINAL = Path("/cygnus/proj/nobackup/swed_db/clean_parquet")  # never written to

# TODO(BLOCKED on Bianca data): two inputs this script can't produce itself --
# see assign_diagnosis_class() and build_complaint_onehot() for where each is
# used, and the module docstring for the full explanation.
#   1. --hdia-top-k-csv (columns Hdia,n): real Hdia frequency counts.
#   2. --complaint-vocab-csv (columns sctid,n): real complaint-code frequency counts.
# Both are produced for you by 03b_pruning_thresholds.py.

VITALS = sorted(["systolic_bp", "diastolic_bp", "spo2", "temperature", "pulse", "resp_rate"])

# ── ICD-10 chapter ranges (public, WHO-published; no icd-10-se.tsv needed) ──
# (start_code, end_code, chapter_name). R*/Z* are NOT excluded here -- the
# R*/Z* exclusion was part of the ICD-10 ROLLUP item that stayed out of
# scope; chapter-bucketing them is a different, lighter thing and we are
# doing it. Source: WHO ICD-10 chapter structure (public knowledge).
ICD10_CHAPTERS = [
    ("A00", "B99", "infectious"),
    ("C00", "D48", "neoplasm"),
    ("D50", "D89", "blood_immune"),
    ("E00", "E90", "endocrine_metabolic"),
    ("F00", "F99", "mental_behavioural"),
    ("G00", "G99", "nervous_system"),
    ("H00", "H59", "eye"),
    ("H60", "H95", "ear"),
    ("I00", "I99", "circulatory"),
    ("J00", "J99", "respiratory"),
    ("K00", "K93", "digestive"),
    ("L00", "L99", "skin"),
    ("M00", "M99", "musculoskeletal"),
    ("N00", "N99", "genitourinary"),
    ("O00", "O99", "pregnancy_childbirth"),
    ("P00", "P96", "perinatal"),
    ("Q00", "Q99", "congenital"),
    ("R00", "R99", "symptoms_signs"),
    ("S00", "T98", "injury_poisoning"),
    ("V01", "Y98", "external_causes"),
    ("Z00", "Z99", "health_status_factors"),
    ("U00", "U99", "special_purposes"),
]

_SCT_RE = re.compile(r"\[(\d+)\]")


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
    """Same lookup as 02/03: locate "<N>K_train_raw.parquet". Read-only here --
    04 is the first stage that does NOT write back into this file."""
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


def _icd_key(code3: str) -> "tuple[str, int]":
    return code3[0], int(code3[1:3])


_ICD10_CHAPTER_BOUNDS = [(_icd_key(s), _icd_key(e), name) for s, e, name in ICD10_CHAPTERS]


def icd10_chapter(code: Optional[str]) -> Optional[str]:
    """Chapter name for an Hdia value. Handles both leaf codes ("I219") and
    the block-range strings Hdia sometimes stores directly ("C15-C26", per
    data_info/swed_profile.txt) by using the range's start code. Not blocked
    on anything -- chapter ranges are public, unlike the full leaf-to-parent
    icd-10-se.tsv hierarchy."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    head = code.split("-")[0].strip().upper()
    m = re.match(r"^([A-Z])(\d{2})", head)
    if not m:
        return "unknown"
    key = (m.group(1), int(m.group(2)))
    for start, end, name in _ICD10_CHAPTER_BOUNDS:
        if start <= key <= end:
            return name
    return "unknown"


def load_visits(train_raw_path: Path) -> "tuple[pd.DataFrame, dict]":
    """train_raw, read-only, with the ed_inexact_time exclusion (decision 14)
    applied. Fails loudly if 02/03 haven't run yet -- 04 needs their columns."""
    visits = pd.read_parquet(train_raw_path)

    required = (
        ["ed_visit_id", "lopnr_swed", "ed_age", "is_male", "ed_inexact_time",
         "ed_presented_complaint_std", "died_30d", "death_days_to_death", "death_date_inexact",
         "diag_hdia_any", "diag_hdia_first"]
        + [f"vital_{v}" for v in VITALS]
    )
    missing = [c for c in required if c not in visits.columns]
    if missing:
        sys.exit(f"train_raw is missing {missing}; run 02_vitals_join.py and "
                  f"03_diagnosis_death_join.py on it first")

    n_total = len(visits)
    visits = visits[~visits["ed_inexact_time"]].copy()
    n_dropped = n_total - len(visits)
    log(f"visits: {n_total:,} total, {n_dropped:,} dropped for ed_inexact_time=true (decision 14), "
        f"{len(visits):,} kept")
    return visits, {"visits_total": n_total, "visits_dropped_inexact_time": n_dropped,
                     "visits_kept": len(visits)}


def parse_complaints(visits: pd.DataFrame) -> pd.DataFrame:
    """ed_presented_complaint_std ("x [sctid]" or "a [x]+b [y]") -> sct1, sct2.
    Parsed once per distinct string (not per row) via a dict lookup, since the
    string parse itself isn't worth vectorizing for ~thousands of distinct
    values. .map() on a dict returns NaN for keys not found (i.e. the
    null-complaint rows), which is exactly what we want -- decision 12
    (null -> "none") is handled by the caller, not here."""
    distinct = visits["ed_presented_complaint_std"].dropna().unique()
    sct1_map, sct2_map = {}, {}
    for raw in distinct:
        codes = [m.group(1) for part in raw.split("+") for m in [_SCT_RE.search(part)] if m]
        sct1_map[raw] = codes[0] if len(codes) >= 1 else None
        sct2_map[raw] = codes[1] if len(codes) >= 2 else None
        if len(codes) > 2:
            log(f"NOTE: more than 2 complaint codes parsed from {raw!r}; keeping only the first 2")

    visits = visits.copy()
    visits["sct1"] = visits["ed_presented_complaint_std"].map(sct1_map)
    visits["sct2"] = visits["ed_presented_complaint_std"].map(sct2_map)
    return visits


def expand_complaint_rows(visits: pd.DataFrame) -> "tuple[pd.DataFrame, dict]":
    """Decisions 10-12: one row per active complaint, sharing ed_visit_id.
    - No complaint (sct1 is null): one row, complaint_sctid="none", weight=1.0.
    - One complaint: one row, complaint_sctid=sct1, weight=1.0.
    - Two complaints: two rows (primary=sct1, secondary=sct2), weight=0.5 each.

    A caller that wants exactly one row per visit (e.g. for evaluation, or a
    val/test split) can always recover it with
    `.drop_duplicates(subset=["ed_visit_id"], keep="first")`, since primary
    rows sort first ("primary" < "secondary") and every visit has exactly
    one."""
    primary = visits.copy()
    primary["complaint_sctid"] = primary["sct1"].fillna("none")
    primary["complaint_role"] = "primary"
    primary["complaint_weight"] = np.where(primary["sct2"].notna(), 0.5, 1.0)
    primary = primary.drop(columns=["sct1", "sct2"])

    secondary = visits[visits["sct2"].notna()].copy()
    secondary["complaint_sctid"] = secondary["sct2"]
    secondary["complaint_role"] = "secondary"
    secondary["complaint_weight"] = 0.5
    secondary = secondary.drop(columns=["sct1", "sct2"])

    n_duplicated = len(secondary)
    out = (
        pd.concat([primary, secondary], ignore_index=True)
        .sort_values(["ed_visit_id", "complaint_role"])
        .reset_index(drop=True)
    )
    log(f"complaint rows: {len(primary):,} visits, {n_duplicated:,} with a second complaint "
        f"-> {len(out):,} total rows (decision 11: duplicated, weight 0.5 each)")
    return out, {"visits_with_two_complaints": n_duplicated, "complaint_rows_total": len(out)}


def build_complaint_onehot(df: pd.DataFrame, vocab_csv: Optional[Path]) -> "tuple[pd.DataFrame, bool]":
    """TODO (BLOCKED on Bianca data, decision 13): vocab_csv must have a
    column "sctid" -- the complaint codes that get their own one-hot column
    (already pruned to whatever frequency threshold you chose). Any
    complaint_sctid not in that list becomes "other_complaint". Without the
    file, this is a no-op and complaint_sctid/_role/_weight are left as the
    only complaint columns, for the model script to encode once the
    vocabulary exists."""
    # TODO(BLOCKED on Bianca data): build the real complaint vocabulary (per-code
    # frequency counts + a pruning threshold) and pass it via --complaint-vocab-csv.
    if vocab_csv is None:
        log("TODO (BLOCKED): no --complaint-vocab-csv given -- skipping one-hot complaint columns. "
            "Only complaint_sctid/_role/_weight are in the output; the model script must build the "
            "vocabulary (and its pruning threshold) from real per-code frequency counts first.")
        return df, False

    vocab = pd.read_csv(vocab_csv)["sctid"].astype(str).tolist()
    df = df.copy()
    df["complaint_bucket"] = np.where(
        df["complaint_sctid"].isin(vocab) | (df["complaint_sctid"] == "none"),
        df["complaint_sctid"],
        "other_complaint",
    )
    onehot = pd.get_dummies(df["complaint_bucket"], prefix="complaint_bucket")
    df = pd.concat([df, onehot], axis=1)
    log(f"complaint one-hot: {len(vocab)} vocabulary codes + none + other_complaint")
    return df, True


def assign_diagnosis_class(visits: pd.DataFrame, top_k_csv: Optional[Path]) -> "tuple[pd.DataFrame, dict]":
    """Decisions 8-9: NODIAG for no match; else the exact Hdia code if it's
    in the (TODO BLOCKED) top-K list; else other_<chapter>."""
    # TODO(BLOCKED on Bianca data): get the real Hdia frequency counts and pass
    # the chosen top-K code list via --hdia-top-k-csv.
    if top_k_csv is None:
        log("TODO(BLOCKED): no --hdia-top-k-csv given -- every Hdia code falls through to "
            "other_<chapter>; no code gets its own class yet. Re-run with the real top-K list "
            "once you have per-code frequency counts from Bianca.")
        top_k_codes = set()
        top_k_applied = False
    else:
        top_k_codes = set(pd.read_csv(top_k_csv)["Hdia"].astype(str).tolist())
        top_k_applied = True

    visits = visits.copy()
    distinct_codes = visits.loc[visits["diag_hdia_first"].notna(), "diag_hdia_first"].unique().tolist()
    chapter_of = {c: icd10_chapter(c) for c in distinct_codes}
    visits["_chapter"] = visits["diag_hdia_first"].map(chapter_of)

    is_top_k = visits["diag_hdia_first"].isin(top_k_codes)
    visits["diag_class"] = np.select(
        [~visits["diag_hdia_any"].values, is_top_k.values],
        ["NODIAG", visits["diag_hdia_first"].values],
        default=("other_" + visits["_chapter"].astype(str)).values,
    )
    visits = visits.drop(columns=["_chapter"])

    n_nodiag = int((visits["diag_class"] == "NODIAG").sum())
    n_top_k = int(is_top_k.sum()) if top_k_codes else 0
    n_other = len(visits) - n_nodiag - n_top_k
    log(f"diag_class: {n_nodiag:,} NODIAG, {n_top_k:,} own-class (top-K), {n_other:,} other_<chapter>")
    return visits, {"diag_nodiag": n_nodiag, "diag_top_k_class": n_top_k, "diag_other_chapter": n_other,
                     "diag_top_k_applied": top_k_applied}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset-dir", type=Path, default=OUT_ROOT / "subset",
                    help="output of 01_subset_v2.py (default $SWED_OUT/subset)")
    ap.add_argument("--label", default=None,
                    help="the <N>K size label (default: auto-detect from the one *_train_raw.parquet present)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write train_clean here instead of <N>K_train_clean.parquet in --subset-dir")
    # TODO(BLOCKED on Bianca data): both of these are optional only because the
    # real data to fill them in doesn't exist yet on this machine -- see
    # assign_diagnosis_class() / build_complaint_onehot().
    ap.add_argument("--hdia-top-k-csv", type=Path, default=None,
                    help="TODO(BLOCKED): columns Hdia,n -- see module docstring")
    ap.add_argument("--complaint-vocab-csv", type=Path, default=None,
                    help="TODO(BLOCKED): columns sctid,n -- see module docstring")
    args = ap.parse_args()
    refuse_to_write_into(args.subset_dir, ORIGINAL)

    train_raw_path, label = find_train_raw(args.subset_dir, args.label)
    log(f"train_raw: {train_raw_path} (label {label}) -- read-only")
    dst = args.out if args.out else args.subset_dir / f"{label}_train_clean.parquet"
    refuse_to_write_into(dst, ORIGINAL, train_raw_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    visits, exclusion_report = load_visits(train_raw_path)
    visits, diag_report = assign_diagnosis_class(visits, args.hdia_top_k_csv)
    visits = parse_complaints(visits)
    rows, complaint_report = expand_complaint_rows(visits)
    rows, complaint_onehot_applied = build_complaint_onehot(rows, args.complaint_vocab_csv)

    keep_cols = (
        ["ed_visit_id", "lopnr_swed", "ed_age", "is_male",
         "complaint_sctid", "complaint_role", "complaint_weight"]
        + [c for c in rows.columns if c.startswith("complaint_bucket_")]
        + [f"vital_{v}" for v in VITALS]
        + ["died_30d", "death_days_to_death", "death_date_inexact",
           "diag_hdia_first", "diag_hdia_any", "diag_class"]
    )
    train_clean = rows[keep_cols].sort_values(["ed_visit_id", "complaint_role"]).reset_index(drop=True)

    tmp = dst.with_suffix(".parquet.tmp")
    train_clean.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.rename(dst)
    log(f"wrote {dst} ({len(train_clean):,} rows x {len(train_clean.columns)} columns); "
        f"{train_raw_path.name} untouched")

    report = {
        **exclusion_report, **complaint_report, **diag_report,
        "complaint_onehot_applied": complaint_onehot_applied,
        "rows_out": len(train_clean),
    }
    report_path = args.subset_dir / f"{label}_04_train_clean_report.csv"
    pd.DataFrame([report]).to_csv(report_path, index=False)
    log(f"report written to {report_path}")

    if not diag_report["diag_top_k_applied"] or not complaint_onehot_applied:
        log("REMINDER: this train_clean has TODO(BLOCKED) stand-ins active -- see the warnings above "
            "and the module docstring. Re-run 04 with --hdia-top-k-csv / --complaint-vocab-csv once "
            "those are available, before using this for anything but a structural dry run.")


if __name__ == "__main__":
    main()
