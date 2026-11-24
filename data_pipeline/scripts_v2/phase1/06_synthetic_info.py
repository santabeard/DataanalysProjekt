#!/usr/bin/env python3
"""Aggregated train_clean summary, for building a synthetic clone later.

Reads "<N>K_train_clean.parquet" (never writes to it) and writes one JSON
file of per-column aggregate statistics -- the same kind of thing
data_info/swed_profile.txt gives for the raw tables, but deliberately
STRICTER, because train_clean is a derived, already-filtered 500K-patient
table where a careless aggregate can re-identify someone. Two guards, applied
to every reported number:

  1. MIN_CELL_COUNT (default 10): a category with fewer than this many rows
     is never reported by name. It's merged into an "OTHER" bucket. This is
     the same threshold the project's own export rule already uses elsewhere
     (">= 10 patients per cell") -- reused here, not invented fresh.
  2. ROUND_TO (default 5): every reported count (row counts, category counts,
     even the count of how many rare categories got merged) is rounded to
     the nearest multiple of this. Same convention, reused.

Numeric columns report mean/std and a SAFE set of percentiles (5/25/50/75/95)
-- deliberately NOT min/max. A literal min or max is frequently a single
record (the one patient with the highest temperature, the longest
days-to-death, ...), which is exactly the kind of value that re-identifies
someone when published. This is the "guard against rare cases" requirement,
applied uniformly rather than column-by-column.

diag_hdia_first and complaint_sctid get the SAME generic categorical
treatment as everything else -- no special-casing needed, because the
MIN_CELL_COUNT rule already does the right thing: both are sparse, long-tail
columns, so most of their mass ends up merged into OTHER, and only codes
that already clear the real threshold in this specific 500K-patient sample
are named.

Hard-excluded, never touched beyond existence (listed in the output's
"excluded_columns" with why):
  - lopnr_swed: the real patient ID. Never aggregated, not even internally.
  - ed_visit_id: a row key, not a variable to model the distribution of.
  - complaint_bucket_* (one-hot columns, if 04 was run with
    --complaint-vocab-csv): fully derivable from complaint_sctid's own
    (already-safe) distribution plus the vocabulary file, so reporting them
    too would be redundant exposure for no extra synthesis information.
  - complaint_role, complaint_weight: fully determined by the two-complaint
    structure (see below), not independent variables.

Structural note on complaint rows: train_clean has 1 or 2 rows per visit
(decision 11 -- two-complaint visits are duplicated, not multi-hot). A
synthetic clone needs the VISIT-level rate of "this visit had 2 complaints",
not a naive per-ROW marginal of complaint_role (which is just an artifact of
that rate). This script reports that rate once, and computes complaint_sctid's
distribution over every occurrence (both roles), matching
03b_pruning_thresholds.py's own counting convention. All other columns are
summarized once per visit (deduplicated on ed_visit_id, primary row kept).

KNOWN LIMITATION, not an oversight: this reports MARGINAL (per-column)
statistics only, no joint/correlation structure (e.g. nothing like "vitals by
age bucket" or "diag_class by complaint"). Cross-tabulating columns multiplies
small-cell risk fast -- a joint table over even two sparse-ish columns can
have many cells with count 1, which is exactly the re-identification problem
this script exists to avoid. A synthetic clone built only from this file will
have realistic per-column distributions but WILL NOT preserve real
correlations between columns. If correlation-preserving synthesis is wanted
later, that needs its own careful, cell-count-guarded design -- not a
quick addition here.

In: "<N>K_train_clean.parquet", read-only.
Out: "<N>K_synthetic_info.json" in --subset-dir.

    python3 06_synthetic_info.py --subset-dir /home/theon/Desktop/pipeline/phase1/out_data/subset
    python3 06_synthetic_info.py --subset-dir ... --min-count 20 --round-to 10   # stricter
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

OUT_ROOT = Path(os.environ.get("SWED_OUT", "/home/theon/Desktop/data_std"))
ORIGINAL = Path("/cygnus/proj/nobackup/swed_db/clean_parquet")  # never written to

MIN_CELL_COUNT_DEFAULT = 10  # same threshold as the project's export rule elsewhere
ROUND_TO_DEFAULT = 5         # same convention

SAFE_PERCENTILES = (5, 25, 50, 75, 95)  # deliberately no 0/100 (min/max)

HARD_EXCLUDE = {"lopnr_swed", "ed_visit_id"}
STRUCTURAL_EXCLUDE = {"complaint_role", "complaint_weight"}  # derived, see docstring


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def refuse_to_write_into(out: Path, *protected: Path) -> None:
    """Outputs must never land in, or overwrite anything in, the source data folders."""
    o = out.expanduser().resolve()
    for p in protected:
        p = p.expanduser().resolve()
        if o == p or p in o.parents:
            sys.exit(f"refusing to write to {out}: it is inside the source data folder {p}")


def find_train_clean(subset_dir: Path, label: Optional[str]) -> "tuple[Path, str]":
    """Locate "<N>K_train_clean.parquet" (04's output), read-only."""
    if label:
        path = subset_dir / f"{label}_train_clean.parquet"
        if not path.exists():
            sys.exit(f"missing {path}")
        return path, label

    candidates = sorted(subset_dir.glob("*_train_clean.parquet"))
    if not candidates:
        sys.exit(f"no *_train_clean.parquet in {subset_dir}; run 04_train_clean.py first")
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        sys.exit(f"more than one *_train_clean.parquet in {subset_dir} ({names}); pass --label to pick one")
    path = candidates[0]
    found_label = path.name[: -len("_train_clean.parquet")]
    return path, found_label


def round_count(n: int, round_to: int) -> int:
    return int(round(n / round_to)) * round_to


def is_categorical_like(series: pd.Series, max_numeric_levels: int = 5) -> bool:
    """Any non-numeric dtype is always categorical -- checked via
    pd.api.types.is_numeric_dtype(), not `dtype == object`, since string
    columns read from parquet can land as classic object dtype (pandas
    1.2.4, Bianca) or an Arrow-backed string dtype (newer pandas), and only
    the former equals `object`. A numeric column with very few distinct
    non-null values (is_male, complaint_weight if it ever got this far,
    died_30d/diag_hdia_any if stored as 0/1 rather than bool) is treated as
    categorical too, rather than relying on column-name guesses."""
    if not pd.api.types.is_numeric_dtype(series):
        return True
    return series.dropna().nunique() <= max_numeric_levels


def summarize_categorical(series: pd.Series, min_count: int, round_to: int) -> dict:
    n_total = len(series)
    non_null = series.dropna()
    null_rate = 1.0 - len(non_null) / n_total if n_total else 0.0

    counts = non_null.astype(str).value_counts()  # str() so True/False and "x" compare uniformly
    kept = counts[counts >= min_count]
    rare = counts[counts < min_count]

    return {
        "type": "categorical",
        "n_rows": round_count(n_total, round_to),
        "null_rate": round(null_rate, 4),
        "categories": {str(cat): round_count(int(cnt), round_to) for cat, cnt in kept.items()},
        "other_bucket_count": round_count(int(rare.sum()), round_to) if len(rare) else 0,
        "n_rare_categories_merged": round_count(len(rare), round_to),
    }


def summarize_numeric(series: pd.Series, min_count: int, round_to: int,
                       percentiles=SAFE_PERCENTILES) -> dict:
    n_total = len(series)
    non_null = series.dropna()
    null_rate = 1.0 - len(non_null) / n_total if n_total else 0.0

    if len(non_null) < min_count:
        return {
            "type": "numeric",
            "n_rows": round_count(n_total, round_to),
            "null_rate": round(null_rate, 4),
            "suppressed": True,
            "reason": f"fewer than {min_count} non-null values",
        }

    out = {
        "type": "numeric",
        "n_rows": round_count(n_total, round_to),
        "null_rate": round(null_rate, 4),
        "n_non_null": round_count(len(non_null), round_to),
        "mean": float(non_null.mean()),
        "std": float(non_null.std()),
    }
    for p in percentiles:
        out[f"p{p}"] = float(non_null.quantile(p / 100))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset-dir", type=Path, default=OUT_ROOT / "subset",
                    help="where <N>K_train_clean.parquet lives (default $SWED_OUT/subset)")
    ap.add_argument("--label", default=None,
                    help="the <N>K size label (default: auto-detect from the one *_train_clean.parquet present)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the JSON here instead of <N>K_synthetic_info.json in --subset-dir")
    ap.add_argument("--min-count", type=int, default=MIN_CELL_COUNT_DEFAULT,
                    help=f"categories/non-null counts below this are suppressed (default {MIN_CELL_COUNT_DEFAULT})")
    ap.add_argument("--round-to", type=int, default=ROUND_TO_DEFAULT,
                    help=f"round every reported count to the nearest multiple of this (default {ROUND_TO_DEFAULT})")
    args = ap.parse_args()
    refuse_to_write_into(args.subset_dir, ORIGINAL)

    train_clean_path, label = find_train_clean(args.subset_dir, args.label)
    log(f"train_clean: {train_clean_path} (label {label}) -- read-only")
    dst = args.out if args.out else args.subset_dir / f"{label}_synthetic_info.json"
    refuse_to_write_into(dst, ORIGINAL, train_clean_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(train_clean_path)
    log(f"loaded {len(df):,} rows x {len(df.columns)} columns")

    bucket_cols = [c for c in df.columns if c.startswith("complaint_bucket_")]
    excluded = {c: "direct patient identifier, never aggregated" for c in HARD_EXCLUDE if c in df.columns}
    excluded.update({c: "derivable from complaint_sctid + the vocabulary file" for c in bucket_cols})
    excluded.update({c: "derived from the two-complaint structure, see structural_stats"
                      for c in STRUCTURAL_EXCLUDE if c in df.columns})

    # -- Structural stats: the two-complaint rate (decision 11) --------------
    sizes = df.groupby("ed_visit_id").size()
    n_visits_total = round_count(len(sizes), args.round_to)
    n_visits_two = round_count(int((sizes == 2).sum()), args.round_to)
    structural_stats = {
        "n_visits_total": n_visits_total,
        "n_visits_with_two_complaints": n_visits_two,
        "p_two_complaints": round(n_visits_two / n_visits_total, 4) if n_visits_total else 0.0,
    }
    log(f"structural: {n_visits_total:,} visits, {n_visits_two:,} with two complaints "
        f"({structural_stats['p_two_complaints']:.1%})")

    # -- complaint_sctid: every occurrence (both roles), matching 03b's count --
    columns = {}
    if "complaint_sctid" in df.columns:
        columns["complaint_sctid"] = summarize_categorical(df["complaint_sctid"], args.min_count, args.round_to)
        log(f"  complaint_sctid: {len(columns['complaint_sctid']['categories'])} category(ies) kept, "
            f"{columns['complaint_sctid']['other_bucket_count']:,} occurrences in OTHER")

    # -- everything else: one row per visit (primary row), generic summary ---
    visit_level = df.drop_duplicates(subset=["ed_visit_id"], keep="first")
    other_cols = [c for c in df.columns
                  if c not in HARD_EXCLUDE and c not in bucket_cols and c not in STRUCTURAL_EXCLUDE
                  and c != "complaint_sctid"]
    for col in other_cols:
        series = visit_level[col]
        if is_categorical_like(series):
            columns[col] = summarize_categorical(series, args.min_count, args.round_to)
        else:
            columns[col] = summarize_numeric(series, args.min_count, args.round_to)
        kind = columns[col]["type"]
        log(f"  {col}: {kind}" + (" [suppressed]" if columns[col].get("suppressed") else ""))

    result = {
        "source": train_clean_path.name,
        "min_count": args.min_count,
        "round_to": args.round_to,
        "note": "Marginal per-column statistics only -- no joint/correlation structure. "
                "See this script's module docstring for why.",
        "structural_stats": structural_stats,
        "excluded_columns": excluded,
        "columns": columns,
    }

    with dst.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, sort_keys=True)
    log(f"wrote {dst} ({len(columns)} column(s) summarized, {len(excluded)} excluded)")


if __name__ == "__main__":
    main()
