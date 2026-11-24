#!/usr/bin/env python3
"""Quick sanity check on <N>K_train_clean.parquet after 04_train_clean.py.

Not part of the pipeline itself -- a throwaway terminal check, not something
04 or any later stage depends on. Re-derives the invariants Section 3.1's
decisions imply and flags anything that doesn't hold, instead of just
dumping stats and leaving you to eyeball them.

    python3 check_train_clean.py /home/theon/Desktop/pipeline/phase1/out_data/subset
    python3 check_train_clean.py /home/theon/Desktop/pipeline/phase1/out_data/subset --label 500K
"""
import sys
from pathlib import Path

import pandas as pd

subset_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else None

if label is None:
    candidates = sorted(subset_dir.glob("*_train_clean.parquet"))
    if len(candidates) != 1:
        sys.exit(f"expected exactly one *_train_clean.parquet in {subset_dir}, found {candidates}")
    path = candidates[0]
else:
    path = subset_dir / f"{label}_train_clean.parquet"

df = pd.read_parquet(path)
problems = []


def check(name, ok, detail=""):
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        problems.append(name)


print(f"=== {path} ===")
print(f"rows: {len(df):,}  columns: {len(df.columns)}")
print()

# -- Row/visit structure (decisions 11-12) --------------------------------
n_visits = df["ed_visit_id"].nunique()
check("every ed_visit_id appears at least once", n_visits == df["ed_visit_id"].nunique())
check("complaint_role only primary/secondary", set(df["complaint_role"].unique()) <= {"primary", "secondary"})

role_counts = df.groupby("ed_visit_id")["complaint_role"].apply(lambda s: tuple(sorted(s.tolist())))
bad_shapes = role_counts[~role_counts.isin([("primary",), ("primary", "secondary")])]
check("each visit is exactly 1 row (primary only) or 2 rows (primary+secondary)",
      len(bad_shapes) == 0, f"{len(bad_shapes)} visit(s) with an unexpected row shape")

weight_sums = df.groupby("ed_visit_id")["complaint_weight"].sum()
bad_weights = weight_sums[(weight_sums - 1.0).abs() > 1e-9]
check("complaint_weight sums to 1.0 per visit", len(bad_weights) == 0,
      f"{len(bad_weights)} visit(s) don't sum to 1.0")

none_two_role = df[(df["complaint_sctid"] == "none") & (df["complaint_role"] == "secondary")]
check('no "none" complaint ever appears as a secondary row', len(none_two_role) == 0)

# -- One-hot complaint columns (decision 13), if present ------------------
bucket_cols = [c for c in df.columns if c.startswith("complaint_bucket_")]
if bucket_cols:
    row_sums = df[bucket_cols].sum(axis=1)
    check(f"exactly one complaint_bucket_* column is 1 per row ({len(bucket_cols)} bucket columns)",
          (row_sums == 1).all(), f"{(row_sums != 1).sum()} row(s) violate this")
else:
    print("[SKIP] no complaint_bucket_* columns -- 04 was run without --complaint-vocab-csv")

# -- Diagnosis class (decisions 8-9) ---------------------------------------
check("diag_hdia_any == False implies diag_class == NODIAG",
      (df.loc[~df["diag_hdia_any"], "diag_class"] == "NODIAG").all())
check("diag_hdia_any == True implies diag_class != NODIAG",
      (df.loc[df["diag_hdia_any"], "diag_class"] != "NODIAG").all())

other_unknown = df[df["diag_class"] == "other_unknown"]
if len(other_unknown):
    print(f"[NOTE] {len(other_unknown)} row(s) have diag_class == 'other_unknown' -- "
          f"an Hdia value icd10_chapter() couldn't parse; worth a look: "
          f"{other_unknown['diag_hdia_first'].unique()[:10]}")

# -- Booleans should have no nulls -----------------------------------------
for col in ["diag_hdia_any", "died_30d"]:
    check(f"{col} has no nulls", df[col].notna().all())

# -- Vitals: just coverage, informational, not pass/fail -------------------
print()
print("vital coverage (fraction of visits with a value -- compare against 02's log):")
visit_level = df.drop_duplicates(subset=["ed_visit_id"])
for col in sorted(c for c in df.columns if c.startswith("vital_")):
    pct = visit_level[col].notna().mean()
    print(f"  {col}: {pct:.1%}")

print()
if problems:
    print(f"{len(problems)} check(s) failed: {problems}")
    sys.exit(1)
print("All checks passed.")
