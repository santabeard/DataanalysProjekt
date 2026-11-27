#!/usr/bin/env python3
"""Sanity-check <N>K_train_clean.parquet against the train_clean design rules.

    python3 check_train_clean.py [label] [train_dir]     # defaults: 500K, data/train
    python3 check_train_clean.py path/to/100K_uu_train_clean.parquet

Diagnosis checks are skipped for tables without diag_* columns (UU).
"""
import sys
from pathlib import Path

import pandas as pd

from common import TRAIN

arg = sys.argv[1] if len(sys.argv) > 1 else "500K"
if arg.endswith(".parquet"):
    path = Path(arg)
else:
    path = (Path(sys.argv[2]) if len(sys.argv) > 2 else TRAIN) / f"{arg}_train_clean.parquet"
df = pd.read_parquet(path)
failed = []


def check(name, ok):
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}")
    if not ok:
        failed.append(name)


print(f"rows: {len(df):,}  visits: {df['ed_visit_id'].nunique():,}  columns: {len(df.columns)}")

shapes = df.groupby("ed_visit_id")["complaint_role"].apply(tuple)
check("each visit is (primary,) or (primary, secondary)", shapes.isin([("primary",), ("primary", "secondary")]).all())
check("complaint_weight sums to 1 per visit", (df.groupby("ed_visit_id")["complaint_weight"].sum() - 1).abs().max() < 1e-9)
check('"none" is never a secondary complaint', not ((df["complaint_sctid"] == "none") & (df["complaint_role"] == "secondary")).any())

buckets = [c for c in df.columns if c.startswith("complaint_bucket_")]
check(f"exactly one of {len(buckets)} complaint_bucket_* columns set per row", (df[buckets].sum(axis=1) == 1).all())

check("died_30d has no nulls", df["died_30d"].notna().all())
if "diag_class" in df.columns:
    check("diag_hdia_any is False exactly when diag_class is NODIAG", ((df["diag_class"] == "NODIAG") == ~df["diag_hdia_any"]).all())
    check("diag_hdia_any has no nulls", df["diag_hdia_any"].notna().all())
    unknown = df.loc[df["diag_class"] == "other_unknown", "diag_hdia_first"].unique()
    if len(unknown):
        print(f"[NOTE] Hdia codes with no ICD-10 chapter: {unknown[:10]}")
else:
    print("[SKIP] no diag_* columns (UU table)")

visits = df.drop_duplicates("ed_visit_id")
for col in sorted(c for c in df.columns if c.startswith("vital_")):
    print(f"  {col}: {visits[col].notna().mean():.1%} of visits")

sys.exit(f"{len(failed)} check(s) failed: {failed}" if failed else 0)
