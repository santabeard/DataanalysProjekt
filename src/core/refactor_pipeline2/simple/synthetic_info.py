#!/usr/bin/env python3
"""Privacy-guarded per-column summary of train_clean, for building a synthetic clone.

Guards, applied to every reported number:
  - Categories with fewer than --min-count rows are merged into OTHER, never named.
  - Every count is rounded to the nearest multiple of --round-to.
  - Numeric columns report mean, std and p5/25/50/75/95, never min/max (often a
    single record).
Marginal statistics only: no cross-tabulation, which multiplies small-cell risk.
So a clone built from this has realistic columns but no real correlations.

complaint_sctid is counted over every row (both roles); other columns once per
visit. lopnr_swed, ed_visit_id, complaint_role/weight and complaint_bucket_*
are excluded (identifiers, or derivable from complaint_sctid).

    python3 synthetic_info.py
    python3 synthetic_info.py --min-count 20 --round-to 10   # stricter
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from common import INTERIM, TRAIN, guard_output, log

EXCLUDE = {"lopnr_swed", "ed_visit_id", "complaint_role", "complaint_weight"}
PERCENTILES = (5, 25, 50, 75, 95)


def summarize(s, min_count, round_to):
    r = lambda n: int(round(n / round_to)) * round_to
    non_null = s.dropna()
    out = {"n_rows": r(len(s)), "null_rate": round(1 - len(non_null) / len(s), 4) if len(s) else 0.0}

    if not pd.api.types.is_numeric_dtype(s) or non_null.nunique() <= 5:
        counts = non_null.astype(str).value_counts()
        rare = counts[counts < min_count]
        return {**out, "type": "categorical",
                "categories": {k: r(v) for k, v in counts[counts >= min_count].items()},
                "other_bucket_count": r(rare.sum()), "n_rare_categories_merged": r(len(rare))}

    if len(non_null) < min_count:
        return {**out, "type": "numeric", "suppressed": True}
    return {**out, "type": "numeric", "n_non_null": r(len(non_null)),
            "mean": float(non_null.mean()), "std": float(non_null.std()),
            **{f"p{p}": float(non_null.quantile(p / 100)) for p in PERCENTILES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path, default=TRAIN, help="where <N>K_train_clean.parquet is")
    ap.add_argument("--out-dir", type=Path, default=INTERIM, help="where the JSON goes")
    ap.add_argument("--label", default="500K")
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--round-to", type=int, default=5)
    args = ap.parse_args()

    src = args.train_dir / f"{args.label}_train_clean.parquet"
    dst = args.out_dir / f"{args.label}_synthetic_info.json"
    guard_output(dst)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src)

    sizes = df.groupby("ed_visit_id").size()
    n_visits = int(round(len(sizes) / args.round_to)) * args.round_to
    n_two = int(round((sizes == 2).sum() / args.round_to)) * args.round_to

    columns = {"complaint_sctid": summarize(df["complaint_sctid"], args.min_count, args.round_to)}
    visit_level = df.drop_duplicates("ed_visit_id")
    for col in df.columns:
        if col in EXCLUDE or col == "complaint_sctid" or col.startswith("complaint_bucket_"):
            continue
        columns[col] = summarize(visit_level[col], args.min_count, args.round_to)

    result = {
        "source": src.name, "min_count": args.min_count, "round_to": args.round_to,
        "structural_stats": {"n_visits_total": n_visits, "n_visits_with_two_complaints": n_two,
                             "p_two_complaints": round(n_two / n_visits, 4) if n_visits else 0.0},
        "excluded_columns": sorted(EXCLUDE | {c for c in df.columns if c.startswith("complaint_bucket_")}),
        "columns": columns,
    }
    dst.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    log(f"wrote {dst} ({len(columns)} columns summarized)")


if __name__ == "__main__":
    main()
