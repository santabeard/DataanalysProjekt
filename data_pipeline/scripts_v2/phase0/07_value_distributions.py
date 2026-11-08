#!/usr/bin/env python3
"""Value distributions per vital x source, to set the plausible limits.

data_plan.pdf Section 4, "Also produced": used together with a cited reference
to set config.yaml's implausible-value limits (Section 2, "Implausible values").

Writes:
    value_distributions.csv   concept x source: n, min, max and a wide
                               percentile ladder of value_num

    SWED_DATA=$SWED_OUT/subset python 07_value_distributions.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, load_vital_rows, log, prepare_out, ENGINE_KW

PERCENTILES = [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    out = prepare_out("07_value_distributions")
    log(f"data: {DATA}")

    vitals = load_vital_rows().filter(pl.col("value_num").is_not_null())
    agg = [pl.len().alias("n"), pl.col("value_num").min().alias("min"), pl.col("value_num").max().alias("max")]
    agg += [pl.col("value_num").quantile(q).alias(f"p{q * 100:g}") for q in PERCENTILES]

    dist = (
        vitals.group_by(["concept", "source"])
        .agg(agg)
        .sort(["concept", "source"])
        .collect(**ENGINE_KW)
    )
    dist.write_csv(out / "value_distributions.csv")
    log(f"wrote value_distributions.csv: {dist.height} rows -> set config.yaml's implausible-value limits")


if __name__ == "__main__":
    main()
