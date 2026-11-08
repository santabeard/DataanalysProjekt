#!/usr/bin/env python3
"""Units table: source x term x unit, row count and value percentiles.

data_plan.pdf Section 4, "Also produced": the input to pipeline/vital_units.csv
(keep / convert / drop), reviewed by hand once this script's output has been
looked at.

Writes:
    units_table.csv   source x term_name x unit x concept, n, and the 1/5/50/95/99
                       percentiles of value_num -- the starting point for a
                       reviewed vital_units.csv

    SWED_DATA=$SWED_OUT/subset python 06_units_table.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, load_vital_rows, log, prepare_out, ENGINE_KW

PERCENTILES = [0.01, 0.05, 0.50, 0.95, 0.99]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    out = prepare_out("06_units_table")
    log(f"data: {DATA}")

    vitals = load_vital_rows()
    agg = [pl.len().alias("n"), pl.col("value_num").is_not_null().sum().alias("n_with_value")]
    agg += [pl.col("value_num").quantile(q).alias(f"p{int(q * 100):02d}") for q in PERCENTILES]

    units = (
        vitals.group_by(["source", "term_name", "concept", "unit"])
        .agg(agg)
        .sort(["concept", "source", "term_name", "n"], descending=[False, False, False, True])
        .collect(**ENGINE_KW)
    )
    units.write_csv(out / "units_table.csv")
    log(f"wrote units_table.csv: {units.height} rows -> review into out/vital_units.csv (keep/convert/drop)")


if __name__ == "__main__":
    main()
