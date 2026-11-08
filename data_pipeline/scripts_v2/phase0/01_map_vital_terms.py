#!/usr/bin/env python3
"""Shared step: map measurement term names to the 7 vital concepts (+ GCS subscores).

data_plan.pdf Section 4, "Shared step". Scans every distinct term_name across the
5 EHR measurement sources, applies the TERM_REGEX candidate map (common.py), and
writes:

    vital_term_candidates.csv   every distinct term_name x source, row count, and
                                 the regex-guessed concept (blank if no match)
    unmapped_terms.csv          the subset with no regex match, sorted by row
                                 count, for hand review

Review both files by hand, fix any wrong guesses, and save the corrected
mapping as $SWED_WORK/out/vitals_concept_map.csv (columns: term_name, concept).
Every other phase0 script picks that file up automatically (common.VITALS_MAP_CSV)
and falls back to the regex guesses if it isn't there yet.

    SWED_DATA=$SWED_OUT/subset python 01_map_vital_terms.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, MEAS_SOURCES, VITALS_MAP_CSV, log, prepare_out, regex_concept_map, table, TERM_REGEX, ENGINE_KW


def term_counts(src: str) -> pl.DataFrame | None:
    if not table(src).exists():
        log(f"skip {src}: not found")
        return None
    return (
        pl.scan_parquet(table(src))
        .select("term_name")
        .group_by("term_name")
        .agg(pl.len().alias("n"))
        .collect(**ENGINE_KW)
        .with_columns(pl.lit(src).alias("source"))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    out = prepare_out("01_vital_term_mapping")
    log(f"data: {DATA}")

    counts = [c for c in (term_counts(s) for s in MEAS_SOURCES) if c is not None]
    counts = pl.concat(counts, how="vertical_relaxed").filter(pl.col("term_name").is_not_null())

    guess = regex_concept_map(counts["term_name"].unique().to_list(), TERM_REGEX)
    joined = counts.join(guess, on="term_name", how="left").sort(["source", "n"], descending=[False, True])
    joined.write_csv(out / "vital_term_candidates.csv")
    log(f"wrote vital_term_candidates.csv: {joined.height} rows")

    unmapped = joined.filter(pl.col("concept").is_null()).sort("n", descending=True)
    unmapped.write_csv(out / "unmapped_terms.csv")
    log(f"wrote unmapped_terms.csv: {unmapped.height} distinct terms with no regex match")

    log(f"NOTE: no reviewed map yet at {VITALS_MAP_CSV}" if not VITALS_MAP_CSV.exists()
        else f"using reviewed map at {VITALS_MAP_CSV} is NOT applied by this script -- it only reports regex guesses")
    log("Review both CSVs by hand, then save corrections as "
        f"{VITALS_MAP_CSV} (columns: term_name, concept)")


if __name__ == "__main__":
    main()
