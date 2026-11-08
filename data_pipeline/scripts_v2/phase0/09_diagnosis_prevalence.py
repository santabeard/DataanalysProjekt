#!/usr/bin/env python3
"""Diagnoses: prevalence of each code in days 0-30, and how many register rows
are block-coded (data_plan.pdf Section 4 "Also produced"; Section 7, open
question 6; Section 2, "Block-coded rows").

icd10_code length is used as the block-coded signal (a 3-character code is
assumed full precision; anything shorter is assumed block/chapter-only) --
THIS IS UNVERIFIED. The length_distribution.csv output exists specifically so
you can check that assumption against real icd10_code values before trusting
block_coded_share.csv. There is no ICD-10 block/chapter reference table in
data_info/ yet; 3-char-to-block-to-chapter rollup is NOT computed here.

Writes:
    icd10_length_distribution.csv   count of register rows by len(icd10_code)
                                     -- look at this first
    diagnosis_prevalence_3char.csv  prevalence of each 3-char code in days 0-30
                                     from arrival, among rows with len >= 3
    block_coded_share.csv           rows with len(icd10_code) < 3, by that
                                     short code -- candidate block identifiers

    SWED_DATA=$SWED_OUT/subset python 09_diagnosis_prevalence.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, apply_export_rule, load_visits, log, prepare_out, table, ENGINE_KW


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    out = prepare_out("09_diagnosis_prevalence")
    log(f"data: {DATA}")

    visits = load_visits(["lopnr_swed", "ed_arrival_datetime"])
    diag = pl.scan_parquet(table("sos_patreg_diag_long")).select(["lopnr_swed", "INDATUM", "icd10_code"])
    diag = diag.with_columns(pl.col("icd10_code").str.to_uppercase().str.replace(".", "", literal=True).str.strip_chars())

    length_dist = (
        diag.with_columns(pl.col("icd10_code").str.len_chars().alias("code_len"))
        .group_by("code_len").agg(pl.len().alias("n_rows"))
        .sort("code_len")
        .collect(**ENGINE_KW)
    )
    length_dist.write_csv(out / "icd10_length_distribution.csv")
    log(f"wrote icd10_length_distribution.csv: {length_dist.height} distinct lengths -- check which length means 'block-coded'")

    linked = (
        visits.join(diag, on="lopnr_swed", how="inner")
        .with_columns(((pl.col("INDATUM") - pl.col("ed_arrival_datetime").dt.date())
                       .dt.total_days()).alias("days_from_arrival"))
        .filter(pl.col("days_from_arrival").is_between(0, 30))
        .with_columns(pl.col("icd10_code").str.len_chars().alias("code_len"))
    )

    full = linked.filter(pl.col("code_len") >= 3).with_columns(pl.col("icd10_code").str.slice(0, 3).alias("icd10_3"))
    prevalence = (
        full.group_by("icd10_3")
        .agg(pl.col("ed_visit_id").n_unique().alias("n_patients"), pl.len().alias("n_rows"))
        .sort("n_patients", descending=True)
        .collect(**ENGINE_KW)
    )
    prevalence = apply_export_rule(prevalence, "n_patients")
    prevalence.write_csv(out / "diagnosis_prevalence_3char.csv")
    log(f"wrote diagnosis_prevalence_3char.csv: {prevalence.height} codes")

    short = linked.filter(pl.col("code_len") < 3)
    block_share = (
        short.group_by("icd10_code")
        .agg(pl.col("ed_visit_id").n_unique().alias("n_patients"), pl.len().alias("n_rows"))
        .sort("n_rows", descending=True)
        .collect(**ENGINE_KW)
    )
    block_share = apply_export_rule(block_share, "n_patients")
    block_share.write_csv(out / "block_coded_share.csv")
    log(f"wrote block_coded_share.csv: {block_share.height} short codes -- "
        f"map these to ICD-10 blocks/chapters by hand once confirmed as block-coded")


if __name__ == "__main__":
    main()
