#!/usr/bin/env python3
"""Question C: is missingness linked to the outcome? (data_plan.pdf Section 4, Question C)

For each vital concept, compares visits with vs. without that vital (present
within +/- 4h of arrival, same definition as Question A) on:

    30-day mortality, and
    the rate of each of the 20 most common diagnosis codes (3-char ICD-10,
    INDATUM in days 0-30 from arrival)

broken down by ed_priority_in and ed_arrival_type.

Decides: whether missing flags are enough, or a model must be restricted to
visits with vitals; what to say about bias in the report.

Writes:
    top20_diagnosis_codes.csv     the 20 codes used below, with overall prevalence
    missingness_vs_mortality.csv  mortality rate x (has_vital, priority, arrival_type)
    missingness_vs_diagnosis.csv  same, per top-20 diagnosis code

    SWED_DATA=$SWED_OUT/subset python 04_question_c_missingness_bias.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, apply_export_rule, link_window, load_vital_rows, load_visits, log, prepare_out, table, ENGINE_KW

WINDOW_H = 4
N_TOP_CODES = 20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    out = prepare_out("04_question_c_missingness_bias")
    log(f"data: {DATA}")

    visits = load_visits(["lopnr_swed", "ed_arrival_datetime", "ed_priority_in", "ed_arrival_type"])
    visits_df = visits.collect(**ENGINE_KW)

    # has_vital: one row per (ed_visit_id, concept) present within +/- 4h.
    vitals = load_vital_rows()
    linked = link_window(visits, vitals, before_min=WINDOW_H * 60, after_min=WINDOW_H * 60)
    has_vital = (
        linked.select(["ed_visit_id", "concept"]).unique()
        .collect(**ENGINE_KW)
        .with_columns(pl.lit(True).alias("has_vital"))
    )

    # Death within 30 days.
    deaths = pl.scan_parquet(table("sos_deathreg_wide")).select(["lopnr_swed", "date_death_minimp"]).collect(**ENGINE_KW)
    base = visits_df.join(deaths, on="lopnr_swed", how="left").with_columns(
        pl.col("ed_arrival_datetime").dt.date().alias("arrival_date"))
    base = base.with_columns(
        ((pl.col("date_death_minimp") - pl.col("arrival_date")).dt.total_days().is_between(0, 30))
        .fill_null(False).alias("died_30d")
    )

    # Top 20 diagnosis codes in days 0-30.
    diag = pl.scan_parquet(table("sos_patreg_diag_long")).select(["lopnr_swed", "INDATUM", "icd10_code"])
    diag_linked = (
        visits.select(["ed_visit_id", "lopnr_swed", "ed_arrival_datetime"])
        .join(diag, on="lopnr_swed", how="inner")
        .with_columns(((pl.col("INDATUM") - pl.col("ed_arrival_datetime").dt.date()).dt.total_days()).alias("days_from_arrival"))
        .filter(pl.col("days_from_arrival").is_between(0, 30))
        .with_columns(pl.col("icd10_code").str.to_uppercase().str.replace(".", "", literal=True).str.slice(0, 3).alias("icd10_3"))
        .collect(**ENGINE_KW)
    )
    top20 = (
        diag_linked.group_by("icd10_3").agg(pl.col("ed_visit_id").n_unique().alias("n_visits"))
        .sort("n_visits", descending=True).head(N_TOP_CODES)
    )
    top20.write_csv(out / "top20_diagnosis_codes.csv")
    log(f"wrote top20_diagnosis_codes.csv: {top20.height} codes")
    top_codes = top20["icd10_3"].to_list()

    def with_flags(concept: str) -> pl.DataFrame:
        hv = has_vital.filter(pl.col("concept") == concept).select(["ed_visit_id", "has_vital"])
        return base.join(hv, on="ed_visit_id", how="left").with_columns(pl.col("has_vital").fill_null(False))

    mort_rows = []
    diag_rows = []
    for concept in sorted(has_vital["concept"].unique().to_list()):
        flagged = with_flags(concept)
        m = (
            flagged.group_by(["has_vital", "ed_priority_in", "ed_arrival_type"])
            .agg(pl.len().alias("n_patients"), pl.col("died_30d").sum().alias("n_died_30d"))
            .with_columns(pl.lit(concept).alias("concept"),
                          (pl.col("n_died_30d") / pl.col("n_patients")).alias("mortality_rate"))
        )
        mort_rows.append(m)

        dcodes = flagged.select(["ed_visit_id", "has_vital", "ed_priority_in", "ed_arrival_type"]).join(
            diag_linked.filter(pl.col("icd10_3").is_in(top_codes)).select(["ed_visit_id", "icd10_3"]).unique(),
            on="ed_visit_id", how="left")
        for code in top_codes:
            d = (
                dcodes.with_columns((pl.col("icd10_3") == code).fill_null(False).alias("has_code"))
                .group_by(["has_vital", "ed_priority_in", "ed_arrival_type"])
                .agg(pl.len().alias("n_patients"), pl.col("has_code").sum().alias("n_with_code"))
                .with_columns(pl.lit(concept).alias("concept"), pl.lit(code).alias("icd10_3"),
                              (pl.col("n_with_code") / pl.col("n_patients")).alias("code_rate"))
            )
            diag_rows.append(d)

    mort_out = apply_export_rule(pl.concat(mort_rows), "n_patients").sort(["concept", "has_vital"])
    mort_out.write_csv(out / "missingness_vs_mortality.csv")
    log(f"wrote missingness_vs_mortality.csv: {mort_out.height} rows")

    diag_out = apply_export_rule(pl.concat(diag_rows), "n_patients").sort(["concept", "icd10_3", "has_vital"])
    diag_out.write_csv(out / "missingness_vs_diagnosis.csv")
    log(f"wrote missingness_vs_diagnosis.csv: {diag_out.height} rows")


if __name__ == "__main__":
    main()
