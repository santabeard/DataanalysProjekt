#!/usr/bin/env python3
"""Question A: which visits have vitals? (data_plan.pdf Section 4, Question A)

For a wide window (arrival +/- 4h), computes:

    vitals_coverage_by_hospital_year.csv   share of visits with each vital
                                            concept, by ed_hospital x year
    hospital_source_candidates.csv         which 'from' sources carry vitals
                                            linked to which ed_hospital, with
                                            row/patient counts and year range
                                            -> fill in Appendix B by hand

Decides: the hospital-source mapping (data_plan.pdf Appendix B) and which
hospitals/years a model can use. Needs a reviewed vitals_concept_map.csv from
01_map_vital_terms.py; falls back to the regex guesses if that isn't there yet.

    SWED_DATA=$SWED_OUT/subset python 02_question_a_vitals_coverage.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, apply_export_rule, link_window, load_vital_rows, load_visits, log, prepare_out, ENGINE_KW

WINDOW_H = 4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    out = prepare_out("02_question_a_vitals_coverage")
    log(f"data: {DATA}")

    visits = load_visits(["lopnr_swed", "ed_hospital", "ed_arrival_datetime"]).with_columns(
        pl.col("ed_arrival_datetime").dt.year().alias("year"))
    vitals = load_vital_rows()

    linked = link_window(visits, vitals, before_min=WINDOW_H * 60, after_min=WINDOW_H * 60)

    n_visits = (
        visits.group_by(["ed_hospital", "year"])
        .agg(pl.col("ed_visit_id").n_unique().alias("n_visits_total"))
    )
    coverage = (
        linked.group_by(["ed_hospital", "year", "concept"])
        .agg(pl.col("ed_visit_id").n_unique().alias("n_patients"),  # distinct visits, used as the export-rule count
             pl.len().alias("n_rows"))
        .join(n_visits, on=["ed_hospital", "year"], how="left")
        .with_columns((pl.col("n_patients") / pl.col("n_visits_total")).alias("share_with_vital"))
        .collect(**ENGINE_KW)
    )
    coverage = apply_export_rule(coverage, "n_patients").sort(["ed_hospital", "year", "concept"])
    coverage.write_csv(out / "vitals_coverage_by_hospital_year.csv")
    log(f"wrote vitals_coverage_by_hospital_year.csv: {coverage.height} rows")

    hospital_source = (
        linked.group_by(["ed_hospital", "source", "from"])
        .agg(pl.col("lopnr_swed").n_unique().alias("n_patients"),
             pl.len().alias("n_rows"),
             pl.col("year").min().alias("year_min"),
             pl.col("year").max().alias("year_max"))
        .collect(**ENGINE_KW)
    )
    hospital_source = apply_export_rule(hospital_source, "n_patients").sort(["ed_hospital", "n_rows"], descending=[False, True])
    hospital_source.write_csv(out / "hospital_source_candidates.csv")
    log(f"wrote hospital_source_candidates.csv: {hospital_source.height} rows -> fill in Appendix B by hand")


if __name__ == "__main__":
    main()
