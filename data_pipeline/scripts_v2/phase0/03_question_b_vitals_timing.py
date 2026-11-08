#!/usr/bin/env python3
"""Question B: when is the first vital measured? (data_plan.pdf Section 4, Question B)

For visits with a value within 4h of arrival: the cumulative share of first
values by minute after arrival, from -30 min, per vital concept and hospital.

Decides the triage cutoff, by the rule fixed in the plan:
    the smallest whole number of minutes at which at least 90% of the first
    values seen within 4 hours have arrived, capped at 120 min, the same for
    all hospitals and vitals.

Writes:
    first_value_minutes.csv     one row per (ed_visit_id, concept): the
                                 earliest minutes_from_arrival in the window
    coverage_curve.csv          cumulative share of first values arrived, by
                                 minute, per concept and per concept x hospital
    cutoff_recommendation.csv   one row: the overall cutoff by the rule above,
                                 plus the per-hospital cutoffs for comparison

    SWED_DATA=$SWED_OUT/subset python 03_question_b_vitals_timing.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, link_window, load_vital_rows, load_visits, log, prepare_out, ENGINE_KW

BEFORE_MIN = 30
WINDOW_H = 4
TARGET_SHARE = 0.90
CAP_MIN = 120


def cutoff_for(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """Smallest minute where the cumulative share of first values >= TARGET_SHARE,
    capped at CAP_MIN. df must have one row per (ed_visit_id, concept[, group])."""
    total = df.group_by(group_cols).agg(pl.len().alias("n_total"))
    curve = (
        df.group_by(group_cols + ["first_minute"])
        .agg(pl.len().alias("n_at_minute"))
        .sort(group_cols + ["first_minute"])
        .with_columns(pl.col("n_at_minute").cum_sum().over(group_cols).alias("n_cum"))
        .join(total, on=group_cols, how="left")
        .with_columns((pl.col("n_cum") / pl.col("n_total")).alias("cum_share"))
    )
    reached = curve.filter(pl.col("cum_share") >= TARGET_SHARE).group_by(group_cols).agg(
        pl.col("first_minute").min().alias("cutoff_min_raw"))
    reached = reached.with_columns(pl.col("cutoff_min_raw").clip(upper_bound=CAP_MIN).alias("cutoff_min"))
    return curve, reached


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    out = prepare_out("03_question_b_vitals_timing")
    log(f"data: {DATA}")

    visits = load_visits(["lopnr_swed", "ed_hospital", "ed_arrival_datetime"])
    vitals = load_vital_rows()
    linked = link_window(visits, vitals, before_min=BEFORE_MIN, after_min=WINDOW_H * 60)

    first_values = (
        linked.group_by(["ed_visit_id", "ed_hospital", "concept"])
        .agg(pl.col("minutes_from_arrival").min().alias("first_minute"))
        .with_columns(pl.col("first_minute").ceil().cast(pl.Int64))
        .collect(**ENGINE_KW)
    )
    first_values.write_csv(out / "first_value_minutes.csv")
    log(f"wrote first_value_minutes.csv: {first_values.height} rows")

    curve_overall, cutoff_overall = cutoff_for(first_values, ["concept"])
    curve_hosp, cutoff_hosp = cutoff_for(first_values, ["concept", "ed_hospital"])

    curve_overall.write_csv(out / "coverage_curve_by_concept.csv")
    curve_hosp.write_csv(out / "coverage_curve_by_concept_hospital.csv")
    log("wrote coverage_curve_by_concept.csv, coverage_curve_by_concept_hospital.csv")

    overall_cutoff = min(int(cutoff_overall["cutoff_min"].max()), CAP_MIN) if cutoff_overall.height else CAP_MIN
    cutoff_overall.write_csv(out / "cutoff_by_concept.csv")
    cutoff_hosp.write_csv(out / "cutoff_by_concept_hospital.csv")
    pl.DataFrame({
        "rule": [f">= {TARGET_SHARE:.0%} of first values within {WINDOW_H}h, cap {CAP_MIN} min"],
        "recommended_cutoff_min": [overall_cutoff],
    }).write_csv(out / "cutoff_recommendation.csv")
    log(f"recommended single cutoff (worst concept, capped): {overall_cutoff} min "
        f"-> compare against coverage_curve_by_concept_hospital.csv before fixing it")


if __name__ == "__main__":
    main()
