#!/usr/bin/env python3
"""Consciousness scales: which scale each hospital uses, and its values.

data_plan.pdf Section 4, "Also produced". Scales differ between hospitals
(RETTS, NEWS, GCS, RLS); this sets the alert/not-alert mapping per scale used
in phase1 Stage 2 (data_plan.pdf Section 2, "Consciousness").

Classifies each consciousness/GCS term_name by SCALE_REGEX (common.py) and
reports, per ed_hospital x scale, the value distribution.

Writes:
    consciousness_scale_by_hospital.csv   ed_hospital x scale, n, distinct values
    consciousness_value_counts.csv        ed_hospital x scale x value, n

    SWED_DATA=$SWED_OUT/subset python 08_consciousness_scales.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, SCALE_REGEX, apply_export_rule, link_window, load_vital_rows, load_visits, log, prepare_out, regex_concept_map, ENGINE_KW

BEFORE_MIN = 30
WINDOW_H = 4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    out = prepare_out("08_consciousness_scales")
    log(f"data: {DATA}")

    visits = load_visits(["lopnr_swed", "ed_hospital"])
    vitals = load_vital_rows().filter(pl.col("concept").is_in(["consciousness", "gcs_subscore"]))
    linked = link_window(visits, vitals, before_min=BEFORE_MIN, after_min=WINDOW_H * 60).collect(**ENGINE_KW)

    terms = linked["term_name"].unique().to_list()
    scale_map = regex_concept_map(terms, SCALE_REGEX).rename({"concept": "scale"})
    linked = linked.join(scale_map, on="term_name", how="left").with_columns(pl.col("scale").fill_null("unclassified"))

    by_hospital = (
        linked.group_by(["ed_hospital", "scale"])
        .agg(pl.len().alias("n_rows"), pl.col("lopnr_swed").n_unique().alias("n_patients"),
             pl.col("value_num").n_unique().alias("n_distinct_values"))
        .sort(["ed_hospital", "n_rows"], descending=[False, True])
    )
    by_hospital = apply_export_rule(by_hospital, "n_patients")
    by_hospital.write_csv(out / "consciousness_scale_by_hospital.csv")
    log(f"wrote consciousness_scale_by_hospital.csv: {by_hospital.height} rows")

    value_counts = (
        linked.group_by(["ed_hospital", "scale", "value_num"])
        .agg(pl.len().alias("n_rows"), pl.col("lopnr_swed").n_unique().alias("n_patients"))
        .sort(["ed_hospital", "scale", "value_num"])
    )
    value_counts = apply_export_rule(value_counts, "n_patients")
    value_counts.write_csv(out / "consciousness_value_counts.csv")
    log(f"wrote consciousness_value_counts.csv: {value_counts.height} rows -> set the alert/not-alert mapping per scale")


if __name__ == "__main__":
    main()
