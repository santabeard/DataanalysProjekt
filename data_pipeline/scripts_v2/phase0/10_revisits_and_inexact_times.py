#!/usr/bin/env python3
"""Revisits and inexact arrival times (data_plan.pdf Section 4, "Also produced").

Revisits: share of visits within 6 hours and within 30 days of the same
patient's previous visit, computed from consecutive arrival times per lopnr_swed.

Inexact times: share of visits with ed_inexact_time = true, by ed_hospital and
year -- the input to the "Inexact arrival time" decision (Section 2): these
visits are dropped because a triage window needs a known arrival time.

Writes:
    revisit_rates.csv        overall share of visits that are a revisit within
                              6h / within 30d of the same patient's prior visit
    inexact_time_by_hospital_year.csv   share of visits with ed_inexact_time,
                                         by ed_hospital x year

    SWED_DATA=$SWED_OUT/subset python 10_revisits_and_inexact_times.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, apply_export_rule, load_visits, log, prepare_out, ENGINE_KW


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    out = prepare_out("10_revisits_and_inexact_times")
    log(f"data: {DATA}")

    visits = load_visits(["lopnr_swed", "ed_hospital", "ed_arrival_datetime", "ed_inexact_time"]).with_columns(
        pl.col("ed_arrival_datetime").dt.year().alias("year"))
    df = visits.sort(["lopnr_swed", "ed_arrival_datetime"]).collect(**ENGINE_KW)

    df = df.with_columns(
        pl.col("ed_arrival_datetime").diff().over("lopnr_swed").dt.total_minutes().alias("prev_visit_minutes")
    )
    revisit = df.with_columns(
        (pl.col("prev_visit_minutes") <= 6 * 60).fill_null(False).alias("revisit_within_6h"),
        (pl.col("prev_visit_minutes") <= 30 * 24 * 60).fill_null(False).alias("revisit_within_30d"),
    )
    n = revisit.height
    summary = pl.DataFrame({
        "n_visits": [n],
        "n_patients": [revisit["lopnr_swed"].n_unique()],
        "share_revisit_within_6h": [revisit["revisit_within_6h"].sum() / n],
        "share_revisit_within_30d": [revisit["revisit_within_30d"].sum() / n],
    })
    summary = apply_export_rule(summary, "n_patients")
    summary.write_csv(out / "revisit_rates.csv")
    log(f"wrote revisit_rates.csv: {summary.to_dicts()[0]}")

    inexact = (
        df.group_by(["ed_hospital", "year"])
        .agg(pl.len().alias("n_rows"), pl.col("lopnr_swed").n_unique().alias("n_patients"),
             pl.col("ed_inexact_time").fill_null(False).sum().alias("n_inexact"))
        .with_columns((pl.col("n_inexact") / pl.col("n_rows")).alias("share_inexact"))
        .sort(["ed_hospital", "year"])
    )
    inexact = apply_export_rule(inexact, "n_patients")
    inexact.write_csv(out / "inexact_time_by_hospital_year.csv")
    log(f"wrote inexact_time_by_hospital_year.csv: {inexact.height} rows")


if __name__ == "__main__":
    main()
