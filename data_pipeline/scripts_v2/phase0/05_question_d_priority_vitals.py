#!/usr/bin/env python3
"""Question D: how strongly does priority depend on vitals? (data_plan.pdf Section 4, Question D)

In RETTS the priority is partly computed from the vitals, so a model that
includes priority may show vitals as useless by comparison. Computes:

    vital_distribution_by_priority.csv   mean/median/std of each vital's
                                          earliest-in-window value, by
                                          ed_priority_in
    priority_r2_from_vitals.csv          one row: R^2 of an OLS regression of
                                          ed_priority_in on the 7 vitals
                                          (visits with all 7 present only)

Decides: whether priority is used as a predictor, and how the report explains it.

    SWED_DATA=$SWED_OUT/subset python 05_question_d_priority_vitals.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, VITAL_CONCEPTS, apply_export_rule, link_window, load_vital_rows, load_visits, log, prepare_out, ENGINE_KW

BEFORE_MIN = 30
WINDOW_H = 4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    out = prepare_out("05_question_d_priority_vitals")
    log(f"data: {DATA}")

    visits = load_visits(["lopnr_swed", "ed_priority_in"])
    vitals = load_vital_rows()
    linked = link_window(visits, vitals, before_min=BEFORE_MIN, after_min=WINDOW_H * 60)

    # Earliest value per (visit, concept) in the window, as a first look (not
    # the final triage-window value, which needs the Question B cutoff).
    earliest = (
        linked.filter(pl.col("value_num").is_not_null())
        .sort("minutes_from_arrival")
        .group_by(["ed_visit_id", "ed_priority_in", "concept"])
        .agg(pl.col("value_num").first().alias("value"))
        .collect(**ENGINE_KW)
    )

    dist = (
        earliest.group_by(["concept", "ed_priority_in"])
        .agg(pl.len().alias("n_patients"), pl.col("value").mean().alias("mean"),
             pl.col("value").median().alias("median"), pl.col("value").std().alias("std"))
        .sort(["concept", "ed_priority_in"])
    )
    dist = apply_export_rule(dist, "n_patients")
    dist.write_csv(out / "vital_distribution_by_priority.csv")
    log(f"wrote vital_distribution_by_priority.csv: {dist.height} rows")

    wide = earliest.pivot(on="concept", index=["ed_visit_id", "ed_priority_in"], values="value")
    concepts_present = [c for c in VITAL_CONCEPTS if c in wide.columns]
    complete = wide.drop_nulls(subset=concepts_present + ["ed_priority_in"])
    log(f"{complete.height} visits with ed_priority_in and all {len(concepts_present)} vitals present")

    if complete.height > len(concepts_present) + 1:
        y = complete["ed_priority_in"].to_numpy().astype(float)
        X = np.column_stack([complete[c].to_numpy().astype(float) for c in concepts_present])
        X = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        r2 = float("nan")

    pl.DataFrame({
        "n_complete_cases": [complete.height],
        "vitals_used": [",".join(concepts_present)],
        "r2_priority_from_vitals": [r2],
    }).write_csv(out / "priority_r2_from_vitals.csv")
    log(f"OLS R^2 of ed_priority_in on vitals: {r2:.3f}" if r2 == r2 else "not enough complete cases for R^2")


if __name__ == "__main__":
    main()
