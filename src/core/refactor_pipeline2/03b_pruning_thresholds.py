#!/usr/bin/env python3
"""Settles 04_train_clean.py's two TODO(BLOCKED) items with real frequency data.

Runs between 03_diagnosis_death_join.py and 04_train_clean.py. Reads
train_raw (read-only, same as 04) and computes the real per-code frequency
counts that were missing on a local machine with no data access:

  - Hdia frequency, among visits with diag_hdia_any == True -- answers
    "which Hdia codes are common enough to get their own class" (Section
    3.1, decision 8 / --hdia-top-k-csv).
  - Complaint-code frequency, counting sct1 and sct2 occurrences separately
    (a two-complaint visit contributes one count to each code it names) --
    answers "which complaint SCTIDs are common enough for their own one-hot
    column" (Section 3.1, decision 13 / --complaint-vocab-csv).

Threshold policy: a minimum-occurrence count (--hdia-min-count,
--complaint-min-count), not a fixed top-K rank -- simplest defensible rule,
and the one generate_dataset.py's own pruning used conceptually (merge
anything under a minimum sample size). The defaults (30) are a starting
point, not a researched number -- the report below shows exactly how many
codes/visits move depending on the min-count you choose, so you can re-run
this with a different threshold before trusting the output.

Tooling: pandas, not polars (see 02_vitals_join.py's docstring for why).

Out (in --subset-dir, alongside train_raw):
    <N>K_hdia_top_k.csv         columns: Hdia, n          (pass --hdia-top-k-csv to 04)
    <N>K_complaint_vocab.csv    columns: sctid, n         (pass --complaint-vocab-csv to 04)
    <N>K_03b_pruning_report.csv coverage stats for both thresholds

    python3 03b_pruning_thresholds.py
    python3 03b_pruning_thresholds.py --hdia-min-count 50 --complaint-min-count 20
    python3 04_train_clean.py --hdia-top-k-csv <subset-dir>/<N>K_hdia_top_k.csv \\
                               --complaint-vocab-csv <subset-dir>/<N>K_complaint_vocab.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

OUT_ROOT = Path(os.environ.get("SWED_OUT", "/home/theon/Desktop/data_std"))
ORIGINAL = Path("/cygnus/proj/nobackup/swed_db/clean_parquet")  # never written to

_SCT_RE = re.compile(r"\[(\d+)\]")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def refuse_to_write_into(out: Path, *protected: Path) -> None:
    """Outputs must never land in, or overwrite anything in, the source data folders."""
    o = out.expanduser().resolve()
    for p in protected:
        p = p.expanduser().resolve()
        if o == p or p in o.parents:
            sys.exit(f"refusing to write to {out}: it is inside the source data folder {p}")


def find_train_raw(subset_dir: Path, label: Optional[str]) -> "tuple[Path, str]":
    """Same lookup as 02/03/04: locate "<N>K_train_raw.parquet". Read-only here."""
    if label:
        path = subset_dir / f"{label}_train_raw.parquet"
        if not path.exists():
            sys.exit(f"missing {path}")
        return path, label

    candidates = sorted(subset_dir.glob("*_train_raw.parquet"))
    if not candidates:
        sys.exit(f"no *_train_raw.parquet in {subset_dir}; run 01_subset_v2.py first")
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        sys.exit(f"more than one *_train_raw.parquet in {subset_dir} ({names}); pass --label to pick one")
    path = candidates[0]
    found_label = path.name[: -len("_train_raw.parquet")]
    return path, found_label


def hdia_frequency(visits: pd.DataFrame, min_count: int) -> "tuple[pd.DataFrame, dict]":
    diag = visits[visits["diag_hdia_any"] & visits["diag_hdia_first"].notna()]
    counts = (
        diag.groupby("diag_hdia_first").size().reset_index(name="n")
        .sort_values("n", ascending=False)
        .rename(columns={"diag_hdia_first": "Hdia"})
    )

    kept = counts[counts["n"] >= min_count]
    n_total_diag_visits = len(diag)
    n_covered = int(kept["n"].sum()) if len(kept) else 0
    report = {
        "hdia_distinct_codes": len(counts),
        "hdia_codes_kept_at_min_count": len(kept),
        "hdia_min_count": min_count,
        "hdia_diagnosed_visits_total": n_total_diag_visits,
        "hdia_visits_covered_by_kept_codes": n_covered,
        "hdia_coverage_pct": round(100 * n_covered / n_total_diag_visits, 1) if n_total_diag_visits else 0.0,
    }
    log(f"Hdia: {len(counts):,} distinct codes among {n_total_diag_visits:,} diagnosed visits; "
        f"{len(kept):,} codes clear min_count={min_count} ({report['hdia_coverage_pct']}% of diagnosed "
        f"visits covered -- the rest fall through to other_<chapter>)")
    return kept[["Hdia", "n"]], report


def complaint_frequency(visits: pd.DataFrame, min_count: int) -> "tuple[pd.DataFrame, dict]":
    distinct_raw = visits["ed_presented_complaint_std"].dropna().unique().tolist()

    # One row per (raw string, code) occurrence, parsed once per distinct string.
    rows = []
    for raw in distinct_raw:
        for part in raw.split("+"):
            m = _SCT_RE.search(part)
            if m:
                rows.append({"ed_presented_complaint_std": raw, "sctid": m.group(1)})
    code_per_raw = pd.DataFrame(rows, columns=["ed_presented_complaint_std", "sctid"])

    # Join back to visits so each occurrence is weighted by how many visits used that raw string.
    raw_counts = visits.groupby("ed_presented_complaint_std").size().reset_index(name="raw_n")
    weighted = code_per_raw.merge(raw_counts, on="ed_presented_complaint_std", how="left")
    counts = (
        weighted.groupby("sctid")["raw_n"].sum().reset_index(name="n")
        .sort_values("n", ascending=False)
    )

    kept = counts[counts["n"] >= min_count]
    n_complaint_occurrences = int(counts["n"].sum()) if len(counts) else 0
    n_covered = int(kept["n"].sum()) if len(kept) else 0
    report = {
        "complaint_distinct_codes": len(counts),
        "complaint_codes_kept_at_min_count": len(kept),
        "complaint_min_count": min_count,
        "complaint_occurrences_total": n_complaint_occurrences,
        "complaint_occurrences_covered_by_kept_codes": n_covered,
        "complaint_coverage_pct": round(100 * n_covered / n_complaint_occurrences, 1) if n_complaint_occurrences else 0.0,
    }
    log(f"complaints: {len(counts):,} distinct SCTIDs across {n_complaint_occurrences:,} occurrences "
        f"(sct1+sct2); {len(kept):,} codes clear min_count={min_count} ({report['complaint_coverage_pct']}% of "
        f"occurrences covered -- the rest fall through to other_complaint)")
    return kept[["sctid", "n"]], report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset-dir", type=Path, default=OUT_ROOT / "subset",
                    help="output of 01_subset_v2.py, with 02/03 already applied (default $SWED_OUT/subset)")
    ap.add_argument("--label", default=None,
                    help="the <N>K size label (default: auto-detect from the one *_train_raw.parquet present)")
    ap.add_argument("--hdia-min-count", type=int, default=30,
                    help="minimum diagnosed-visit count for an Hdia code to get its own class (default 30)")
    ap.add_argument("--complaint-min-count", type=int, default=30,
                    help="minimum occurrence count for a complaint SCTID to get its own column (default 30)")
    args = ap.parse_args()
    refuse_to_write_into(args.subset_dir, ORIGINAL)

    train_raw_path, label = find_train_raw(args.subset_dir, args.label)
    log(f"train_raw: {train_raw_path} (label {label}) -- read-only")

    visits = pd.read_parquet(
        train_raw_path,
        columns=["diag_hdia_any", "diag_hdia_first", "ed_presented_complaint_std"],
    )

    hdia_top_k, hdia_report = hdia_frequency(visits, args.hdia_min_count)
    complaint_vocab, complaint_report = complaint_frequency(visits, args.complaint_min_count)

    hdia_path = args.subset_dir / f"{label}_hdia_top_k.csv"
    complaint_path = args.subset_dir / f"{label}_complaint_vocab.csv"
    hdia_top_k.to_csv(hdia_path, index=False)
    complaint_vocab.to_csv(complaint_path, index=False)
    log(f"wrote {hdia_path} ({len(hdia_top_k):,} codes)")
    log(f"wrote {complaint_path} ({len(complaint_vocab):,} codes)")

    report_path = args.subset_dir / f"{label}_03b_pruning_report.csv"
    pd.DataFrame([{**hdia_report, **complaint_report}]).to_csv(report_path, index=False)
    log(f"report written to {report_path}")

    log("Next: 04_train_clean.py "
        f"--hdia-top-k-csv {hdia_path} --complaint-vocab-csv {complaint_path}")


if __name__ == "__main__":
    main()
