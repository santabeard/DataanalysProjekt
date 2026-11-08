"""Shared paths, constants and helpers for scripts_v2 (phase0 exploration, phase1 pipeline).

Implements notes/data_plan.pdf. Mirrors the conventions already established in
scripts_v1: SWED_DATA / SWED_OUT / SWED_WORK env vars, refuse_to_write_into() so
nothing is ever written under the source folder, and the export rule (>= 10
patients per cell, counts rounded to 5) for any table meant to leave the cluster.

Uses polars (lazy scans, streaming collect), per data_plan.pdf Section 5 "Tools".
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import polars as pl

DATA = Path(os.environ.get("SWED_DATA", "/cygnus/proj/nobackup/swed_db/clean_parquet"))
ORIGINAL = Path("/cygnus/proj/nobackup/swed_db/clean_parquet")  # never written to
OUT_ROOT = Path(os.environ.get("SWED_OUT", "/home/theon/Desktop/data_std"))
WORK = Path(os.environ.get("SWED_WORK", Path.cwd()))
REF = Path(os.environ.get("SWED_REF", Path(__file__).resolve().parent.parent / "data_info"))

EXPLORE_OUT = OUT_ROOT / "explore" / "phase0"   # phase0 output root, one subfolder per script
VITALS_MAP_CSV = WORK / "out" / "vitals_concept_map.csv"   # reviewed output of 01_map_vital_terms.py
EXCLUDE_CSV = WORK / "out" / "exclude_lopnr.csv"            # test patients, as in scripts_v1

MEAS_SOURCES = ["ehr_karolinska_meas", "ehr_danderyd_lab_meas", "ehr_sa_vgr_lab",
                "ehr_karolinska_lab", "ehr_sahlgrenska_lab"]

# The 7 vital concepts and their SNOMED codes, per data_plan.pdf Section 4 "Shared step".
# These codes differ from scripts_v1/02_build.py's VITAL_CONCEPTS (data_plan.pdf
# supersedes it) -- verify both in the SNOMED browser before trusting either.
VITAL_CONCEPTS: dict[str, str] = {
    "systolic_bp":   "271649006",
    "diastolic_bp":  "271650006",
    "spo2":          "284032008",
    "temperature":   "56342008",
    "pulse":         "65653002",
    "resp_rate":     "86290005",
    "consciousness": "6942003",
}

# term_name regex -> concept, candidate matches only. First match wins. The plan
# requires every unmapped term name to be listed and reviewed by hand before
# being trusted (01_map_vital_terms.py); these regexes just seed that review.
TERM_REGEX: list[tuple[str, str]] = [
    ("bp_combined",   r"^blodtryck$|^blodtryck\b(?!.*(?:systol|diastol))"),
    ("systolic_bp",   r"systol"),
    ("diastolic_bp",  r"diastol"),
    ("pulse",         r"\bpuls\b|hjärtfrekv"),
    ("resp_rate",     r"andningsfrekv|\bandning"),
    ("temperature",   r"\btemp"),
    ("spo2",          r"saturation|spo2"),
    ("consciousness", r"medvetande|\bretts\b|\brls\b"),
    ("gcs_subscore",  r"glasgow|\bgcs\b"),
]

# Consciousness-scale name patterns, for 08_consciousness_scales.py.
SCALE_REGEX: list[tuple[str, str]] = [
    ("RETTS", r"\bretts\b"),
    ("NEWS",  r"\bnews\b"),
    ("GCS",   r"glasgow|\bgcs\b"),
    ("RLS",   r"\brls\b"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _detect_engine_kw() -> dict:
    """Polars' streaming-collect API changed across versions (streaming=True ->
    engine="streaming"), and the cluster's polars version is itself an open
    question (data_plan.pdf Section 7, item 7). Detect once, at import time,
    which kwarg this installation wants; every script below does
    `.collect(**ENGINE_KW)` instead of a hardcoded one."""
    import inspect
    try:
        params = inspect.signature(pl.LazyFrame.collect).parameters
    except (TypeError, ValueError):
        return {}
    if "engine" in params:
        return {"engine": "streaming"}
    if "streaming" in params:
        return {"streaming": True}
    return {}


ENGINE_KW = _detect_engine_kw()


def refuse_to_write_into(out: Path, *protected: Path) -> None:
    """Outputs must never land in, or overwrite anything in, the source data folders."""
    o = out.expanduser().resolve()
    for p in protected:
        p = p.expanduser().resolve()
        if o == p or p in o.parents:
            sys.exit(f"refusing to write to {out}: it is inside the source data folder {p}")


def table(name: str) -> Path:
    return DATA / f"{name}.parquet"


def prepare_out(subdir: str) -> Path:
    """Standard output folder for one phase0 script: $SWED_OUT/explore/phase0/<subdir>/."""
    out = EXPLORE_OUT / subdir
    refuse_to_write_into(out, DATA, ORIGINAL)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_exclude() -> list[float]:
    if not EXCLUDE_CSV.exists():
        log(f"WARNING: {EXCLUDE_CSV} not found; test patients are NOT excluded")
        return []
    ids = pl.read_csv(EXCLUDE_CSV)["lopnr_swed"].cast(pl.Float64).to_list()
    log(f"excluding {len(ids)} lopnr from {EXCLUDE_CSV}")
    return ids


def apply_export_rule(df: pl.DataFrame, patients_col: str = "n_patients",
                       round_to: int = 5, min_patients: int = 10) -> pl.DataFrame:
    """Cells with fewer than min_patients distinct patients are dropped; every
    n_*/n count column is rounded to round_to. Call this on every table before
    it leaves the cluster (data_plan.pdf "Export rule")."""
    df = df.filter(pl.col(patients_col) >= min_patients)
    count_cols = [c for c in df.columns if c == "n" or c.startswith("n_")]
    exprs = [((pl.col(c) / round_to).round(0).cast(pl.Int64) * round_to).alias(c) for c in count_cols]
    return df.with_columns(exprs) if exprs else df


def regex_concept_map(term_names: list[str], patterns: list[tuple[str, str]]) -> pl.DataFrame:
    rows = []
    for t in term_names:
        if t is None:
            continue
        concept = None
        for c, rx in patterns:
            if re.search(rx, t, re.IGNORECASE):
                concept = c
                break
        rows.append({"term_name": t, "concept": concept})
    return pl.DataFrame(rows, schema={"term_name": pl.Utf8, "concept": pl.Utf8})


def vital_term_map(term_names: list[str], reviewed_csv: Path = VITALS_MAP_CSV) -> pl.DataFrame:
    """term_name -> concept. A reviewed CSV (term_name, concept) wins outright;
    otherwise falls back to the TERM_REGEX guesses over the given term names."""
    if reviewed_csv.exists():
        return pl.read_csv(reviewed_csv).select(["term_name", "concept"])
    return regex_concept_map(term_names, TERM_REGEX)


def load_visits(cols: list[str] | None = None) -> pl.LazyFrame:
    """swed_ed_core with a row-order ed_visit_id. Exploration only: the real
    pipeline (phase1, Stage 1) assigns ed_visit_id after sorting and exclusions.

    lopnr_swed and ed_arrival_datetime are always included, whatever `cols`
    says: every phase0 script joins on lopnr_swed and most call link_window(),
    which needs ed_arrival_datetime -- dropping either silently breaks that."""
    lf = pl.scan_parquet(table("swed_ed_core"))
    if cols:
        lf = lf.select(sorted(set(cols) | {"lopnr_swed", "ed_arrival_datetime"}))
    return lf.with_row_index("ed_visit_id")


def load_vital_rows(reviewed_csv: Path = VITALS_MAP_CSV) -> pl.LazyFrame:
    """Every vital-sign row from the 5 EHR measurement sources, concept-mapped.
    Columns: lopnr_swed, datetime_analysis, source, from, term_name, term_id,
    value_num, value_str, unit, concept."""
    frames = []
    for src in MEAS_SOURCES:
        if not table(src).exists():
            log(f"skip {src}: not found")
            continue
        schema = pl.scan_parquet(table(src)).collect_schema().names()
        cols = [c for c in ["lopnr_swed", "datetime_analysis", "from", "term_name", "term_id",
                             "value_num", "value_str", "unit"] if c in schema]
        lf = pl.scan_parquet(table(src)).select(cols).with_columns(pl.lit(src).alias("source"))
        for missing in ["from", "value_str", "unit", "term_id"]:
            if missing not in cols:
                lf = lf.with_columns(pl.lit(None).alias(missing))
        frames.append(lf)
    meas = pl.concat(frames, how="vertical_relaxed")

    terms = meas.select("term_name").unique().collect(**ENGINE_KW)["term_name"].to_list()
    vmap = vital_term_map(terms, reviewed_csv=reviewed_csv)
    return meas.join(vmap.lazy(), on="term_name", how="inner").filter(pl.col("concept").is_not_null())


def link_window(visits: pl.LazyFrame, meas: pl.LazyFrame, before_min: float, after_min: float) -> pl.LazyFrame:
    """Join meas to visits on lopnr_swed, keep rows with datetime_analysis in
    [arrival - before_min, arrival + after_min]. Adds minutes_from_arrival.

    This is an equi-join on lopnr_swed followed by a time filter, not an asof
    join: a patient's measurement rows are compared against every one of their
    own visits, which is what we want for exploration (a value may fall inside
    more than one visit's window for patients with closely spaced revisits)."""
    j = visits.join(meas, on="lopnr_swed", how="inner")
    j = j.with_columns(
        ((pl.col("datetime_analysis") - pl.col("ed_arrival_datetime")).dt.total_minutes())
        .alias("minutes_from_arrival")
    )
    return j.filter(
        pl.col("minutes_from_arrival").is_not_null()
        & (pl.col("minutes_from_arrival") >= -before_min)
        & (pl.col("minutes_from_arrival") <= after_min)
    )
