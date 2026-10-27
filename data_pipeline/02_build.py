#!/usr/bin/env python3
"""Build the analysis dataset: harmonised vital signs and labs, linked to ED visits.

Reads the SWED Parquet files in SWED_DATA (the subset from 01_subset.py while
developing, the full clean_parquet folder for the final run) and writes:

    patients.parquet      one row per patient: sex, birth date, death, number of ED visits
    ed_visits.parquet     swed_ed_core plus an ed_visit_id
    vitals.parquet        vital signs from every hospital, one concept and unit per measurement
    lab_meas.parquet      all other lab/measurement rows, with SNOMED codes and normalised units
    diagnoses.parquet     sos_patreg_diag_long, ED patients only
    deaths.parquet        sos_deathreg_wide, one row per person
    death_causes.parquet  sos_deathreg_diag_long
    build_report.csv      rows read, dropped and changed per step (check this every run)
    unmapped_terms.csv    lab/measurement terms with no SNOMED code, by source

Output goes to $SWED_OUT/built_<name of SWED_DATA folder> (SWED_OUT defaults to
/home/theon/Desktop/data_std), so a subset build never overwrites the full one.
Source files are only read; the script refuses to write into a source folder.
Everything in the output is real patient data.

Rules applied, in order:
  - Patients in out/exclude_lopnr.csv (Tolvan etc., from swed_tasks.ipynb) are removed everywhere.
  - Only patients with an ED visit are kept.
  - A lab/meas row is a vital sign if its term_name matches a concept in VITAL_CONCEPTS,
    or is listed in out/vitals_snomed_map.csv (your reviewed copy of the notebook's
    vitals_snomed_map_draft.csv; columns term_name, concept). The CSV wins over the regexes.
  - Vital units: aliases are normalised (slag/min -> /min); a missing unit is assumed to
    be the concept's canonical unit; a few known conversions are applied (°F, fractions,
    metres, grams), detected from the unit or, when the unit is missing, from the value.
    Values outside PLAUSIBLE are kept but flagged implausible=True.
  - Combined blood pressure strings ("120/80") are split into systolic and diastolic rows.
  - Labs get snomed_id from data_info/swed_lab_meas_to_snomed.csv and units from
    data_info/lab_unit_normalization_07sep2026.csv ("convert" rows rescaled, "drop" rows removed).
  - Vitals and labs get the ed_visit_id of the visit whose window contains them:
    from VISIT_BEFORE_MIN before arrival until arrival + ed_time_total_minutes
    (VISIT_FALLBACK_MIN when that is missing). Rows outside every window get null.

    SWED_DATA=/home/theon/Desktop/data_std/subset python 02_build.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

DATA = Path(os.environ.get("SWED_DATA", "/cygnus/proj/nobackup/swed_db/clean_parquet"))
WORK = Path(os.environ.get("SWED_WORK", Path.cwd()))  # where out/exclude_lopnr.csv etc. are read from
ORIGINAL = Path("/cygnus/proj/nobackup/swed_db/clean_parquet")  # never written to
OUT_ROOT = Path(os.environ.get("SWED_OUT", "/home/theon/Desktop/data_std"))  # everything that stays on Bianca
REF =Path(os.environ.get("SWED_REF", Path(__file__).resolve().parent / "data_info"))
EXCLUDE_CSV = WORK / "out" / "exclude_lopnr.csv"
VITAL_MAP_CSV = WORK / "out" / "vitals_snomed_map.csv"

MEAS_SOURCES = ["ehr_danderyd_lab_meas", "ehr_karolinska_meas", "ehr_karolinska_lab",
                "ehr_sahlgrenska_lab", "ehr_sa_vgr_lab"]
BATCH_ROWS = 2_000_000
VISIT_BEFORE_MIN = 60
VISIT_FALLBACK_MIN = 24 * 60

# =============================================================================
# Vital-sign concepts. SNOMED codes are observable entities, as proposed in
# swed_tasks.ipynb §3. They were written from memory: VERIFY each one in the
# SNOMED browser (Swedish edition) before publishing anything.
# Order matters: the first matching regex wins.
# =============================================================================

#                  concept           term_name regex                      snomed       canonical unit
VITAL_CONCEPTS = [
    ("pulse_irregular", r"oregelbund",                                "61086009",  None),
    ("bp_combined",     r"^blodtryck$|^blodtryck\b(?!.*(?:systol|diastol))", "",       "mmHg"),
    ("systolic_bp",     r"systol",                                    "271649006", "mmHg"),
    ("diastolic_bp",    r"diastol",                                   "271650006", "mmHg"),
    ("heart_rate",      r"hjärtfrekv",                                "364075005", "/min"),
    ("pulse_rate",      r"\bpuls",                                    "78564009",  "/min"),
    ("resp_rate",       r"andningsfrekv|andning",                     "86290005",  "/min"),
    ("body_temp",       r"temp",                                      "386725007", "°C"),
    ("spo2",            r"saturation|spo2",                           "431314004", "%"),
    ("consciousness",   r"medvetande|retts|\brls",                    "6942003",   "score"),
    ("gcs_total",       r"glasgow|gcs.*(?:tot|summa)|^gcs$",            "248241002", "score"),
    ("pain_score",      r"smärt|\bvas\b|\bnrs\b",                     "225908003", "score"),
    ("news_total",      r"\bnews",                                    "",          "score"),
    ("body_weight",     r"^vikt|kroppsvikt",                          "27113001",  "kg"),
    ("body_height",     r"^längd|kroppslängd",                        "50373000",  "cm"),
]
LAB_PREFIX = re.compile(r"^(?:a|v|c|k)?B-|^(?:f?P|S|U|Pt)-|^P/S-|^Csv-|^Ery-", re.I)  # blood gas "temp korr" etc.

PLAUSIBLE = {
    "systolic_bp": (40, 300), "diastolic_bp": (20, 200), "heart_rate": (20, 250), "pulse_rate": (20, 250),
    "resp_rate": (3, 80), "body_temp": (30, 44), "spo2": (50, 100), "consciousness": (1, 8),
    "gcs_total": (3, 15), "pain_score": (0, 10), "news_total": (0, 20),
    "body_weight": (1, 350), "body_height": (40, 230),
}

UNIT_ALIASES = {
    "mmhg": "mmHg", "mm hg": "mmHg",
    "/min": "/min", "slag/min": "/min", "slag/minut": "/min", "bpm": "/min", "1/min": "/min",
    "andetag/min": "/min", "andetag/minut": "/min", "andetag": "/min",
    "°c": "°C", "c": "°C", "grader": "°C", "grader c": "°C", "celsius": "°C", "°f": "°F", "f": "°F",
    "%": "%", "procent": "%",
    "kg": "kg", "g": "g", "gram": "g", "cm": "cm", "m": "m",
    "p": "score", "poäng": "score", "score": "score",
}
UNIT_CONVERSIONS = {  # (concept, unit as written) -> function to the canonical unit
    ("body_temp", "°F"): lambda v: (v - 32) * 5 / 9,
    ("body_weight", "g"): lambda v: v / 1000,
    ("body_height", "m"): lambda v: v * 100,
}
VALUE_REPAIRS = {  # unit missing and the value is clearly in another unit
    "body_temp": (lambda v: v > 50, lambda v: (v - 32) * 5 / 9, "°F"),
    "spo2": (lambda v: v <= 1.0, lambda v: v * 100, "fraction"),
    "body_height": (lambda v: v < 3, lambda v: v * 100, "m"),
}

VITALS_SCHEMA = pa.schema([
    ("lopnr_swed", pa.float64()), ("ed_visit_id", pa.int64()), ("source", pa.string()), ("from", pa.string()),
    ("datetime_analysis", pa.timestamp("us", tz="UTC")), ("term_name", pa.string()), ("term_id", pa.string()),
    ("concept", pa.string()), ("snomed_id", pa.string()), ("value", pa.float64()), ("unit", pa.string()),
    ("unit_original", pa.string()), ("unit_status", pa.string()), ("implausible", pa.bool_()),
])
LABS_SCHEMA = pa.schema([
    ("lopnr_swed", pa.float64()), ("ed_visit_id", pa.int64()), ("source", pa.string()), ("from", pa.string()),
    ("datetime_analysis", pa.timestamp("us", tz="UTC")), ("term_name", pa.string()), ("term_id", pa.string()),
    ("snomed_id", pa.string()), ("snomed_swe", pa.string()), ("snomed_quality", pa.string()),
    ("value_num", pa.float64()), ("value_str", pa.string()), ("unit", pa.string()), ("unit_original", pa.string()),
    ("unit_status", pa.string()), ("lt_flag", pa.bool_()), ("gt_flag", pa.bool_()),
])

REPORT: list[dict] = []


def refuse_to_write_into(out: Path, *protected: Path) -> None:
    """Outputs must never land in, or overwrite anything in, the source data folders."""
    o = out.expanduser().resolve()
    for p in protected:
        p = p.expanduser().resolve()
        if o == p or p in o.parents:
            sys.exit(f"refusing to write to {out}: it is inside the source data folder {p}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def report(step: str, source: str, metric: str, n: int) -> None:
    REPORT.append({"step": step, "source": source, "metric": metric, "n": int(n)})


def path(name: str) -> Path:
    return DATA / f"{name}.parquet"


class Sink:
    """Appends pandas batches to one Parquet file with a fixed schema."""

    def __init__(self, dst: Path, schema: pa.Schema):
        self.dst, self.schema, self.rows = dst, schema, 0
        self.tmp = dst.with_suffix(".parquet.tmp")
        self.writer = pq.ParquetWriter(self.tmp, schema, compression="zstd")

    def write(self, df: pd.DataFrame) -> None:
        if len(df):
            self.writer.write_table(pa.Table.from_pandas(df[self.schema.names], schema=self.schema, preserve_index=False))
            self.rows += len(df)

    def close(self) -> None:
        self.writer.close()
        self.tmp.rename(self.dst)
        log(f"wrote {self.dst.name}: {self.rows:,} rows")


# =============================================================================
# Patients and ED visits
# =============================================================================

def read_exclude() -> list[float]:
    if not EXCLUDE_CSV.exists():
        log(f"WARNING: {EXCLUDE_CSV} not found; test patients (Tolvan) are NOT excluded")
        return []
    ids = pd.read_csv(EXCLUDE_CSV)["lopnr_swed"].astype(float).tolist()
    log(f"excluding {len(ids)} lopnr from {EXCLUDE_CSV}")
    return ids


def build_visits(exclude: list[float]) -> pd.DataFrame:
    ed = pq.read_table(path("swed_ed_core")).to_pandas()
    report("ed_visits", "swed_ed_core", "rows read", len(ed))
    drop = ed["lopnr_swed"].isna() | ed["lopnr_swed"].isin(exclude)
    report("ed_visits", "swed_ed_core", "rows dropped: no lopnr or excluded", drop.sum())
    ed = ed[~drop].sort_values(["lopnr_swed", "ed_arrival_datetime"], kind="stable").reset_index(drop=True)
    ed.insert(0, "ed_visit_id", np.arange(len(ed), dtype="int64"))
    return ed


def build_patients(ed: pd.DataFrame, deaths: pd.DataFrame) -> pd.DataFrame:
    p = ed.groupby("lopnr_swed").agg(
        is_male=("is_male", lambda s: s.mode().iat[0] if s.notna().any() else pd.NA),
        n_ed_visits=("ed_visit_id", "size"),
        first_ed_arrival=("ed_arrival_datetime", "min"),
        last_ed_arrival=("ed_arrival_datetime", "max"),
    ).reset_index()
    sex_conflict = ed.groupby("lopnr_swed")["is_male"].nunique() > 1
    report("patients", "swed_ed_core", "patients with conflicting is_male (mode used)", sex_conflict.sum())
    if path("swed_ed_core_birthdate").exists():
        bd = pq.read_table(path("swed_ed_core_birthdate")).to_pandas().drop_duplicates("lopnr_swed")
        p = p.merge(bd, on="lopnr_swed", how="left")
    else:
        p["birth_date"] = pd.NaT
    p = p.merge(deaths[["lopnr_swed", "date_death_minimp", "inexact_death_date", "ULORSAK"]], on="lopnr_swed", how="left")
    report("patients", "", "patients", len(p))
    report("patients", "", "patients with a death record", p["date_death_minimp"].notna().sum())
    return p


# =============================================================================
# Linking measurements to ED visits
# =============================================================================

def visit_windows(ed: pd.DataFrame) -> pd.DataFrame:
    w = ed[["lopnr_swed", "ed_visit_id", "ed_arrival_datetime", "ed_time_total_minutes"]].dropna(subset=["ed_arrival_datetime"])
    minutes = w["ed_time_total_minutes"].clip(lower=0).fillna(VISIT_FALLBACK_MIN)
    w = w.assign(win_start=w["ed_arrival_datetime"] - pd.Timedelta(minutes=VISIT_BEFORE_MIN),
                 win_end=w["ed_arrival_datetime"] + pd.to_timedelta(minutes, unit="min"))
    return w[["lopnr_swed", "ed_visit_id", "win_start", "win_end"]].sort_values("win_start")


def link_visits(df: pd.DataFrame, windows: pd.DataFrame) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")
    has_t = df["datetime_analysis"].notna()
    if not has_t.any():
        return out
    left = df.loc[has_t, ["lopnr_swed", "datetime_analysis"]].reset_index().sort_values("datetime_analysis")
    # pandas < 2.2 can't merge_asof on a float by-key; lopnr_swed holds whole numbers, so use int64.
    left["lopnr_swed"] = left["lopnr_swed"].astype("int64")
    windows = windows.assign(lopnr_swed=windows["lopnr_swed"].astype("int64"))
    # merge_asof needs identical datetime resolutions; the real files mix us and ns.
    left["datetime_analysis"] = left["datetime_analysis"].astype("datetime64[ns, UTC]")
    windows = windows.assign(win_start=windows["win_start"].astype("datetime64[ns, UTC]"),
                             win_end=windows["win_end"].astype("datetime64[ns, UTC]"))
    m = pd.merge_asof(left, windows, left_on="datetime_analysis", right_on="win_start",
                      by="lopnr_swed", direction="backward")
    inside = m["datetime_analysis"] <= m["win_end"]
    out.loc[m.loc[inside, "index"].to_numpy()] = m.loc[inside, "ed_visit_id"].astype("int64").to_numpy()
    return out


# =============================================================================
# Vital signs
# =============================================================================

def vital_term_map(term_names: pd.Series) -> pd.DataFrame:
    """term_name -> concept. A reviewed out/vitals_snomed_map.csv overrides the regexes."""
    terms = pd.Series(term_names.dropna().unique(), name="term_name")
    concept = pd.Series(pd.NA, index=terms.index, dtype="object")
    is_lab = terms.str.contains(LAB_PREFIX)
    for key, rx, _, _ in reversed(VITAL_CONCEPTS):  # reversed so the first rule wins
        concept[terms.str.contains(rx, case=False, regex=True) & ~is_lab] = key
    m = pd.DataFrame({"term_name": terms, "concept": concept})
    if VITAL_MAP_CSV.exists():
        reviewed = pd.read_csv(VITAL_MAP_CSV, dtype=str)[["term_name", "concept"]]
        m = m.drop(columns="concept").merge(reviewed, on="term_name", how="left")
        m["concept"] = m["concept"].replace({"": pd.NA})
    return m.dropna(subset=["concept"])


def split_combined_bp(v: pd.DataFrame) -> pd.DataFrame:
    bp = v["concept"] == "bp_combined"
    if not bp.any():
        return v
    parts = v.loc[bp, "value_str"].astype("string").str.extract(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})\s*$").astype(float)
    ok = parts.notna().all(axis=1)
    sys_ = v.loc[bp][ok].assign(concept="systolic_bp", value=parts.loc[ok, 0])
    dia_ = v.loc[bp][ok].assign(concept="diastolic_bp", value=parts.loc[ok, 1])
    return pd.concat([v[~bp], sys_, dia_, v.loc[bp][~ok]], ignore_index=True)


def harmonise_vitals(v: pd.DataFrame) -> pd.DataFrame:
    canonical = {k: u for k, _, _, u in VITAL_CONCEPTS}
    snomed = {k: s for k, _, s, _ in VITAL_CONCEPTS}
    v = split_combined_bp(v)
    v["snomed_id"] = v["concept"].map(snomed).replace({"": pd.NA})
    v["unit_original"] = v["unit"]
    written = v["unit"].astype("string").str.strip().str.lower().map(UNIT_ALIASES)
    v["unit"] = v["concept"].map(canonical)
    v["unit_status"] = np.where(v["unit_original"].isna(), "assumed", np.where(written.isna(), "unknown", "given"))

    for (concept, unit), f in UNIT_CONVERSIONS.items():
        hit = (v["concept"] == concept) & (written == unit)
        v.loc[hit, "value"] = f(v.loc[hit, "value"])
        v.loc[hit, "unit_status"] = "converted"
    for concept, (test, f, _) in VALUE_REPAIRS.items():
        hit = (v["concept"] == concept) & v["unit_original"].isna() & v["value"].notna()
        hit &= test(v["value"].where(hit, np.nan)).fillna(False)
        v.loc[hit, "value"] = f(v.loc[hit, "value"])
        v.loc[hit, "unit_status"] = "inferred_from_value"
    # A given unit that disagrees with the concept (e.g. "kg" on a pulse) is not trusted.
    mismatch = written.notna() & ~written.isin([None, "score"]) & (written != v["unit"]) & (v["unit_status"] == "given")
    v.loc[mismatch, "unit_status"] = "unit_mismatch"

    lo = v["concept"].map(lambda c: PLAUSIBLE.get(c, (-np.inf, np.inf))[0])
    hi = v["concept"].map(lambda c: PLAUSIBLE.get(c, (-np.inf, np.inf))[1])
    v["implausible"] = v["value"].notna() & ((v["value"] < lo) | (v["value"] > hi))
    return v


# =============================================================================
# Labs and other measurements
# =============================================================================

def load_lab_maps() -> tuple[pd.DataFrame, pd.DataFrame]:
    sn = pd.read_csv(REF / "swed_lab_meas_to_snomed.csv", dtype=str)
    dup = sn["term_name"].duplicated(keep="first")
    report("setup", "swed_lab_meas_to_snomed.csv", "duplicate term_name rows ignored (first kept)", dup.sum())
    sn = sn[~dup]
    flags = sn[["missing_compartment", "not_perfect", "problematic"]].notna()
    sn["snomed_quality"] = flags.apply(lambda r: ",".join(c for c in flags.columns if r[c]) or pd.NA, axis=1)
    sn = sn.rename(columns={"snomed_id": "snomed_id", "snomed_swe": "snomed_swe"})[
        ["term_name", "snomed_id", "snomed_swe", "snomed_quality"]]
    un = pd.read_csv(REF / "lab_unit_normalization_07sep2026.csv", dtype={"unit": str, "canonical_unit": str})
    un = un.drop_duplicates(["term_name", "unit"])
    return sn, un


def harmonise_labs(lab: pd.DataFrame, snomed: pd.DataFrame, units: pd.DataFrame, src: str) -> pd.DataFrame:
    lab = lab.merge(snomed, on="term_name", how="left")
    lab = lab.merge(units, on=["term_name", "unit"], how="left")
    drop = lab["action"] == "drop"
    report("lab_meas", src, "rows dropped: unit marked 'drop' in lab_unit_normalization", drop.sum())
    lab = lab[~drop].copy()
    conv = lab["action"] == "convert"
    lab["unit_original"] = lab["unit"]
    lab.loc[conv, "value_num"] = lab.loc[conv, "value_num"] * lab.loc[conv, "factor"]
    lab.loc[conv, "unit"] = lab.loc[conv, "canonical_unit"]
    lab["unit_status"] = np.where(conv, "converted", np.where(lab["unit_original"].isna(), "missing", "given"))
    report("lab_meas", src, "rows with unit converted", conv.sum())
    return lab


# =============================================================================
# Main pass over the measurement files
# =============================================================================

def normalise_columns(df: pd.DataFrame, src: str) -> pd.DataFrame:
    for c in ["from", "unit", "value_str", "term_id", "lt_flag", "gt_flag", "value_num"]:
        if c not in df:
            df[c] = pd.NA
    if src == "ehr_sahlgrenska_lab" and df["datetime_analysis"].isna().any() and "date_lab" in df:
        filled = pd.to_datetime(df["date_lab"]).dt.tz_localize("UTC")
        df["datetime_analysis"] = df["datetime_analysis"].fillna(filled)
    df["from"] = df["from"].fillna(src.removeprefix("ehr_").split("_")[0])
    df["term_id"] = df["term_id"].map(lambda x: pd.NA if pd.isna(x) else (str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)))
    # value_num is sometimes empty while value_str holds "37,5"
    parsed = pd.to_numeric(df["value_str"].astype("string").str.replace(",", ".", regex=False).str.strip(), errors="coerce").astype("float64")
    df["value_num"] = df["value_num"].astype("float64").fillna(parsed)
    return df


def process_measurements(exclude: list[float], cohort: np.ndarray, windows: pd.DataFrame, out: Path) -> None:
    snomed, units = load_lab_maps()
    vitals = Sink(out / "vitals.parquet", VITALS_SCHEMA)
    labs = Sink(out / "lab_meas.parquet", LABS_SCHEMA)
    unmapped: Counter = Counter()
    filt = ds.field("lopnr_swed").isin(cohort)
    if exclude:
        filt &= ~ds.field("lopnr_swed").isin(exclude)

    for src in MEAS_SOURCES:
        if not path(src).exists():
            log(f"skip {src}: not found")
            continue
        dset = ds.dataset(path(src), format="parquet")
        cols = [c for c in ["from", "lopnr_swed", "datetime_analysis", "date_lab", "term_name", "term_id", "value_str",
                            "value_num", "unit", "lt_flag", "gt_flag"] if c in dset.schema.names]
        report("measurements", src, "rows in file", dset.count_rows())
        log(f"{src}: reading")
        for batch in dset.to_batches(columns=cols, filter=filt, batch_size=BATCH_ROWS):
            if not batch.num_rows:
                continue
            df = normalise_columns(batch.to_pandas(), src)
            df["source"] = src
            report("measurements", src, "rows read (cohort, not excluded)", len(df))
            df["ed_visit_id"] = link_visits(df, windows)

            vmap = vital_term_map(df["term_name"])
            df = df.merge(vmap, on="term_name", how="left")
            is_vital = df["concept"].notna()

            v = harmonise_vitals(df[is_vital].rename(columns={"value_num": "value"}).copy())
            for status, n in v["unit_status"].value_counts().items():
                report("vitals", src, f"unit_status={status}", n)
            report("vitals", src, "rows implausible (kept, flagged)", v["implausible"].sum())
            report("vitals", src, "rows linked to an ED visit", v["ed_visit_id"].notna().sum())
            vitals.write(v)

            lab = harmonise_labs(df[~is_vital].copy(), snomed, units, src)
            report("lab_meas", src, "rows linked to an ED visit", lab["ed_visit_id"].notna().sum())
            unmapped.update(lab.loc[lab["snomed_id"].isna(), "term_name"].fillna("<null>").map(lambda t: (src, t)))
            labs.write(lab)
        log(f"{src}: done")

    vitals.close()
    labs.close()
    with (out / "unmapped_terms.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "term_name", "n"])
        w.writerows([s, t, n] for (s, t), n in unmapped.most_common())


# =============================================================================
# Registers
# =============================================================================

def process_diagnoses(exclude: list[float], cohort: np.ndarray, out: Path) -> None:
    src = "sos_patreg_diag_long"
    if not path(src).exists():
        log(f"skip {src}: not found")
        return
    dset = ds.dataset(path(src), format="parquet")
    cols = [c for c in ["lopnr_swed", "INDATUM", "UTDATUM", "care_level", "SJUKHUS", "KTYP", "diag_pos", "icd10_code"]
            if c in dset.schema.names]
    filt = ds.field("lopnr_swed").isin(cohort)
    if exclude:
        filt &= ~ds.field("lopnr_swed").isin(exclude)
    report("diagnoses", src, "rows in file", dset.count_rows())
    writer, n = None, 0
    tmp = out / "diagnoses.parquet.tmp"
    for batch in dset.to_batches(columns=cols, filter=filt, batch_size=BATCH_ROWS):
        if not batch.num_rows:
            continue
        t = batch.to_pandas()
        t["diag_pos"] = t["diag_pos"].astype("string")
        t["icd10_code"] = t["icd10_code"].str.upper().str.replace(".", "", regex=False).str.strip()
        t["icd10_3"] = t["icd10_code"].str[:3]
        table = pa.Table.from_pandas(t, preserve_index=False)
        writer = writer or pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.write_table(table.cast(writer.schema))
        n += len(t)
    if writer:
        writer.close()
        tmp.rename(out / "diagnoses.parquet")
    report("diagnoses", src, "rows kept (cohort, not excluded)", n)
    log(f"wrote diagnoses.parquet: {n:,} rows")


def process_deaths(exclude: list[float], cohort: np.ndarray, out: Path) -> pd.DataFrame:
    cols = ["lopnr_swed", "date_death_minimp", "inexact_death_date", "ULORSAK"]
    if not path("sos_deathreg_wide").exists():
        log("skip sos_deathreg_wide: not found")
        return pd.DataFrame(columns=cols)
    dr = pq.read_table(path("sos_deathreg_wide")).to_pandas()
    report("deaths", "sos_deathreg_wide", "rows read", len(dr))
    dr = dr[dr["lopnr_swed"].isin(cohort) & ~dr["lopnr_swed"].isin(exclude)]
    dup = dr["lopnr_swed"].duplicated(keep="first")
    report("deaths", "sos_deathreg_wide", "rows dropped: second death record for the same person", dup.sum())
    dr = dr[~dup]
    dr.to_parquet(out / "deaths.parquet", index=False)
    report("deaths", "sos_deathreg_wide", "rows kept", len(dr))
    log(f"wrote deaths.parquet: {len(dr):,} rows")

    if path("sos_deathreg_diag_long").exists():
        dl = pq.read_table(path("sos_deathreg_diag_long")).to_pandas()
        dl = dl[dl["lopnr_swed"].isin(cohort) & ~dl["lopnr_swed"].isin(exclude)]
        dl["diag_pos"] = dl["diag_pos"].astype("string")
        dl.to_parquet(out / "death_causes.parquet", index=False)
        report("deaths", "sos_deathreg_diag_long", "rows kept", len(dl))
        log(f"wrote death_causes.parquet: {len(dl):,} rows")
    return dr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_ROOT / f"built_{DATA.name}",
                    help="output folder (default $SWED_OUT/built_<data folder name>)")
    args = ap.parse_args()
    refuse_to_write_into(args.out, DATA, ORIGINAL)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"data: {DATA}")
    log(f"ref:  {REF}")
    log(f"out:  {args.out}")
    if not VITAL_MAP_CSV.exists():
        log(f"NOTE: {VITAL_MAP_CSV} not found; vital signs are picked by the VITAL_CONCEPTS regexes")

    exclude = read_exclude()
    ed = build_visits(exclude)
    cohort = ed["lopnr_swed"].unique()
    ed.to_parquet(args.out / "ed_visits.parquet", index=False)
    log(f"wrote ed_visits.parquet: {len(ed):,} rows, {len(cohort):,} patients")

    deaths = process_deaths(exclude, cohort, args.out)
    build_patients(ed, deaths).to_parquet(args.out / "patients.parquet", index=False)
    log("wrote patients.parquet")

    process_measurements(exclude, cohort, visit_windows(ed), args.out)
    process_diagnoses(exclude, cohort, args.out)

    rep = pd.DataFrame(REPORT).groupby(["step", "source", "metric"], sort=False, as_index=False)["n"].sum()
    rep.to_csv(args.out / "build_report.csv", index=False)
    log(f"done in {time.time() - t0:.0f}s; check {args.out / 'build_report.csv'}")


if __name__ == "__main__":
    main()
