#!/usr/bin/env python3
"""Synthetic version of the built dataset, generated from exported statistics.

Runs on your own machine. Reads the CSVs written by 03_export_stats.py and
writes Parquet files with the same names, columns and types as 02_build.py:

    patients.parquet  ed_visits.parquet  vitals.parquet  lab_meas.parquet
    diagnoses.parquet  deaths.parquet

so code developed against these files runs unchanged on the real build.
Every file carries the Parquet metadata  synthetic = true.

How values are drawn
  - Every variable follows its exported distribution: category counts, or the
    percentiles q01..q99, sampled by linear interpolation between them. Tails
    beyond the outermost percentile are extended by one segment width.
  - When 03c_export_joint_stats.py has run, latent vitals and labs use exported
    correlations and conditional quantiles, and visits/deaths/diagnoses use
    joint categorical and conditional tables. Suppressed groups still fall
    back to independent draws. Use the data to test code, not findings.
  - Vital signs are taken in rounds that share a timestamp, as in the real files.
  - Categories that were suppressed on Bianca (fewer than 10 patients) never appear.

Assumptions not backed by the statistics are named constants marked ASSUMPTION.

    python 04_synthesize.py --stats export_stats_built_subset --n-patients 5000
    python 04_synthesize.py --stats export_stats_built_subset --validate
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REF = HERE / "data_info"

# 02_build.py defines the vitals/labs schemas and the SNOMED codes per concept; reuse them.
_spec = importlib.util.spec_from_file_location("build", HERE / "02_build.py")
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

from joint_synth import CONDITION_KEYS, JointStats, context_keys
from missingness_synth import MissingnessModel, VITAL_CONCEPTS, VITAL_VARIABLES, add_base_design

HOME_HOSPITAL_P = 0.8        # ASSUMPTION: share of a patient's visits at their most-used hospital
TOP_CODE_EXTRA_P = 0.3       # ASSUMPTION: "10+" becomes 10 + Geometric(p) - 1
UNLINKED_DAYS = (2, 60)      # ASSUMPTION: measurements outside a visit fall this many days after one
INPATIENT_DAYS_P = 0.25      # ASSUMPTION: inpatient length of stay ~ Geometric(p) days
DECIMALS = {"body_temp": 1}  # other vital concepts are whole numbers
STOCKHOLM = "Europe/Stockholm"

ED_SCHEMA = pa.schema([
    ("ed_visit_id", pa.int64()), ("ed_hospital", pa.string()), ("lopnr_swed", pa.float64()),
    ("ed_arrival_datetime", pa.timestamp("us", tz="UTC")), ("ed_inexact_time", pa.bool_()), ("ed_age", pa.float64()),
    ("is_male", pa.int32()), ("ed_presented_complaint_std", pa.string()), ("ed_arrival_date", pa.date32()),
    ("ed_arrival_time", pa.string()), ("ed_time_to_doctor_minutes", pa.float64()), ("ed_time_total_minutes", pa.float64()),
    ("ed_priority_in", pa.float64()), ("ed_arrival_type", pa.string()), ("care_unit", pa.string()),
    ("ed_priority_out", pa.float64()), ("ed_exit_to", pa.string()), ("ed_exit_to_care_unit", pa.string()),
    ("ed_presented_complaint_nonstd", pa.string()),
])
PATIENTS_SCHEMA = pa.schema([
    ("lopnr_swed", pa.float64()), ("is_male", pa.int32()), ("n_ed_visits", pa.int64()),
    ("first_ed_arrival", pa.timestamp("us", tz="UTC")), ("last_ed_arrival", pa.timestamp("us", tz="UTC")),
    ("birth_date", pa.date32()), ("date_death_minimp", pa.date32()), ("inexact_death_date", pa.bool_()),
    ("ULORSAK", pa.string()),
])
DIAG_SCHEMA = pa.schema([
    ("lopnr_swed", pa.float64()), ("INDATUM", pa.date32()), ("UTDATUM", pa.date32()), ("care_level", pa.string()),
    ("SJUKHUS", pa.string()), ("KTYP", pa.int32()), ("diag_pos", pa.string()), ("icd10_code", pa.string()),
    ("icd10_3", pa.string()),
])
MORSAK = [f"MORSAK{i}" for i in range(1, 49)]
DEATHS_SCHEMA = pa.schema(
    [("from", pa.string()), ("source_row", pa.int32()), ("lopnr_swed", pa.float64()), ("inexact_death_date", pa.bool_()),
     ("date_death_meanimp", pa.date32()), ("date_death_minimp", pa.date32()), ("ULORSAK", pa.string())]
    + [(c, pa.string()) for c in MORSAK]
    + [("KAP19", pa.string()), ("LK", pa.string()), ("DODSPL", pa.int32()), ("DBGRUND1", pa.int32()), ("is_female", pa.int32())]
)

QCOLS = ["q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Stats:
    def __init__(self, folder: Path):
        self.folder = folder

    def __call__(self, name: str) -> pd.DataFrame | None:
        f = self.folder / f"{name}.csv"
        if not f.exists():
            log(f"  no {name}.csv; that part is skipped")
            return None
        return pd.read_csv(f, keep_default_na=True, dtype={"value": "string", "unit_original": "string"})


# =============================================================================
# Samplers
# =============================================================================

def sample_quantiles(row: pd.Series | None, n: int, rng: np.random.Generator,
                     u: np.ndarray | None = None) -> np.ndarray:
    """n draws from the distribution described by a row of percentiles and null_rate."""
    if row is None or n == 0:
        return np.full(n, np.nan)
    qs = [(int(c[1:]) / 100, row[c]) for c in QCOLS if c in row and pd.notna(row[c])]
    if len(qs) < 2:
        return np.full(n, np.nan)
    p, x = np.array([q[0] for q in qs]), np.array([q[1] for q in qs], dtype=float)
    p = np.r_[0.0, p, 1.0]
    x = np.r_[x[0] - (x[1] - x[0]), x, x[-1] + (x[-1] - x[-2])]
    out = np.interp(rng.random(n) if u is None else u, p, x)
    null = row.get("null_rate", 0.0)
    if pd.notna(null) and null > 0:
        out[rng.random(n) < null] = np.nan
    return out


def choose(df: pd.DataFrame, n: int, rng: np.random.Generator, weight: str = "n") -> pd.DataFrame:
    """n rows drawn from df with probability proportional to df[weight]."""
    w = df[weight].to_numpy(dtype=float)
    return df.iloc[rng.choice(len(df), size=n, p=w / w.sum())].reset_index(drop=True)


def draw(df: pd.DataFrame | None, col: str, n: int, rng: np.random.Generator, fallback) -> np.ndarray:
    """n values of df[col] weighted by n, or fallback(n) when the statistics are missing or all suppressed."""
    if df is None or df.empty or df["n"].sum() <= 0:
        return np.asarray(fallback(n))
    return choose(df, n, rng)[col].to_numpy()


def scalar(df: pd.DataFrame | None, col: str, default: float) -> float:
    return default if df is None or df.empty or pd.isna(df[col].iloc[0]) else float(df[col].iloc[0])


def top_coded_counts(values: pd.Series, rng: np.random.Generator) -> np.ndarray:
    v = values.astype(str).to_numpy()
    out = np.zeros(len(v), dtype=np.int64)
    plus = np.char.endswith(v.astype(str), "+")
    out[~plus] = v[~plus].astype(float).astype(np.int64)
    if plus.any():
        base = np.char.rstrip(v[plus].astype(str), "+").astype(float).astype(np.int64)
        out[plus] = base + rng.geometric(TOP_CODE_EXTRA_P, plus.sum()) - 1
    return out


def lookup(df: pd.DataFrame | None, **keys) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    m = np.ones(len(df), dtype=bool)
    for k, v in keys.items():
        m &= (df[k].astype(str) == str(v)).to_numpy()
    return df[m]


def first_row(df: pd.DataFrame) -> pd.Series | None:
    return None if df.empty else df.iloc[0]


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype("string").str.lower().map({"true": True, "false": False}).astype("boolean")


def nullable(s: pd.Series) -> pd.Series:
    return s.astype("string").replace({"<missing>": pd.NA})


def complaints_from_stats(st: Stats) -> list[str]:
    cat = st("ed_visits_categorical")
    if cat is None or cat.empty:
        return []
    sel = cat[cat["column"].astype(str) == "ed_presented_complaint_std"]["value"]
    return [str(x) for x in sel.dropna().tolist()]


def _draw_latent(domain: str, source: str, variables: list[str], ed: pd.DataFrame,
                 marginal_rows: dict[str, pd.Series | None], joint: JointStats,
                 rng: np.random.Generator) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=ed.index, columns=variables)
    ctx = context_keys(ed)
    u = joint.uniforms(domain, "all", source, variables, len(ed), rng)
    for j, name in enumerate(variables):
        for key, labels in ctx.groupby(CONDITION_KEYS, dropna=False).groups.items():
            idx = np.asarray(list(labels), dtype=int)
            row_ctx = dict(zip(CONDITION_KEYS, key))
            row = joint.conditional_row(domain, "all", source, name, row_ctx)
            if row is None:
                row = marginal_rows.get(name)
            out.iloc[idx, j] = sample_quantiles(row, len(idx), rng, u[idx, j])
    return out


def latent_vital_values(st: Stats, ed: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    values = st("vitals_values")
    joint = JointStats(st.folder)
    rows = {}
    for name in VITAL_VARIABLES:
        candidates = lookup(values, concept=VITAL_CONCEPTS[name])
        rows[name] = first_row(choose(candidates, len(ed), rng)) if not candidates.empty else None
    return _draw_latent("vital", "", VITAL_VARIABLES, ed, rows, joint, rng)


def latent_lab_values(st: Stats, ed: pd.DataFrame, source: str, concepts: list[str],
                      rng: np.random.Generator) -> pd.DataFrame:
    values = st("labs_values")
    joint = JointStats(st.folder)
    variables = [f"lab_{c}" for c in concepts]
    rows = {}
    for concept, name in zip(concepts, variables):
        candidates = lookup(values, source=source, term_name=concept)
        rows[name] = first_row(choose(candidates, len(ed), rng)) if not candidates.empty else None
    return _draw_latent("lab", source, variables, ed, rows, joint, rng)


# =============================================================================
# Patients, visits, deaths
# =============================================================================

def make_patients(st: Stats, n: int, rng: np.random.Generator) -> pd.DataFrame:
    sex = draw(st("patients_sex"), "is_male", n, rng, lambda k: rng.integers(0, 2, k)).astype(int)
    n_visits = np.maximum(1, top_coded_counts(pd.Series(draw(st("patients_n_ed_visits"), "n_ed_visits", n, rng,
                                                             lambda k: np.ones(k, int))), rng))
    ages = st("patients_age_first_visit")
    age = np.empty(n)
    for s in (0, 1):
        m = sex == s
        age[m] = sample_quantiles(first_row(lookup(ages, is_male=s)), m.sum(), rng)
    age = np.clip(np.where(np.isnan(age), np.nanmedian(age), age), 0, 110)
    return pd.DataFrame({"lopnr_swed": np.arange(1, n + 1, dtype=float), "is_male": sex,
                         "n_ed_visits": n_visits, "age_first_visit": age})


def random_datetimes(hospitals: np.ndarray, st: Stats, rng: np.random.Generator) -> pd.Series:
    """Arrival times following the year, month, weekday and hour distributions per hospital."""
    by_year, by_wd, by_hour = st("ed_visits_hospital_year"), st("ed_visits_hospital_weekday"), st("ed_visits_hospital_hour")
    joint = JointStats(st.folder)
    out = pd.Series(pd.NaT, index=range(len(hospitals)), dtype="datetime64[us, UTC]")
    for h in np.unique(hospitals):
        idx = np.flatnonzero(hospitals == h)
        k = len(idx)
        year = draw(lookup(by_year, ed_hospital=h), "year", k, rng, lambda m: draw(by_year, "year", m, rng, lambda _: np.full(m, 2015)))
        wd = draw(lookup(by_wd, ed_hospital=h), "weekday", k, rng, lambda m: rng.integers(0, 7, m))
        hour = draw(lookup(by_hour, ed_hospital=h), "hour", k, rng, lambda m: rng.integers(0, 24, m))
        months = joint.month_weights(h)
        if months.empty:
            day = pd.to_datetime(year.astype(str)) + pd.to_timedelta(rng.integers(0, 358, k), unit="D")
            day = day + pd.to_timedelta((wd - day.day_of_week.to_numpy()) % 7, unit="D")
        else:
            month = draw(months, "month", k, rng, lambda m: rng.integers(1, 13, m)).astype(int)
            first = pd.DatetimeIndex(pd.to_datetime(dict(year=np.asarray(year, dtype=int), month=month, day=np.ones(k, dtype=int))))
            offset = (wd - first.dayofweek) % 7
            weeks = ((first.days_in_month - 1 - offset) // 7) + 1
            day = first + pd.to_timedelta(offset + 7 * rng.integers(0, weeks), unit="D")
        local = day + pd.to_timedelta(hour * 60 + rng.integers(0, 60, k), unit="min")
        utc = local.tz_localize(STOCKHOLM, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
        out.iloc[idx] = utc.to_numpy()
    return out.fillna(out.dropna().iloc[0] if out.notna().any() else pd.Timestamp("2015-01-01", tz="UTC"))


def hospital_weights(st: Stats) -> pd.Series:
    """Visits per hospital, from the first ED table that survived suppression."""
    for name in ["ed_visits_hospital_year", "ed_visits_hospital_weekday", "ed_visits_hospital_hour", "ed_visits_numeric"]:
        df = st(name)
        if df is not None and not df.empty:
            if name == "ed_visits_numeric":
                df = df[df["column"] == df["column"].iloc[0]]
            return df.groupby("ed_hospital")["n"].sum()
    log("  WARNING: no hospital counts survived suppression; all visits get hospital 'unknown'")
    return pd.Series({"unknown": 1})


def make_visits(st: Stats, pts: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    hosp_w = hospital_weights(st)
    names, w = hosp_w.index.to_numpy(), hosp_w.to_numpy(dtype=float) / hosp_w.sum()
    home = rng.choice(names, len(pts), p=w)
    lop = np.repeat(pts["lopnr_swed"].to_numpy(), pts["n_ed_visits"].to_numpy())
    hosp = np.repeat(home, pts["n_ed_visits"].to_numpy())
    away = rng.random(len(lop)) > HOME_HOSPITAL_P
    hosp[away] = rng.choice(names, away.sum(), p=w)

    ed = pd.DataFrame({"lopnr_swed": lop, "ed_hospital": hosp})
    ed["ed_arrival_datetime"] = random_datetimes(hosp, st, rng).to_numpy()

    cat = st("ed_visits_categorical")
    for col in ["ed_arrival_type", "care_unit", "ed_exit_to", "ed_exit_to_care_unit", "ed_priority_in",
                "ed_priority_out", "ed_presented_complaint_std", "ed_inexact_time"]:
        ed[col] = pd.Series(pd.NA, index=ed.index, dtype="object")
        for h, idx in ed.groupby("ed_hospital").groups.items():
            opts = lookup(cat, ed_hospital=h, column=col)
            if not opts.empty:
                ed.loc[idx, col] = choose(opts, len(idx), rng)["value"].to_numpy()
    joint = JointStats(st.folder)
    for a, b in [("ed_presented_complaint_std", "ed_priority_in"),
                 ("ed_presented_complaint_std", "ed_arrival_type"),
                 ("ed_presented_complaint_std", "ed_exit_to"),
                 ("ed_age_band", "ed_priority_in"),
                 ("is_male", "ed_priority_in"),
                 ("ed_priority_in", "ed_exit_to")]:
        if a not in ed:
            if a == "ed_age_band":
                ed[a] = context_keys(ed)["age_band"].to_numpy()
            else:
                continue
        for (h, va), labels in ed.groupby(["ed_hospital", a], dropna=False).groups.items():
            opts = joint.categorical_options(str(h), a, str(va), b)
            if opts.empty:
                continue
            ed.loc[list(labels), b] = choose(opts, len(labels), rng)["value_b"].to_numpy()
    for col in ["ed_priority_in", "ed_priority_out"]:
        ed[col] = pd.to_numeric(ed[col], errors="coerce")
    ed["ed_inexact_time"] = as_bool(ed["ed_inexact_time"]).fillna(False).astype(bool)

    num = st("ed_visits_numeric")
    for col in ["ed_time_to_doctor_minutes", "ed_time_total_minutes"]:
        ed[col] = np.nan
        for h, idx in ed.groupby("ed_hospital").groups.items():
            ed.loc[idx, col] = np.round(np.clip(sample_quantiles(first_row(lookup(num, ed_hospital=h, column=col)), len(idx), rng), 0, None))

    # inexact times keep only the date, at 00:00 UTC, like the real files
    inexact = ed["ed_inexact_time"].to_numpy()
    ed.loc[inexact, "ed_arrival_datetime"] = ed.loc[inexact, "ed_arrival_datetime"].dt.floor("D")
    local = ed["ed_arrival_datetime"].dt.tz_convert(STOCKHOLM)
    ed["ed_arrival_date"] = local.dt.date
    ed["ed_arrival_time"] = local.dt.strftime("%H:%M").where(~inexact, None)

    ed = ed.sort_values(["lopnr_swed", "ed_arrival_datetime"], kind="stable").reset_index(drop=True)
    first = ed.groupby("lopnr_swed")["ed_arrival_datetime"].transform("min")
    p = pts.set_index("lopnr_swed")
    ed["is_male"] = ed["lopnr_swed"].map(p["is_male"]).astype("int32")
    years = (ed["ed_arrival_datetime"] - first).dt.total_seconds() / (365.25 * 86400)
    ed["ed_age"] = np.floor(ed["lopnr_swed"].map(p["age_first_visit"]) + years)
    ed["ed_presented_complaint_nonstd"] = None
    ed.insert(0, "ed_visit_id", np.arange(len(ed), dtype="int64"))
    return ed


def make_deaths(st: Stats, pts: pd.DataFrame, ed: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    died = st("patients_died")
    share = scalar(died, "share_died", 0.0)
    joint = JointStats(st.folder)
    first = ed.sort_values("ed_arrival_datetime").drop_duplicates("lopnr_swed").reset_index(drop=True)
    ctx = context_keys(first)
    profile = pd.DataFrame({
        "ed_hospital": first["ed_hospital"].astype(str).to_numpy(),
        "age_band": ctx["age_band"].to_numpy(), "sex": ctx["sex"].to_numpy(), "complaint": ctx["complaint"].to_numpy(),
    }, index=first["lopnr_swed"].to_numpy()).reindex(pts["lopnr_swed"].to_numpy())
    raw = np.array([joint.mortality_rate(str(r.ed_hospital), r.age_band, r.sex, r.complaint, share)
                    for r in profile.itertuples()], dtype=float)
    target = int(round(share * len(pts)))
    scale = target / max(raw.sum(), 1e-12)
    prob = np.clip(raw * scale, 0, 1)
    for _ in range(5):
        scale *= target / max(prob.sum(), 1e-12)
        prob = np.clip(raw * scale, 0, 1)
    cand = pts[rng.random(len(pts)) < prob][["lopnr_swed", "is_male"]].copy()
    last = ed.groupby("lopnr_swed")["ed_arrival_datetime"].max().dt.tz_localize(None).dt.normalize()
    after = st("deaths_days_after_last_visit")
    days = np.maximum(0, np.round(sample_quantiles(None if after is None else first_row(after), len(cand), rng)))
    cand["date_death_minimp"] = cand["lopnr_swed"].map(last) + pd.to_timedelta(np.nan_to_num(days), unit="D")
    years = st("deaths_year")
    if years is not None and not years.empty:  # everyone in the register died before it ends: redraw within the observed period
        end = pd.Timestamp(f"{int(years['year'].max())}-12-31")
        late = cand["date_death_minimp"] > end
        room = (end - cand.loc[late, "lopnr_swed"].map(last)).dt.days.clip(lower=0)
        cand.loc[late, "date_death_minimp"] = cand.loc[late, "lopnr_swed"].map(last) + pd.to_timedelta(
            np.floor(rng.random(late.sum()) * (room + 1)), unit="D")
    k = len(cand)

    def pick(name: str, col: str):
        df = st(name)
        return draw(df, col, k, rng, lambda m: np.full(m, None))

    dr = pd.DataFrame({"from": "synthetic", "source_row": np.arange(k, dtype="int32"), "lopnr_swed": cand["lopnr_swed"].to_numpy()})
    dr["inexact_death_date"] = as_bool(pd.Series(pick("deaths_inexact_date", "inexact_death_date"))).fillna(False).astype(bool).to_numpy()
    dr["date_death_minimp"] = cand["date_death_minimp"].dt.date.to_numpy()
    dr["date_death_meanimp"] = dr["date_death_minimp"]
    dr["ULORSAK"] = pick("deaths_underlying_cause_3", "ulorsak_3")
    for c in MORSAK:
        dr[c] = None
    dr["MORSAK1"] = dr["ULORSAK"]
    dr["KAP19"], dr["LK"] = None, None
    dr["DODSPL"] = pd.to_numeric(pd.Series(pick("deaths_place", "DODSPL")), errors="coerce").astype("Int32").to_numpy()
    dr["DBGRUND1"] = pd.to_numeric(pd.Series(pick("deaths_autopsy", "DBGRUND1")), errors="coerce").astype("Int32").to_numpy()
    dr["is_female"] = (1 - cand["is_male"].to_numpy()).astype("int32")
    return dr


def finish_patients(pts: pd.DataFrame, ed: pd.DataFrame, dr: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    g = ed.groupby("lopnr_swed")["ed_arrival_datetime"]
    p = pts.assign(first_ed_arrival=pts["lopnr_swed"].map(g.min()), last_ed_arrival=pts["lopnr_swed"].map(g.max()))
    birth = p["first_ed_arrival"].dt.tz_localize(None) - pd.to_timedelta(p["age_first_visit"] * 365.25 + rng.uniform(0, 365, len(p)), unit="D")
    p["birth_date"] = birth.dt.date
    p = p.merge(dr[["lopnr_swed", "date_death_minimp", "inexact_death_date", "ULORSAK"]], on="lopnr_swed", how="left")
    return p.drop(columns="age_first_visit")


# =============================================================================
# Vital signs and labs
# =============================================================================

def _make_vitals_legacy(st: Stats, ed: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    share, counts, timing = st("vitals_share_of_visits"), st("vitals_count_per_visit"), st("vitals_timing")
    src, units, values = st("vitals_source_by_hospital"), st("vitals_units"), st("vitals_values")
    implaus, linked_share = st("vitals_implausible"), st("vitals_linked")
    if share is None or src is None:
        return pd.DataFrame(columns=build.VITALS_SCHEMA.names)
    snomed = {k: s or None for k, _, s, _ in build.VITAL_CONCEPTS}

    # 1. which concepts each visit gets, and how often
    per = []
    for r in share.itertuples():
        vis = ed.loc[ed["ed_hospital"] == r.ed_hospital, ["ed_visit_id", "ed_hospital", "lopnr_swed", "ed_arrival_datetime"]]
        vis = vis[rng.random(len(vis)) < r.share_visits_with]
        opts = lookup(counts, ed_hospital=r.ed_hospital, concept=r.concept)
        k = top_coded_counts(pd.Series(draw(opts, "per_visit", len(vis), rng, lambda m: np.ones(m, int))), rng)
        per.append(vis.assign(concept=r.concept, k=k))
    per = pd.concat(per, ignore_index=True)

    # 2. measurement rounds per visit; a concept measured k times uses the first k rounds
    rounds = per.groupby("ed_visit_id")["k"].max()
    hosp = ed.set_index("ed_visit_id")["ed_hospital"]
    anchor = timing.sort_values("n").groupby("ed_hospital").last() if timing is not None else pd.DataFrame()
    rv = np.repeat(rounds.index.to_numpy(), rounds.to_numpy())
    off = np.empty(len(rv))
    rh = hosp.loc[rv].to_numpy()
    for h in np.unique(rh):
        m = rh == h
        off[m] = sample_quantiles(anchor.loc[h] if h in anchor.index else None, m.sum(), rng)
    off = np.nan_to_num(off, nan=0.0)
    order = np.lexsort((off, rv))
    rv, off = rv[order], off[order]
    start = pd.Series(np.flatnonzero(np.r_[True, rv[1:] != rv[:-1]]), index=np.unique(rv))

    rows = per.loc[per.index.repeat(per["k"])].reset_index(drop=True)
    rows["round"] = rows.groupby(["ed_visit_id", "concept"]).cumcount()
    rows["offset"] = off[start.loc[rows["ed_visit_id"]].to_numpy() + rows["round"].to_numpy()]
    rows["datetime_analysis"] = rows["ed_arrival_datetime"] + pd.to_timedelta(np.round(rows["offset"]), unit="min")

    # 3. source, term, unit and value per row
    parts = []
    for (h, c), g in rows.groupby(["ed_hospital", "concept"]):
        opts = lookup(src, ed_hospital=h, concept=c)
        opts = opts if not opts.empty else lookup(src, ed_hospital=h)
        if opts.empty:
            continue
        g = g.reset_index(drop=True)
        g[["source", "from"]] = choose(opts, len(g), rng)[["source", "from"]].to_numpy()
        for (s, f), gg in g.groupby(["source", "from"]):
            u = lookup(units, source=s, concept=c, **{"from": f})
            gg = gg.copy()
            if not u.empty:
                pick = choose(u, len(gg), rng)
                gg["term_name"], gg["unit_original"] = pick["term_name"].to_numpy(), nullable(pick["unit_original"]).to_numpy()
                gg["unit"], gg["unit_status"] = pick["unit"].to_numpy(), pick["unit_status"].to_numpy()
            else:
                gg["term_name"], gg["unit_original"], gg["unit"], gg["unit_status"] = c, None, None, "assumed"
            val = sample_quantiles(first_row(lookup(values, source=s, concept=c, **{"from": f})), len(gg), rng)
            lo, hi = build.PLAUSIBLE.get(c, (-np.inf, np.inf))
            val = np.round(np.clip(val, lo, hi), DECIMALS.get(c, 0))
            ir = first_row(lookup(implaus, source=s, concept=c, **{"from": f}))
            bad = rng.random(len(gg)) < (ir["share_implausible"] if ir is not None else 0)
            val[bad & np.isfinite(hi)] = hi * 3
            gg["value"], gg["implausible"] = val, bad & np.isfinite(hi)
            parts.append(gg)
    v = pd.concat(parts, ignore_index=True)
    v["concept"] = v["concept"].astype(str)
    v["snomed_id"] = v["concept"].map(snomed)
    v["term_id"] = None
    return add_unlinked(v, linked_share, rng)


def make_vitals(st: Stats, ed: pd.DataFrame, rng: np.random.Generator, y: np.ndarray,
                missingness: MissingnessModel | None = None, gx_scale: float = 1.0) -> pd.DataFrame:
    if missingness is None or not missingness.available:
        return _make_vitals_legacy(st, ed, rng)
    counts, timing = st("vitals_count_per_visit"), st("vitals_timing")
    src, units, values = st("vitals_source_by_hospital"), st("vitals_units"), st("vitals_values")
    implaus, linked_share = st("vitals_implausible"), st("vitals_linked")
    if values is None or src is None:
        return pd.DataFrame(columns=build.VITALS_SCHEMA.names)
    snomed = {k: s or None for k, _, s, _ in build.VITAL_CONCEPTS}

    base = ed.reset_index(drop=True).copy()
    base["died30"] = np.asarray(y, dtype=float)
    latent = latent_vital_values(st, base, rng)
    mask = missingness.sample(base, latent, rng, gx_scale=gx_scale, group_col="ed_hospital")
    per = []
    for j, name in enumerate(VITAL_VARIABLES):
        use = np.flatnonzero(mask[name].to_numpy())
        if not len(use):
            continue
        d = base.iloc[use][["ed_visit_id", "ed_hospital", "lopnr_swed", "ed_arrival_datetime"]].copy()
        d["concept"] = VITAL_CONCEPTS[name]
        d["latent_value"] = latent.iloc[use, j].to_numpy()
        per.append(d)
    if not per:
        return pd.DataFrame(columns=build.VITALS_SCHEMA.names)
    per = pd.concat(per, ignore_index=True)
    per["k"] = 1
    if counts is not None:
        for (h, c), loc in per.groupby(["ed_hospital", "concept"]).groups.items():
            opts = lookup(counts, ed_hospital=h, concept=c)
            k = top_coded_counts(pd.Series(draw(opts, "per_visit", len(loc), rng,
                                                lambda m: np.ones(m, int))), rng)
            per.loc[loc, "k"] = k

    rounds = per.groupby("ed_visit_id")["k"].max()
    hosp = base.set_index("ed_visit_id")["ed_hospital"]
    anchor = timing.sort_values("n").groupby("ed_hospital").last() if timing is not None else pd.DataFrame()
    rv = np.repeat(rounds.index.to_numpy(), rounds.to_numpy())
    off = np.empty(len(rv))
    rh = hosp.loc[rv].to_numpy()
    for h in np.unique(rh):
        m = rh == h
        off[m] = sample_quantiles(anchor.loc[h] if h in anchor.index else None, m.sum(), rng)
    off = np.nan_to_num(off, nan=0.0)
    order = np.lexsort((off, rv))
    rv, off = rv[order], off[order]
    start = pd.Series(np.flatnonzero(np.r_[True, rv[1:] != rv[:-1]]), index=np.unique(rv))

    rows = per.loc[per.index.repeat(per["k"])].reset_index(drop=True)
    rows["round"] = rows.groupby(["ed_visit_id", "concept"]).cumcount()
    rows["offset"] = off[start.loc[rows["ed_visit_id"]].to_numpy() + rows["round"].to_numpy()]
    rows["datetime_analysis"] = rows["ed_arrival_datetime"] + pd.to_timedelta(np.round(rows["offset"]), unit="min")

    parts = []
    for (h, c), g in rows.groupby(["ed_hospital", "concept"]):
        opts = lookup(src, ed_hospital=h, concept=c)
        opts = opts if not opts.empty else lookup(src, ed_hospital=h)
        if opts.empty:
            continue
        g = g.reset_index(drop=True)
        g[["source", "from"]] = choose(opts, len(g), rng)[["source", "from"]].to_numpy()
        for (s, f), gg in g.groupby(["source", "from"]):
            gg = gg.copy()
            u = lookup(units, source=s, concept=c, **{"from": f})
            if not u.empty:
                pick = choose(u, len(gg), rng)
                gg["term_name"], gg["unit_original"] = pick["term_name"].to_numpy(), nullable(pick["unit_original"]).to_numpy()
                gg["unit"], gg["unit_status"] = pick["unit"].to_numpy(), pick["unit_status"].to_numpy()
            else:
                gg["term_name"], gg["unit_original"], gg["unit"], gg["unit_status"] = c, None, None, "assumed"
            lo, hi = build.PLAUSIBLE.get(c, (-np.inf, np.inf))
            val = np.round(np.clip(gg["latent_value"].to_numpy(dtype=float), lo, hi), DECIMALS.get(c, 0))
            ir = first_row(lookup(implaus, source=s, concept=c, **{"from": f}))
            bad = rng.random(len(gg)) < (ir["share_implausible"] if ir is not None else 0)
            val[bad & np.isfinite(hi)] = hi * 3
            gg["value"], gg["implausible"] = val, bad & np.isfinite(hi)
            parts.append(gg.drop(columns="latent_value"))
    if not parts:
        return pd.DataFrame(columns=build.VITALS_SCHEMA.names)
    v = pd.concat(parts, ignore_index=True)
    v["concept"] = v["concept"].astype(str)
    v["snomed_id"] = v["concept"].map(snomed)
    v["term_id"] = None
    return add_unlinked(v, linked_share, rng)


def make_labs(st: Stats, ed: pd.DataFrame, rng: np.random.Generator, y: np.ndarray,
              missingness: list[MissingnessModel] | None = None, gx_scale: float = 1.0) -> pd.DataFrame:
    share, per_visit, timing = st("labs_share_of_visits"), st("labs_rows_per_visit"), st("labs_timing")
    src, terms, values, linked_share = st("labs_source_by_hospital"), st("labs_terms"), st("labs_values"), st("labs_linked")
    models = [m for m in (missingness or []) if m.available]
    if terms is None or (share is None and not models):
        return pd.DataFrame(columns=build.LABS_SCHEMA.names)
    sn = pd.read_csv(REF / "swed_lab_meas_to_snomed.csv", dtype=str).drop_duplicates("term_name").set_index("term_name") \
        if (REF / "swed_lab_meas_to_snomed.csv").exists() else pd.DataFrame(columns=["snomed_swe"])

    if models:
        parts = []
        for model in models:
            concepts = [v.replace("lab_", "", 1) for v in model.variables]
            sources = sorted({key[1] for key in model.patterns_by_key})
            for source in sources:
                hospitals = sorted({key[0] for key in model.patterns_by_key if key[1] == source})
                vis = ed[ed["ed_hospital"].isin(hospitals)].reset_index(drop=True).copy()
                if vis.empty:
                    continue
                vis["source"] = source
                if "died30" not in vis:
                    vis["died30"] = 0.0
                latent = latent_lab_values(st, vis, source, concepts, rng)
                mask = model.sample(vis, latent, rng, gx_scale=gx_scale,
                                    group_col="ed_hospital", source_col="source")
                for j, concept in enumerate(concepts):
                    use = np.flatnonzero(mask[f"lab_{concept}"].to_numpy())
                    if not len(use):
                        continue
                    g = vis.iloc[use].copy()
                    g["concept"] = concept
                    g["latent_value"] = latent.iloc[use, j].to_numpy()
                    g["k"] = 1
                    if per_visit is not None:
                        for h, loc in g.groupby("ed_hospital").groups.items():
                            row = first_row(lookup(per_visit, ed_hospital=h, source=source))
                            k = np.maximum(1, np.round(np.nan_to_num(
                                sample_quantiles(row, len(loc), rng), nan=1))).astype(int)
                            g.loc[loc, "k"] = k
                    g = g.loc[g.index.repeat(g["k"])].reset_index(drop=True)
                    g["from"] = None
                    if src is not None:
                        for h, loc in g.groupby("ed_hospital").groups.items():
                            opts = lookup(src, ed_hospital=h, source=source)
                            if not opts.empty:
                                g.loc[loc, "from"] = choose(opts, len(loc), rng)["from"].to_numpy()
                    off = np.zeros(len(g))
                    if timing is not None:
                        for h, loc in g.groupby("ed_hospital").groups.items():
                            row = first_row(lookup(timing, ed_hospital=h, source=source))
                            off[loc] = np.nan_to_num(sample_quantiles(row, len(loc), rng))
                    g["datetime_analysis"] = g["ed_arrival_datetime"] + pd.to_timedelta(np.round(off), unit="min")
                    for f, gg0 in g.groupby("from", dropna=False):
                        gg = gg0.copy()
                        term = first_row(lookup(terms, source=source, term_name=concept, **{"from": f}))
                        gg["term_name"] = concept
                        gg["snomed_id"] = term["snomed_id"] if term is not None else None
                        gg["unit"] = term["unit"] if term is not None else None
                        gg["unit_status"] = term["unit_status"] if term is not None else "assumed"
                        gg["value_num"] = np.round(np.asarray(gg["latent_value"], dtype=float), 2)
                        parts.append(gg)
        if not parts:
            return pd.DataFrame(columns=build.LABS_SCHEMA.names)
        lab = pd.concat(parts, ignore_index=True)
        lab["snomed_swe"] = lab["term_name"].map(sn["snomed_swe"]) if "snomed_swe" in sn else None
        lab["snomed_quality"] = None
        lab["value_str"] = lab["value_num"].map(lambda x: None if pd.isna(x) else f"{x:g}")
        lab["lt_flag"], lab["gt_flag"], lab["term_id"] = False, False, None
        return add_unlinked(lab, linked_share, rng)

    parts = []
    for r in share.itertuples():
        vis = ed.loc[ed["ed_hospital"] == r.ed_hospital, ["ed_visit_id", "ed_hospital", "lopnr_swed", "ed_arrival_datetime"]]
        vis = vis[rng.random(len(vis)) < r.share_visits_with]
        k = np.maximum(1, np.round(np.nan_to_num(sample_quantiles(first_row(lookup(per_visit, ed_hospital=r.ed_hospital, source=r.source)),
                                                                  len(vis), rng), nan=1))).astype(int)
        rows = vis.loc[vis.index.repeat(k)].reset_index(drop=True)
        off = sample_quantiles(first_row(lookup(timing, ed_hospital=r.ed_hospital, source=r.source)), len(rows), rng)
        rows["datetime_analysis"] = rows["ed_arrival_datetime"] + pd.to_timedelta(np.round(np.nan_to_num(off)), unit="min")
        rows["source"] = r.source
        froms = lookup(src, ed_hospital=r.ed_hospital, source=r.source)
        rows["from"] = choose(froms, len(rows), rng)["from"].to_numpy() if not froms.empty else None
        for f, g in rows.groupby("from"):
            t = lookup(terms, source=r.source, **{"from": f})
            if t.empty:
                continue
            g = g.reset_index(drop=True)
            pick = choose(t, len(g), rng)
            for c in ["term_name", "snomed_id", "unit", "unit_status"]:
                g[c] = pick[c].to_numpy()
            g["value_num"] = np.nan
            for key, gi in g.groupby(["term_name", "snomed_id", "unit"]).groups.items():
                row = first_row(lookup(values, source=r.source, term_name=key[0], snomed_id=key[1], unit=key[2], **{"from": f}))
                g.loc[gi, "value_num"] = np.round(sample_quantiles(row, len(gi), rng), 2)
            parts.append(g)
    lab = pd.concat(parts, ignore_index=True)
    lab["snomed_id"] = lab["snomed_id"].replace({"<unmapped>": None})
    lab["unit"] = lab["unit"].replace({"<missing>": None})
    lab["unit_original"] = lab["unit"].where(lab["unit_status"] != "converted", None)
    lab["snomed_swe"] = lab["term_name"].map(sn["snomed_swe"]) if "snomed_swe" in sn else None
    lab["snomed_quality"] = None
    lab["value_str"] = lab["value_num"].map(lambda x: None if pd.isna(x) else f"{x:g}")
    lab["lt_flag"], lab["gt_flag"], lab["term_id"] = False, False, None
    return add_unlinked(lab, linked_share, rng)


def add_unlinked(df: pd.DataFrame, linked_share: pd.DataFrame | None, rng: np.random.Generator) -> pd.DataFrame:
    """Rows outside any ED visit: copies of linked rows moved UNLINKED_DAYS later."""
    df = df.assign(ed_visit_id=df["ed_visit_id"].astype("Int64"))
    if linked_share is None:
        return df
    extra = []
    for _, r in linked_share.iterrows():
        g = df[(df["source"] == r["source"]) & (df["from"] == r["from"])]
        s = r["share_linked_to_visit"]
        if len(g) == 0 or s <= 0 or s >= 1:
            continue
        n = int(round(len(g) * (1 - s) / s))
        e = g.sample(n, replace=True, random_state=int(rng.integers(1 << 31))).copy()
        e["datetime_analysis"] += pd.to_timedelta(rng.integers(*UNLINKED_DAYS, n), unit="D")
        e["ed_visit_id"] = pd.array([pd.NA] * len(e), dtype="Int64")
        extra.append(e)
    return pd.concat([df, *extra], ignore_index=True) if extra else df


# =============================================================================
# Diagnoses
# =============================================================================

def make_diagnoses(st: Stats, pts: pd.DataFrame, ed: pd.DataFrame, dr: pd.DataFrame,
                   rng: np.random.Generator) -> pd.DataFrame:
    icd, years, per, share = st("diagnoses_icd10_3"), st("diagnoses_year"), st("diagnoses_per_patient"), st("diagnoses_share_patients")
    if icd is None or icd.empty:
        return pd.DataFrame(columns=DIAG_SCHEMA.names)
    s = scalar(share, "share_with_diagnosis", 1.0)
    who = pts["lopnr_swed"].to_numpy()[rng.random(len(pts)) < s]
    k = np.maximum(1, top_coded_counts(pd.Series(draw(per, "per_patient", len(who), rng, lambda m: np.ones(m, int))), rng))
    joint = JointStats(st.folder)
    first = ed.sort_values("ed_arrival_datetime").drop_duplicates("lopnr_swed").reset_index(drop=True)
    ctx = context_keys(first)
    death_map = pd.Series(pd.to_datetime(dr["date_death_minimp"]).to_numpy(), index=dr["lopnr_swed"].to_numpy())
    start = first["ed_arrival_datetime"].dt.tz_convert(STOCKHOLM).dt.tz_localize(None).dt.normalize()
    days = (first["lopnr_swed"].map(death_map) - start).dt.days
    outcome = days.between(0, 30).astype(int)
    profile = pd.DataFrame({
        "age_band": ctx["age_band"].to_numpy(), "sex": ctx["sex"].to_numpy(), "complaint": ctx["complaint"].to_numpy(),
        "outcome": outcome.reindex(first["lopnr_swed"]).to_numpy(),
    }, index=first["lopnr_swed"].to_numpy()).reindex(who)
    cols = ["care_level", "diag_main", "icd10_3"]
    parts = []
    for pid, n in zip(who, k):
        r = profile.loc[pid]
        opts = joint.diagnosis_options(str(r.age_band), str(r.sex), str(r.complaint), str(r.outcome))
        opts = opts if not opts.empty else icd
        parts.append(choose(opts, int(n), rng)[cols].assign(lopnr_swed=pid))
    d = pd.concat(parts, ignore_index=True)
    d["INDATUM"] = pd.NaT
    for cl, idx in d.groupby("care_level").groups.items():
        y = lookup(years, care_level=cl)
        yr = draw(y, "year", len(idx), rng, lambda m: draw(years, "year", m, rng, lambda _: np.full(m, 2015)))
        d.loc[idx, "INDATUM"] = pd.to_datetime(yr.astype(str)) + pd.to_timedelta(rng.integers(0, 365, len(idx)), unit="D")
    inpat = d["care_level"].astype(str).str.contains("in", case=False).to_numpy()
    d["UTDATUM"] = pd.NaT
    d.loc[inpat, "UTDATUM"] = d.loc[inpat, "INDATUM"] + pd.to_timedelta(rng.geometric(INPATIENT_DAYS_P, inpat.sum()), unit="D")
    d["INDATUM"], d["UTDATUM"] = pd.to_datetime(d["INDATUM"]).dt.date, pd.to_datetime(d["UTDATUM"]).dt.date
    d["diag_pos"] = np.where(as_bool(d["diag_main"]).fillna(False), "Hdia", "DIAGNOS1")
    d["icd10_code"] = d["icd10_3"]
    d["SJUKHUS"], d["KTYP"] = None, pd.array([pd.NA] * len(d), dtype="Int32")
    return d.sort_values(["lopnr_swed", "INDATUM"]).reset_index(drop=True)


# =============================================================================
# Output and validation
# =============================================================================

def write(df: pd.DataFrame, schema: pa.Schema, dst: Path, stats_dir: Path,
          extra: dict[str, str] | None = None) -> None:
    t = pa.Table.from_pandas(df.reindex(columns=schema.names), schema=schema, preserve_index=False)
    meta = {**(t.schema.metadata or {}), b"synthetic": b"true",
            b"synthetic_from": str(stats_dir.name).encode()}
    for k, v in (extra or {}).items():
        meta[k.encode()] = str(v).encode()
    t = t.replace_schema_metadata(meta)
    pq.write_table(t, dst, compression="zstd")
    log(f"wrote {dst.name}: {len(df):,} rows")


def validate(st: Stats, out: Path, models: list[MissingnessModel] | None = None) -> None:
    log("validation: synthetic vs exported statistics")
    ed = pd.read_parquet(out / "ed_visits.parquet")
    hy = st("ed_visits_hospital_year").groupby("ed_hospital")["n"].sum()
    cmp = pd.DataFrame({"stats_share": hy / hy.sum(), "synth_share": ed["ed_hospital"].value_counts(normalize=True)}).round(3)
    print(cmp.to_string())
    joint = JointStats(st.folder)
    months = st("ed_visits_hospital_month")
    if months is not None and not months.empty:
        month = ed["ed_arrival_datetime"].dt.tz_convert(STOCKHOLM).dt.month
        ms = pd.DataFrame({"stats_share": months.groupby("month")["n"].sum() / months["n"].sum(),
                           "synth_share": month.value_counts(normalize=True).sort_index()}).round(3)
        print(ms.to_string())
    vals = st("vitals_values")
    if vals is not None and (out / "vitals.parquet").exists():
        v = pd.read_parquet(out / "vitals.parquet")
        v = v[~v["implausible"]]
        q = v.groupby(["source", "from", "concept"])["value"].median().rename("synth_q50").reset_index()
        print(vals[["source", "from", "concept", "q50"]].merge(q, on=["source", "from", "concept"]).to_string(index=False))
        corr = joint.correlation
        if not corr.empty:
            wide = v.pivot_table(index="ed_visit_id", columns="concept", values="value", aggfunc="first")
            rows = []
            for r in corr[(corr["domain"] == "vital") & (corr["source"].fillna("") == "")].itertuples():
                a = str(r.var_a) if str(r.var_a) in wide else VITAL_CONCEPTS.get(str(r.var_a))
                b = str(r.var_b) if str(r.var_b) in wide else VITAL_CONCEPTS.get(str(r.var_b))
                if a not in wide or b not in wide:
                    continue
                synth = wide[[a, b]].corr(method="spearman").iloc[0, 1]
                rows.append({"pair": f"{a}-{b}", "exported_rho": round(float(r.rho), 3),
                             "synth_rho": round(float(synth), 3), "abs_diff": round(abs(float(r.rho) - float(synth)), 3)})
            if rows:
                print(pd.DataFrame(rows).to_string(index=False))
    labs = st("labs_terms")
    if labs is not None and (out / "lab_meas.parquet").exists():
        lab = pd.read_parquet(out / "lab_meas.parquet")
        top = labs.groupby("term_name")["n"].sum().nlargest(8)
        syn = lab["term_name"].value_counts(normalize=True)
        print(pd.DataFrame({"stats_share": (top / labs["n"].sum()).round(3), "synth_share": syn.reindex(top.index).round(3)}).to_string())
    for model in models or []:
        if model is None or not model.available:
            continue
        rows = []
        source = pd.read_parquet(out / ("vitals.parquet" if model.domain == "vital" else "lab_meas.parquet"))
        for (group, lab_source, variable), target in model.rates.items():
            visits = ed[ed["ed_hospital"] == group]
            if visits.empty:
                continue
            if model.domain == "vital":
                seen = source[source["concept"] == VITAL_CONCEPTS.get(variable)]["ed_visit_id"]
            else:
                seen = source[(source["source"] == lab_source)
                              & (source["term_name"] == variable.replace("lab_", "", 1))]["ed_visit_id"]
            synth = float(seen[seen.isin(visits["ed_visit_id"])].nunique() / len(visits))
            rows.append({"domain": model.domain, "hospital": group, "source": lab_source,
                         "variable": variable, "target_rate": round(target, 3), "synth_rate": round(synth, 3)})
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", type=Path, required=True, help="folder written by 03_export_stats.py")
    ap.add_argument("--n-patients", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--out", type=Path, help="default: synthetic_<stats folder name>")
    ap.add_argument("--validate", action="store_true", help="print synthetic vs exported statistics")
    ap.add_argument("--missingness", choices=("auto", "legacy", "fit"), default="auto",
                    help="use exported missingness assumptions when present")
    ap.add_argument("--gx-scale", type=float, default=1.0,
                    help="scale fitted MNAR own-value coefficients for sensitivity analysis")
    args = ap.parse_args()
    out = args.out or Path(f"synthetic_{args.stats.name}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    st = Stats(args.stats)
    log(f"stats: {args.stats}")
    log(f"out:   {out}")

    vital_model, lab_models = None, []
    if args.missingness != "legacy":
        vital_model = MissingnessModel(args.stats, "vital", VITAL_VARIABLES, "missingness_patterns")
        config_path = args.stats / "missingness_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            for source, concepts in sorted(config.get("lab_concepts", {}).items()):
                lab_models.append(MissingnessModel(
                    args.stats, "lab", [f"lab_{c}" for c in concepts], "lab_patterns", source=source))
        if args.missingness == "fit" and not any(m and m.available for m in [vital_model, *lab_models]):
            raise SystemExit("no usable missingness assumptions found in the stats folder")
    meta = {"missingness_mode": args.missingness, "gx_scale": str(args.gx_scale)}
    joint = JointStats(args.stats)
    if joint.available:
        meta["joint_stats"] = "true"
    if vital_model and vital_model.available:
        meta["vital_missingness"] = "fit"
    lab_fit_sources = [m.source for m in lab_models if m.available and m.source]
    if lab_fit_sources:
        meta["lab_missingness"] = "fit"
        meta["lab_sources"] = ",".join(sorted(lab_fit_sources))

    pts = make_patients(st, args.n_patients, rng)
    ed = make_visits(st, pts, rng)
    dr = make_deaths(st, pts, ed, rng)
    death_by_patient = pd.Series(pd.to_datetime(dr["date_death_minimp"]).to_numpy(),
                                 index=dr["lopnr_swed"].to_numpy())
    arrival = ed["ed_arrival_datetime"].dt.tz_convert(STOCKHOLM).dt.tz_localize(None).dt.normalize()
    days = (pd.to_datetime(ed["lopnr_swed"].map(death_by_patient)) - arrival).dt.days
    y = days.between(0, 30).astype(float).to_numpy()
    ed = add_base_design(ed, complaints_from_stats(st))
    ed["died30"] = y

    write(ed, ED_SCHEMA, out / "ed_visits.parquet", args.stats, meta)
    write(dr, DEATHS_SCHEMA, out / "deaths.parquet", args.stats, meta)
    write(finish_patients(pts, ed, dr, rng), PATIENTS_SCHEMA, out / "patients.parquet", args.stats, meta)
    write(make_vitals(st, ed, rng, y, vital_model, args.gx_scale), build.VITALS_SCHEMA,
          out / "vitals.parquet", args.stats, meta)
    write(make_labs(st, ed, rng, y, lab_models, args.gx_scale), build.LABS_SCHEMA,
          out / "lab_meas.parquet", args.stats, meta)
    write(make_diagnoses(st, pts, ed, dr, rng), DIAG_SCHEMA, out / "diagnoses.parquet", args.stats, meta)
    if args.validate:
        validate(st, out, [m for m in (vital_model, *lab_models) if m])
    log("done")


if __name__ == "__main__":
    main()
