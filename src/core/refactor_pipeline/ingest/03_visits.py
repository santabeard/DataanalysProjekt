"""Visits file: one row per ED visit with complaint, vitals and diagnosis.

In:  swed_ed_core, every lab/meas file in VITAL_SOURCES, sos_patreg_wide
Out: OUT_DIR/visits.parquet
     visit_id             int64     row number in swed_ed_core (stable: the source is read-only)
     lopnr_swed           int64
     ed_hospital          category
     ed_arrival_datetime  timestamp, UTC
     age                  float64   ed_age at this visit
     complaint_sctid      category  first SNOMED code in ed_presented_complaint_std (null if none)
     complaint_text       category  its text
     <vital>              float64   first plausible reading in the visit window, one column per vital:
     <vital>_minutes      float64   systolic_bp, diastolic_bp, pulse, resp_rate, spo2, temperature;
                                    _minutes = minutes from arrival to that reading (negative = before)
     diagnosis            string    Hdia of the first patreg contact within 30 days, or
                                    "NODIAG" = patreg covers the 30 days but has no contact, or
                                    null     = the 30 days run past the end of patreg (2016-12-31)
     diagnosis_days       float64   days from arrival to that contact (0 = same day)

Rules (from v2/phase1 where it settled them):
  - Visits with ed_inexact_time = TRUE are dropped: their clock time is unknown, so
    the vitals window can't be placed (the documentation says to remove them).
  - Vitals window: arrival - 60 min until arrival + ed_time_total_minutes
    (24 h when that is missing). First reading per vital, by time. Readings with
    inexact_time = TRUE, no value, or outside PLAUSIBLE are dropped. SpO2 written
    as a fraction (0-1) is scaled to percent. A reading goes to the patient's latest
    visit whose window has opened; overlapping visits are not split further.
  - Diagnosis: Hdia only, INDATUM in [arrival local date, + 30 days], both ends
    included. Ties on the same INDATUM keep the row read first.

Memory: holds the ~7.4M visits plus reduced vitals in memory (several GB on the full data;
run with sbatch -n 8). Every file is read in batches of at most BATCH_SIZE rows.

    python 03_visits.py
"""
import pandas as pd
import ingest_common as C

ED_COLUMNS = ["lopnr_swed", "ed_hospital", "ed_arrival_datetime", "ed_inexact_time",
              "ed_age", "ed_presented_complaint_std", "ed_time_total_minutes"]
VITAL_COLUMNS = ["lopnr_swed", "datetime_analysis", "term_name", "value_num"]
PATREG_COLUMNS = ["lopnr_swed", "INDATUM", "Hdia"]
COMPACT_EVERY = 20  # batches between merges of the partial vitals/diagnosis results


# ------------------------------------------------------------------- visits ---
def load_visits():
    path = C.source_path(C.ED_CORE)
    C.log(f"reading visits from {path}")
    parts = []
    offset = n_inexact = n_excluded = 0
    for b in C.iter_batches(path, ED_COLUMNS):
        b["visit_id"] = range(offset, offset + len(b))
        offset += len(b)
        b["lopnr_swed"] = C.to_lopnr(b["lopnr_swed"])
        inexact = b["ed_inexact_time"].fillna(False).astype(bool)
        excluded = ~C.not_excluded(b)
        n_inexact += int(inexact.sum())
        n_excluded += int((excluded & ~inexact).sum())
        b = b.loc[~inexact & ~excluded].copy()
        complaint = b["ed_presented_complaint_std"]
        b["complaint_sctid"] = complaint.str.extract(r"\[(\d+)\]", expand=False)
        b["complaint_text"] = complaint.str.extract(r"^\s*([^\[]*?)\s*\[", expand=False)
        parts.append(b.drop(columns=["ed_presented_complaint_std", "ed_inexact_time"]))

    v = pd.concat(parts, ignore_index=True)
    for col in ["ed_hospital", "complaint_sctid", "complaint_text"]:
        v[col] = v[col].astype("category")
    v["visit_id"] = v["visit_id"].astype("int64")
    v["ed_arrival_datetime"] = C.as_utc(v["ed_arrival_datetime"])
    n_dup = int(v.duplicated(["lopnr_swed", "ed_arrival_datetime"]).sum())
    C.log(f"visits read {offset:,} | inexact time dropped {n_inexact:,} | excluded {n_excluded:,} | "
          f"kept {len(v):,} | same patient+arrival duplicates {n_dup:,}")
    C.log(f"visits without a complaint code: {int(v['complaint_sctid'].isna().sum()):,}")
    return v


# ------------------------------------------------------------------- vitals ---
def visit_windows(v):
    arrival = v["ed_arrival_datetime"].dt.tz_convert(None)  # naive UTC, for merge_asof
    total = v["ed_time_total_minutes"]
    C.log(f"visits without ed_time_total_minutes (window = {C.VITALS_FALLBACK_MIN} min): "
          f"{int(total.isna().sum()):,}")
    total = total.fillna(C.VITALS_FALLBACK_MIN)
    w = pd.DataFrame({
        "visit_id": v["visit_id"].values,
        "lopnr_swed": v["lopnr_swed"].values,
        "arrival": arrival.values,
        "win_start": (arrival - pd.Timedelta(minutes=C.VITALS_BEFORE_MIN)).values,
        "win_end": (arrival + pd.to_timedelta(total, unit="m")).values,
    })
    return w.sort_values("win_start", kind="mergesort").reset_index(drop=True)


def clean_vital_batch(b, stats, unmatched):
    """Keep vital rows with a usable time and a plausible value; returns lopnr_swed, t, vital, value."""
    vital = b["term_name"].map(C.VITAL_TERMS)

    other = b.loc[vital.isna(), "term_name"].value_counts()
    if len(other):
        like = pd.Series(other.index, dtype="object").str.contains(C.VITAL_LIKE_PATTERN, case=False, na=False)
        for term, n in other[like.values].items():
            unmatched[term] = unmatched.get(term, 0) + int(n)

    b = b.loc[vital.notna()].copy()
    b["vital"] = vital[vital.notna()]
    stats["vital_rows"] += len(b)
    if "inexact_time" in b.columns:
        inexact = b["inexact_time"].fillna(False).astype(bool)
        stats["inexact"] += int(inexact.sum())
        b = b.loc[~inexact]

    value = pd.to_numeric(b["value_num"], errors="coerce")
    no_value = value.isna() | b["datetime_analysis"].isna()
    stats["no_value"] += int(no_value.sum())
    b, value = b.loc[~no_value], value[~no_value]

    fraction = (b["vital"] == "spo2") & (value > 0) & (value <= 1)
    value = value.where(~fraction, value * 100)
    names = b["vital"].astype("object")  # newer pandas keeps a string dtype through .map()
    lo = names.map({k: lim[0] for k, lim in C.PLAUSIBLE.items()}).astype("float64")
    hi = names.map({k: lim[1] for k, lim in C.PLAUSIBLE.items()}).astype("float64")
    ok = (value >= lo) & (value <= hi)
    stats["implausible"] += int((~ok).sum())
    b, value = b.loc[ok], value[ok]

    out = pd.DataFrame({
        "lopnr_swed": C.to_lopnr(b["lopnr_swed"]).values,
        "t": C.as_utc(b["datetime_analysis"]).dt.tz_convert(None).values,
        "vital": b["vital"].values,
        "value": value.astype("float64").values,
    })
    return out.loc[C.not_excluded(out)]


def match_to_visits(r, w):
    """Attach each reading to the patient's latest visit whose window has opened,
    keep it if it is before that window closes, then keep the first reading per visit and vital."""
    r = r.sort_values("t", kind="mergesort")
    ws = w.loc[w["lopnr_swed"].isin(r["lopnr_swed"].unique())]
    m = pd.merge_asof(r, ws, left_on="t", right_on="win_start", by="lopnr_swed", direction="backward")
    m = m.loc[m["visit_id"].notna() & (m["t"] <= m["win_end"])]
    if m.empty:
        return None
    m = m.drop_duplicates(["visit_id", "vital"], keep="first")  # already sorted by time
    return pd.DataFrame({
        "visit_id": m["visit_id"].astype("int64").values,
        "vital": m["vital"].values,
        "value": m["value"].values,
        "minutes": ((m["t"] - m["arrival"]) / pd.Timedelta(minutes=1)).values,
    })


def collect_vitals(v):
    w = visit_windows(v)
    found, unmatched = [], {}
    for table in C.VITAL_SOURCES:
        path = C.source_path(table)
        if not path.exists():
            C.log(f"{table}: file not found, skipped ({path})")
            continue
        names = C.file_columns(path)
        missing = [c for c in VITAL_COLUMNS if c not in names]
        if missing:
            C.log(f"{table}: no column(s) {missing}, skipped")
            continue
        columns = VITAL_COLUMNS + (["inexact_time"] if "inexact_time" in names else [])
        stats = {"read": 0, "vital_rows": 0, "inexact": 0, "no_value": 0, "implausible": 0, "matched": 0}
        n_batches = 0
        for b in C.iter_batches(path, columns):
            stats["read"] += len(b)
            r = clean_vital_batch(b, stats, unmatched)
            m = match_to_visits(r, w) if len(r) else None
            if m is not None:
                stats["matched"] += len(m)
                found.append(m)
            n_batches += 1
            if len(found) >= COMPACT_EVERY:
                found = [C.keep_first(found, ["visit_id", "vital"], "minutes")]
            if n_batches % 50 == 0:
                C.log(f"  {table}: {stats['read']:,} rows read")
        C.log(f"{table}: read {stats['read']:,} | vital rows {stats['vital_rows']:,} | "
              f"inexact time {stats['inexact']:,} | no value {stats['no_value']:,} | "
              f"implausible {stats['implausible']:,} | first-in-window readings {stats['matched']:,}")

    if unmatched:
        C.log("unmapped vital-like term names (add to VITAL_TERMS if they are vitals):")
        for term, n in sorted(unmatched.items(), key=lambda kv: -kv[1])[:40]:
            C.log(f"    {n:>12,}  {term!r}")

    if not found:
        C.log("no vital readings matched any visit")
        return pd.DataFrame(index=pd.Index([], name="visit_id", dtype="int64"))
    first = C.keep_first(found, ["visit_id", "vital"], "minutes")
    values = first.pivot(index="visit_id", columns="vital", values="value")
    minutes = first.pivot(index="visit_id", columns="vital", values="minutes")
    wide = pd.DataFrame(index=values.index)
    for vital in C.VITALS:
        wide[vital] = values[vital] if vital in values.columns else float("nan")
        wide[vital + "_minutes"] = minutes[vital] if vital in minutes.columns else float("nan")
    return wide


# ---------------------------------------------------------------- diagnosis ---
def collect_diagnoses(v):
    path = C.source_path(C.PATREG)
    C.log(f"reading diagnoses from {path}")
    d = pd.DataFrame({
        "visit_id": v["visit_id"].values,
        "lopnr_swed": v["lopnr_swed"].values,
        "arrival_date": C.to_local_date(v["ed_arrival_datetime"]).values,
    }).sort_values("arrival_date", kind="mergesort").reset_index(drop=True)
    window = pd.Timedelta(days=C.DIAG_WINDOW_DAYS)
    first_day, last_day = d["arrival_date"].min(), d["arrival_date"].max() + window
    patients = pd.Index(d["lopnr_swed"].unique())

    found = []
    coverage_end = None
    n_read = n_used = n_batches = 0
    for b in C.iter_batches(path, PATREG_COLUMNS):
        n_read += len(b)
        dated = b["INDATUM"].notna()
        if dated.any():
            batch_max = b.loc[dated, "INDATUM"].max()
            coverage_end = batch_max if coverage_end is None else max(coverage_end, batch_max)
        b = b.loc[dated & b["Hdia"].notna()].copy()
        b["lopnr_swed"] = C.to_lopnr(b["lopnr_swed"])
        b = b.loc[(b["INDATUM"] >= first_day) & (b["INDATUM"] <= last_day)
                  & b["lopnr_swed"].isin(patients) & C.not_excluded(b)]
        n_batches += 1
        if b.empty:
            continue
        n_used += len(b)
        b = b.sort_values("INDATUM", kind="mergesort")
        left = d.loc[d["lopnr_swed"].isin(b["lopnr_swed"].unique())]
        m = pd.merge_asof(left, b, left_on="arrival_date", right_on="INDATUM", by="lopnr_swed",
                          direction="forward", tolerance=window)
        m = m.loc[m["INDATUM"].notna(), ["visit_id", "arrival_date", "INDATUM", "Hdia"]]
        if len(m):
            found.append(m)
        if len(found) >= COMPACT_EVERY:
            found = [C.keep_first(found, "visit_id", "INDATUM")]
        if n_batches % 50 == 0:
            C.log(f"  patreg: {n_read:,} rows read")

    if coverage_end is None:
        raise SystemExit(f"{path.name} has no INDATUM values")
    C.log(f"patreg: read {n_read:,} | Hdia rows in the visits' date range {n_used:,} | "
          f"last INDATUM {coverage_end:%Y-%m-%d}")

    if found:
        first = C.keep_first(found, "visit_id", "INDATUM")
    else:
        first = pd.DataFrame(columns=["visit_id", "arrival_date", "INDATUM", "Hdia"])
    diag = d[["visit_id", "arrival_date"]].merge(
        first[["visit_id", "INDATUM", "Hdia"]], on="visit_id", how="left")
    covered = diag["arrival_date"] + window <= coverage_end
    diag["diagnosis"] = diag["Hdia"].astype("object")
    diag.loc[diag["diagnosis"].isna() & covered, "diagnosis"] = "NODIAG"
    diag["diagnosis_days"] = (diag["INDATUM"] - diag["arrival_date"]).dt.days.astype("float64")
    n_found = int(diag["Hdia"].notna().sum())
    n_nodiag = int((diag["diagnosis"] == "NODIAG").sum())
    C.log(f"diagnosis: Hdia found {n_found:,} | NODIAG {n_nodiag:,} | "
          f"not covered by patreg (null) {len(diag) - n_found - n_nodiag:,} | "
          f"found exactly on day {C.DIAG_WINDOW_DAYS}: {int((diag['diagnosis_days'] == C.DIAG_WINDOW_DAYS).sum()):,}")
    return diag[["visit_id", "diagnosis", "diagnosis_days"]]


# --------------------------------------------------------------------- main ---
def main():
    C.check_settings()
    v = load_visits()
    vitals = collect_vitals(v)
    diag = collect_diagnoses(v)

    out = v[["visit_id", "lopnr_swed", "ed_hospital", "ed_arrival_datetime", "ed_age",
             "complaint_sctid", "complaint_text"]].rename(columns={"ed_age": "age"})
    del v
    out = out.merge(vitals, left_on="visit_id", right_index=True, how="left")
    out = out.merge(diag, on="visit_id", how="left")
    for vital in C.VITALS:
        if vital not in out.columns:
            out[vital] = float("nan")
            out[vital + "_minutes"] = float("nan")
    out = out.sort_values("visit_id").reset_index(drop=True)

    C.log("share of visits with each vital: " + ", ".join(
        f"{vital} {out[vital].notna().mean():.3f}" for vital in C.VITALS))
    C.log("visits per hospital with any vital: " + ", ".join(
        f"{h} {g[C.VITALS].notna().any(axis=1).mean():.2f}"
        for h, g in out.groupby("ed_hospital", observed=True)))
    C.write_parquet(out, "visits.parquet")


if __name__ == "__main__":
    main()
