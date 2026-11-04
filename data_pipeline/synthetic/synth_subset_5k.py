#!/usr/bin/env python3
"""Synthetic version of subset/ (the 10 raw tables from 01_subset.py) for 5,000 patients.

Same file names, columns and types as the real subset, so 02_build.py and the later steps run on it:

    python synthetic/synth_subset_5k.py            # run from data_pipeline/ -> synthetic/synth_subset_5k/
    SWED_DATA=synthetic/synth_subset_5k SWED_WORK=synthetic/synth_subset_5k SWED_OUT=synthetic python 02_build.py
                                                   # -> synthetic/built_synth_subset_5k/

Sizes, dates, null rates and quantiles are rough targets from data_info/swed_profile.txt; nothing is
copied from real rows. Vitals, labs, diagnoses and complaints have made-up names (V1.., L1.., D1.., C1..),
so the files carry no real codes. out/vitals_snomed_map.csv tells 02_build.py which V is which vital.
There are no real correlations: only age raises the risk of death, and vitals shift in the 30 days
before a death, so the models have something to find. Use it to test code, not findings.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DAY, MIN = np.timedelta64(1, "D"), np.timedelta64(1, "m")
S, F, B, I32, DT = pa.string(), pa.float64(), pa.bool_(), pa.int32(), pa.date32()
TS = pa.timestamp("us", tz="UTC")
DICT = pa.dictionary(pa.int8(), pa.string())

# =============================================================================
# Targets from data_info/swed_profile.txt (rounded). ASSUMPTION marks guesses.
# =============================================================================

HOSPITALS = {"karolinska": 22.6, "södersjukhuset": 17.7, "sahlgrenska": 16.2, "danderyd": 14.4, "st_göran": 11.9,
             "västerås": 3.9, "hosp_07": 2.2, "hosp_08": 2.2, "hosp_09": 2.2, "hosp_10": 2.2, "hosp_11": 2.2, "hosp_12": 2.2}
HOME_HOSPITAL = 0.85            # ASSUMPTION: share of a patient's visits at their usual hospital
VISITS_PER_PATIENT = 3.59       # 7.80M visits / 2.17M patients
MALE = 0.46
AGE_Q = [(0, 18), (.05, 22), (.25, 35), (.5, 54), (.75, 72), (.95, 88), (1, 108)]
ARRIVAL_Q = [(0, "2003-04-02"), (.05, "2005-10-20"), (.25, "2009-07-14"), (.5, "2012-09-07"),
             (.75, "2015-03-25"), (.95, "2017-03-10"), (1, "2017-12-18")]
TO_DOCTOR_Q = [(0, 0), (.05, 5), (.5, 60), (.95, 291), (1, 900)]      # minutes
TOTAL_Q = [(0, 5), (.05, 33), (.5, 191), (.95, 527), (1, 1800)]       # minutes
EPISODES_PER_PATIENT = 28       # 61.6M register episodes / 2.17M patients
EPISODE_Q = [(0, "1996-01-01"), (.05, "2001-12-05"), (.5, "2010-12-02"), (.95, "2016-06-02"), (1, "2016-12-31")]
DEATH_END = np.datetime64("2017-12-31")
MORSAK_P = [.89, .76, .59, .41, .27, .16, .10, .06, .03, .017, .009, .005]  # share of deaths with MORSAK1, 2 ...
MUNICIPALITIES = ["0114", "0117", "0123", "0126", "0127", "0136", "0138", "0160"]
UNLINKED = 0.25                 # ASSUMPTION: share of measurements taken outside any ED visit

# Made-up names only. Vital: name, unit, mean, sd, decimals, (lo, hi), shift before death, share of rounds, concept
VITALS = [
    ("V1", "mmHg", 135, 22, 0, (50, 250), -15, 1.00, "systolic_bp"),
    ("V2", "mmHg", 78, 13, 0, (30, 150), -8, 0.95, "diastolic_bp"),
    ("V3", "%", 96.5, 2.5, 0, (70, 100), -3, 0.90, "spo2"),
    ("V4", "andetag/min", 17, 4, 0, (6, 50), 4, 0.80, "resp_rate"),
    ("V5", "°C", 36.9, 0.6, 1, (34, 41.5), 0.5, 0.80, "body_temp"),
    ("V6", "slag/min", 84, 17, 0, (30, 200), 15, 0.60, "pulse_rate"),
    ("V7", "p", 1.1, 0.4, 0, (1, 8), 1, 0.30, "consciousness"),
]
# Lab: name, unit, median, log-sd, share of panels
LABS = [("L1", "g/L", 135, .12, .95), ("L2", "x10(9)/L", 8, .4, .95), ("L3", "x10(9)/L", 240, .35, .9),
        ("L4", "mmol/L", 139, .03, .9), ("L5", "mmol/L", 4.0, .1, .9), ("L6", "mikromol/L", 80, .35, .85),
        ("L7", "mg/L", 5, 1.2, .7), ("L8", "mmol/L", 6, .25, .5), ("L9", "mikrokat/L", .4, .5, .4),
        ("L10", "mmol/L", 2.3, .08, .3), ("L11", "ng/L", 10, 1.0, .3), ("L12", "mmol/L", 1.5, .6, .2)]
DIAGNOSES = [f"D{i}" for i in range(1, 41)]
COMPLAINTS = list(range(1, 31))  # written as "C1 [1]" (std) and "C1" (free text)


# =============================================================================
# Helpers
# =============================================================================

def quantiles(rng, n, q):
    """n draws by linear interpolation between (probability, value) knots."""
    p, x = zip(*q)
    return np.interp(rng.random(n), p, np.asarray(x, float))


def dates(rng, n, q):
    p, x = zip(*q)
    days = [np.datetime64(d, "D").astype(np.int64) for d in x]
    return np.interp(rng.random(n), p, days).astype(np.int64).astype("datetime64[D]")


def choose(rng, labels, n, weights=None, null=0.0):
    """n labels; without weights the first ones are most common (1, 1/2, 1/3 ...). A share `null` is None."""
    w = np.asarray(weights, float) if weights is not None else 1 / np.arange(1, len(labels) + 1)
    out = np.asarray(labels, dtype=object)[rng.choice(len(labels), n, p=w / w.sum())]
    out[rng.random(n) < null] = None
    return out


def numbers(rng, values, weights, n, null=0.0):
    out = rng.choice(np.asarray(values, float), n, p=np.asarray(weights, float) / sum(weights))
    out[rng.random(n) < null] = np.nan
    return out


def cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def to_arrow(v, typ):
    if pa.types.is_dictionary(typ):
        cats, idx = np.unique(np.asarray(v, dtype=str), return_inverse=True)
        return pa.DictionaryArray.from_arrays(pa.array(idx.astype(np.int8)), pa.array(cats))
    v = np.asarray(v)
    if v.dtype.kind == "M":
        return pa.array(v.astype("datetime64[us]" if pa.types.is_timestamp(typ) else "datetime64[D]"), typ, mask=np.isnat(v))
    if v.dtype.kind == "f":  # NaN becomes null; also for int columns such as KTYP
        mask = np.isnan(v)
        return pa.array(np.where(mask, 0, v).astype(typ.to_pandas_dtype()), typ, mask=mask)
    if v.dtype.kind in "iu":
        return pa.array(v.astype(typ.to_pandas_dtype()), typ)
    return pa.array(v, typ)  # strings (None = null), bools


def write(path, cols):
    """cols: {name: (values, arrow type)} in file order."""
    table = pa.Table.from_arrays([to_arrow(v, t) for v, t in cols.values()],
                                 schema=pa.schema([pa.field(k, t) for k, (_, t) in cols.items()]))
    pq.write_table(table, path, compression="zstd")
    print(f"  {path.name}: {table.num_rows:,} rows")


# =============================================================================
# Measurements
# =============================================================================

def measure_times(rng, start, stay, before):
    off = rng.uniform(before, stay)
    off += np.where(rng.random(len(off)) < UNLINKED, rng.integers(2, 60, len(off)) * 1440, 0)  # days after a visit
    return start + off.astype(np.int64) * MIN


def vital_rows(rng, visits, arrival, total, sick):
    """Rounds of vital signs during each visit (ASSUMPTION: ~2 rounds); one row per round and vital."""
    rv = np.repeat(visits, rng.geometric(0.45, len(visits)))
    time = measure_times(rng, arrival[rv], np.nan_to_num(total[rv], nan=240.0), -10)
    parts = []
    for j, (name, unit, mean, sd, dec, (lo, hi), shift, share, _) in enumerate(VITALS, 1):
        has = rng.random(len(rv)) < share
        k = has.sum()
        value = rng.normal(mean + shift * sick[rv[has]], sd)
        parts.append({"visit": rv[has], "time": time[has], "name": np.full(k, name, object),
                      "value": np.round(np.clip(value, lo, hi), dec), "unit": np.full(k, unit, object),
                      "tid": np.full(k, 1000.0 + j)})
    return cat(parts)


def lab_rows(rng, visits, arrival):
    """One lab panel per visit, 5-120 min after arrival; one row per lab in the panel."""
    time = measure_times(rng, arrival[visits], np.full(len(visits), 120.0), 5)
    parts = []
    for j, (name, unit, median, logsd, share) in enumerate(LABS, 1):
        has = rng.random(len(visits)) < share
        k = has.sum()
        parts.append({"visit": visits[has], "time": time[has], "name": np.full(k, name, object),
                      "value": np.round(median * np.exp(rng.normal(0, logsd, k)), 2), "unit": np.full(k, unit, object),
                      "tid": np.full(k, 2000.0 + j)})
    return cat(parts)


# =============================================================================
# Main
# =============================================================================

def main(n: int, out: Path, seed: int = 20261104) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{n:,} synthetic patients -> {out}")

    # --- patients and ED visits ---
    lopnr = np.arange(1, n + 1, dtype=float)
    male = (rng.random(n) < MALE).astype(np.int32)
    k = rng.geometric(1 / VISITS_PER_PATIENT, n)
    pid = np.repeat(np.arange(n), k)
    arrival = dates(rng, len(pid), ARRIVAL_Q).astype("datetime64[m]") \
        + (rng.normal(14 * 60, 300, len(pid)) % 1440).astype(np.int64) * MIN  # ASSUMPTION: busiest mid-afternoon
    order = np.lexsort((arrival, pid))
    pid, arrival = pid[order], arrival[order]
    starts = np.r_[0, np.cumsum(k)[:-1]]
    age_first = quantiles(rng, n, AGE_Q)
    birth = arrival[starts].astype("datetime64[D]") - (age_first * 365.25 + rng.uniform(0, 365, n)).astype(np.int64) * DAY
    ed_age = np.floor((arrival.astype("datetime64[D]") - birth[pid]) / DAY / 365.25)
    keep = ed_age <= 109
    pid, arrival, ed_age = pid[keep], arrival[keep], ed_age[keep]
    m = len(pid)

    names = np.array(list(HOSPITALS), dtype=object)
    w = np.array(list(HOSPITALS.values())) / sum(HOSPITALS.values())
    h = rng.choice(len(names), n, p=w)[pid]
    away = rng.random(m) > HOME_HOSPITAL
    h[away] = rng.choice(len(names), away.sum(), p=w)
    hospital = names[h]

    # --- deaths: older patients die more often, some soon after their last visit ---
    last = np.zeros(n, np.int64)
    np.maximum.at(last, pid, arrival.astype(np.int64))
    age_last = np.zeros(n)
    np.maximum.at(age_last, pid, ed_age)
    dead = rng.random(n) < 1 / (1 + np.exp(-(-2.6 + 0.06 * (age_last - 55))))  # ASSUMPTION: ~12% die by 2017
    gap = np.where(rng.random(n) < 0.35, rng.integers(0, 31, n), 31 + rng.exponential(900, n).astype(np.int64))
    death = last.astype("datetime64[m]").astype("datetime64[D]") + gap * DAY
    dead &= death <= DEATH_END
    alive_until = np.where(dead, death, np.datetime64("2100-01-01"))
    sick = (alive_until[pid] - arrival.astype("datetime64[D]")) <= 30 * DAY  # visit within 30 days of death

    # --- swed_ed_core, swed_ed_core_birthdate ---
    inexact = rng.random(m) < 0.05
    shown = arrival.astype("datetime64[us]")
    shown[inexact] = shown[inexact].astype("datetime64[D]")  # inexact times keep only the date
    local = pd.DatetimeIndex(shown).tz_localize("UTC").tz_convert("Europe/Stockholm")
    arrival_time = np.array(local.strftime("%H:%M"), dtype=object)
    arrival_time[inexact] = None
    ci = choose(rng, COMPLAINTS, m)
    std = np.array([f"C{c} [{c}]" for c in ci], dtype=object)
    std[rng.random(m) < 0.166] = None
    nonstd = np.array([f"C{c}" for c in ci], dtype=object)
    nonstd[rng.random(m) < 0.091] = None
    to_doctor, total = np.round(quantiles(rng, m, TO_DOCTOR_Q)), np.round(quantiles(rng, m, TOTAL_Q))
    to_doctor[rng.random(m) < 0.409] = np.nan
    total[rng.random(m) < 0.052] = np.nan
    write(out / "swed_ed_core.parquet", {
        "ed_hospital": (hospital, S), "lopnr_swed": (lopnr[pid], F), "ed_arrival_datetime": (shown, TS),
        "ed_inexact_time": (inexact, B), "ed_age": (ed_age, F), "is_male": (male[pid], I32),
        "ed_presented_complaint_std": (std, S),
        "ed_arrival_date": (local.tz_localize(None).normalize().values.astype("datetime64[D]"), DT),
        "ed_arrival_time": (arrival_time, S), "ed_time_to_doctor_minutes": (to_doctor, F), "ed_time_total_minutes": (total, F),
        "ed_priority_in": (numbers(rng, [1, 2, 3, 4, 5], [8, 25, 35, 25, 7], m, null=0.217), F),
        "ed_arrival_type": (choose(rng, ["Gående", "Ambulans", "Sjuktransport"], m, [76, 22, 2], null=0.366), S),
        "care_unit": (choose(rng, ["MED", "KIR", "AKUTEN", "ORT"], m, [30, 25, 25, 20], null=0.115), S),
        "ed_priority_out": (numbers(rng, [1, 2, 3, 4, 5], [3, 12, 30, 35, 20], m, null=0.516), F),
        "ed_exit_to": (choose(rng, ["Hem", "Intagen", "Avvikit"], m, [82, 14, 4], null=0.322), S),
        "ed_exit_to_care_unit": (choose(rng, ["Medicin", "Kirurgi", "Ortopedi"], m, [50, 30, 20], null=0.904), S),
        "ed_presented_complaint_nonstd": (nonstd, S),
    })
    write(out / "swed_ed_core_birthdate.parquet", {"lopnr_swed": (lopnr, F), "birth_date": (birth, DT)})

    # --- hospital EHR files: vitals at Karolinska and Danderyd/SÖS, labs everywhere they are recorded ---
    def at(hosp, share):
        return np.flatnonzero(np.isin(hospital, hosp) & (rng.random(m) < share))

    ks_v = vital_rows(rng, at(["karolinska"], .5), arrival, total, sick)
    ds = cat([vital_rows(rng, at(["danderyd", "södersjukhuset"], .5), arrival, total, sick),
              lab_rows(rng, at(["danderyd", "södersjukhuset"], .6), arrival)])
    ds["tid"][ds["tid"] >= 2000] = np.nan  # labs have no term_id in this file
    ks_l = lab_rows(rng, at(["karolinska"], .6), arrival)
    su = lab_rows(rng, at(["sahlgrenska"], .7), arrival)
    vgr = lab_rows(rng, at(["hosp_11", "hosp_12"], .6), arrival)  # ASSUMPTION: which hospitals send VGR labs

    def null(k):
        return np.full(k, None, object)

    def false(k):
        return np.full(k, False)

    k = len(ks_v["visit"])
    write(out / "ehr_karolinska_meas.parquet", {
        "from": (np.full(k, "karolinska_meas", object), S), "source_row": (np.arange(k), I32),
        "lopnr_swed": (lopnr[pid[ks_v["visit"]]], F), "datetime_analysis": (ks_v["time"], TS),
        "term_name": (ks_v["name"], S), "term_id": (ks_v["tid"], F), "value_num": (ks_v["value"], F), "unit": (ks_v["unit"], S)})

    k = len(ds["visit"])
    write(out / "ehr_danderyd_lab_meas.parquet", {
        "from": (np.char.add(choose(rng, ["ds3_", "ds4_"], k, [60, 40]).astype(str), hospital[ds["visit"]].astype(str)).astype(object), S),
        "lopnr_swed": (lopnr[pid[ds["visit"]]], F), "inexact_time": (false(k), B), "datetime_analysis": (ds["time"], TS),
        "term_name": (ds["name"], S), "term_id": (ds["tid"], F), "value_str": (null(k), S), "value_num": (ds["value"], F),
        "unit": (ds["unit"], S), "lt_flag": (false(k), B), "gt_flag": (false(k), B),
        "other_flag": (null(k), S), "ref_val1": (null(k), S)})

    k = len(ks_l["visit"])
    write(out / "ehr_karolinska_lab.parquet", {
        "from": (choose(rng, ["karolinska_lab1", "karolinska_lab2", "karolinska_lab3"], k, [29, 32, 39]), S),
        "source_row": (np.arange(k), I32), "lopnr_swed": (lopnr[pid[ks_l["visit"]]], F), "inexact_time": (false(k), B),
        "datetime_analysis": (ks_l["time"], TS), "date_analysis": (ks_l["time"], DT), "term_name": (ks_l["name"], S),
        "value_str": (null(k), S), "value_num": (ks_l["value"], F), "unit": (ks_l["unit"], S),
        "lt_flag": (false(k), B), "gt_flag": (false(k), B), "other_flag": (null(k), S), "ref_val1": (null(k), S)})

    k = len(su["visit"])
    write(out / "ehr_sahlgrenska_lab.parquet", {
        "from": (choose(rng, ["sahlgrenska_drift", "sahlgrenska_clab"], k, [65, 35]), S),
        "lopnr_swed": (lopnr[pid[su["visit"]]], F), "inexact_time": (false(k), B), "datetime_analysis": (su["time"], TS),
        "term_id": (np.char.add("T", su["name"].astype(str)).astype(object), S), "term_name": (su["name"], S),
        "method": (null(k), pa.large_string()), "value_str": (np.array([f"{x:g}" for x in su["value"]], dtype=object), S),
        "value_num": (su["value"], F), "lt_flag": (false(k), B), "gt_flag": (false(k), B),
        "star_flag": (false(k), B), "approx_flag": (false(k), B), "other_flag": (null(k), S),
        "ref_val1": (null(k), S), "f1_str": (np.char.zfill(rng.integers(0, 10**9, k).astype(str), 10).astype(object), S),
        "date_lab": (su["time"], DT)})

    k = len(vgr["visit"])
    write(out / "ehr_sa_vgr_lab.parquet", {
        "from": (np.full(k, "vgr_sa_lab", object), S), "lopnr_swed": (lopnr[pid[vgr["visit"]]], F),
        "datetime_analysis": (vgr["time"], TS), "term_name": (vgr["name"], S), "unit": (vgr["unit"], S),
        "value_num": (vgr["value"], F), "lt_flag": (false(k), B), "gt_flag": (false(k), B),
        "other_flag": (null(k), S)})

    # --- sos_patreg_diag_long: register episodes between birth and death, 1-4 diagnoses each ---
    epid = np.repeat(np.arange(n), rng.geometric(1 / EPISODES_PER_PATIENT, n))
    indat = dates(rng, len(epid), EPISODE_Q)
    ok = (indat >= birth[epid]) & (indat <= alive_until[epid])
    epid, indat = epid[ok], indat[ok]
    e = len(epid)
    inpat = rng.random(e) < 0.15
    utdat = np.where(inpat, indat + rng.geometric(0.25, e) * DAY, np.datetime64("NaT", "D"))
    ktyp = numbers(rng, [0, 1, 2, 3, 8, 9], [80, 4, 4, 4, 4, 4], e, null=0.62)
    insatt = np.where(inpat, choose(rng, ["1", "2", "3"], e, [14, 1, 85]), None)
    utsatt = np.where(inpat, choose(rng, ["1", "2", "3", "4"], e, [13, 4, 81, 2]), None)
    sjukhus = np.array([f"{i:05d}" for i in range(1, 41)], dtype=object)[rng.integers(0, 40, e)]
    extra = rng.poisson(0.6, e)
    rows = [(np.flatnonzero(rng.random(e) < 0.85), "Hdia"), (np.arange(e), "DIAGNOS1")] \
        + [(np.flatnonzero(extra >= j), f"DIAGNOS{j + 1}") for j in range(1, extra.max() + 1)]
    ei = np.concatenate([r for r, _ in rows])
    pos = np.concatenate([np.full(len(r), p, object) for r, p in rows])
    order = np.argsort(ei, kind="stable")
    ei, pos = ei[order], pos[order]
    k = len(ei)
    write(out / "sos_patreg_diag_long.parquet", {
        "from": (np.full(k, "patreg", object), S), "source_row": (ei, I32), "lopnr_swed": (lopnr[epid[ei]], F),
        "INDATUM": (indat[ei], DT), "UTDATUM": (utdat[ei], DT),
        "care_level": (np.where(inpat[ei], "inpatient", "spec_outpatient").astype(object), S),
        "SJUKHUS": (sjukhus[ei], S), "KTYP": (ktyp[ei], I32), "INSATT": (insatt[ei], S), "UTSATT": (utsatt[ei], S),
        "diag_pos": (pos, DICT), "icd10_code": (choose(rng, DIAGNOSES, k, null=0.061), S)})

    # --- sos_deathreg_wide and sos_deathreg_diag_long ---
    di = np.flatnonzero(dead)
    d = len(di)
    causes = {"ULORSAK": choose(rng, DIAGNOSES, d, null=0.102)}
    u = rng.random(d)  # the same u for every position, so MORSAK3 implies MORSAK2 etc.
    for j in range(1, 49):
        c = choose(rng, DIAGNOSES, d)
        c[u >= (MORSAK_P[j - 1] if j <= len(MORSAK_P) else 0)] = None
        causes[f"MORSAK{j}"] = c
    base = {"from": (np.full(d, "deathreg", object), S), "source_row": (np.arange(d), I32), "lopnr_swed": (lopnr[di], F),
            "inexact_death_date": (rng.random(d) < 0.01, B), "date_death_meanimp": (death[di], DT),
            "date_death_minimp": (death[di], DT)}
    lk = choose(rng, MUNICIPALITIES, d, np.ones(len(MUNICIPALITIES)), null=0.0001)
    dodspl = numbers(rng, [1, 2, 3, 4], [40, 35, 20, 5], d, null=0.18)
    dbgrund = numbers(rng, [1, 2, 3, 4], [40, 10, 45, 5], d, null=0.90)
    write(out / "sos_deathreg_wide.parquet", {
        **base, **{c: (v, S) for c, v in causes.items()},
        "KAP19": (choose(rng, DIAGNOSES, d, null=0.95), S), "LK": (lk, S), "DODSPL": (dodspl, I32),
        "DBGRUND1": (dbgrund, I32), "is_female": (1 - male[di], I32)})

    long = [(np.flatnonzero(v != None), c) for c, v in causes.items()]  # noqa: E711 (object array)
    r = np.concatenate([i for i, _ in long])
    write(out / "sos_deathreg_diag_long.parquet", {
        **{c: (v[r], t) for c, (v, t) in base.items()},
        "LK": (lk[r], S), "DODSPL": (dodspl[r], I32), "DBGRUND1": (dbgrund[r], I32),
        "diag_pos": (np.concatenate([np.full(len(i), c, object) for i, c in long]), DICT),
        "icd10_code": (np.concatenate([causes[c][i] for i, c in long]), S)})

    # --- tells 02_build.py which made-up term is which vital ---
    (out / "out").mkdir(exist_ok=True)
    with (out / "out" / "vitals_snomed_map.csv").open("w", newline="") as f:
        csv.writer(f).writerows([("term_name", "concept")] + [(v[0], v[-1]) for v in VITALS])
    print(f"done: {n:,} patients, {m:,} ED visits, {d:,} deaths")


if __name__ == "__main__":
    main(5_000, HERE / "synth_subset_5k")
