#!/usr/bin/env python3
"""Generate a synthetic SWED dataset with a chosen number of patients and seed, as one CSV file.

You choose the size and the seed; the script runs the whole pipeline and writes the result to
Data/synth/ as <size>_<seed>_<date>.csv, e.g. Data/synth/15K_42_2026-11-24.csv.

    python src/core/data_pipeline/synthetic/generate_data.py                       # asks for size and seed
    python src/core/data_pipeline/synthetic/generate_data.py --n-patients 15000 --seed 42
    python src/core/data_pipeline/synthetic/generate_data.py -n 15000 -s 42 --keep-raw
    python src/core/data_pipeline/synthetic/generate_data.py --help

What runs, in order:
    1. synth_subset_5k.py  makes the 10 raw tables (fake patients, visits, vitals, labs, deaths, ...)
    2. 02_build.py         cleans them (units, plausible values, links vitals to the ED visit)
    3. this script         joins them into one table: one row per ED visit

The CSV has one row per ED visit:
    ed_visit_id, patient_id, hospital, arrival, age, male, complaint, complaints,
    one column per vital sign (first plausible value in the visit), n_vitals (of temperature,
    systolic blood pressure, pulse), days_to_death, died_30d

From Python (with PYTHONPATH=src):
    from core.data_pipeline.synthetic.generate_data import generate
    path = generate(n_patients=15000, seed=42)

The same size and seed always give exactly the same data. The data is made up: use it to test
code, never as results.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent              # src/core/data_pipeline/synthetic
REPO_ROOT = HERE.parents[3]
BUILD_SCRIPT = HERE.parent / "02_build.py"
DEFAULT_OUT = REPO_ROOT / "Data" / "synth"
DEFAULT_REF = REPO_ROOT / "Data" / "data_info"
LAB_MAPS = ["swed_lab_meas_to_snomed.csv", "lab_unit_normalization_07sep2026.csv"]
MAIN_VITALS = ["body_temp", "systolic_bp", "pulse_rate"]  # the 3 vitals the models use


def size_label(n: int) -> str:
    """15000 -> '15K', 2500 -> '2.5K', 800 -> '800'."""
    return f"{n / 1000:g}K" if n >= 1000 else str(n)


def file_name(n_patients: int, seed: int, day: date | None = None) -> str:
    """The output name, e.g. '15K_42_2026-11-24.csv'."""
    return f"{size_label(n_patients)}_{seed}_{(day or date.today()).isoformat()}.csv"


def _run_generator(n_patients: int, seed: int, raw: Path) -> None:
    """Step 1: the 10 raw tables, from main() in synth_subset_5k.py."""
    spec = importlib.util.spec_from_file_location("synth_subset_5k", HERE / "synth_subset_5k.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(n_patients, raw, seed)


def _run_build(raw: Path, work: Path, ref: Path) -> Path:
    """Step 2: 02_build.py; returns the folder with the cleaned tables."""
    if not all((ref / f).exists() for f in LAB_MAPS):
        # 02_build.py also cleans labs and needs two mapping files that are kept out of git.
        # The CSV has no lab columns, so placeholders are enough.
        ref = work / "placeholder_data_info"
        ref.mkdir(exist_ok=True)
        (ref / LAB_MAPS[0]).write_text("term_name,snomed_id,snomed_swe,missing_compartment,not_perfect,problematic\n__none__,0,none,,,\n")
        (ref / LAB_MAPS[1]).write_text("term_name,unit,canonical_unit,action,factor\n__none__,x,x,keep,1.0\n")
    env = {**os.environ,
           "SWED_DATA": str(raw),
           "SWED_WORK": str(raw),          # holds out/vitals_snomed_map.csv; without it there are no vitals
           "SWED_OUT": str(work),
           "SWED_REF": str(ref)}
    run = subprocess.run([sys.executable, str(BUILD_SCRIPT)], env=env, capture_output=True, text=True)
    if run.returncode:
        print(run.stderr[-3000:], file=sys.stderr)
        raise RuntimeError("02_build.py failed (see the error above)")
    return work / f"built_{raw.name}"


def _one_row_per_visit(built: Path) -> pd.DataFrame:
    """Step 3: join the cleaned tables into one row per ED visit."""
    ed = pd.read_parquet(built / "ed_visits.parquet",
                         columns=["ed_visit_id", "lopnr_swed", "ed_hospital", "ed_arrival_datetime",
                                  "ed_age", "is_male", "ed_presented_complaint_std"])
    codes = ed["ed_presented_complaint_std"].fillna("").astype(str).str.findall(r"\[(\d+)\]")
    df = pd.DataFrame({
        "ed_visit_id": ed["ed_visit_id"].astype("int64"),
        "patient_id": ed["lopnr_swed"].astype("int64"),
        "hospital": ed["ed_hospital"],
        "arrival": ed["ed_arrival_datetime"],
        "age": ed["ed_age"],
        "male": ed["is_male"],
        "complaint": codes.str[0],          # first SNOMED code of the visit
        "complaints": codes.str.join(";"),  # all codes
    })

    # first plausible value of each vital sign in the visit
    v = pd.read_parquet(built / "vitals.parquet", columns=["ed_visit_id", "datetime_analysis", "concept", "value", "implausible"])
    v["concept"] = v["concept"].replace({"heart_rate": "pulse_rate"})
    v = v[v["ed_visit_id"].notna() & v["value"].notna() & ~v["implausible"].astype(bool)]
    v["ed_visit_id"] = v["ed_visit_id"].astype("int64")
    first = v.sort_values("datetime_analysis").groupby(["ed_visit_id", "concept"])["value"].first().unstack()
    df = df.join(first, on="ed_visit_id")
    for c in MAIN_VITALS:
        if c not in df:
            df[c] = float("nan")
    df["n_vitals"] = df[MAIN_VITALS].notna().sum(axis=1)

    # death within 30 days of arrival
    deaths = pd.read_parquet(built / "deaths.parquet", columns=["lopnr_swed", "date_death_minimp"])
    death = pd.to_datetime(df["patient_id"].map(deaths.set_index("lopnr_swed")["date_death_minimp"]))
    arrival_day = df["arrival"].dt.tz_convert("Europe/Stockholm").dt.tz_localize(None).dt.normalize()
    df["days_to_death"] = (death - arrival_day).dt.days
    df["died_30d"] = df["days_to_death"].between(0, 30).astype(int)
    return df.sort_values(["patient_id", "arrival"]).reset_index(drop=True)


def generate(n_patients: int, seed: int, out: Path | str | None = None, keep_raw: bool = False,
             ref: Path | str | None = None, overwrite: bool = False, day: date | None = None) -> Path:
    """Run the pipeline and return the path of the CSV, e.g. Data/synth/15K_42_2026-11-24.csv.

    n_patients  number of patients (each has one or more ED visits)
    seed        random seed: the same n_patients and seed always give the same data
    out         folder for the CSV (default Data/synth)
    keep_raw    also save the 10 raw tables as CSV in a folder with the same name
    ref         folder with the lab mapping files for 02_build.py (default Data/data_info)
    overwrite   replace the CSV if it already exists
    """
    if n_patients < 1:
        raise ValueError("n_patients must be at least 1")
    out = Path(out or DEFAULT_OUT)
    path = out / file_name(n_patients, seed, day)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (use --overwrite to replace it)")
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        raw = work / "raw"
        raw.mkdir()
        print(f"1/3 generating {n_patients:,} patients with seed {seed}")
        _run_generator(n_patients, seed, raw)
        print("2/3 cleaning with 02_build.py")
        built = _run_build(raw, work, Path(ref or DEFAULT_REF))
        print("3/3 joining into one row per ED visit")
        df = _one_row_per_visit(built)
        if df[MAIN_VITALS].notna().sum().sum() == 0:
            raise RuntimeError("no vital signs in the result: check out/vitals_snomed_map.csv in the raw data")
        df.to_csv(path, index=False)
        if keep_raw:
            raw_dir = path.with_suffix("")
            raw_dir.mkdir(exist_ok=True)
            for f in sorted(raw.glob("*.parquet")):
                pd.read_parquet(f).to_csv(raw_dir / f"{f.stem}.csv", index=False)

    print(f"done: {path}  ({len(df):,} ED visits, {df['patient_id'].nunique():,} patients, "
          f"{df['died_30d'].sum():,} deaths within 30 days, {path.stat().st_size / 1e6:.1f} MB)")
    return path


def ask_int(question: str, minimum: int) -> int:
    """Ask in the terminal until the answer is a whole number >= minimum."""
    while True:
        try:
            answer = input(question).strip().replace(",", "").replace("_", "")
        except EOFError:
            raise ValueError("no answer given") from None
        if answer.lstrip("-").isdigit() and int(answer) >= minimum:
            return int(answer)
        print(f"  please type a whole number of at least {minimum}, e.g. 15000")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Generate synthetic SWED data (made-up patients) as one CSV: Data/synth/<size>_<seed>_<date>.csv. "
                    "Run it without arguments and it asks for the number of patients and the seed.",
        epilog="examples:  python src/core/data_pipeline/synthetic/generate_data.py            (asks you)\n"
               "           python src/core/data_pipeline/synthetic/generate_data.py -n 15000 -s 42",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--n-patients", type=int, help="number of patients, e.g. 15000 (asked if left out)")
    ap.add_argument("-s", "--seed", type=int, help="random seed, e.g. 42; same seed = same data (asked if left out)")
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT, help="output folder (default Data/synth)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="also save the 10 raw tables as CSV in a folder named like the file")
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF,
                    help="folder with the lab mapping files for 02_build.py (default Data/data_info)")
    ap.add_argument("--overwrite", action="store_true", help="replace the CSV if it already exists")
    a = ap.parse_args(argv)
    try:
        if a.n_patients is None:
            a.n_patients = ask_int("Number of patients: ", 1)
        if a.seed is None:
            a.seed = ask_int("Seed: ", 0)
        generate(a.n_patients, a.seed, a.out, a.keep_raw, a.ref, a.overwrite)
    except (ValueError, FileExistsError, RuntimeError) as e:
        ap.exit(1, f"error: {e}\n")


if __name__ == "__main__":
    main()
