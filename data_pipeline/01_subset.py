#!/usr/bin/env python3
"""Patient-level subset of the SWED Parquet files, for fast iteration on Bianca.

Samples N patients (lopnr_swed) from swed_ed_core and copies every row those
patients have in each table below. Sampling by patient keeps joins intact:
a sampled patient brings their ED visits, labs, diagnoses and death record.

Outputs go to $SWED_OUT/subset (default /home/theon/Desktop/data_std/subset) and
keep the source file names and schemas, so later scripts and the notebook run on
the subset just by pointing SWED_DATA at it:

    SWED_DATA=/home/theon/Desktop/data_std/subset python 02_build.py

The source files are only read, never changed; the script refuses to write into
the source folder. The output is real patient data. It stays on Bianca.

Big files are streamed in batches, so memory stays at a few GB even for
the 146M-row patient register. Each table is written to a .tmp file and
renamed when done; finished tables are skipped on rerun (use --overwrite).

    python 01_subset.py                          # 5000 patients, all tables
    python 01_subset.py --n 500 --tables swed_ed_core sos_deathreg_wide
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATA = Path(os.environ.get("SWED_DATA", "/cygnus/proj/nobackup/swed_db/clean_parquet"))
WORK = Path(os.environ.get("SWED_WORK", Path.cwd()))  # where out/exclude_lopnr.csv is read from
ORIGINAL = Path("/cygnus/proj/nobackup/swed_db/clean_parquet")  # never written to
OUT_ROOT = Path(os.environ.get("SWED_OUT", "/home/theon/Desktop/data_std"))  # everything that stays on Bianca
EXCLUDE_CSV = WORK / "out" / "exclude_lopnr.csv"  # written by swed_tasks.ipynb (Tolvan etc.)

SAMPLE_FROM = "swed_ed_core"
TABLES = [
    "swed_ed_core",
    "swed_ed_core_birthdate",
    "ehr_karolinska_meas",      # vital parameters
    "ehr_karolinska_lab",
    "ehr_danderyd_lab_meas",    # labs and vital parameters
    "ehr_sahlgrenska_lab",
    "ehr_sa_vgr_lab",
    "sos_patreg_diag_long",
    "sos_deathreg_wide",
    "sos_deathreg_diag_long",
]

BATCH_ROWS = 2_000_000
READ_BUFFER = 64 * 1024 * 1024  # buffered reads: don't load a whole row group's bytes at once


def refuse_to_write_into(out: Path, *protected: Path) -> None:
    """Outputs must never land in, or overwrite anything in, the source data folders."""
    o = out.expanduser().resolve()
    for p in protected:
        p = p.expanduser().resolve()
        if o == p or p in o.parents:
            sys.exit(f"refusing to write to {out}: it is inside the source data folder {p}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_exclude() -> set[float]:
    if not EXCLUDE_CSV.exists():
        log(f"WARNING: {EXCLUDE_CSV} not found; test patients (Tolvan) are NOT excluded from the sample")
        return set()
    with EXCLUDE_CSV.open() as f:
        ids = {float(r["lopnr_swed"]) for r in csv.DictReader(f)}
    log(f"excluding {len(ids)} lopnr from {EXCLUDE_CSV}")
    return ids


def sample_patients(n: int, seed: int, exclude: set[float]) -> np.ndarray:
    col = pq.read_table(DATA / f"{SAMPLE_FROM}.parquet", columns=["lopnr_swed"])["lopnr_swed"]
    ids = pc.unique(col.drop_null()).to_numpy()
    ids = ids[~np.isin(ids, list(exclude))] if exclude else ids
    if n > len(ids):
        sys.exit(f"asked for {n} patients but {SAMPLE_FROM} only has {len(ids)}")
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(ids, size=n, replace=False))
    log(f"sampled {n:,} of {len(ids):,} patients in {SAMPLE_FROM} (seed {seed})")
    return picked


def subset_table(name: str, ids: pa.Array, out_dir: Path, overwrite: bool) -> dict:
    src = DATA / f"{name}.parquet"
    dst = out_dir / f"{name}.parquet"
    if not src.exists():
        log(f"skip {name}: {src} not found")
        return {"table": name, "status": "missing"}
    if dst.exists() and not overwrite:
        log(f"skip {name}: already done ({pq.read_metadata(dst).num_rows:,} rows)")
        return {"table": name, "status": "exists", "rows_out": pq.read_metadata(dst).num_rows}

    t0 = time.time()
    pf = pq.ParquetFile(src, buffer_size=READ_BUFFER, pre_buffer=False)
    rows_in = pf.metadata.num_rows
    log(f"{name}: scanning {rows_in:,} rows in {pf.metadata.num_row_groups} row group(s)")

    tmp = dst.with_suffix(".parquet.tmp")
    rows_out = seen = 0
    next_report = 0.1
    with pq.ParquetWriter(tmp, pf.schema_arrow, compression="zstd") as writer:
        for batch in pf.iter_batches(batch_size=BATCH_ROWS):
            seen += batch.num_rows
            keep = batch.filter(pc.is_in(batch.column("lopnr_swed"), value_set=ids))
            if keep.num_rows:
                writer.write_batch(keep)
                rows_out += keep.num_rows
            if rows_in and seen / rows_in >= next_report:
                log(f"  {name}: {seen / rows_in:4.0%} read, {rows_out:,} rows kept")
                next_report += 0.1
    tmp.rename(dst)

    secs = time.time() - t0
    log(f"{name}: kept {rows_out:,} of {rows_in:,} rows in {secs:.0f}s")
    return {"table": name, "status": "done", "rows_in": rows_in, "rows_out": rows_out, "seconds": round(secs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000, help="number of patients (default 5000)")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--out", type=Path, default=OUT_ROOT / "subset", help="output folder (default $SWED_OUT/subset)")
    ap.add_argument("--tables", nargs="+", default=TABLES, help="tables to subset (default: all)")
    ap.add_argument("--overwrite", action="store_true", help="redo tables that are already written")
    args = ap.parse_args()

    log(f"data: {DATA}")
    log(f"out:  {args.out}")
    refuse_to_write_into(args.out, DATA, ORIGINAL)
    args.out.mkdir(parents=True, exist_ok=True)

    # Reuse the existing sample on rerun, so all tables share one patient set.
    ids_csv = args.out / "sample_lopnr.csv"
    if ids_csv.exists() and not args.overwrite:
        with ids_csv.open() as f:
            picked = np.array([float(r["lopnr_swed"]) for r in csv.DictReader(f)])
        log(f"reusing {len(picked):,} patients from {ids_csv}")
        if len(picked) != args.n:
            log(f"NOTE: --n {args.n} ignored; delete {ids_csv} or pass --overwrite to resample")
    else:
        picked = sample_patients(args.n, args.seed, read_exclude())
        with ids_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lopnr_swed"])
            w.writerows([x] for x in picked)

    ids = pa.array(picked, type=pa.float64())
    results = [subset_table(t, ids, args.out, args.overwrite) for t in args.tables]

    manifest = args.out / "manifest.csv"
    fields = ["table", "status", "rows_in", "rows_out", "seconds"]
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    log(f"done; summary in {manifest}")


if __name__ == "__main__":
    main()
