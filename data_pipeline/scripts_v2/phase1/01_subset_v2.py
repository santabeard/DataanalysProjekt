#!/usr/bin/env python3
"""Patient-level subset of the SWED Parquet files, for fast iteration on Bianca.

Samples N patients (lopnr_swed) from swed_ed_core and copies every row those
patients have in each table below. Sampling by patient keeps joins intact:
a sampled patient brings their ED visits, vitals, diagnoses and death record.

Table list is scoped to vitals-augmented ED Atlas only:
  - swed_ed_core: the anchor (visit logistics, age, sex, complaint). Not
    swed_ed_core_birthdate -- ed_age (already in ed_core) covers age; nothing
    in the plan calls for the exact birth date.
  - ehr_karolinska_meas / ehr_danderyd_lab_meas: the two vitals sources (Danderyd's
    table mixes in labs too; isolating vitals by SNOMED happens downstream, not here).
  - sos_patreg_wide / sos_deathreg_wide: diagnosis (Hdia) and mortality. The
    "_diag_long" / "_wide" pair for both patreg and deathreg are confirmed
    duplicates (diag_pos values match the wide table's column names exactly, i.e.
    "Hdia", "DIAGNOS1", ... / "ULORSAK", "MORSAK1", ...) so only "_wide" is pulled.
  - All other EHR lab tables (karolinska_lab, sahlgrenska_lab, sa_vgr_lab, sos_lab,
    stgoran_lab) are NOT needed: ed_atlas and its vitals augmentation use no lab
    values at all.

Outputs go to $SWED_OUT/subset (default /home/theon/Desktop/data_std/subset).
Each table keeps its schema but NOT its original filename: a subset must never
be mistaken for the real table it was sampled from, so every file is written
as "<N>K_SUB_<original name>.parquet", where <N>K is the patient count in
thousands (e.g. 500,000 patients -> "500K_SUB_swed_ed_core.parquet"). A plain
copy of the swed_ed_core subset is also written as "<N>K_train_raw.parquet":
this is the file every later stage (vitals join, diagnosis join, death join)
reads from and writes its new columns into, so the "<N>K_SUB_*" files stay
exactly what they were the day they were sampled, for comparison/debugging.

The source files are only read, never changed; the script refuses to write into
the source folder. The output is real patient data. It stays on Bianca.

Big files are streamed in batches, so memory stays at a few GB even for
the 146M-row patient register. Each table is written to a .tmp file and
renamed when done; finished tables are skipped on rerun (use --overwrite).

    python 01_subset_v2.py                          # 500,000 patients, all tables
    python 01_subset_v2.py --n 500 --tables swed_ed_core sos_deathreg_wide
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
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
    "ehr_karolinska_meas",      # vitals (Karolinska)
    "ehr_danderyd_lab_meas",    # vitals + labs mixed (Danderyd/SÖS/Södertälje)
    "sos_patreg_wide",          # diagnosis (Hdia); _diag_long is a confirmed duplicate
    "sos_deathreg_wide",        # mortality; _diag_long is a confirmed duplicate
]

BATCH_ROWS = 2_000_000
READ_BUFFER = 64 * 1024 * 1024  # buffered reads: don't load a whole row group's bytes at once


def size_label(n: int) -> str:
    """500_000 -> "500K", 50_000 -> "50K". Patient count in thousands, rounded
    (sizes are always chosen in round thousands in practice, but round rather
    than truncate so an odd n doesn't silently collide with a different size)."""
    return f"{round(n / 1000)}K"


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


def subset_table(name: str, ids: pa.Array, out_dir: Path, overwrite: bool, label: str) -> dict:
    src = DATA / f"{name}.parquet"
    dst = out_dir / f"{label}_SUB_{name}.parquet"
    if not src.exists():
        log(f"skip {name}: {src} not found")
        return {"table": name, "status": "missing"}
    if dst.exists() and not overwrite:
        log(f"skip {name}: already done ({pq.read_metadata(dst).num_rows:,} rows)")
        return {"table": name, "status": "exists", "rows_out": pq.read_metadata(dst).num_rows, "path": str(dst)}

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
    return {"table": name, "status": "done", "rows_in": rows_in, "rows_out": rows_out,
            "seconds": round(secs), "path": str(dst)}


def write_train_raw(sub_path: Path, out_dir: Path, label: str, overwrite: bool) -> None:
    """<N>K_train_raw.parquet: initially a plain copy of the swed_ed_core
    subset. This is the file later stages (vitals join, diagnosis join, death
    join) read from and add their own columns to -- the "<N>K_SUB_*" files are
    never touched again after this script writes them."""
    dst = out_dir / f"{label}_train_raw.parquet"
    if dst.exists() and not overwrite:
        log(f"skip train_raw: already exists ({pq.read_metadata(dst).num_rows:,} rows)")
        return
    tmp = dst.with_suffix(".parquet.tmp")
    shutil.copyfile(sub_path, tmp)
    tmp.rename(dst)
    log(f"wrote {dst} (copy of {sub_path.name})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500_000, help="number of patients (default 500,000)")
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

    label = size_label(len(picked))
    log(f"size label: {label} ({len(picked):,} patients)")

    ids = pa.array(picked, type=pa.float64())
    results = [subset_table(t, ids, args.out, args.overwrite, label) for t in args.tables]

    manifest = args.out / "manifest.csv"
    fields = ["table", "status", "rows_in", "rows_out", "seconds", "path"]
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    log(f"done; summary in {manifest}")

    ed_core_result = next((r for r in results if r["table"] == SAMPLE_FROM and "path" in r), None)
    if ed_core_result is None:
        log(f"NOTE: no train_raw written -- {SAMPLE_FROM} was not in --tables or its subset is missing")
    else:
        write_train_raw(Path(ed_core_result["path"]), args.out, label, args.overwrite)


if __name__ == "__main__":
    main()