#!/usr/bin/env python3
"""Patient-level subset of every SWED table in data/raw/swed_data/.

Samples N patients from swed_ed_core and copies every row they have in each
raw table to "<label>_SUB_<table>.parquet". The tables are found
automatically: every copy_<table>.parquet in the raw folder that has a
lopnr_swed column. Large tables are streamed in batches, so memory stays low
even for the 146M-row patient register.

Tables with a term_name column get one extra column, vitals_snomed: the SNOMED
ID from swed_lab_meas_to_snomed.csv when the row is one of the 6 vitals, null
otherwise (labs).

The label is the exact patient count: 500 -> "500", 100000 -> "100K". Each
size keeps its own sample in "<label>_sample_lopnr.csv", reused on rerun, so
runs of different sizes can share one folder. Finished tables are skipped on
rerun (pass --overwrite to redo them).

    python3 subset.py --n 100000                  # all raw tables
    python3 subset.py --n 500 --tables swed_ed_core sos_deathreg_wide
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import DATA, INTERIM, RAW_PREFIX, REF, VITAL_SNOMED, WORK, guard_output, log, raw_table


def find_tables():
    """Every copy_<table>.parquet in the raw folder, as table names: ['ehr_sos_lab', ...]."""
    files = sorted(DATA.glob(f"{RAW_PREFIX}*.parquet"))
    return [f.name[len(RAW_PREFIX):-len(".parquet")] for f in files]


def size_label(n):
    """100000 -> '100K', 500 -> '500', 1500 -> '1500'. Exact, so two sizes never share a label."""
    return f"{n // 1000}K" if n % 1000 == 0 else str(n)


def sample_patients(n, seed):
    ids = pc.unique(pq.read_table(raw_table("swed_ed_core"), columns=["lopnr_swed"])["lopnr_swed"].drop_null())
    ids = ids.to_numpy().astype(float)
    exclude_csv = WORK / "out" / "exclude_lopnr.csv"  # test patients (Tolvan etc.)
    if exclude_csv.exists():
        ids = ids[~np.isin(ids, pd.read_csv(exclude_csv)["lopnr_swed"].astype(float))]
    else:
        log(f"WARNING: {exclude_csv} not found; test patients are NOT excluded")
    if n > len(ids):
        sys.exit(f"asked for {n} patients but only {len(ids)} exist")
    return np.sort(np.random.default_rng(seed).choice(ids, size=n, replace=False))


def vital_term_map():
    """term_name -> SNOMED ID, for the term_names that are one of the 6 vitals."""
    snomed = pd.read_csv(REF / "swed_lab_meas_to_snomed.csv", dtype=str)
    snomed = snomed[snomed["snomed_id"].isin(VITAL_SNOMED)].drop_duplicates("term_name")
    log(f"vitals SNOMED map: {len(snomed)} term_names -> {snomed['snomed_id'].nunique()} vitals")
    return pa.array(snomed["term_name"].tolist()), pa.array(snomed["snomed_id"].tolist())


def subset_table(name, ids, dst, terms, snomed_ids):
    pf = pq.ParquetFile(raw_table(name), buffer_size=64 * 1024 * 1024)
    schema = pf.schema_arrow
    if "lopnr_swed" not in schema.names:
        log(f"skip {name}: no lopnr_swed column, so it can't be matched to patients")
        return
    add_snomed = "term_name" in schema.names
    if add_snomed:
        schema = schema.append(pa.field("vitals_snomed", pa.string()))

    tmp = dst.with_suffix(".tmp")
    kept = n_vitals = 0
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for batch in pf.iter_batches(batch_size=2_000_000):
            # cast so tables storing lopnr_swed as integers match the float sample IDs too
            batch = batch.filter(pc.is_in(batch.column("lopnr_swed").cast(pa.float64()), value_set=ids))
            if add_snomed:
                # position of each term_name in the vitals map (null if not a vital) -> its SNOMED ID
                pos = pc.index_in(batch.column("term_name").cast(pa.string()), value_set=terms)
                vitals_snomed = snomed_ids.take(pos)
                n_vitals += len(vitals_snomed) - vitals_snomed.null_count
                batch = pa.RecordBatch.from_arrays(batch.columns + [vitals_snomed], schema=schema)
            writer.write_batch(batch)
            kept += batch.num_rows
    tmp.rename(dst)  # only a finished table gets its final name, so reruns can skip it
    log(f"{name}: kept {kept:,} of {pf.metadata.num_rows:,} rows"
        + (f", {n_vitals:,} of them vitals (vitals_snomed set)" if add_snomed else ""))


def main():
    available = find_tables()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--out", type=Path, default=INTERIM)
    ap.add_argument("--tables", nargs="+", default=available, help="default: every table in the raw folder")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    guard_output(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    if "swed_ed_core" not in available:
        sys.exit(f"{raw_table('swed_ed_core')} not found: patients are sampled from it")
    unknown = sorted(set(args.tables) - set(available))
    if unknown:
        sys.exit(f"not in {DATA}: {unknown}. Available: {available}")
    log(f"{len(args.tables)} table(s): {', '.join(args.tables)}")

    label = size_label(args.n)
    ids_csv = args.out / f"{label}_sample_lopnr.csv"
    if ids_csv.exists():
        picked = pd.read_csv(ids_csv)["lopnr_swed"].to_numpy(dtype=float)
        log(f"reusing {len(picked):,} patients from {ids_csv}")
    else:
        picked = sample_patients(args.n, args.seed)
        pd.DataFrame({"lopnr_swed": picked}).to_csv(ids_csv, index=False)
        log(f"sampled {len(picked):,} patients (seed {args.seed}) -> {ids_csv.name}")

    ids = pa.array(picked, type=pa.float64())
    terms, snomed_ids = vital_term_map()
    for name in args.tables:
        dst = args.out / f"{label}_SUB_{name}.parquet"
        if dst.exists() and not args.overwrite:
            log(f"skip {name}: {dst.name} already exists")
        else:
            subset_table(name, ids, dst, terms, snomed_ids)


if __name__ == "__main__":
    main()
