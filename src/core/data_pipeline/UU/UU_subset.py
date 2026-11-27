#!/usr/bin/env python3
"""Patient-level subset of the UU tables copy_edvisit and copy_measurements.

Samples N patients (PersonId) from copy_edvisit and writes to data/interim:

  <label>_uu_sample_personid.csv       the sampled PersonIds, reused on rerun
  <label>_UU_SUB_edvisit.parquet        their ED visits: PersonId, ContactId, visit time and
                                        duration, age, sex, complaint, date of death, plus
                                        complaint_snomed (see below)
  <label>_UU_SUB_measurements.parquet   their vitals, one reading per row, with vitals_snomed
  <label>_UU_SUB_healthcarevisits.parquet  their care contacts (all types): PersonId,
                                        HospitalEpisodeId, AdmissionDate, MainDiagnosis, Type.
                                        The build keeps OV/SV and maps the codes to SWED's level.

complaint_snomed: ED_PresentedComplaint is plain Swedish text. It is mapped through
docs/data_info/complaints_combined.csv (exact text match) into SWED's format,
"text [sctid]" or "a [x]+b [y]" when the mapping gives two codes, so the SWED
complaint handling works unchanged. Complaints with no SNOMED code stay null and are
counted in the log.

vitals_snomed: only the names in UU_MEASUREMENT_SNOMED (common.py) are kept, each
turned into one reading per (result column, SNOMED ID) pair. Blodtryck gives two
readings (systolic, diastolic). Same SNOMED IDs as SWED.

The label is the exact patient count (500 -> "500", 100000 -> "100K"). Finished
files are skipped on rerun (pass --overwrite to redo them).

    python3 UU_subset.py --n 100000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# --- make shared pipeline modules importable from any working directory ---
import sys as _sys
from pathlib import Path as _Path
_PR = _Path(__file__).resolve().parents[1]  # src/core/data_pipeline/
for _d in (_PR, _PR / "SWED"):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))
# --------------------------------------------------------------------------

from common import INTERIM, REF, UU_MEASUREMENT_SNOMED, guard_output, log, uu_raw_table
from SWED_subset import size_label

EDVISIT_COLUMNS = ["PersonId", "ContactId", "ED_VisitDateTime", "ED_VisitDuration", "AgeAtED_Visit",
                   "Sex", "ED_PresentedComplaint", "DateOfDeath"]


def sample_patients(n, seed):
    ids = pc.unique(pq.read_table(uu_raw_table("edvisit"), columns=["PersonId"])["PersonId"].drop_null()).to_numpy()
    if n > len(ids):
        sys.exit(f"asked for {n} patients but only {len(ids)} exist")
    return np.sort(np.random.default_rng(seed).choice(ids, size=n, replace=False))


def complaint_map():
    """Complaint text -> "text [sctid]" (or "a [x]+b [y]"), from complaints_combined.csv."""
    m = pd.read_csv(REF / "complaints_combined.csv", dtype=str)
    out = {}
    for _, row in m.iterrows():
        parts = []
        for i in (1, 2):
            sctid, text = row[f"SNOMED_ID{i}_merged"], row[f"SNOMED_ID{i}_text_merged"]
            if pd.notna(sctid) and sctid.strip():
                sctid = sctid.strip()
                sctid = sctid[:-2] if sctid.endswith(".0") else sctid
                parts.append(f"{text.strip() if pd.notna(text) else ''} [{sctid}]".strip())
        if parts and pd.notna(row["Presented_complaint"]):
            out[row["Presented_complaint"].strip()] = "+".join(parts)
    log(f"complaint map: {len(out):,} complaint texts with a SNOMED code")
    return out


def as_strings(table):
    """Dictionary-encoded columns -> plain strings, so pandas gets str, not categorical."""
    for i, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type):
            table = table.set_column(i, field.name, pc.cast(table.column(i), pa.string()))
    return table


def subset_edvisit(ids, dst):
    table = pq.read_table(uu_raw_table("edvisit"), columns=EDVISIT_COLUMNS)
    table = as_strings(table.filter(pc.is_in(table.column("PersonId"), value_set=ids)))
    visits = table.to_pandas()

    raw = visits["ED_PresentedComplaint"].str.strip()
    visits["complaint_snomed"] = raw.map(complaint_map())
    unmapped = raw[raw.notna() & visits["complaint_snomed"].isna()]
    log(f"edvisit: {len(visits):,} visits; complaint mapped to SNOMED for "
        f"{visits['complaint_snomed'].notna().mean():.1%}, no complaint {raw.isna().mean():.1%}, "
        f"text with no SNOMED code {len(unmapped) / max(len(visits), 1):.1%}")
    for text, n in unmapped.value_counts().head(15).items():
        if n >= 10:  # complaint category names, not patient data
            log(f"  unmapped complaint {text!r}: {n:,} visits")

    tmp = dst.with_suffix(".tmp")
    visits.to_parquet(tmp, index=False)
    tmp.rename(dst)
    log(f"wrote {dst.name}")


def subset_measurements(ids, dst):
    pf = pq.ParquetFile(uu_raw_table("measurements"), buffer_size=64 * 1024 * 1024)
    result_cols = sorted({col for pairs in UU_MEASUREMENT_SNOMED.values() for col, _ in pairs})
    read_cols = ["PersonId", "NameOfMeasurement", "DateTimeOfMeasurement"] + result_cols
    schema = pa.schema([
        ("PersonId", pf.schema_arrow.field("PersonId").type),
        ("DateTimeOfMeasurement", pf.schema_arrow.field("DateTimeOfMeasurement").type),
        ("NameOfMeasurement", pa.string()),
        ("source_column", pa.string()),
        ("value_num", pa.float64()),
        ("vitals_snomed", pa.string()),
    ])

    tmp = dst.with_suffix(".tmp")
    counts = {}
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for batch in pf.iter_batches(batch_size=2_000_000, columns=read_cols):
            batch = batch.filter(pc.is_in(batch.column("PersonId"), value_set=ids))
            names = pc.cast(batch.column("NameOfMeasurement"), pa.string())
            for name, pairs in UU_MEASUREMENT_SNOMED.items():
                sub = batch.filter(pc.equal(names, pa.scalar(name, pa.string())))
                n = sub.num_rows
                if not n:
                    continue
                for col, sctid in pairs:  # one reading per (result column, SNOMED ID)
                    writer.write_batch(pa.RecordBatch.from_arrays([
                        sub.column("PersonId"),
                        sub.column("DateTimeOfMeasurement"),
                        pa.array([name] * n, pa.string()),
                        pa.array([col] * n, pa.string()),
                        pc.cast(sub.column(col), pa.float64()),
                        pa.array([sctid] * n, pa.string()),
                    ], schema=schema))
                    counts[(name, col)] = counts.get((name, col), 0) + n
    tmp.rename(dst)
    for (name, col), n in counts.items():
        log(f"  {name} / {col}: {n:,} readings")
    log(f"wrote {dst.name} ({sum(counts.values()):,} readings)")


def subset_healthcarevisits(ids, dst):
    """Care contacts of the sampled patients, streamed (19M rows in the full table)."""
    pf = pq.ParquetFile(uu_raw_table("healthcarevisits"), buffer_size=64 * 1024 * 1024)
    src = pf.schema_arrow
    columns = ["PersonId", "HospitalEpisodeId", "AdmissionDate", "MainDiagnosis", "Type"]
    # text columns as plain strings: dictionary columns can't be appended batch by batch reliably
    types = [pa.string() if (pa.types.is_dictionary(src.field(c).type) or pa.types.is_large_string(src.field(c).type))
             else src.field(c).type for c in columns]
    schema = pa.schema(list(zip(columns, types)))

    tmp = dst.with_suffix(".tmp")
    kept = 0
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for batch in pf.iter_batches(batch_size=2_000_000, columns=columns):
            batch = batch.filter(pc.is_in(batch.column("PersonId"), value_set=ids))
            arrays = [pc.cast(batch.column(c), t) for c, t in zip(columns, types)]
            writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
            kept += batch.num_rows
    tmp.rename(dst)
    log(f"wrote {dst.name} ({kept:,} of {pf.metadata.num_rows:,} care contacts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--out", type=Path, default=INTERIM)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    guard_output(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    label = size_label(args.n)
    ids_csv = args.out / f"{label}_uu_sample_personid.csv"
    if ids_csv.exists():
        picked = pd.read_csv(ids_csv)["PersonId"].to_numpy(dtype=np.int64)
        log(f"reusing {len(picked):,} patients from {ids_csv}")
    else:
        picked = sample_patients(args.n, args.seed)
        pd.DataFrame({"PersonId": picked}).to_csv(ids_csv, index=False)
        log(f"sampled {len(picked):,} patients (seed {args.seed}) -> {ids_csv.name}")
    ids = pa.array(picked, type=pa.int64())

    for name, fn in [("edvisit", subset_edvisit), ("measurements", subset_measurements),
                     ("healthcarevisits", subset_healthcarevisits)]:
        dst = args.out / f"{label}_UU_SUB_{name}.parquet"
        if dst.exists() and not args.overwrite:
            log(f"skip {name}: {dst.name} already exists")
        else:
            fn(ids, dst)


if __name__ == "__main__":
    main()
