"""Shared settings and helpers for the ingestion scripts.

    01_deaths.py    -> deaths.parquet     one row per dead patient
    02_patients.py  -> patients.parquet   one row per patient: age, is_male
    03_visits.py    -> visits.parquet     one row per ED visit: hospital, time, complaint, vitals, diagnosis

Edit the SETTINGS block before running. All scripts read their input in batches of
at most BATCH_SIZE rows and only import pandas, pyarrow and pathlib.
Source files are only read; check_settings() refuses to write into SOURCE_DIR.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# ----------------------------------------------------------------- SETTINGS ---
# Folder with the clean_parquet files (the 2026-06 profile ran on
# /proj/nobackup/sens2020598/swed_db/clean_parquet, which is the same folder).
SOURCE_DIR = Path("/cygnus/proj/nobackup/swed_db/clean_parquet")
FILE_PREFIX = ""  # "copy_" to read copy_swed_ed_core.parquet etc. instead
OUT_DIR = Path.home() / "Desktop" / "data_std" / "ingest"
BATCH_SIZE = 300_000  # rows per read batch; must not exceed 300K

EXCLUDE_LOPNR = {701403}  # test patient "Tolvan Tolvansson" (out/exclude_lopnr.csv)

# ------------------------------------------------------------------ TABLES ---
ED_CORE = "swed_ed_core"
DEATHREG = "sos_deathreg_wide"
PATREG = "sos_patreg_wide"
# Every lab/meas file is scanned for vitals. Per the profile only the first three
# hold vitals; the lab-only files are scanned anyway and simply yield no rows.
# Missing files are skipped with a message.
VITAL_SOURCES = [
    "ehr_karolinska_meas",
    "ehr_danderyd_lab_meas",
    "ehr_sa_vgr_lab",
    "ehr_karolinska_lab",
    "ehr_sahlgrenska_lab",
    "ehr_sos_lab",
    "ehr_stgoran_lab",
]

# ------------------------------------------------------------------ VITALS ---
# term_name -> vital. Spellings are every term in data_info/swed_lab_meas_to_snomed.csv
# with one of the six vital SNOMED IDs (as in v2/phase1/02_vitals_join.py), except
# "Saturation med syrgas": same SNOMED ID as Saturation but measured on oxygen,
# which a model should not mix with room-air saturation. Add it here if wanted.
VITAL_TERMS = {
    "Blodtryck systoliskt": "systolic_bp",
    "Blodtryck systoliskt - övre": "systolic_bp",
    "Blodtryck diastoliskt": "diastolic_bp",
    "Blodtryck diastoliskt - nedre": "diastolic_bp",
    "Puls": "pulse",
    "Pulsfrekvens": "pulse",
    "Andningsfrekvens": "resp_rate",
    "Andningsfrekvens (PNA)": "resp_rate",
    "Saturation": "spo2",
    "Kroppstemp": "temperature",
    "Kroppstemp (PNA)": "temperature",
    "Kroppstemperatur": "temperature",
    "Temperatur": "temperature",
}
VITALS = ["systolic_bp", "diastolic_bp", "pulse", "resp_rate", "spo2", "temperature"]

# Readings outside these limits are dropped (limits from archive_v1/02_build.py).
PLAUSIBLE = {
    "systolic_bp": (40, 300),
    "diastolic_bp": (20, 200),
    "pulse": (20, 250),
    "resp_rate": (3, 80),
    "spo2": (50, 100),
    "temperature": (30, 44),
}
# Unmapped term names matching this pattern are listed at the end of 03_visits.py,
# so spellings missing from VITAL_TERMS show up in the log.
VITAL_LIKE_PATTERN = "blodtryck|puls|satur|andning|temp|syre"

# --------------------------------------------------------------- WINDOWS ---
VITALS_BEFORE_MIN = 60  # vitals window opens 60 min before arrival (v2 decision)
VITALS_FALLBACK_MIN = 24 * 60  # window length when ed_time_total_minutes is missing (v1)
DIAG_WINDOW_DAYS = 30  # diagnosis: first Hdia with INDATUM in [arrival day, arrival day + 30]
LOCAL_TZ = "Europe/Stockholm"  # registry dates are Swedish civil dates (v2 decision)


# ----------------------------------------------------------------- HELPERS ---
def log(msg):
    print(f"[{pd.Timestamp.now():%H:%M:%S}] {msg}", flush=True)


def check_settings():
    """Fail early on a bad batch size or an output folder inside the source data."""
    if not 0 < BATCH_SIZE <= 300_000:
        raise SystemExit(f"BATCH_SIZE must be between 1 and 300,000, got {BATCH_SIZE}")
    src = SOURCE_DIR.expanduser().resolve()
    out = OUT_DIR.expanduser().resolve()
    if out == src or src in out.parents:
        raise SystemExit(f"refusing to write to {OUT_DIR}: it is inside the source folder {SOURCE_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def source_path(table):
    return SOURCE_DIR / f"{FILE_PREFIX}{table}.parquet"


def file_columns(path):
    return pq.ParquetFile(path).schema_arrow.names


def iter_batches(path, columns):
    """Yield the file as pandas DataFrames of at most BATCH_SIZE rows.
    Dates and timestamps come back as datetime64[ns] (newer pandas would otherwise
    keep the file's ms/us resolution, which merge_asof refuses to mix)."""
    if not path.exists():
        raise SystemExit(f"missing input file {path}")
    pf = pq.ParquetFile(path)
    missing = [c for c in columns if c not in pf.schema_arrow.names]
    if missing:
        raise SystemExit(f"{path.name} has no column(s) {missing}")
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        df = batch.to_pandas(date_as_object=False)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                tz = df[col].dt.tz
                df[col] = df[col].astype(f"datetime64[ns, {tz}]" if tz is not None else "datetime64[ns]")
        yield df


def to_lopnr(s):
    """lopnr_swed is stored as double in most files; join on int64 everywhere."""
    if s.isna().any():
        raise ValueError(f"{int(s.isna().sum())} null lopnr_swed values")
    s = s.astype("float64")
    if (s != s.round()).any():
        raise ValueError("non-integer lopnr_swed values")
    return s.astype("int64")


def not_excluded(df):
    return ~df["lopnr_swed"].isin(EXCLUDE_LOPNR)


def as_utc(s):
    """Timestamps as tz-aware UTC. Naive input is assumed to already be UTC."""
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


def to_local_date(s):
    """UTC timestamp -> Swedish civil date (tz-naive midnight), comparable with INDATUM."""
    return as_utc(s).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()


def keep_first(parts, keys, order):
    """Concatenate partial results and keep the row with the smallest `order` per key."""
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(order, kind="mergesort").drop_duplicates(keys, keep="first")


def write_parquet(df, name):
    """Write df to OUT_DIR/name in BATCH_SIZE-row chunks, via a temp file so a
    crashed run never leaves a half-written output behind."""
    final = OUT_DIR / name
    tmp = OUT_DIR / (name + ".tmp")
    df = df.reset_index(drop=True)
    schema = pa.Schema.from_pandas(df, preserve_index=False)
    writer = pq.ParquetWriter(str(tmp), schema)
    try:
        for start in range(0, max(len(df), 1), BATCH_SIZE):
            chunk = df.iloc[start:start + BATCH_SIZE]
            writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
    finally:
        writer.close()
    tmp.replace(final)
    log(f"wrote {final}  ({len(df):,} rows, {len(df.columns)} columns)")
