"""Paths and helpers shared by the pipeline scripts.

Repo layout (this file lives in root/src/core/data_pipeline/):
  data/raw/swed_data/  SWED source tables, read-only (named copy_<table>.parquet)
  data/raw/uu_data/    UU source tables, read-only (named copy_<table>.parquet)
  data/interim/        everything produced that is not model-ready
  data/train/          model-ready tables only
  docs/data_info/      reference files (swed_lab_meas_to_snomed.csv, complaints_combined.csv)
Every path can be overridden with an environment variable, e.g. on Bianca.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # root/src/core/data_pipeline/common.py -> root

DATA = Path(os.environ.get("SWED_DATA", ROOT / "data" / "raw" / "swed_data"))  # read-only
UU_DATA = Path(os.environ.get("UU_DATA", ROOT / "data" / "raw" / "uu_data"))  # read-only
RAW_PREFIX = os.environ.get("SWED_RAW_PREFIX", "copy_")  # set to "" where tables have no prefix
INTERIM = Path(os.environ.get("SWED_INTERIM", ROOT / "data" / "interim"))
TRAIN = Path(os.environ.get("SWED_TRAIN", ROOT / "data" / "train"))
REF = Path(os.environ.get("SWED_REF", ROOT / "docs" / "data_info"))
WORK = Path(os.environ.get("SWED_WORK", Path.cwd()))  # where out/exclude_lopnr.csv lives

# The 6 vitals, by SNOMED ID. SWED_subset.py writes these IDs into vitals_snomed.
VITAL_SNOMED = {
    "271649006": "systolic_bp",
    "271650006": "diastolic_bp",
    "284032008": "spo2",
    "56342008": "temperature",
    "65653002": "pulse",
    "86290005": "resp_rate",
    "6942003":  "consciousness",
}

# UU measurements: NameOfMeasurement -> [(result column, SNOMED ID), ...]. Same SNOMED IDs as
# VITAL_SNOMED, so UU and SWED end up with the same vital_* columns. UU_subset.py turns each row
# into one reading per pair (Blodtryck -> systolic + diastolic). Names not listed are dropped.
UU_MEASUREMENT_SNOMED = {
    "Blodtryck":         [("ResultOfMeasurement1", "271649006"),   # systolic, mmHg
                          ("ResultOfMeasurement2", "271650006")],  # diastolic, mmHg
    "Puls":              [("ResultOfMeasurement1", "65653002")],   # /min
    "Temperatur":        [("ResultOfMeasurement1", "56342008")],   # °C
    "Indirekt oximetri": [("ResultOfMeasurement1", "284032008")],  # SpO2 %; Result2 (O2 flow l/min) not used
    "Andning":           [("ResultOfMeasurement1", "86290005")],   # /min
    "AVPU":              [("ResultOfMeasurement1", "6942003")],    # consciousness; numeric coding unknown (5 = Alert?)
}


def raw_table(name):
    """Path of a source table, e.g. 'swed_ed_core' -> data/raw/swed_data/copy_swed_ed_core.parquet."""
    return DATA / f"{RAW_PREFIX}{name}.parquet"


def uu_raw_table(name):
    """Path of a UU source table, e.g. 'edvisit' -> data/raw/uu_data/copy_edvisit.parquet."""
    return UU_DATA / f"{RAW_PREFIX}{name}.parquet"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def guard_output(path):
    """Exit if path is inside the raw data folders: outputs must never land there."""
    p = Path(path).expanduser().resolve()
    for src in (ROOT / "data" / "raw", DATA, UU_DATA):
        src = src.expanduser().resolve()
        if p == src or src in p.parents:
            sys.exit(f"refusing to write to {path}: it is inside the raw data folder {src}")
