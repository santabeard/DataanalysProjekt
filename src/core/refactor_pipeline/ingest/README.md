# Ingestion: deaths, patients, visits

Three scripts turn the SWED clean_parquet files into three files for the
age + sex + complaint + vitals → diagnosis / 30-day mortality models.

| Script | Reads | Writes | One row per |
|---|---|---|---|
| `01_deaths.py` | `sos_deathreg_wide` | `deaths.parquet`: `lopnr_swed`, `date_of_death`, `inexact_death_date` | dead patient |
| `02_patients.py` | `swed_ed_core` | `patients.parquet`: `lopnr_swed`, `age` (at first ED visit), `is_male` | patient |
| `03_visits.py` | `swed_ed_core`, all lab/meas files, `sos_patreg_wide` | `visits.parquet` (columns below) | ED visit |

`visits.parquet`: `visit_id`, `lopnr_swed`, `ed_hospital`, `ed_arrival_datetime` (UTC), `age` (at this visit),
`complaint_sctid`, `complaint_text`, then per vital (`systolic_bp`, `diastolic_bp`, `pulse`, `resp_rate`,
`spo2`, `temperature`) the first plausible reading and `<vital>_minutes` from arrival, then `diagnosis` and
`diagnosis_days`.

`diagnosis` is the Hdia of the first patreg contact within 30 days of arrival, `"NODIAG"` when patreg covers
those 30 days but has no contact, and null when the 30 days run past the end of patreg (2016-12-31).

30-day mortality: join `deaths.parquet` on `lopnr_swed` and check whether `date_of_death` falls within
0–30 days of the arrival's local date.

## Run (on Bianca)

1. Edit the SETTINGS block in `ingest_common.py`: `SOURCE_DIR`, `FILE_PREFIX` (`"copy_"` for the copy files),
   `OUT_DIR`.
2. `sbatch run_ingest.sbatch`, or run the scripts one by one with `python3`.
3. Read the log: every step prints rows read, dropped (and why), and kept, plus any vital-like term names that
   are not mapped (`VITAL_TERMS` in `ingest_common.py`).

## Constraints and rules

- Only `pandas`, `pyarrow` and `pathlib` are imported, plus the local `ingest_common.py`.
- Every file is read in batches of at most `BATCH_SIZE` (300,000) rows; the scripts refuse a larger value.
- Source files are only read; the scripts refuse an `OUT_DIR` inside `SOURCE_DIR`.
- Patient 701403 (test patient) is removed everywhere.
- Rules taken from `v2/phase1`: vitals window from 60 min before arrival to the end of the visit,
  first reading per vital, Hdia only, 30-day windows on Swedish local dates, `date_death_minimp`.
- Visits with `ed_inexact_time = TRUE` are dropped (their clock time is unknown).
- Tested locally on the synthetic 5k/50k fixtures with Python 3.9 + pandas 1.2.4 + pyarrow 4, and with
  pandas 3.1: the output matched a brute-force recomputation for 1,500 sampled visits.
