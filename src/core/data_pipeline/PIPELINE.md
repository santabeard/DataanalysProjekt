# ED data pipeline: design and reference

## 1. What this project is about

When a patient arrives at a hospital **emergency department (ED)**, staff record a few things within minutes: age, sex, why the patient came (the *presenting complaint*, e.g. "chest pain") and basic measurements such as pulse and blood pressure (*vital signs*). The project builds statistical models that use exactly this early information to estimate two things:

- **Which diagnosis** the patient will get in specialist care within 30 days.
- **Whether the patient dies** within 30 days.

The models are Bayesian logistic regressions. A published model called the **ED Atlas** (age, sex and complaint only) serves as the starting point; this project adds vital signs.

This pipeline turns raw hospital and register data into one clean, model-ready table per data source, called **`train_clean`**. It supports two data sources, **SWED** and **UU** (section 4), and produces the same columns from both wherever the data allows.

---

## 2. Glossary

### Healthcare terms

| Term | Meaning |
|---|---|
| ED | Emergency department (Swedish: *akutmottagning*) |
| Visit | One ED visit: one patient, one arrival time |
| Presenting complaint | The reason for the visit, as registered on arrival ("Bröstsmärta" = chest pain). A symptom, not a diagnosis |
| Triage | Quick sorting of ED patients by urgency on arrival |
| RETTS | The Swedish triage system. Gives a priority colour (red, orange, yellow, green, blue) |
| Vital signs | Basic bedside measurements: blood pressure, pulse, temperature, oxygen saturation, breathing rate, consciousness |
| SpO2 | Oxygen saturation of the blood, in %. Swedish: *saturation*, *indirekt oximetri* |
| Systolic / diastolic BP | The upper / lower blood pressure value, in mmHg |
| Level of consciousness | How awake and responsive the patient is. Scored on a scale, not measured by a device |
| AVPU | Consciousness scale: **A**lert / responds to **V**oice / responds to **P**ain / **U**nresponsive |
| RLS-85 | Swedish consciousness scale, 1 (alert) to 8 (unresponsive) |
| NEWS / NEWS2 | National Early Warning Score: one number summarising several vital signs. Higher means sicker |
| Diagnosis | What the doctor concludes the patient has (e.g. heart attack) |
| Main diagnosis | The single most important diagnosis of a care contact. SWED: `Hdia`, UU: `MainDiagnosis` |
| Primary care | Local health centres (Swedish: *vårdcentral*). UU type `PV` |
| Specialist outpatient care | Hospital clinics without admission. UU type `OV` |
| Inpatient care | Admitted to a hospital ward. UU type `SV` |
| Patient register | Socialstyrelsen's national register of specialist care contacts and their diagnoses |
| Death register | Socialstyrelsen's national register of deaths |
| Lab test | Blood sample analysed in a laboratory (e.g. CRP, haemoglobin). Not used by the models yet |

### Coding systems

| Term | Meaning | Example |
|---|---|---|
| SNOMED CT | International system of numeric codes for clinical concepts. Used here for **complaints** and **vital signs** | `29857009` = chest pain, `65653002` = pulse |
| SCTID | A SNOMED CT ID (the number) | `29857009` |
| ICD-10 | WHO's system of codes for **diagnoses**. A letter + digits | `I21` = heart attack |
| ICD-10-SE | The Swedish edition of ICD-10. Adds detail digits and sometimes a letter | `R104X`, `F900B` |
| ICD-10 chapter | The 22 top-level groups of ICD-10, by first letter | `I00–I99` = circulatory diseases |
| ICD-10 block | A range of related codes inside a chapter | `C15-C26` = cancer of the digestive organs |
| 3-character code | ICD-10 category: letter + 2 digits | `I48` = atrial fibrillation |
| 4-character code | More detail: letter + 3 digits (no dot in this data) | `I489` = atrial fibrillation, unspecified |

### Data and technical terms

| Term | Meaning |
|---|---|
| `lopnr_swed` | SWED's pseudonymised patient ID (a number that replaces the personal identity number) |
| `PersonId` | UU's pseudonymised patient ID. A different ID system from `lopnr_swed` |
| Subset | A random sample of patients with **all** their rows in every table, for faster work |
| Label | The size tag in file names: exact patient count (`500`, `100K`) |
| Parquet | Column-based file format for large tables |
| Bianca | UPPMAX's computer cluster for sensitive data. No internet access. All patient data stays here |
| Wharf | The only file-transfer folder in and out of Bianca |
| Slurm / sbatch | Bianca's job queue. `sbatch file.sbatch` runs a job in the background and writes a log |

---

## 3. Folder layout

```
root/
├── data/
│   ├── raw/                 source tables, READ-ONLY (scripts refuse to write here)
│   │   ├── swed_data/       copy_<table>.parquet
│   │   └── uu_data/         copy_<table>.parquet
│   ├── interim/             everything produced that is not model-ready (subsets, samples, caches)
│   └── train/               model-ready tables only (*_train_clean.parquet)
├── docs/data_info/          reference files (SNOMED mappings)
└── src/core/data_pipeline/  the scripts in this folder (+ batch/ with Slurm jobs)
```

- The scripts must sit directly in `root/src/core/data_pipeline/`: `common.py` finds `root` by going three folders up.
- Every path can be overridden with an environment variable (section 13).

---

## 4. Data sources

### SWED (Stockholm region, national registers)

| Table | What it holds | Used for |
|---|---|---|
| `swed_ed_core` | One row per ED visit: arrival time, visit length, age, sex, complaint | Visits, complaints, patient sampling |
| `ehr_karolinska_meas` | Bedside measurements from Karolinska | Vital signs |
| `ehr_danderyd_lab_meas` | Measurements **and** lab tests mixed, from Danderyd / SÖS / Södertälje | Vital signs (lab rows are ignored) |
| `sos_patreg_wide` | Patient register: one row per specialist care contact, main diagnosis `Hdia` + up to 30 more | Diagnosis target |
| `sos_deathreg_wide` | Death register: date of death | Mortality target |
| `ehr_karolinska_lab`, `ehr_sos_lab`, `ehr_sahlgrenska_lab` | Lab tests only | Subset, not used yet |

### UU (Akademiska sjukhuset, Uppsala)

| Table | What it holds | Used for |
|---|---|---|
| `edvisit` | One row per ED visit (608K, 2014-02 to 2024-09): time, length, age, sex, complaint (plain text), date of death | Visits, complaints, mortality, patient sampling |
| `measurements` | Vital signs (24M readings, 2017-11 to 2024-12) | Vital signs |
| `healthcarevisits` | All care contacts (19M) incl. primary care, with `MainDiagnosis` | Diagnosis target |

### Reference files (`docs/data_info/`)

| File | What it holds | Used by |
|---|---|---|
| `swed_lab_meas_to_snomed.csv` | SWED measurement/lab names (`term_name`) → SNOMED ID | `subset.py` (vitals) |
| `complaints_combined.csv` | Complaint texts → SNOMED ID(s), incl. the UU texts | `uu_subset.py` (complaints) |
| `swed_profile.txt`, `*.xlsx` | Column statistics and documentation of the SWED tables | Humans only |

---

## 5. Pipeline overview

```
SWED                                            UU
raw/swed_data/copy_*.parquet                    raw/uu_data/copy_*.parquet
        │ subset.py                                     │ uu_subset.py
        ▼                                               ▼
interim/<label>_SUB_*.parquet                   interim/<label>_UU_SUB_*.parquet
        │ build_train_clean.py                          │ uu_build_train_clean.py
        ▼                                               ▼
train/<label>_train_clean.parquet               train/<label>_uu_train_clean.parquet
        │                                               │
        └──────────── check_train_clean.py ─────────────┘
        │
        └─ synthetic_info.py → interim/<label>_synthetic_info.json
```

| Script | Reads | Writes | What it does |
|---|---|---|---|
| `common.py` | – | – | Shared paths, the SNOMED vital list, the UU measurement map, safety guard |
| `subset.py` | `raw/swed_data/` | `interim/<label>_SUB_*`, `<label>_sample_lopnr.csv` | Samples N SWED patients, copies all their rows from every table, adds `vitals_snomed` |
| `build_train_clean.py` | SWED subsets | `train/<label>_train_clean.parquet` | Joins vitals, diagnosis, death; builds complaint rows and target classes |
| `uu_subset.py` | `raw/uu_data/` | `interim/<label>_UU_SUB_*`, `<label>_uu_sample_personid.csv` | Samples N UU patients; maps complaints and vitals to SNOMED |
| `uu_build_train_clean.py` | UU subsets (+ SWED code list) | `train/<label>_uu_train_clean.parquet` | Same as the SWED build, using the same shared functions |
| `check_train_clean.py` | a `train_clean` | screen | Checks the design rules hold (section 11) |
| `synthetic_info.py` | SWED `train_clean` | `interim/<label>_synthetic_info.json` | Privacy-guarded summary statistics, for building synthetic data |
| `batch/*.sbatch` | – | Slurm logs | Run the scripts as Bianca jobs (section 12) |
| `batch/uu_profile.sbatch` | `raw/uu_data/` | log only | One-off, privacy-safe description of the UU tables |

---

## 6. Design decisions

### Sampling (subset scripts)

| Decision | Choice | Why |
|---|---|---|
| Unit of sampling | **Patients**, not visits or rows | A sampled patient brings all their visits, vitals, diagnoses and death record, so joins stay complete |
| Sampled from | The ED visit table (`swed_ed_core` / `edvisit`) | Every sampled patient has at least one ED visit |
| Randomness | Fixed seed `20261026` | Same size + same data = same patients, every time |
| Sample file | `<label>_sample_lopnr.csv` / `<label>_uu_sample_personid.csv`, reused on rerun | All tables share one patient set; sizes can live side by side |
| Label | Exact patient count: `500`, `1500`, `100K` | Two sizes can never get the same file names |
| Test patients | SWED IDs in `out/exclude_lopnr.csv` are removed before sampling | Test records (e.g. "Tolvan") are not real patients |
| Which SWED tables | Every `copy_*.parquet` in `raw/swed_data/` that has `lopnr_swed` | New tables are picked up without code changes |
| Large tables | Streamed in batches of 2M rows | The 146M-row patient register fits in a few GB of memory |
| Reruns | Finished files are skipped; a file only gets its final name when complete | A job killed halfway can simply be resubmitted |

### Vital signs

| Decision | Choice | Why |
|---|---|---|
| Identify vitals by | SNOMED ID, written into a `vitals_snomed` column at subset time | Different hospitals spell the same vital differently; the code is the common key |
| Which vitals | 7 (table in section 7) | The ED Atlas extension uses these bedside measurements |
| Time window | From **60 min before arrival** to **arrival + visit length** | Captures triage measurements, including those taken just before registration |
| Several readings in the window | Keep the **first** by time | Closest to what is known at triage |
| Hospital match | Not used as a filter | Kept simple; a reading belongs to the patient, not the hospital |
| Non-numeric values | Dropped (`value_num` must be filled) | The models need numbers |
| Missing vitals | Left empty (null); no imputation | Imputation is a modelling decision, not a data-cleaning one |
| Scaling | None; raw units | Same reason |

### Diagnosis (the classification target)

| Decision | Choice | Why |
|---|---|---|
| Source | Main diagnosis only (SWED `Hdia`, UU `MainDiagnosis`) | One clear diagnosis per contact; secondary diagnoses are noisier |
| Which care | Specialist care only (SWED: the register only has this; UU: `Type` OV/SV) | Primary care and phone contacts would dominate UU; keeps SWED and UU comparable |
| Time window | **Day 0 to day 30** after the arrival date, both included | Same-day diagnoses are the central case |
| Several matches | Keep the earliest by date | Closest to the ED visit |
| Code detail | As SWED stores it (section 8); UU codes are mapped to SWED's level | One model can train on both sources |
| R and Z codes (symptoms, follow-ups) | **Kept** | The ED Atlas drops them; kept here so the choice can be made later |
| Target classes | `diag_class` (section 8) | Few enough classes to model; rare codes grouped by chapter |

### Death (the mortality target)

| Decision | Choice | Why |
|---|---|---|
| Window | Death date **day 0 to day 30** after the arrival date | Standard 30-day mortality |
| SWED date column | `date_death_minimp` (the earliest possible date when only part of the date is known) | Conservative: never places a death later than it could have been |
| SWED inexact dates | Kept, flagged in `death_date_inexact` | Lets the model stage decide |
| UU source | `DateOfDeath` in `edvisit` | UU has no separate death register |
| Death date before the visit (UU) | Counted as not died, reported in the log | Data error; should not create a false positive |

### Visits kept

| Rule | SWED | UU | Why |
|---|---|---|---|
| Visit length missing | Dropped | Dropped | The vitals window needs an end time |
| Arrival time only approximate | Dropped (`ed_inexact_time`) | No such flag | The 60-min vitals window needs an exact time |
| Sex unknown | – | Dropped (`U`, 15 visits) | `is_male` must be true/false |
| Visits before vitals exist | – | Kept, vitals empty (UU vitals start 2017-11) | Diagnosis and mortality are still valid |

### Time zones

| Source | Stored as | Handling |
|---|---|---|
| SWED | UTC | Converted to Stockholm local date before the 30-day comparisons, so a visit at 00:30 local time is not counted as the previous day |
| UU | Local time, no time zone | Used as is |

### Safety and tooling

| Decision | Choice | Why |
|---|---|---|
| Raw data | Scripts refuse to write anywhere under `data/raw/` | Source data must never be changed |
| Library | pandas + pyarrow (+ numpy for sampling) | Available on Bianca (Python 3.9.5, pandas 1.2.4, pyarrow 7). polars would need a manual install |
| Shared code | SWED and UU builds call the same functions for vitals, complaints, the 30-day match and diagnosis classes | A rule changes in one place for both sources |
| Intermediate files | `interim/` only; `train/` gets model-ready tables only | Clear separation of finished and unfinished data |

---

## 7. Vital signs: names and codes

| `vital_*` column | SNOMED | Meaning | SWED `term_name`(s) | UU `NameOfMeasurement` → column | Unit |
|---|---|---|---|---|---|
| `vital_systolic_bp` | 271649006 | Upper blood pressure | `Blodtryck systoliskt`, `Blodtryck systoliskt - övre` | `Blodtryck` → Result1 | mmHg |
| `vital_diastolic_bp` | 271650006 | Lower blood pressure | `Blodtryck diastoliskt`, `Blodtryck diastoliskt - nedre` | `Blodtryck` → Result2 | mmHg |
| `vital_pulse` | 65653002 | Heart beats per minute | `Puls`, `Pulsfrekvens` | `Puls` → Result1 | /min |
| `vital_temperature` | 56342008 | Body temperature | `Temperatur`, `Kroppstemperatur`, `Kroppstemp`, `Kroppstemp (PNA)` | `Temperatur` → Result1 | °C |
| `vital_spo2` | 284032008 | Oxygen saturation | `Saturation`, `Saturation med syrgas` | `Indirekt oximetri` → Result1 | % |
| `vital_resp_rate` | 86290005 | Breaths per minute | `Andningsfrekvens`, `Andningsfrekvens (PNA)` | `Andning` → Result1 | /min |
| `vital_consciousness` | 6942003 | Level of consciousness | `Medvetandegrad enl Retts`, `Medvetandegrad, NEWS` | `AVPU` → Result1 | scale |

- **Where the mapping lives:** SWED: `swed_lab_meas_to_snomed.csv` + `VITAL_SNOMED` in `common.py`. UU: `UU_MEASUREMENT_SNOMED` in `common.py`.
- **`(PNA)`** = point-of-care measurement (taken next to the patient).
- **UU `Blodtryck`** is one row with two values, so each row becomes two readings.
- **UU names not used:** `NEWS`, `NEWS2` (summary scores), `Metvetenhet` (coding unknown). UU `Indirekt oximetri` Result2 (oxygen flow, l/min) is not used.

### Lab test name prefixes (for the lab tables)

| Prefix | Sample type | Example |
|---|---|---|
| `B-` | Whole blood | `B-Hemoglobin` |
| `P-` | Plasma | `P-CRP` |
| `S-` | Serum (Sahlgrenska's naming) | `S-ALAT` |
| `P(aB)-` / `P(vB)-` | Plasma from an arterial / venous blood gas | `P(vB)-Laktat /PNA` |

---

## 8. Diagnosis codes

### How the codes look

| Source | Format | Examples | Notes |
|---|---|---|---|
| SWED `Hdia` | Mixed levels | `C15-C26` (block), `F32` (3-char), `I489` (4-char) | Whole chapters (cancer, eye, skin, pregnancy, symptoms…) appear only as blocks, probably for privacy. Exact rules are not documented |
| UU `MainDiagnosis` | Full ICD-10-SE | `I109`, `R104X`, `F900B` | Full detail; Swedish extra letters; 66% empty (mostly administrative contacts) |

### Mapping UU codes to SWED's level (`to_swed_level`)

The build reads every distinct `Hdia` value in SWED's raw patient register once and caches the list as `interim/swed_hdia_vocab.csv` (codes only). Each UU code then:

| Step | Rule | Example |
|---|---|---|
| 0 | Drop the Swedish extra letter(s) | `R104X` → `R104` |
| 1 | SWED has the 4-character code → keep it | `I489` → `I489` |
| 2 | Else SWED has the 3-character code → use it | `F329` → `F32` |
| 3 | Else SWED has a block containing it → use the narrowest | `C189` → `C15-C26` |
| 4 | Else (no match) → the 3-character code, reported as `fallback` in the log | `F900B` → `F90` |

### Target classes (`diag_class`)

| Situation | `diag_class` | Example |
|---|---|---|
| No specialist-care diagnosis within 30 days | `NODIAG` | – |
| Code occurs ≥ 30 times in this dataset (`--hdia-min-count`) | The code itself | `I489`, `C15-C26` |
| Rarer code | `other_<chapter>` | `other_circulatory` |

ICD-10 chapters used for `other_<chapter>`:

| Codes | Chapter name | | Codes | Chapter name |
|---|---|---|---|---|
| A00–B99 | `infectious` | | M00–M99 | `musculoskeletal` |
| C00–D48 | `neoplasm` | | N00–N99 | `genitourinary` |
| D50–D89 | `blood_immune` | | O00–O99 | `pregnancy_childbirth` |
| E00–E90 | `endocrine_metabolic` | | P00–P96 | `perinatal` |
| F00–F99 | `mental_behavioural` | | Q00–Q99 | `congenital` |
| G00–G99 | `nervous_system` | | R00–R99 | `symptoms_signs` |
| H00–H59 | `eye` | | S00–T98 | `injury_poisoning` |
| H60–H95 | `ear` | | U00–U99 | `special_purposes` |
| I00–I99 | `circulatory` | | V01–Y98 | `external_causes` |
| J00–J99 | `respiratory` | | Z00–Z99 | `health_status_factors` |
| K00–K93 | `digestive` | | other / unreadable | `unknown` |
| L00–L99 | `skin` | | | |

Block codes like `C15-C26` use their first code (`C15`) to find the chapter.

---

## 9. Complaints

| Step | SWED | UU |
|---|---|---|
| Raw value | Already SNOMED: `"Bröstsmärta [29857009]"`, or two joined by `+` | Plain text: `"Bröstsmärta"` |
| Subset | Copied unchanged | Mapped through `complaints_combined.csv` (exact text match) into SWED's format, new column `complaint_snomed` |
| Build | `complaint_rows()` | The same `complaint_rows()` |

What `complaint_rows()` produces:

| Visit has | Rows | `complaint_sctid` | `complaint_role` | `complaint_weight` |
|---|---|---|---|---|
| No complaint | 1 | `none` | `primary` | 1.0 |
| One code | 1 | the code | `primary` | 1.0 |
| Two codes | 2 | each code | `primary`, `secondary` | 0.5 each |

- **Why duplicate instead of one row with two complaints:** this matches the ED Atlas, which averages the two complaints' predictions in probability space. The weights make each visit count once in total.
- **One-hot columns:** each row gets `complaint_bucket_<sctid>` columns (exactly one is true). Codes seen fewer than 30 times (`--complaint-min-count`) go to `complaint_bucket_other_complaint`.
- **One row per visit:** `df.drop_duplicates("ed_visit_id")` keeps the primary row.
- **In UU, two codes come from the mapping file**, e.g. `Extremitetssvullnad/värk` → swelling + pain of a limb.

---

## 10. `train_clean` columns

One row per (visit, complaint). Most visits have one row; two-complaint visits have two.

| Column | Type | Meaning | SWED | UU |
|---|---|---|---|---|
| `ed_visit_id` | int | Row number of the visit in the subset; unique per visit | ✓ | ✓ |
| `lopnr_swed` | float | SWED patient ID | ✓ | – |
| `PersonId`, `ContactId` | int | UU patient ID and UU visit ID | – | ✓ |
| `ed_age` | int | Age at the visit, years | ✓ | ✓ (from 16) |
| `is_male` | bool | Sex | ✓ | ✓ |
| `complaint_sctid` | text | SNOMED code of this row's complaint, or `none` | ✓ | ✓ |
| `complaint_role` | text | `primary` / `secondary` | ✓ | ✓ |
| `complaint_weight` | float | 1.0 or 0.5 (sums to 1 per visit) | ✓ | ✓ |
| `complaint_bucket_*` | bool | One-hot complaint columns | ✓ | ✓ |
| `vital_*` (7 columns) | float | First reading in the window, raw units; empty if none | ✓ | ✓ |
| `died_30d` | bool | Died within 30 days | ✓ | ✓ |
| `death_days_to_death` | float | Days from arrival to death, if within 30 | ✓ | ✓ |
| `death_date_inexact` | bool | Death date only partly known | ✓ | – |
| `diag_hdia_first` | text | First specialist-care main diagnosis within 30 days | ✓ | ✓ (mapped to SWED level) |
| `diag_hdia_any` | bool | Whether there was such a diagnosis | ✓ | ✓ |
| `diag_class` | text | Target class: `NODIAG`, a code, or `other_<chapter>` | ✓ | ✓ |

---

## 11. Checks (`check_train_clean.py`)

| Check | Meaning |
|---|---|
| Each visit is `(primary)` or `(primary, secondary)` | The complaint duplication worked |
| `complaint_weight` sums to 1 per visit | Each visit counts once |
| `none` is never a secondary complaint | No-complaint visits have one row |
| Exactly one `complaint_bucket_*` is set per row | The one-hot encoding is valid |
| `died_30d` has no nulls | Every visit has a mortality outcome |
| `diag_hdia_any` false ⇔ `diag_class` = `NODIAG` | The target classes are consistent |

It also prints the share of visits with each vital sign.

---

## 12. How to run (on Bianca)

Submit every job from the pipeline folder: `cd ~/Desktop/pda/src/core/data_pipeline`

| Goal | Command |
|---|---|
| SWED subset, 100,000 patients | `sbatch batch/subset.sbatch --n 100000` |
| SWED train_clean + checks | `sbatch batch/build_train_clean.sbatch 100K` (after the subset finished) |
| UU subset + train_clean + checks | `sbatch batch/uu_pipeline.sbatch 100000` |
| Redo some SWED tables | `sbatch batch/subset.sbatch --n 100000 --overwrite --tables ehr_karolinska_meas` |
| Watch the queue | `watch -n 5 squeue -u theon` |
| Follow a log | `tail -f <jobname>_<jobid>.log` |

- `subset.sbatch` passes everything after the file name to `subset.py`.
- The build job takes the **label** (`100K`), the UU job takes the **patient count** (`100000`).
- Scripts can also be run directly: `python3 subset.py --n 1000`, `python3 build_train_clean.py --label 1K`.

### Main options

| Script | Option | Default | Meaning |
|---|---|---|---|
| `subset.py` / `uu_subset.py` | `--n` | 500,000 / 100,000 | Number of patients |
| | `--seed` | 20261026 | Random seed |
| | `--overwrite` | off | Redo finished files (the patient sample is still reused) |
| `subset.py` | `--tables` | all raw tables | Only these tables |
| builds | `--label` | 500K / 100K | Which subset to use |
| | `--hdia-min-count` | 30 | Minimum count for a diagnosis code to get its own class |
| | `--complaint-min-count` | 30 | Minimum count for a complaint code to get its own column |
| `synthetic_info.py` | `--min-count`, `--round-to` | 10, 5 | Privacy guards (section 14) |

---

## 13. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SWED_DATA` | `root/data/raw/swed_data` | SWED source tables |
| `UU_DATA` | `root/data/raw/uu_data` | UU source tables |
| `SWED_RAW_PREFIX` | `copy_` | File name prefix of the source tables (`""` for none) |
| `SWED_INTERIM` | `root/data/interim` | Subsets and other intermediate files |
| `SWED_TRAIN` | `root/data/train` | `train_clean` outputs |
| `SWED_REF` | `root/docs/data_info` | Reference CSV files |
| `SWED_WORK` | current folder | Where `out/exclude_lopnr.csv` is read from |

---

## 14. Privacy rules

| Rule | Detail |
|---|---|
| Patient-level data stays on Bianca | Subsets and `train_clean` files are never exported |
| Export only through the wharf | And only data with no patient-level content |
| `synthetic_info.py` guards | Categories with < 10 rows merged into OTHER; counts rounded to 5; percentiles only, no min/max; no cross-tables |
| `uu_profile.sbatch` guards | ID columns never printed; values shown only if ≥ 10 rows share them; long values, 6+ digit numbers, name-like values and free-text/staff columns never printed; formats shown as shapes (`I219` → `A999`) |
| Always read a log before exporting it | The guards catch known patterns, not everything |

---

## 15. Known limitations and open questions

| Topic | Issue |
|---|---|
| Consciousness | SWED (RETTS / NEWS scale) and UU (AVPU as a number, coding unknown) share one column and code, but may use different scales |
| SpO2 | Readings with and without extra oxygen share one code; a low value on oxygen means something different |
| UU complaints without a SNOMED code | End up as `none`, the same as no complaint |
| Diagnosis detail | `diag_class` mixes broad blocks (`R10-R19`) and specific codes (`I489`) |
| SWED code coarsening | Inferred from the data, not documented |
| R / Z codes | Kept here, excluded by the ED Atlas |
| ED Atlas comparison | Mapping `diag_hdia_first` onto the Atlas's 930 outcome codes is not built yet |
| UU vitals before 2017-11 | Missing (the measurement table starts then) |
| Age | UU starts at 16, the ED Atlas at 18 |
| Same-time readings | When two readings share the earliest time, which one is kept is arbitrary |
| Lab tables | Subset but not used |
| Train / validation / test split | Not done here; a later stage |
