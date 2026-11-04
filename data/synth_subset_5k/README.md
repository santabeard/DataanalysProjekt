# synth_subset_5k

Synthetic version of `subset/` (the output of `01_subset.py`) for **5,000 made-up patients**.
Made by `data_pipeline/synthetic/synth_subset_5k.py` (seed 20261104). No real rows: use it to test code, not findings.

## Files

Same file names, columns and types as the real subset (checked against `data_info/swed_profile.txt`).

| File | Rows |
|---|---|
| `swed_ed_core.parquet` | 17,913 ED visits |
| `swed_ed_core_birthdate.parquet` | 5,000 |
| `ehr_karolinska_meas.parquet` | 25,419 (vitals) |
| `ehr_danderyd_lab_meas.parquet` | 60,665 (vitals + labs) |
| `ehr_karolinska_lab.parquet` | 19,954 |
| `ehr_sahlgrenska_lab.parquet` | 16,178 |
| `ehr_sa_vgr_lab.parquet` | 3,530 |
| `sos_patreg_diag_long.parquet` | 337,832 |
| `sos_deathreg_wide.parquet` | 500 deaths |
| `sos_deathreg_diag_long.parquet` | 2,099 |
| `out/vitals_snomed_map.csv` | tells `02_build.py` which V is which vital |

## Made-up names

| Real thing | Here |
|---|---|
| Vital signs | `V1`–`V7` (V1 systolic BP, V2 diastolic BP, V3 SpO2, V4 resp. rate, V5 temperature, V6 pulse, V7 consciousness) |
| Lab tests | `L1`–`L12` |
| Diagnoses and causes of death | `D1`–`D40` |
| Presenting complaints | `C1 [1]`–`C30 [30]` (ed_atlas maps these to its "other" category) |

## What is realistic and what is not

- **From the profile:** rows per patient, share of visits per hospital, date ranges, age and waiting-time quantiles, null rates.
- **Made up:** value distributions; death depends only on age; vitals shift in the 30 days before a death (lower BP, higher pulse, ...) so models have a signal to find.
- **Differs from the real subset:** fewer lab rows (one panel per visit), ~10% deaths instead of ~13%, no real correlations.

## Run the pipeline on it

From `data_pipeline/`:

```
SWED_DATA=../data/synth_subset_5k SWED_WORK=../data/synth_subset_5k SWED_OUT=../data python 02_build.py
# -> ../data/built_synth_subset_5k/
```
