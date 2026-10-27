# SWED data pipeline: notes (2026-10-26)

## Goal

The full SWED files on Bianca are too big to iterate on quickly. The plan:

1. **On Bianca:** take a patient-level subset, harmonise it into a usable dataset, and compute aggregated statistics.
2. **Off Bianca:** generate synthetic data from those statistics, for fast local development.

Code developed on the synthetic data runs unchanged on the real build, because the synthetic files copy its file names, columns and types.

## Supervisor's tasks and where they stand

| Task | Status |
|---|---|
| Map vital parameters to SNOMED | Done in `02_build.py` (`VITAL_CONCEPTS`). The codes are observable entities written from memory: **verify them in the SNOMED browser**. NEWS has no code yet. |
| Remove Tolvan Tolvansson | `lopnr_swed` **701403**, listed in `out/exclude_lopnr.csv` on Bianca. `02` removes him everywhere. |
| Synthetic dataset for local work | `03_export_stats.py` (Bianca) → `04_synthesize.py` (local). |
| Same units for vitals across hospitals? Sahlgrenska? | `02` records the unit status per row, and `03` writes `vitals_units.csv`. The answer comes from the subset build report. |
| Variable list for the death register | Compared in `swed_tasks.ipynb` §5. |
| Load diagnoses from patreg, fat node if needed | `02` streams `sos_patreg_diag_long`. A fat node turned out not to be needed for the subset. |

## Workflow

| Step | Where | Command | Output |
|---|---|---|---|
| 1. Subset | Bianca | `sbatch 01_subset.sbatch` | `~/Desktop/data_std/subset/` |
| 2. Build | Bianca | `sbatch 02_build.sbatch` | `~/Desktop/data_std/built_subset/` |
| 3. Stats | Bianca | `python 03_export_stats.py` | `~/Desktop/data_std/export_stats_built_subset/` |
| 4. Review and export | Bianca → laptop | read every CSV, follow the project's export routine | |
| 5. Synthesize | laptop | `python 04_synthesize.py --stats export_stats_built_subset --validate` | `synthetic_export_stats_built_subset/` |

`check_env.py` checks the Python packages, paths and input files. Run it on Bianca from `data_std` before the pipeline.

### What each script does

- **`01_subset.py`** samples 5,000 patients from `swed_ed_core` and copies **all** rows for those patients from 10 tables, so joins stay intact. It streams the files in 2M-row chunks; only the `lopnr_swed` column is read whole. Finished tables are skipped on rerun.
- **`02_build.py`** writes `patients`, `ed_visits`, `vitals`, `lab_meas`, `diagnoses`, `deaths` and `death_causes`, plus `build_report.csv` and `unmapped_terms.csv`. It:
  - removes the patients in `out/exclude_lopnr.csv`
  - normalises vital units, splits "120/80" blood pressures, and flags implausible values without removing them
  - maps labs with `data_info/swed_lab_meas_to_snomed.csv` and `lab_unit_normalization_07sep2026.csv`
  - links vitals and labs to ED visits (from 60 min before arrival until arrival + `ed_time_total_minutes`)
  - uses a reviewed `out/vitals_snomed_map.csv`, if present, instead of the regex rules
- **`03_export_stats.py`** writes aggregated CSVs only, then self-checks them. Rules:
  - every group needs **10 or more distinct patients**, and counts are rounded to 5
  - percentiles need 50 or more values (1st/99th need 1,000 or more), and min/max are never written
  - dates appear only as year, hour or weekday
  - no IDs and no free text
- **`04_synthesize.py`** draws each variable from its exported distribution, per hospital, source and concept. Vitals come in rounds that share a timestamp. Parquet metadata is tagged `synthetic=true`. Unbacked guesses are marked `ASSUMPTION` in the code.

### Safety

- The original files are **only read**. Scripts 01–03 refuse to write inside the source folder or their own input folder. This was tested, and the checksums were identical before and after a full run.
- Everything that stays on Bianca goes under `/home/theon/Desktop/data_std` (override with `SWED_OUT`).
- Claude Code has a hook (`~/.claude/hooks/block-bianca.sh`) that blocks any shell command mentioning Bianca. Claude writes the commands; you run them.

## Paths and environment on Bianca

| What | Path |
|---|---|
| Original data (read-only) | `/cygnus/proj/nobackup/swed_db/clean_parquet/` (30 files, 9.8 GB) |
| Work and output folder | `/home/theon/Desktop/data_std/` (really `/castor/project/home/theon/...`) |
| Wharf (incoming files) | `/home/theon/Desktop/wharf/theon/theon-sens2020598/` |
| Project | `sens2020598` |

Bianca's software, from `check_env.py`: Python 3.13.11, numpy 2.2.6, pandas 2.2.3, pyarrow 20.0.0. All checks pass.

The old path in the notebook and `swed_profile.txt` (`/proj/nobackup/sens2020598/swed_db/clean_parquet`) is out of date. Update the notebook's `DATA` line if it fails.

## Run log

| Job | What | Result |
|---|---|---|
| 38117 | `01_subset.py`, whole node | COMPLETED in 1 min 30 s, peak memory **1.8 GB**. All 10 tables done, 8.9 MB total. |
| 38118 | `02_build.py` on the subset, 2 cores | Submitted; was pending (Resources) when these notes were written. |

Subset of 5,000 patients (0.23% of 2.17M). Every table kept 0.22–0.26% of its rows, as expected:

| Table | Rows in | Rows kept |
|---|---|---|
| swed_ed_core | 7,801,285 | 18,537 |
| swed_ed_core_birthdate | 2,172,844 | 5,000 |
| ehr_karolinska_meas | 55,868,222 | 129,522 |
| ehr_karolinska_lab | 46,510,003 | 104,478 |
| ehr_danderyd_lab_meas | 16,599,550 | 42,788 |
| ehr_sahlgrenska_lab | 81,002,366 | 183,268 |
| ehr_sa_vgr_lab | 13,028,632 | 31,215 |
| sos_patreg_diag_long | 146,108,589 | 348,792 |
| sos_deathreg_wide | 275,954 | 666 |
| sos_deathreg_diag_long | 1,201,391 | 3,029 |

Lesson: memory use is small. Use `-p core -n 2` for subset work. Try a normal node before a fat node, even for the full build.

## Open questions and caveats

- **SNOMED codes for vitals:** verify each one. The existing mapping file uses procedure codes for pulse, temperature and saturation, while `02` uses observable entities. Discuss with your supervisor.
- **Blood pressure at VGR:** `ehr_sa_vgr_lab` has no `value_str`, so a plain "Blodtryck" row can't be split and ends up with no value. Check how VGR stores BP in the subset.
- **Exporting synthetic data:** generating it from aggregates (as done here) is the safer route. The disclosure rules in `03` should be agreed with your supervisor.
- **Limitations of the synthetic data:**
  - no correlations between variables, so use it to test code, not findings
  - categories with fewer than 10 patients are missing
  - `term_id` and `SJUKHUS` are blank, and ICD-10 codes have only 3 characters
- **Statistics from the full build** (`03 --built .../built_clean_parquet`) would keep more rare categories than the 5,000-patient subset.

## Bianca how-to

- **Transfer files in:** from the laptop, run `sftp theon-sens2020598@bianca-sftp.uppmax.uu.se`, then `cd theon-sens2020598`, then `put` / `put -r` (the top level `/` is not writable). On Bianca, copy from the wharf path above into `data_std`.
- **Several terminals:** with the SSH ControlMaster config, a second `ssh bianca` opens instantly without asking for 2FA again.
- **tmux:** run `tmux new -s work`, detach with Ctrl+b d, and come back with `tmux attach -t work`. Sessions are tied to one login node.
- **"Not allowed at this time" when connecting:** this was a temporary block, probably from the auto-reconnect loop. Stop retrying, wait, then try once. It was not the eduroam/student network.

### Slurm cheat sheet

| Command | What |
|---|---|
| `sbatch file.sbatch` | submit a batch job; options on the command line override the file |
| `interactive -A sens2020598 -n 2 -t 00:30:00` | small interactive session |
| `sbatch -p node -N 1 -C mem256GB ...` | whole fat node (check feature names with `sinfo -o "%20N %10m %f"`) |
| `squeue -u $USER` | status: `PD` = waiting, `R` = running; the reason is shown in brackets |
| `squeue -u $USER --start` | estimated start time |
| `sacct -j <id> --format=JobID,State,Elapsed,MaxRSS` | result and peak memory after the job ends |
| `scancel <id>` | cancel a job |
| `sinfo -s` | partitions; `*` marks the default |

`-p core` shares a node and takes only the cores in `-n`. `-p node` takes whole nodes.

## Next steps

1. When job 38118 finishes: check `swed_build_<id>.log` (it should say `excluding 1 lopnr`) and read `built_subset/build_report.csv`.
2. Answer the unit questions from the `unit_status` rows per hospital. Check Sahlgrenska and VGR blood pressure.
3. Review the vitals list (the notebook's `vitals_snomed_map_draft.csv` → `out/vitals_snomed_map.csv`) and verify the SNOMED codes.
4. Run `03_export_stats.py`, review the CSVs, and agree on the export with your supervisor.
5. Locally: run `04_synthesize.py --validate` on the exported statistics.
