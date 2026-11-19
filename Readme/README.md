# DataanalysProjekt

Bayesian updating of a baseline emergency-department (ED) risk model with routinely recorded
vital signs under realistic missing data. Group 36 project, Uppsala University.

The outcome is **30-day mortality** per ED visit. A baseline model on triage information is
updated with the routinely recorded vital signs. The questions are whether the vitals add
predictive value, whether modular updating beats refitting everything jointly, and how much the
answers depend on how missing vitals are handled.

## Repository layout

```text
Data/       input data, synthetic subsets, generated simulation inputs
Docs/       notebooks and project documentation
Results/    generated outputs, tables and figures
Tests/      placeholder for future tests (no source tests currently exist)
Readme/     this file
src/        all Python source and SLURM scripts
```

Project code is grouped under `src/`: `data_pipeline/`, `preprocessing/`, `evaluation/`,
`ed_atlas/`, and `bayesianUpdating/`.

Only commands supported by files currently in the repository are listed below. Commands for
missing modules (`evaluation/orchestrator.py`, `03b_export_missingness.py`,
`03c_export_joint_stats.py`, `report.tex`) are intentionally omitted.

## Setup

Run all commands from the repository root.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Synthetic raw-table generators

These write local, regenerated tables under `Data/local/data_pipeline/`. The committed
canonical raw subsets are under `Data/synth_subset_5k/` and `Data/synth_subset_50k/`.

```bash
python3 src/data_pipeline/synthetic/synth_subset_5k.py
python3 src/data_pipeline/synthetic/synth_subset_50k.py
python3 src/data_pipeline/synthetic/make_synthetic_swed.py --n-persons 100000
```

## Preprocessing / missing-data Gibbs sampler

```bash
python3 src/preprocessing/simulate_data.py --out Data/preprocessing
python3 src/preprocessing/run_analysis.py \
  --data-dir Data/preprocessing \
  --out-dir Results/preprocessing
```

Use `--skip-sensitivity` for a faster run, or `--iter`/`--burn`/`--chains` to change the sampler.

## Hospital/missingness comparison

```bash
python3 src/evaluation/simulate_hospital.py --out Data/evaluation/data_h
python3 src/evaluation/run_hospital_comparison.py \
  --data-dir Data/evaluation/data_h \
  --out Results/evaluation/results_h \
  --quick
```

Drop `--quick` for the full comparison. Add `--with-standard` for ridge/boosting baselines.

## Data pipeline

The pipeline needs a `data_info/` folder with the real variable and lab/SNOMED mappings. It is
not in git; point `SWED_REF` at it. On Bianca, use the real SWED path for `SWED_DATA`.

```bash
export SWED_DATA="$PWD/Data/synth_subset_50k"
export SWED_WORK="$PWD/src/data_pipeline"
export SWED_OUT="$PWD/Results/data_pipeline"
export SWED_REF="/path/to/data_info"

python3 src/data_pipeline/check_env.py
python3 src/data_pipeline/02_build.py
python3 src/data_pipeline/03_export_stats.py \
  --built "$SWED_OUT/built_synth_subset_50k"
python3 src/data_pipeline/04_synthesize.py \
  --stats "$SWED_OUT/export_stats_built_synth_subset_50k" \
  --validate
```

`01_subset.py` is only runnable on Bianca against the full SWED extract. The SLURM wrappers
must be submitted from `src/data_pipeline`:

```bash
cd src/data_pipeline
sbatch 01_subset.sbatch
sbatch 02_build.sbatch
sbatch build_and_stats.sbatch
sbatch 06_joint_hierarchical.sbatch
sbatch 07_compare_models.sbatch
cd ../..
```

## Models

Steps `05` and `07` need `atlas_mortality_posterior.npz` in `Data/ed_atlas/` (or pass
`--atlas`). Posterior `.npz` artifacts are not committed.

```bash
python3 src/data_pipeline/05_atlas_vitals.py \
  --built "$SWED_OUT/built_synth_subset_50k"
python3 src/data_pipeline/06_joint_hierarchical.py \
  --built "$SWED_OUT/built_synth_subset_50k"
python3 src/data_pipeline/07_compare_models.py \
  --built "$SWED_OUT/built_synth_subset_50k"
```

## Notebooks

```bash
jupyter lab
```

Notebooks live under `Docs/notebooks/`. `explore_built.ipynb` reads
`Results/data_pipeline/built_synth_subset_50k/`; `evaluation.ipynb` reads
`Data/healthcare/healthcare_dataset_2_missingness.csv`; `ex.ipynb` adds `src/` to `sys.path`
and writes to `Data/preprocessing/` and `Results/preprocessing/`.

## Tests

`Tests/` is currently a placeholder. No source test files exist, so there is no `pytest`
command to run.
