#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 6                # one core per chain
#SBATCH -t 00:30:00
#SBATCH -J swed_model_3_4
#SBATCH -o swed_model_3_4_%j.log

# Submit from the folder holding 01_fit_model_3_4.py:   sbatch 01_fit_model_3_4.sbatch
# Requires <N>K_train_clean.parquet (scripts_v2/phase1/04_train_clean.py) to already exist.

set -euo pipefail

module load python          # CHANGE if pymc/numpyro come from another module or conda environment

export SWED_WORK="$SLURM_SUBMIT_DIR"
cd "$SWED_WORK"

python3 -c "import pymc, numpyro; print('pymc', pymc.__version__, '| numpyro', numpyro.__version__)"
python3 01_fit_model_3_4.py "$@"
