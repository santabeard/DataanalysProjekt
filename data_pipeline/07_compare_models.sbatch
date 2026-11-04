#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 6                # one core per chain
#SBATCH -t 01:00:00
#SBATCH -J swed_compare
#SBATCH -o swed_compare_%j.log

# Submit from the folder holding 07_compare_models.py:   sbatch 07_compare_models.sbatch

set -euo pipefail

module load python          # CHANGE if pymc comes from another module or conda environment

cd "$SLURM_SUBMIT_DIR"
python -c "import pymc, numpyro; print('pymc', pymc.__version__, '| numpyro', numpyro.__version__)"
python 07_compare_models.py
