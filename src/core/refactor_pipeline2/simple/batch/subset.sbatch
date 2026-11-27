#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 4                # streams the big tables, so no fat node needed
#SBATCH -t 04:00:00
#SBATCH -J swed_subset
#SBATCH -o swed_subset_%j.log

# Submit from the folder holding subset.py. Arguments after the file name go to subset.py:
#   sbatch batch/subset.sbatch --n 500000
#   sbatch batch/subset.sbatch --n 10000 --out ../../../data/interim/10K
# If it times out, resubmit: finished tables are skipped.
set -euo pipefail
module load python_ML_packages/3.9.5-cpu
export SWED_WORK="$SLURM_SUBMIT_DIR"
cd "$SWED_WORK"
python3 subset.py "$@"
