#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 00:45:00
#SBATCH -J swed_build_stats
#SBATCH -o swed_build_stats_%j.log

# Runs 02_build.py, then 03_export_stats.py only if the build succeeded.
# Subset (default):  sbatch build_and_stats.sbatch
# Full data:         sbatch -p node -N 1 -t 04:00:00 --export=ALL,SWED_DATA=/cygnus/proj/nobackup/swed_db/clean_parquet build_and_stats.sbatch

set -euo pipefail

module load python          # CHANGE to the same module line as in 01_subset.sbatch

export SWED_WORK="$SLURM_SUBMIT_DIR"
export SWED_DATA="${SWED_DATA:-$HOME/Desktop/data_std/subset}"
cd "$SWED_WORK"

python 02_build.py
python 03_export_stats.py --built "$HOME/Desktop/data_std/built_$(basename "$SWED_DATA")"
