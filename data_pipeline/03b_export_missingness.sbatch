#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 01:00:00
#SBATCH -J swed_missingness_stats
#SBATCH -o swed_missingness_%j.log

# Runs after 03_export_stats.py and writes into the same export_stats folder.
# Full data: sbatch -p node -N 1 -t 04:00:00 --export=ALL,SWED_DATA=/cygnus/proj/nobackup/swed_db/clean_parquet 03b_export_missingness.sbatch

set -euo pipefail

module load python

export SWED_WORK="$SLURM_SUBMIT_DIR"
export SWED_DATA="${SWED_DATA:-$HOME/Desktop/data_std/subset}"
cd "$SWED_WORK"

BUILT="$HOME/Desktop/data_std/built_$(basename "$SWED_DATA")"
OUT="$HOME/Desktop/data_std/export_stats_$(basename "$BUILT")"

python 03b_export_missingness.py --built "$BUILT" --out "$OUT"
