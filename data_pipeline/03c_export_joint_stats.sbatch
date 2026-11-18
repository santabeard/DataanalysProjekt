#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 01:00:00
#SBATCH -J swed_joint_stats
#SBATCH -o swed_joint_%j.log

set -euo pipefail

module load python

export SWED_WORK="$SLURM_SUBMIT_DIR"
export SWED_DATA="${SWED_DATA:-$HOME/Desktop/data_std/subset}"
cd "$SWED_WORK"

BUILT="$HOME/Desktop/data_std/built_$(basename "$SWED_DATA")"
OUT="$HOME/Desktop/data_std/export_stats_$(basename "$BUILT")"

python 03c_export_joint_stats.py --built "$BUILT" --out "$OUT"
