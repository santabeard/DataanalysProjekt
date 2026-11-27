#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 8                # the diagnosis join's pre-filter merge is the memory peak
#SBATCH -t 01:00:00
#SBATCH -J swed_train_clean
#SBATCH -o swed_train_clean_%j.log

# Submit from the SWED pipeline folder (src/core/data_pipeline/SWED/):
#   sbatch batch/SWED_build_train_clean.sbatch
# Reads <N>K_SUB_* from $SWED_INTERIM, writes <N>K_train_clean.parquet to $SWED_TRAIN
# (defaults: <repo>/data/interim and <repo>/data/train, see common.py).
set -euo pipefail
module load python_ML_packages/3.9.5-cpu
export SWED_WORK="$SLURM_SUBMIT_DIR"
cd "$SWED_WORK"
LABEL="${1:-500K}"   # e.g. sbatch batch/SWED_build_train_clean.sbatch 100K
python3 SWED_build_train_clean.py --label "$LABEL"
python3 SWED_check_train_clean.py "$LABEL"
