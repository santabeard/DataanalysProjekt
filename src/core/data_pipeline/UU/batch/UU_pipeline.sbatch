#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 4                # the UU tables are small (21 MB + 255 MB)
#SBATCH -t 01:00:00
#SBATCH -J uu_pipeline
#SBATCH -o uu_pipeline_%j.log

# Submit from the UU pipeline folder (src/core/data_pipeline/UU/):
#   sbatch batch/UU_pipeline.sbatch 100000
# The argument is the patient count (default 100000). Runs UU_subset.py,
# UU_build_train_clean.py and SWED_check_train_clean.py in a row.
set -euo pipefail
module load python_ML_packages/3.9.5-cpu
cd "$SLURM_SUBMIT_DIR"

# common.py lives in the pipeline root; the shared SWED modules in SWED/.
# Put both on PYTHONPATH so the inline `-c` helpers below resolve them too.
PIPELINE_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"
export PYTHONPATH="$PIPELINE_ROOT:$PIPELINE_ROOT/SWED:${PYTHONPATH:-}"

N="${1:-100000}"
LABEL=$(python3 -c "from SWED_subset import size_label; print(size_label($N))")

python3 UU_subset.py --n "$N"
python3 UU_build_train_clean.py --label "$LABEL"
TRAIN_DIR=$(python3 -c "from common import TRAIN; print(TRAIN)")
python3 "$PIPELINE_ROOT/SWED/SWED_check_train_clean.py" "$TRAIN_DIR/${LABEL}_uu_train_clean.parquet"
