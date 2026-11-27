#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 4                # the UU tables are small (21 MB + 255 MB)
#SBATCH -t 01:00:00
#SBATCH -J uu_pipeline
#SBATCH -o uu_pipeline_%j.log

# Submit from the pipeline folder:   sbatch batch/uu_pipeline.sbatch 100000
# The argument is the patient count (default 100000). Runs uu_subset.py,
# uu_build_train_clean.py and check_train_clean.py in a row.
set -euo pipefail
module load python_ML_packages/3.9.5-cpu
cd "$SLURM_SUBMIT_DIR"

N="${1:-100000}"
LABEL=$(python3 -c "from subset import size_label; print(size_label($N))")

python3 uu_subset.py --n "$N"
python3 uu_build_train_clean.py --label "$LABEL"
TRAIN_DIR=$(python3 -c "from common import TRAIN; print(TRAIN)")
python3 check_train_clean.py "$TRAIN_DIR/${LABEL}_uu_train_clean.parquet"
