#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 4                # ~32 GB RAM; the script streams, so no fat node needed
#SBATCH -t 04:00:00
#SBATCH -J swed_subset
#SBATCH -o swed_subset_%j.log

# Submit from the folder holding 01_subset.py:   sbatch 01_subset.sbatch
# Follow progress:                               tail -f swed_subset_<jobid>.log
# If it times out, resubmit: finished tables are skipped.

set -euo pipefail

module load python          # CHANGE to the module(s) you load for swed_tasks.ipynb (needs pyarrow + numpy)

export SWED_WORK="$SLURM_SUBMIT_DIR"
cd "$SWED_WORK"

python -c "import pyarrow, numpy; print('pyarrow', pyarrow.__version__, '| numpy', numpy.__version__)"
python 01_subset.py --n 5000
