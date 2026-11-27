#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 4                # ~32 GB RAM; the script streams, so no fat node needed
#SBATCH -t 04:00:00
#SBATCH -J swed_subset_v2
#SBATCH -o swed_subset_v2_%j.log

# Submit from the folder holding 01_subset_v2.py:   sbatch 01_subset_v2.sbatch
# Follow progress:                                  tail -f swed_subset_v2_<jobid>.log
# If it times out, resubmit: finished tables are skipped.

set -euo pipefail

module load python_ML_packages/3.9.5-cpu   # Python 3.9.5 + pyarrow + numpy already bundled, no internet needed

export SWED_WORK="$SLURM_SUBMIT_DIR"
cd "$SWED_WORK"

python3 -c "import pyarrow, numpy; print('pyarrow', pyarrow.__version__, '| numpy', numpy.__version__)"
python3 01_subset_v2.py --n 500000
