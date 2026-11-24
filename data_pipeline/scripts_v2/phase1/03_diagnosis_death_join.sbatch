#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 8                # bumped from 4: the diagnosis-join merge builds a large
                             # pre-filter intermediate (~1.7M visits x ~12M patreg rows,
                             # joined on lopnr_swed, before the 30-day window cuts it down)
#SBATCH -t 01:00:00
#SBATCH -J swed_diag_death_join
#SBATCH -o swed_diag_death_join_%j.log

# Submit from the folder holding 03_diagnosis_death_join.py:   sbatch 03_diagnosis_death_join.sbatch
# Follow progress:                                               tail -f swed_diag_death_join_<jobid>.log

set -euo pipefail

module load python_ML_packages/3.9.5-cpu

export SWED_WORK="$SLURM_SUBMIT_DIR"
cd "$SWED_WORK"

python3 -c "import pandas, pyarrow, numpy; print('pandas', pandas.__version__, '| pyarrow', pyarrow.__version__)"
python3 03_diagnosis_death_join.py --subset-dir "$SWED_WORK/out_data/subset"
