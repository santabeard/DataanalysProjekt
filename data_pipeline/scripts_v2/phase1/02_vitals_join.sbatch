#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 4                # the subset, not the full 55M/16M-row raw tables; pandas loads it in memory
#SBATCH -t 01:00:00
#SBATCH -J swed_vitals_join
#SBATCH -o swed_vitals_join_%j.log

# Submit from the folder holding 02_vitals_join.py:   sbatch 02_vitals_join.sbatch
# Follow progress:                                     tail -f swed_vitals_join_<jobid>.log
# Requires 01_subset_v2.py to have already produced <N>K_train_raw.parquet
# and the <N>K_SUB_ehr_karolinska_meas / <N>K_SUB_ehr_danderyd_lab_meas files
# in the same $SWED_OUT/subset. Rerunning is safe: 02_vitals_join.py drops its
# own columns from a previous run before rejoining (see its docstring).

set -euo pipefail

module load python_ML_packages/3.9.5-cpu   # Python 3.9.5 + pandas 1.2.4 + pyarrow + numpy, no internet needed

export SWED_WORK="$SLURM_SUBMIT_DIR"
cd "$SWED_WORK"

python3 -c "import pandas, pyarrow, numpy; print('pandas', pandas.__version__, '| pyarrow', pyarrow.__version__)"
python3 02_vitals_join.py
