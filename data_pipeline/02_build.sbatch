#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 2                # the subset build peaks well below 16 GB; use -p node for the full data
#SBATCH -t 00:30:00
#SBATCH -J swed_build
#SBATCH -o swed_build_%j.log

# Submit from the folder holding 02_build.py, data_info/ and out/:   sbatch 02_build.sbatch
# Full data instead of the subset:   sbatch -p node -N 1 -t 04:00:00 --export=ALL,SWED_DATA=/cygnus/proj/nobackup/swed_db/clean_parquet 02_build.sbatch

set -euo pipefail

module load python          # CHANGE to the same module line as in 01_subset.sbatch

export SWED_WORK="$SLURM_SUBMIT_DIR"
export SWED_DATA="${SWED_DATA:-$HOME/Desktop/data_std/subset}"
cd "$SWED_WORK"

python -c "import pandas, pyarrow; print('pandas', pandas.__version__, '| pyarrow', pyarrow.__version__)"
python 02_build.py
