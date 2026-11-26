#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 8                # ~64 GB: 03_visits.py keeps the ~7.4M visits plus reduced vitals in memory
#SBATCH -t 06:00:00
#SBATCH -J swed_ingest
#SBATCH -o swed_ingest_%j.log

# Submit from the folder holding the scripts:   sbatch run_ingest.sbatch
# Follow progress:                              tail -f swed_ingest_<jobid>.log
# Edit SOURCE_DIR / FILE_PREFIX / OUT_DIR in ingest_common.py first.
# Each script can also be run on its own:       python3 03_visits.py

set -euo pipefail

module load python_ML_packages/3.9.5-cpu   # Python 3.9.5 + pandas 1.2.4 + pyarrow

cd "$SLURM_SUBMIT_DIR"
python3 -c "import pandas, pyarrow; print('pandas', pandas.__version__, '| pyarrow', pyarrow.__version__)"

python3 01_deaths.py
python3 02_patients.py
python3 03_visits.py
