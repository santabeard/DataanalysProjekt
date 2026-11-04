#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 6                # one core per chain
#SBATCH -t 01:00:00
#SBATCH -J swed_joint
#SBATCH -o swed_joint_%j.log

# Submit from the folder holding 05_atlas_vitals.py and 06_joint_hierarchical.py:   sbatch 06_joint_hierarchical.sbatch
# Other vitals:   sbatch 06_joint_hierarchical.sbatch --vitals temp,sbp,pulse,spo2

set -euo pipefail

module load python          # CHANGE if pymc comes from another module or conda environment

cd "$SLURM_SUBMIT_DIR"
python -c "import pymc; print('pymc', pymc.__version__)"
# numpyro is much faster; without it, add  --sampler pymc  to the sbatch line
python 06_joint_hierarchical.py "$@"
