#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 02:00:00
#SBATCH -J swed_phase0
#SBATCH -o swed_phase0_%j.log

# Runs every phase0 script, 01-10, in order. set -e stops the chain on the
# first failure, so a broken script never lets a later one run on bad input.
#
# Submit from the folder holding these scripts (data_pipeline/scripts_v2/phase0/):
#
#   Subset (default):   sbatch run_phase0.sbatch
#   Full data:           sbatch -p node -N 1 -t 08:00:00 \
#                           --export=ALL,SWED_DATA=/cygnus/proj/nobackup/swed_db/clean_parquet run_phase0.sbatch
#   One script only:     sbatch --export=ALL,ONLY=03_question_b_vitals_timing.py run_phase0.sbatch
#   From a given step:   sbatch --export=ALL,FROM=06_units_table.py run_phase0.sbatch

set -euo pipefail

module load python          # CHANGE to the same module line as in scripts_v1/01_subset.sbatch

export SWED_WORK="$SLURM_SUBMIT_DIR"
export SWED_DATA="${SWED_DATA:-$HOME/Desktop/data_std/subset}"
export SWED_OUT="${SWED_OUT:-$HOME/Desktop/data_std}"
cd "$SWED_WORK"

echo "data: $SWED_DATA"
echo "out:  $SWED_OUT"
python -c "import polars, pyarrow; print('polars', polars.__version__, '| pyarrow', pyarrow.__version__)"

SCRIPTS=(
    01_map_vital_terms.py
    02_question_a_vitals_coverage.py
    03_question_b_vitals_timing.py
    04_question_c_missingness_bias.py
    05_question_d_priority_vitals.py
    06_units_table.py
    07_value_distributions.py
    08_consciousness_scales.py
    09_diagnosis_prevalence.py
    10_revisits_and_inexact_times.py
)

if [[ -n "${ONLY:-}" ]]; then
    SCRIPTS=("$ONLY")
elif [[ -n "${FROM:-}" ]]; then
    found=0
    REMAINING=()
    for script in "${SCRIPTS[@]}"; do
        [[ "$script" == "$FROM" ]] && found=1
        [[ $found -eq 1 ]] && REMAINING+=("$script")
    done
    [[ ${#REMAINING[@]} -eq 0 ]] && { echo "FROM=$FROM matched no script" >&2; exit 1; }
    SCRIPTS=("${REMAINING[@]}")
fi

if [[ ! -f "$SWED_WORK/out/vitals_concept_map.csv" ]]; then
    echo "NOTE: no reviewed $SWED_WORK/out/vitals_concept_map.csv yet -- 02-05 and 08 will use the raw regex guesses from 01"
fi

for script in "${SCRIPTS[@]}"; do
    echo "=== $(date '+%H:%M:%S') running $script ==="
    python "$script"
done

echo "=== $(date '+%H:%M:%S') phase0 done -- output under $SWED_OUT/explore/phase0/ ==="
