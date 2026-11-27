#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 4                # the subset-sized train_raw, not a full raw table; pandas loads it in memory
#SBATCH -t 00:30:00
#SBATCH -J swed_train_clean
#SBATCH -o swed_train_clean_%j.log

# Submit from the folder holding 04_train_clean.py:   sbatch 04_train_clean.sbatch
# Follow progress:                                      tail -f swed_train_clean_<jobid>.log
#
# Requires 01_subset_v2.py's <N>K_train_raw.parquet to already have the
# vitals (02) and diagnosis/death (03) columns joined in. Unlike 02/03, this
# stage reads train_raw READ-ONLY and writes a NEW file,
# <N>K_train_clean.parquet -- train_raw is never modified, so this is safe
# to resubmit as often as you like.
#
# TODO(BLOCKED on Bianca data): run 03b_pruning_thresholds.py first to
# produce the two files below from real per-code frequency counts, so this
# isn't a structural dry run -- see 04_train_clean.py's module docstring.
#   --hdia-top-k-csv <path/to/<N>K_hdia_top_k.csv>
#   --complaint-vocab-csv <path/to/<N>K_complaint_vocab.csv>
# Fill in below once those files exist (03b writes them into --subset-dir):
HDIA_TOP_K_CSV=""        # e.g. "$SLURM_SUBMIT_DIR/500K_hdia_top_k.csv"
COMPLAINT_VOCAB_CSV=""   # e.g. "$SLURM_SUBMIT_DIR/500K_complaint_vocab.csv"

set -euo pipefail

module load python_ML_packages/3.9.5-cpu   # Python 3.9.5 + pandas 1.2.4 + pyarrow + numpy, no internet needed

export SWED_WORK="$SLURM_SUBMIT_DIR"
cd "$SWED_WORK"

python3 -c "import pandas, pyarrow, numpy; print('pandas', pandas.__version__, '| pyarrow', pyarrow.__version__)"

EXTRA_ARGS=()
if [ -n "$HDIA_TOP_K_CSV" ]; then
    EXTRA_ARGS+=(--hdia-top-k-csv "$HDIA_TOP_K_CSV")
fi
if [ -n "$COMPLAINT_VOCAB_CSV" ]; then
    EXTRA_ARGS+=(--complaint-vocab-csv "$COMPLAINT_VOCAB_CSV")
fi

python3 04_train_clean.py "${EXTRA_ARGS[@]}"
