#!/usr/bin/env python3
"""Synthetic version of subset/ for 50,000 patients. Same generator as synth_subset_5k.py.

    python src/core/data_pipeline/synthetic/synth_subset_50k.py
    SWED_DATA=Data/local/synth/synth_subset_50k SWED_OUT=Results/data_pipeline python src/core/data_pipeline/02_build.py
"""

from pathlib import Path

from synth_subset_5k import main

REPO_ROOT = Path(__file__).resolve().parents[4]
main(50_000, REPO_ROOT / "Data" / "local" / "data_pipeline" / "synth_subset_50k")
