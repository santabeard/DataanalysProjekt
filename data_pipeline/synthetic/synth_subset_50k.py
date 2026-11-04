#!/usr/bin/env python3
"""Synthetic version of subset/ for 50,000 patients. Same generator as synth_subset_5k.py.

    python synthetic/synth_subset_50k.py           # run from data_pipeline/ -> synthetic/synth_subset_50k/
    SWED_DATA=synthetic/synth_subset_50k SWED_WORK=synthetic/synth_subset_50k SWED_OUT=synthetic python 02_build.py
"""

from pathlib import Path

from synth_subset_5k import main

main(50_000, Path(__file__).resolve().parent / "synth_subset_50k")
