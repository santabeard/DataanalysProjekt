import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import src.orchestrator as orchestrator
from src.orchestrator import load_data


class LoadDataTests(unittest.TestCase):
    def test_uses_latest_day_when_unspecified(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_dir = Path.cwd()
            try:
                os.chdir(temp_dir)
                data_dir = Path("Data/synth")
                data_dir.mkdir(parents=True)
                pd.DataFrame({"value": [1]}).to_csv(data_dir / "10K_7_2025-01-01.csv", index=False)
                pd.DataFrame({"value": [2]}).to_csv(data_dir / "10K_7_2025-01-02.csv", index=False)

                loaded = load_data(n_patients=10000, seed=7)

                self.assertEqual(loaded.loc[0, "value"], 2)
            finally:
                os.chdir(original_dir)

    def test_uses_requested_day_when_provided(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_dir = Path.cwd()
            try:
                os.chdir(temp_dir)
                data_dir = Path("Data/synth")
                data_dir.mkdir(parents=True)
                pd.DataFrame({"value": [3]}).to_csv(data_dir / "10K_7_2025-01-01.csv", index=False)

                loaded = load_data(n_patients=10000, seed=7, day=date(2025, 1, 1))

                self.assertEqual(loaded.loc[0, "value"], 3)
            finally:
                os.chdir(original_dir)

    def test_generates_data_when_no_matching_file_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_dir = Path.cwd()
            try:
                os.chdir(temp_dir)
                data_dir = Path("Data/synth")
                data_dir.mkdir(parents=True)

                def generate(n_patients, seed, out, day=None, **kwargs):
                    pd.DataFrame({"value": [4]}).to_csv(
                        out / orchestrator.generate_data.file_name(n_patients, seed, day), index=False
                    )

                with patch.object(orchestrator.generate_data, "generate", generate):
                    loaded = load_data(n_patients=10000, seed=7)

                self.assertEqual(loaded.loc[0, "value"], 4)
            finally:
                os.chdir(original_dir)


if __name__ == "__main__":
    unittest.main()
