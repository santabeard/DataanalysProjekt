import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import evaluator
import evaluation.calibration_slope as calibration_slope_module
from evaluation.plots import calibration_bins, plot_calibration


class EvaluateModelTests(unittest.TestCase):
    def test_fitted_posterior_models_are_scored_via_predict_proba(self):
        labels = pd.Series([0, 0, 1, 1], name="mortality")

        class FixedClassifier:
            classes_ = np.array([0, 1])

            def __init__(self, probabilities):
                self.probabilities = probabilities

            def predict_proba(self, X):
                return np.column_stack(
                    [1 - self.probabilities, self.probabilities]
                )

        models = [
            FixedClassifier(np.array([0.1, 0.3, 0.6, 0.8])),
            FixedClassifier(np.array([0.2, 0.4, 0.7, 0.9])),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluator.evaluate_model(
                models,
                np.zeros((len(labels), 1)),
                labels,
                Path(temp_dir) / "metrics.csv",
            )

        self.assertAlmostEqual(result.loc[0, "roc_auc"], 1.0)

    def test_binary_posteriors_return_metrics_and_plots(self):
        labels = pd.Series([0, 0, 1, 1], name="mortality")
        probabilities = np.array(
            [
                [0.1, 0.4, 0.35, 0.8],
                [0.2, 0.3, 0.45, 0.9],
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "metrics.csv"
            result = evaluator.evaluate_model(
                probabilities,
                np.zeros((len(labels), 1)),
                labels,
                output,
            )

            self.assertEqual(result.loc[0, "target"], "mortality")
            self.assertAlmostEqual(result.loc[0, "roc_auc"], 0.875)
            expected_oe = np.median(2 / probabilities.sum(axis=1))
            self.assertAlmostEqual(result.loc[0, "oe_ratio"], expected_oe)
            self.assertTrue(output.is_file())
            self.assertTrue(Path(temp_dir, "metrics_calibration.png").is_file())
            self.assertTrue(Path(temp_dir, "metrics_roc.png").is_file())

    def test_multilabel_posteriors_produce_micro_macro_metrics_and_roc_plot(self):
        labels = pd.DataFrame(
            {
                "dx_a": [0, 0, 1, 1, 0, 1],
                "dx_b": [0, 1, 0, 1, 1, 0],
            }
        )
        probabilities = np.array(
            [
                [
                    [0.1, 0.2],
                    [0.3, 0.7],
                    [0.6, 0.4],
                    [0.8, 0.8],
                    [0.2, 0.6],
                    [0.7, 0.3],
                ],
                [
                    [0.2, 0.1],
                    [0.4, 0.6],
                    [0.7, 0.3],
                    [0.9, 0.9],
                    [0.1, 0.7],
                    [0.8, 0.2],
                ],
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "diagnosis.csv"
            result = evaluator.evaluate_model(
                probabilities,
                np.zeros((len(labels), 1)),
                labels,
                output,
            )

            self.assertEqual(set(result["target"]), {"dx_a", "dx_b", "macro", "micro"})
            self.assertTrue(result.loc[result["target"] == "macro", "roc_auc"].notna().all())
            self.assertTrue(result.loc[result["target"] == "micro", "roc_auc"].notna().all())
            self.assertTrue(Path(temp_dir, "diagnosis_roc.png").is_file())

    def test_standalone_calibration_slope_saves_plot(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        probabilities = np.array(
            [
                [0.1, 0.2, 0.3, 0.6, 0.7, 0.8],
                [0.15, 0.25, 0.35, 0.55, 0.75, 0.85],
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "calibration.csv"
            result = calibration_slope_module.calibration_slope(
                labels,
                probabilities,
                out=output,
            )

            self.assertIn("slope", result.columns)
            self.assertTrue(output.is_file())
            self.assertTrue(
                Path(temp_dir, "calibration_calibration.png").is_file()
            )
    def test_calibration_bins_count_observations_events_and_write_plot(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        probabilities = np.array([0.05, 0.1, 0.2, 0.5, 0.7, 0.9])

        grouped, edges = calibration_bins(labels, probabilities, bins=2)

        self.assertEqual(int(grouped["n"].sum()), len(labels))
        self.assertEqual(int(grouped["events"].sum()), int(labels.sum()))
        self.assertEqual(len(edges), 3)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "calibration_bins.png"
            plot_calibration(labels, probabilities, 1.0, 0.0, output, "Test calibration")
            self.assertTrue(output.is_file())

    def test_build_scoreboards_writes_all_three_views(self):
        rng = np.random.default_rng(0)
        n = 40
        df = pd.DataFrame({
            "age": np.arange(n, dtype=float),
            "body_temp": np.where(rng.random(n) < 0.3, np.nan, rng.normal(37, 0.5, n)),
            "died_30d": np.arange(n) % 2,
            "hospital": ["a" if i % 2 == 0 else "b" for i in range(n)],
        })
        idx = np.arange(n)
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "metrics.csv"
            boards = evaluator.build_scoreboards(
                df, ["age", "body_temp"], "died_30d", out, idx[:30], idx[30:],
                method="laplace", iter=20, burn=5, chains=1, seed=7,
            )
            for column in (
                "AUC as recorded", "Brier score as recorded",
                "Negative log likelihood as recorded", "O:E ratio as recorded",
            ):
                self.assertIn(column, boards["scoreboard"].columns)
            self.assertEqual(len(boards["by_strategy"]), 15)
            self.assertEqual(set(boards["by_hospital"]["Hospital"]), {"a", "b"})
            for path in boards["paths"].values():
                self.assertTrue(Path(path).is_file())


if __name__ == "__main__":
    unittest.main()
