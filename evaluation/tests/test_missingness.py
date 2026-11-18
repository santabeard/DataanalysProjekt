import numpy as np
import pandas as pd

from evaluation.metrics import calibration_table
from preprocessing.bayes_missing import NORMAL_Q10_Q90, prepare, summarize_imputations


def test_robust_imputation_sd_uses_quantiles():
    stats = np.zeros((2, 50, 1, 5))
    stats[..., 0] = 0.5
    stats[..., 1] = 25.0              # one inflated draw must not dominate
    stats[..., 2] = -1.2815515655
    stats[..., 3] = 0.0
    stats[..., 4] = 1.2815515655
    out = summarize_imputations(stats)
    assert np.isclose(out[0, 0], 0.5)
    assert np.isclose(out[0, 1], 2.563103131 / NORMAL_Q10_Q90)
    assert np.isclose(out[0, 1], 1.0)


def test_summarize_imputations_skips_all_missing_vital():
    stats = np.full((1, 10, 2, 5), np.nan)
    stats[:, :, 0, :] = [1.0, 1.0, -1.0, 0.0, 1.0]
    out = summarize_imputations(stats)
    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()


def test_prepare_returns_selection_covariates():
    df = pd.DataFrame({
        "age": [40.0, 50.0, 60.0, 70.0],
        "sex": [0.0, 1.0, 0.0, 1.0],
        "hosp": [0.0, 1.0, 0.0, 1.0],
        "v": [1.0, np.nan, 3.0, 4.0],
        "y": [0.0, 0.0, 1.0, 1.0],
    })
    data = prepare(df, ["age", "sex"], ["v"], "y", selection_cols=["hosp"])
    assert data["S"].shape == (4, 1)
    assert np.array_equal(data["S"][:, 0], df["hosp"].to_numpy())


def test_calibration_table_is_reproducible_and_ordered():
    rng = np.random.default_rng(0)
    n = 400
    p = rng.uniform(0.005, 0.05, n)
    y = rng.binomial(1, p)
    pid = np.repeat(np.arange(100), 4)
    a = calibration_table(y, p, pid, bins=10, B=50, seed=3)
    b = calibration_table(y, p, pid, bins=10, B=50, seed=3)
    assert a["lo"].tolist() == b["lo"].tolist()
    assert ((a["lo"] <= a["obs"]) & (a["obs"] <= a["hi"])).all()
