from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import calibration_fit, get_labels, get_probability_draws
from .plots import plot_calibration, plot_path


def calibration_slope(
    y_test,
    posterior_samples,
    out="Results/calibration.csv",
    X_test=None,
):
    """Plot calibration from posterior risk draws produced by the main pipeline."""
    labels, target_names = get_labels(y_test)
    draws = get_probability_draws(
        posterior_samples,
        X_test,
        n_observations=labels.shape[0],
        n_targets=labels.shape[1],
    )
    multi_label = labels.shape[1] > 1
    y = labels.ravel() if multi_label else labels[:, 0]
    probabilities = draws.mean(axis=0).ravel() if multi_label else draws.mean(axis=0)[:, 0]
    slope, intercept = calibration_fit(y, probabilities)
    result = pd.DataFrame(
        [
            {
                "target": "micro" if multi_label else target_names[0],
                "n_test": len(y),
                "events": int(y.sum()),
                "slope": slope,
                "intercept": intercept,
                "pred_mean": probabilities.mean(),
                "obs_mean": y.mean(),
            }
        ]
    ).round(4)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    plot_calibration(
        y,
        probabilities,
        slope,
        intercept,
        plot_path(out, "_calibration"),
        title="Diagnosis micro calibration" if multi_label else f"{target_names[0]} calibration",
    )
    print(f"\nwritten -> {out}")
    print(f"calibration plot -> {plot_path(out, '_calibration')}")
    return result
