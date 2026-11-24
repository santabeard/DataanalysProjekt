from pathlib import Path

from .metrics import (
    compute_metrics,
    get_labels,
    get_probability_draws,
)
from .plots import plot_calibration, plot_path, plot_roc


def evaluate_model(posterior_samples, X_test, y_test, out_path):
    """Write posterior metric summaries and target-appropriate evaluation plots."""
    labels, target_names = get_labels(y_test)
    probabilities = get_probability_draws(
        posterior_samples,
        X_test,
        n_observations=labels.shape[0],
        n_targets=labels.shape[1],
    )
    results = compute_metrics(labels, probabilities, target_names)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    posterior_mean = probabilities.mean(axis=0)
    multi_label = labels.shape[1] > 1
    if multi_label:
        calibration_target = "micro"
        calibration_y = labels.ravel()
        calibration_probabilities = posterior_mean.ravel()
    else:
        calibration_target = target_names[0]
        calibration_y = labels[:, 0]
        calibration_probabilities = posterior_mean[:, 0]

    calibration = results.loc[results["target"] == calibration_target].iloc[0]
    plot_calibration(
        calibration_y,
        calibration_probabilities,
        calibration["calibration_slope"],
        calibration["calibration_intercept"],
        plot_path(out_path, "_calibration"),
        title="Diagnosis micro calibration" if multi_label else "Mortality calibration",
    )
    plot_roc(
        labels,
        probabilities,
        results,
        target_names,
        plot_path(out_path, "_roc"),
    )
    return results
