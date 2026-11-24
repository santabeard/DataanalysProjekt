import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def get_labels(y_test):
    if isinstance(y_test, pd.DataFrame):
        labels = y_test.to_numpy(dtype=float)
        names = [str(column) for column in y_test.columns]
    elif isinstance(y_test, pd.Series):
        labels = y_test.to_numpy(dtype=float)
        names = [str(y_test.name or "outcome")]
    else:
        labels = np.asarray(y_test, dtype=float)
        names = []

    if labels.ndim == 1:
        labels = labels[:, None]
    if labels.ndim != 2 or not labels.size:
        raise ValueError("y_test must contain at least one binary target and observation.")
    if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("y_test must contain only finite binary labels (0 or 1).")
    if len(names) != labels.shape[1]:
        names = [f"target_{index}" for index in range(labels.shape[1])]
    return labels, names


def _positive_probability(scores, classes=None):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        return scores
    if scores.ndim != 2:
        raise ValueError("Each predict_proba result must be a vector or 2D array.")
    if scores.shape[1] == 1:
        if classes is not None and len(classes) == 1:
            return np.full(scores.shape[0], float(classes[0] == 1))
        return scores[:, 0]
    if classes is not None:
        positive = np.flatnonzero(np.asarray(classes) == 1)
        if len(positive) == 1:
            return scores[:, positive[0]]
    if scores.shape[1] == 2:
        return scores[:, 1]
    raise ValueError("Could not identify the positive-class probability column.")


def _model_probabilities(model, X_test, n_targets):
    predictions = model.predict_proba(X_test)
    if isinstance(predictions, (list, tuple)):
        classes = getattr(model, "classes_", None)
        if not isinstance(classes, (list, tuple)) or len(classes) != len(predictions):
            classes = [None] * len(predictions)
        return np.column_stack(
            [
                _positive_probability(output, class_values)
                for output, class_values in zip(predictions, classes)
            ]
        )

    predictions = np.asarray(predictions, dtype=float)
    if n_targets == 1:
        return _positive_probability(
            predictions, getattr(model, "classes_", None)
        )[:, None]
    if predictions.ndim == 2 and predictions.shape[1] == n_targets:
        return predictions
    raise ValueError(
        "For multi-label targets, each model must return one positive-class "
        "probability per target."
    )


def get_probability_draws(posterior_samples, X_test, n_observations, n_targets):
    if hasattr(posterior_samples, "predict_proba"):
        models = [posterior_samples]
    elif isinstance(posterior_samples, np.ndarray):
        models = None
    else:
        samples = list(posterior_samples)
        if not samples:
            raise ValueError("posterior_samples must contain at least one draw.")
        if all(hasattr(sample, "predict_proba") for sample in samples):
            models = samples
        else:
            models = None
            posterior_samples = np.asarray(samples, dtype=float)

    if models is not None:
        probabilities = np.stack(
            [_model_probabilities(model, X_test, n_targets) for model in models],
            axis=0,
        )
    else:
        probabilities = np.asarray(posterior_samples, dtype=float)
        if probabilities.ndim == 1:
            probabilities = probabilities[None, :, None]
        elif probabilities.ndim == 2:
            if probabilities.shape == (n_observations, n_targets):
                probabilities = probabilities[None, :, :]
            elif n_targets == 1 and probabilities.shape[1] == n_observations:
                probabilities = probabilities[:, :, None]
            else:
                raise ValueError(
                    "2D posterior probabilities must have shape "
                    "(draws, observations) or (observations, targets)."
                )

    if probabilities.ndim != 3 or probabilities.shape[1:] != (
        n_observations,
        n_targets,
    ):
        raise ValueError(
            "Posterior probabilities must have shape "
            f"(draws, {n_observations}, {n_targets})."
        )
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0).any() or (probabilities > 1).any()
    ):
        raise ValueError("Posterior probabilities must be finite values in [0, 1].")
    return probabilities


def calibration_fit(y, probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
    if np.unique(y).size != 2 or np.ptp(probabilities) == 0:
        return np.nan, np.nan

    logits = np.log(probabilities / (1 - probabilities))
    fit = LogisticRegression(C=1e6, solver="lbfgs").fit(logits[:, None], y)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def _draw_metrics(y, probabilities):
    events = int(y.sum())
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    aucs = np.full(probabilities.shape[0], np.nan)
    if 0 < events < len(y):
        aucs = np.asarray(
            [roc_auc_score(y, draw) for draw in probabilities], dtype=float
        )
    expected = probabilities.sum(axis=1)
    return {
        "roc_auc": aucs,
        "brier_score": np.mean((probabilities - y[None, :]) ** 2, axis=1),
        "negative_log_likelihood": -np.mean(
            y[None, :] * np.log(clipped)
            + (1 - y[None, :]) * np.log(1 - clipped),
            axis=1,
        ),
        "oe_ratio": np.divide(
            events,
            expected,
            out=np.full(expected.shape, np.inf if events else np.nan),
            where=expected != 0,
        ),
    }


def _mean_draws(values, axis=0):
    values = np.asarray(values, dtype=float)
    valid = ~np.isnan(values)
    counts = valid.sum(axis=axis)
    totals = np.where(valid, values, 0).sum(axis=axis)
    return np.divide(
        totals,
        counts,
        out=np.full(np.shape(totals), np.nan, dtype=float),
        where=counts > 0,
    )


def _metric_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if not values.size:
        return np.nan, np.nan, np.nan
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def _result_row(name, y, metrics, slope, intercept):
    row = {
        "target": name,
        "n": len(y),
        "events": int(y.sum()),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }
    for metric, values in metrics.items():
        median, lower, upper = _metric_summary(values)
        row[metric] = median
        row[f"{metric}_lower_95"] = lower
        row[f"{metric}_upper_95"] = upper
    return row


def compute_metrics(labels, probabilities, target_names):
    per_target = [
        _draw_metrics(labels[:, index], probabilities[:, :, index])
        for index in range(labels.shape[1])
    ]
    posterior_mean = probabilities.mean(axis=0)
    rows = []

    for index, name in enumerate(target_names):
        slope, intercept = calibration_fit(
            labels[:, index], posterior_mean[:, index]
        )
        rows.append(
            _result_row(name, labels[:, index], per_target[index], slope, intercept)
        )

    if labels.shape[1] > 1:
        macro = {
            metric: _mean_draws(
                np.stack([result[metric] for result in per_target]),
                axis=0,
            )
            for metric in per_target[0]
        }
        slopes, intercepts = zip(
            *[
                calibration_fit(labels[:, index], posterior_mean[:, index])
                for index in range(labels.shape[1])
            ]
        )
        macro_row = _result_row(
            "macro",
            labels[:, 0],
            macro,
            float(_mean_draws(slopes)),
            float(_mean_draws(intercepts)),
        )
        macro_row.update(events=np.nan, n_labels=labels.shape[1])
        rows.append(macro_row)

        micro = _draw_metrics(
            labels.ravel(), probabilities.reshape(probabilities.shape[0], -1)
        )
        slope, intercept = calibration_fit(labels.ravel(), posterior_mean.ravel())
        micro_row = _result_row(
            "micro",
            labels.ravel(),
            micro,
            slope,
            intercept,
        )
        micro_row["n_labels"] = labels.shape[1]
        rows.append(micro_row)

    return pd.DataFrame(rows)
