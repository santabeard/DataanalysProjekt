from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import roc_auc_score

from shared.sampling import fit_model

from .metrics import (
    calibration_fit,
    compute_metrics,
    get_labels,
    get_probability_draws,
)
from .plots import plot_calibration, plot_path, plot_roc

LEVELS = [0.0, 0.25, 0.5]
METRICS = ["AUC", "Brier score", "Negative log likelihood", "O:E ratio"]
STRATEGIES = [
    ("median", "Median imputation"),
    ("mean", "Mean imputation"),
    ("mice", "MICE imputation"),
    ("mice_flags", "MICE imputation + missing flags"),
    ("complete_case", "Complete-case"),
]


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


def _point_metrics(y, probabilities):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1 - 1e-12)
    if not len(y):
        return {"n": 0, "events": 0, "AUC": np.nan, "Brier score": np.nan,
                "Negative log likelihood": np.nan, "O:E ratio": np.nan, "Calibration slope": np.nan}
    slope, _ = calibration_fit(y, p)
    expected = p.sum()
    return {
        "n": len(y),
        "events": int(y.sum()),
        "AUC": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
        "Brier score": float(np.mean((p - y) ** 2)),
        "Negative log likelihood": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "O:E ratio": float(y.sum() / expected) if expected > 0 and y.sum() > 0 else np.nan,
        "Calibration slope": slope,
    }


def _prepare_train(key, X_train, y_train, seed):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if key == "complete_case":
        keep = ~np.isnan(X).any(axis=1)
        return X[keep], y[keep], None
    if key in ("median", "mean"):
        imputer = SimpleImputer(strategy=key)
        return imputer.fit_transform(X), y, imputer
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=seed)
    fitted = imputer.fit_transform(X)
    if key == "mice_flags":
        fitted = np.column_stack([fitted, np.isnan(X).astype(float)])
    return fitted, y, (imputer, key == "mice_flags")


def _transform_test(key, prepared, X_test):
    X = np.asarray(X_test, dtype=float)
    if key == "complete_case":
        keep = ~np.isnan(X).any(axis=1)
        return X[keep], keep
    if key in ("median", "mean"):
        return prepared.transform(X), np.ones(len(X), dtype=bool)
    imputer, with_flags = prepared
    transformed = imputer.transform(X)
    if with_flags:
        transformed = np.column_stack([transformed, np.isnan(X).astype(float)])
    return transformed, np.ones(len(X), dtype=bool)


def build_scoreboards(
    data_df,
    features,
    target,
    out_path,
    idx_train,
    idx_test,
    method="laplace",
    iter=4000,
    burn=1500,
    chains=2,
    prior_std=1.0,
    seed=7,
    group_col="hospital",
):
    X = data_df[features].apply(pd.to_numeric, errors="coerce").astype(float)
    y = data_df[target].to_numpy(dtype=float)
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    groups = data_df[group_col].astype(str).to_numpy()[idx_test]

    rng = np.random.default_rng(seed + 1)
    deleted = []
    for level in LEVELS:
        Xd = X_test.to_numpy(copy=True)
        if level:
            Xd[~np.isnan(Xd) & (rng.random(Xd.shape) < level)] = np.nan
        deleted.append((level, Xd))

    model_name = f"{method} logistic regression"
    rows, hospital_rows = [], []
    for key, label in STRATEGIES:
        X_fit, y_fit, prepared = _prepare_train(key, X_train, y_train, seed)
        posterior = fit_model(
            X_fit, y_fit, method=method, iter=iter, burn=burn, chains=chains,
            prior_std=prior_std, seed=seed,
        )
        for level, Xd in deleted:
            Xt, keep = _transform_test(key, prepared, Xd)
            yt, ht = y_test[keep], groups[keep]
            probabilities = get_probability_draws(
                posterior, Xt, n_observations=len(yt), n_targets=1
            ).mean(axis=0)[:, 0]
            metrics = _point_metrics(yt, probabilities)
            metrics.update({
                "Prediction model": model_name,
                "Missing-data handling": label,
                "Key": key,
                "Deletion": level,
            })
            rows.append(metrics)
            for hospital in np.unique(ht):
                subset = ht == hospital
                hospital_metrics = _point_metrics(yt[subset], probabilities[subset])
                hospital_metrics.update({
                    "Hospital": hospital,
                    "Prediction model": model_name,
                    "Missing-data handling": label,
                    "Key": key,
                    "Deletion": level,
                })
                hospital_rows.append(hospital_metrics)

    long_cols = ["Prediction model", "Missing-data handling", "Key", "Deletion", "n", "events"] + METRICS + ["Calibration slope"]
    by_strategy = pd.DataFrame(rows)[long_cols]
    by_hospital = pd.DataFrame(hospital_rows)[["Hospital"] + long_cols]

    lookup = by_strategy.set_index(["Key", "Deletion"])
    get = lambda key, level, col: float(lookup.loc[(key, level), col])
    wide_rows = []
    for key, label in STRATEGIES:
        row = {"Prediction model": model_name, "Missing-data handling": label, "Key": key}
        for metric in METRICS:
            row[f"{metric} as recorded"] = get(key, 0.0, metric)
            row[f"{metric}, 50% more deleted"] = get(key, 0.5, metric)
        row["AUC drop"] = get(key, 0.5, "AUC") - get(key, 0.0, "AUC")
        row["Calibration slope (1 = ideal)"] = get(key, 0.0, "Calibration slope")
        row["n"] = int(get(key, 0.0, "n"))
        row["events"] = int(get(key, 0.0, "events"))
        wide_rows.append(row)
    scoreboard = pd.DataFrame(wide_rows)
    scoreboard["Rank"] = pd.Series(pd.NA, index=scoreboard.index, dtype="Int64")
    rankable = scoreboard["events"] >= 5
    scoreboard.loc[rankable, "Rank"] = scoreboard.loc[rankable, "Negative log likelihood as recorded"].rank(method="min").astype("Int64")
    wide_cols = (
        ["Rank", "Prediction model", "Missing-data handling"]
        + [f"{metric} as recorded" for metric in METRICS]
        + [f"{metric}, 50% more deleted" for metric in METRICS]
        + ["AUC drop", "Calibration slope (1 = ideal)", "n", "events", "Key"]
    )
    scoreboard = scoreboard[wide_cols].sort_values("Rank").reset_index(drop=True)

    out_path = Path(out_path)
    paths = {
        "scoreboard": out_path.with_name("orchestrator_scoreboard.csv"),
        "by_strategy": out_path.with_name("orchestrator_scoreboard_by_strategy.csv"),
        "by_hospital": out_path.with_name("orchestrator_scoreboard_by_hospital.csv"),
    }
    scoreboard.round(4).to_csv(paths["scoreboard"], index=False)
    by_strategy.round(4).to_csv(paths["by_strategy"], index=False)
    by_hospital.round(4).to_csv(paths["by_hospital"], index=False)
    return {"scoreboard": scoreboard, "by_strategy": by_strategy, "by_hospital": by_hospital, "paths": paths}
