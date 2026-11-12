from collections.abc import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from sklearn.metrics import roc_auc_score


def fit_calibration_intercept_slope(y_true, prob_pred):
    p = np.clip(prob_pred, 1e-12, 1.0 - 1e-12)
    logit_p = np.log(p / (1.0 - p))
    design = sm.add_constant(logit_p)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            result = sm.GLM(
                y_true, design, family=sm.families.Binomial()
            ).fit()
        return float(result.params[0]), float(result.params[1])
    except (
        PerfectSeparationWarning,
        ValueError,
        np.linalg.LinAlgError,
        ZeroDivisionError,
    ):
        return np.nan, np.nan


def _format_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return "N/A"
    median, lower, upper = np.percentile(values, [50, 2.5, 97.5])
    return f"{median:.3f} [{lower:.3f}, {upper:.3f}]"


def evaluate_empirical_clinical_risks_df(
    Y_test,
    posterior_risks: Mapping[str, np.ndarray],
    diagnosis_names: Sequence[str],
    metadata_df=None,
    strata_cols=None,
    min_events=10,
):
    """Compare posterior draws shaped (draws, patients, diagnoses).

    For one diagnosis, (draws, patients) arrays are also accepted. Use NaN
    predictions for patients excluded from a fit, such as complete-case testing.
    Reported intervals summarize posterior draws, not a patient bootstrap.
    """
    Y = np.asarray(Y_test)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2 or not np.isfinite(Y).all() or not np.isin(Y, [0, 1]).all():
        raise ValueError("Y_test must be a binary array with shape (patients, diagnoses).")

    n_patients, n_diagnoses = Y.shape
    if n_patients == 0 or n_diagnoses == 0:
        raise ValueError("Y_test must contain at least one patient and diagnosis.")
    if len(diagnosis_names) != n_diagnoses:
        raise ValueError("diagnosis_names must have one name per outcome column.")
    if not posterior_risks:
        raise ValueError("posterior_risks must contain at least one fitted model.")
    if min_events < 1:
        raise ValueError("min_events must be at least 1.")

    model_draws = {}
    for model_name, risks in posterior_risks.items():
        draws = np.asarray(risks, dtype=float)
        if draws.ndim == 1:
            draws = draws[None, :]
        if n_diagnoses == 1 and draws.ndim == 2:
            draws = draws[:, :, None]
        if draws.ndim != 3 or draws.shape[1:] != (n_patients, n_diagnoses):
            raise ValueError(
                f"Posterior risks for {model_name!r} must have shape "
                f"(draws, {n_patients}, {n_diagnoses})."
            )
        if draws.shape[0] == 0:
            raise ValueError(f"Posterior risks for {model_name!r} must include draws.")
        if np.isinf(draws).any() or np.any(
            (draws[np.isfinite(draws)] < 0) | (draws[np.isfinite(draws)] > 1)
        ):
            raise ValueError(f"Posterior risks for {model_name!r} must be in [0, 1].")
        model_draws[model_name] = draws

    main_results = []
    subgroup_results = []

    for model_name, draws in model_draws.items():
        for k, diagnosis in enumerate(diagnosis_names):
            risks = draws[:, :, k]
            valid = np.isfinite(risks).all(axis=0)
            y = Y[valid, k]
            risks = np.clip(risks[:, valid], 1e-12, 1.0 - 1e-12)
            n = len(y)
            n_events = int(y.sum())
            prevalence = float(y.mean()) if n else np.nan
            enough_events = (
                n_events >= min_events and (n - n_events) >= min_events
            )

            if n:
                nll = -np.mean(
                    y[None, :] * np.log(risks)
                    + (1 - y[None, :]) * np.log1p(-risks),
                    axis=1,
                )
                brier = np.mean((risks - y[None, :]) ** 2, axis=1)
                base_brier = prevalence * (1.0 - prevalence)
                brier_skill = (
                    1.0 - brier / base_brier
                    if base_brier > 0
                    else np.full(len(risks), np.nan)
                )
                observed_expected = (
                    n_events / risks.sum(axis=1)
                    if n_events
                    else np.full(len(risks), np.nan)
                )
                auc = (
                    np.array([roc_auc_score(y, draw) for draw in risks])
                    if np.unique(y).size == 2
                    else np.full(len(risks), np.nan)
                )
                mean_risk = risks.mean(axis=0)
                _, calibration_slope = fit_calibration_intercept_slope(
                    y, mean_risk
                )
            else:
                nll = brier = brier_skill = observed_expected = auc = np.array([])
                calibration_slope = np.nan

            main_results.append(
                {
                    "Fit": model_name,
                    "Diagnosis": diagnosis,
                    "N": n,
                    "Events": n_events,
                    "Prevalence": f"{prevalence:.1%}" if n else "N/A",
                    "ROC-AUC": _format_interval(auc),
                    "Neg Log Likelihood": _format_interval(nll),
                    "Brier Skill Score": _format_interval(brier_skill),
                    "O:E Ratio": _format_interval(observed_expected),
                    "Calib Slope": (
                        f"{calibration_slope:.3f}"
                        if np.isfinite(calibration_slope)
                        else "N/A"
                    ),
                    "Note": (
                        ""
                        if enough_events
                        else f"Fewer than {min_events} events or non-events"
                    ),
                }
            )

        if metadata_df is not None and strata_cols is not None:
            if len(metadata_df) != n_patients:
                raise ValueError("metadata_df must have one row per test patient.")
            metadata = metadata_df.reset_index(drop=True)
            for col in strata_cols:
                if col not in metadata.columns:
                    continue
                for group, group_indices in metadata.groupby(col).groups.items():
                    group_mask = np.zeros(n_patients, dtype=bool)
                    group_mask[np.asarray(list(group_indices), dtype=int)] = True
                    for k, diagnosis in enumerate(diagnosis_names):
                        risk_draws = draws[:, :, k]
                        valid_group = group_mask & np.isfinite(risk_draws).all(axis=0)
                        y_group = Y[valid_group, k]
                        n_group = len(y_group)
                        events_group = int(y_group.sum())
                        if (
                            n_group < min_events
                            or events_group < min_events
                            or n_group - events_group < min_events
                        ):
                            continue
                        mean_risk = np.clip(
                            risk_draws[:, valid_group].mean(axis=0),
                            1e-12,
                            1.0 - 1e-12,
                        )
                        _, slope = fit_calibration_intercept_slope(
                            y_group, mean_risk
                        )
                        subgroup_results.append(
                            {
                                "Fit": model_name,
                                "Factor": col,
                                "Subgroup": group,
                                "Diagnosis": diagnosis,
                                "N": n_group,
                                "Events": events_group,
                                "Prevalence": f"{y_group.mean():.1%}",
                                "Neg Log Likelihood": round(
                                    -np.mean(
                                        y_group * np.log(mean_risk)
                                        + (1 - y_group) * np.log1p(-mean_risk)
                                    ),
                                    3,
                                ),
                                "O:E Ratio": round(
                                    events_group / mean_risk.sum(), 3
                                ),
                                "Calib Slope": round(slope, 3),
                                "ROC-AUC": round(
                                    roc_auc_score(y_group, mean_risk), 3
                                ),
                            }
                        )

    metrics_df = pd.DataFrame(main_results).set_index(["Fit", "Diagnosis"])
    subgroups_df = pd.DataFrame(subgroup_results)
    if not subgroups_df.empty:
        subgroups_df = subgroups_df.set_index(
            ["Fit", "Factor", "Subgroup", "Diagnosis"]
        )
    return {"metrics_df": metrics_df, "subgroups_df": subgroups_df}
