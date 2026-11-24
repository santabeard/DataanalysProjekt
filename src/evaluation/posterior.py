#!/usr/bin/env python3
"""
posterior.py -- fit the Bayesian risk model once and hand back posterior predictions on held-out
visits, as draws, so any evaluation metric can be computed draw by draw.

    from evaluation.posterior import posterior_predictions

    y, p = posterior_predictions()          # p has one row per posterior draw
    aucs = np.array([roc_auc_score(y, p[i]) for i in range(len(p))])
    print(aucs.mean(), np.quantile(aucs, [0.025, 0.975]))

Vitals missing on the held-out rows are filled with the training medians; the sampler's own
imputation only covers the rows it was fitted on.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer

from core.missingness_detection import bayes_missing as bm

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = ["age", "sex", "comorbidity"]
VITALS = ["heart_rate", "resp_rate", "spo2", "temperature", "sbp", "dbp",
          "glucose", "lactate", "creatinine", "wbc", "sodium"]
LOG_VARS = ("glucose", "lactate", "creatinine", "wbc")


def posterior_predictions(test_frac=0.3, seed=7, n_iter=4000, n_burn=1500, chains=2,
                          data_dir=None, verbose=False):
    """Fit the model on (1 - test_frac) of the visits and predict the rest.

    Returns
    -------
    y : (n_test,)           observed outcomes of the held-out visits
    p : (n_draws, n_test)   predicted risk of each held-out visit, one row per posterior draw
    """
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / "Data" / "missingness_detection"
    df = pd.read_csv(data_dir / "synthetic_ehr.csv")
    data = bm.prepare(df, BASE, VITALS, "outcome", LOG_VARS)

    held = np.random.default_rng(seed).random(len(data["y"])) < test_frac
    train = dict(data, Z=data["Z"][~held], X=data["X"][~held], y=data["y"][~held])

    beta = bm.fit(train, chains=chains, n_iter=n_iter, n_burn=n_burn, seed=seed,
                  verbose=verbose)["beta"]
    beta = beta.reshape(-1, 1 + len(BASE) + len(VITALS))    # one row per posterior draw

    X = SimpleImputer(strategy="median").fit(train["X"]).transform(data["X"][held])
    design = np.column_stack([np.ones(held.sum()), data["Z"][held], X])
    return data["y"][held], expit(design @ beta.T).T
