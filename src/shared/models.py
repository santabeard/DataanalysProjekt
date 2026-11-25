

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


def build_model(X_train, y_train, method="laplace", prior_std=1.0):
    """
    Build the model specification for the chosen inference method.
    """
    if method == "bootstrap":
        # L2 penalty C = prior_std^2 acts as a Gaussian prior N(0, prior_std^2)
        return LogisticRegression(penalty="l2", C=prior_std ** 2, max_iter=1000)

    # Shared Bayesian log-posterior for Laplace and MCMC
    X_design = np.column_stack([np.ones(len(X_train)), np.asarray(X_train, dtype=float)])
    y = np.asarray(y_train, dtype=float)
    prior_prec = 1.0 / (prior_std ** 2)

    def neg_log_posterior(w):
        z = X_design @ w
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * prior_prec * np.dot(w, w)

    def grad(w):
        p = expit(X_design @ w)
        return X_design.T @ (p - y) + prior_prec * w

    return {
        "X": X_design,
        "y": y,
        "prior_prec": prior_prec,
        "neg_log_posterior": neg_log_posterior,
        "grad": grad,
    }
