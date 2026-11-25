

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.utils import resample

from .models import build_model

def _fit_bootstrap(X_train, y_train, iter=1000, prior_std=1.0, seed=7):
    posterior_samples = []
    for i in range(iter):
        X_boot, y_boot = resample(X_train, y_train, random_state=seed + i)
        if len(np.unique(y_boot)) < 2:
            continue
        model = build_model(X_boot, y_boot, method="bootstrap", prior_std=prior_std)
        model.fit(X_boot, y_boot)
        weights = np.hstack([model.intercept_, model.coef_.ravel()])
        posterior_samples.append(weights)
    return np.array(posterior_samples)


def _fit_laplace(X_train, y_train, iter=4000, prior_std=1.0, seed=7):
    model = build_model(X_train, y_train, method="laplace", prior_std=prior_std)
    X = model["X"]
    n_params = X.shape[1]

    # 1. Find MAP estimate
    res = minimize(
        model["neg_log_posterior"],
        x0=np.zeros(n_params),
        jac=model["grad"],
        method="BFGS",
    )
    w_map = res.x

    # 2. Compute inverse Hessian (posterior covariance)
    p = expit(X @ w_map)
    W = p * (1 - p)
    H = (X.T * W) @ X + model["prior_prec"] * np.eye(n_params)
    cov = np.linalg.inv(H)

    # 3. Sample from N(w_map, cov)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=w_map, cov=cov, size=iter)


def _fit_mcmc(X_train, y_train, iter=4000, burn=1500, chains=2, prior_std=1.0, seed=7):
    """
    Random-walk Metropolis-Hastings MCMC using iter, burn, and chains.
    """
    model = build_model(X_train, y_train, method="laplace", prior_std=prior_std)
    n_params = model["X"].shape[1]
    neg_log_post = model["neg_log_posterior"]
    rng = np.random.default_rng(seed)

    step_size = 0.1 / np.sqrt(n_params)
    all_samples = []

    for _ in range(chains):
        w_curr = rng.normal(0, 0.1, size=n_params)
        lp_curr = -neg_log_post(w_curr)

        chain_samples = []
        for step in range(iter):
            w_prop = w_curr + rng.normal(0, step_size, size=n_params)
            lp_prop = -neg_log_post(w_prop)

            if np.log(rng.random()) < (lp_prop - lp_curr):
                w_curr, lp_curr = w_prop, lp_prop

            if step >= burn:
                chain_samples.append(w_curr.copy())

        all_samples.append(np.array(chain_samples))

    return np.vstack(all_samples)


def fit_model(
    X_train,
    y_train,
    method="laplace",
    iter=4000,
    burn=1500,
    chains=2,
    prior_std=1.0,
    seed=7,
):
    """Dispatch to the requested sampler and return posterior weight draws."""
    if method == "bootstrap":
        return _fit_bootstrap(
            X_train, y_train, iter=iter, prior_std=prior_std, seed=seed
        )
    if method == "laplace":
        return _fit_laplace(
            X_train, y_train, iter=iter, prior_std=prior_std, seed=seed
        )
    if method == "mcmc":
        return _fit_mcmc(
            X_train,
            y_train,
            iter=iter,
            burn=burn,
            chains=chains,
            prior_std=prior_std,
            seed=seed,
        )
    raise ValueError(
        f"Unknown method {method!r}; choose 'bootstrap', 'laplace', or 'mcmc'."
    )

