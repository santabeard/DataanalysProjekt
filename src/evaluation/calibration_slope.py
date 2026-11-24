#!/usr/bin/env python3
"""
calibration_slope.py -- check whether the Bayesian risk model's probabilities are on the right scale.

The model and its held-out predictions come from `evaluation.posterior`.  Regress the held-out
outcome on the predicted log-odds:

    logit P(y = 1) = a + b * logit(p)

b is the calibration slope: 1 means the risks are correctly scaled, b < 1 means they are too
extreme (over-confident), b > 1 too flat (under-confident).  a is the calibration intercept,
0 when the risks are right on average.

Usage
    PYTHONPATH=src python3 src/evaluation/calibration_slope.py
    PYTHONPATH=src python3 src/evaluation/calibration_slope.py --test-frac 0.4 --quick
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluation.posterior import REPO_ROOT, posterior_predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-frac", type=float, default=0.3, help="share of visits held out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--iter", type=int, default=4000)
    ap.add_argument("--burn", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="short sampler run, for a smoke test")
    ap.add_argument("--out", default=str(REPO_ROOT / "Results" / "evaluation" / "calibration_slope.csv"))
    a = ap.parse_args()
    if a.quick:
        a.iter, a.burn, a.chains = 200, 80, 1

    print(f"fitting the model, holding out {a.test_frac:.0%} of the visits")
    y, p = posterior_predictions(test_frac=a.test_frac, seed=a.seed, n_iter=a.iter,
                                 n_burn=a.burn, chains=a.chains)
    p = p.mean(axis=0)                      # posterior-mean risk of each held-out visit

    # calibration slope: regress the outcome on the predicted log-odds
    fit = LogisticRegression(C=1e6).fit(np.log(p / (1 - p))[:, None], y)   # C large = unpenalised
    slope, intercept = fit.coef_[0, 0], fit.intercept_[0]

    print(f"\ntest visits {len(y)}  events {int(y.sum())}  mean risk {p.mean():.3f} vs observed {y.mean():.3f}")
    print(f"calibration slope     {slope:.3f}   (1 = perfect, < 1 = over-confident)")
    print(f"calibration intercept {intercept:+.3f}   (0 = perfect)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(n_test=len(y), events=int(y.sum()), slope=slope, intercept=intercept,
                       pred_mean=p.mean(), obs_mean=y.mean())]).round(4).to_csv(out, index=False)
    print(f"\nwritten -> {out}")


if __name__ == "__main__":
    main()
