import argparse
import pathlib as pl
import random
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core.data_pipeline.synthetic import generate_data
from evaluation.evaluator import build_scoreboards, evaluate_model
from shared.sampling import fit_model

DATA_DIR = pl.Path("Data/synth")
FEATURES = ["age", "male", "body_temp", "systolic_bp", "pulse_rate", "n_vitals"]


def _log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def load_data(n_patients=None, seed=None, day=None):
    """Load generated data, generating it first if the matching CSV is missing."""
    if day is None:
        pattern = f"{generate_data.size_label(n_patients)}_{seed}_*.csv"
        matches = sorted(DATA_DIR.glob(pattern))
        if not matches:
            generate_data.generate(n_patients, seed, out=DATA_DIR)
            matches = sorted(DATA_DIR.glob(pattern))
        path = matches[-1]
    else:
        path = DATA_DIR / generate_data.file_name(n_patients, seed, day)
        if not path.is_file():
            generate_data.generate(n_patients, seed, out=DATA_DIR, day=day)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_orchestrator():
    """
    Run the orchestrator.
    """
    parser = argparse.ArgumentParser(description="Run the orchestrator.")
    parser.add_argument(
        "--method",
        type=str,
        choices=["bootstrap", "laplace", "mcmc"],
        default="laplace",
        help="Inference method for Bayesian logistic regression.",
    )
    parser.add_argument("--prior-std", type=float, default=1.0, help="Gaussian prior standard deviation.")
    parser.add_argument("--test-frac", type=float, default=0.3, help="Share of visits held out for testing.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--iter", type=int, default=4000, help="Number of iterations for the sampler.")
    parser.add_argument("--burn", type=int, default=1500, help="Number of burn-in iterations for the sampler.")
    parser.add_argument("--chains", type=int, default=2, help="Number of chains for the sampler.")
    parser.add_argument("--n-patients", type=int, default=15000, help="Number of patients to generate/load.")
    parser.add_argument(
        "--timestamp",
        type=date.fromisoformat,
        default=None,
        help="Data date (YYYY-MM-DD) to generate/load. If omitted, the latest date is used.",
    )
    parser.add_argument("--target", type=str, default="died_30d", help="Target variable for prediction.")
    parser.add_argument("--quick", action="store_true", help="Run a quick test with fewer iterations.")
    parser.add_argument(
        "--out",
        default=str(pl.Path("Results") / "evaluation" / "calibration_slope.csv"),
        help="Output file path for calibration slope results.",
    )

    args = parser.parse_args()
    started = time.perf_counter()
    _log(
        f"start n_patients={args.n_patients} seed={args.seed} target={args.target} "
        f"method={args.method}"
    )

    if args.quick:
        args.iter, args.burn, args.chains = 200, 80, 1
        _log(f"quick mode iter={args.iter} burn={args.burn} chains={args.chains}")

    random.seed(args.seed)

    _log("loading or generating synthetic data")
    data_df = load_data(n_patients=args.n_patients, seed=args.seed, day=args.timestamp)
    _log(f"loaded data rows={len(data_df):,} positives={int(data_df[args.target].sum())}")
    X = data_df[FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
    y = data_df[args.target]
    stratify = y if y.value_counts().min() >= 2 else None
    idx_train, idx_test = train_test_split(
        np.arange(len(X)), test_size=args.test_frac, random_state=args.seed,
        stratify=stratify,
    )
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    _log(
        f"split train n={len(idx_train):,} events={int(y_train.sum())} "
        f"test n={len(idx_test):,} events={int(y_test.sum())}"
    )
    train_median = X_train.median()
    X_train = X_train.fillna(train_median).fillna(0)
    X_test = X_test.fillna(train_median).fillna(0)

    _log(f"fitting {args.method} logistic regression")
    fit_started = time.perf_counter()
    posterior_samples = fit_model(
        X_train,
        y_train,
        method=args.method,
        iter=args.iter,
        burn=args.burn,
        chains=args.chains,
        prior_std=args.prior_std,
        seed=args.seed,
    )
    _log(
        f"fit done draws={len(posterior_samples)} "
        f"elapsed={time.perf_counter() - fit_started:.1f}s"
    )

    _log("evaluating held-out probabilities and writing plots")
    results = evaluate_model(posterior_samples, X_test, y_test, out_path=args.out)
    _log(f"wrote {args.out}")

    _log("building missingness scoreboards")
    build_scoreboards(
        data_df, FEATURES, args.target, args.out, idx_train, idx_test,
        method=args.method, iter=args.iter, burn=args.burn, chains=args.chains,
        prior_std=args.prior_std, seed=args.seed,
    )
    _log(
        "wrote 3 scoreboards in "
        f"{pl.Path(args.out).parent} total_elapsed={time.perf_counter() - started:.1f}s"
    )
    return results


if __name__ == "__main__":
    run_orchestrator()
