import argparse
import pathlib as pl
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from src.evaluation.simulate_hospital import main

def load_data(n_samples=None, seed=None, setting=None, timestamp=None):
    """
    Load synthetic data named setting_samples_seed_timestamp.csv.

    If timestamp is omitted, load the file with the latest timestamp in its name.
    """
    data_dir = pl.Path("Data/synth")
    if timestamp is None:
        pattern = f"{setting}_{n_samples}_{seed}_*.csv"
        matches = sorted(data_dir.glob(pattern))
        if not matches:
            generate_data(n_samples=n_samples, seed=seed, setting=setting)
            matches = sorted(data_dir.glob(pattern))
            if not matches:
                raise FileNotFoundError(
                    f"No data files found matching {data_dir / pattern}"
                )
        path = matches[-1]
    else:
        path = data_dir / f"{setting}_{n_samples}_{seed}_{timestamp}.csv"
        if not path.is_file():
            generate_data(n_samples=n_samples, seed=seed, setting=setting)
            if not path.is_file():
                raise FileNotFoundError(f"Data file not found: {path}")

    return pd.read_csv(path)

def fit_model(model=None, X_train=None, y_train=None, iter=4000, burn=1500, chains=2):
    """
    Fit the model using the bootstrap method.
    """
    # Placeholder for model fitting logic

    pass


def run_orchestrator():
    """
    Run the orchestrator.
    """
    parser = argparse.ArgumentParser(description="Run the orchestrator.")
    parser.add_argument("--test-frac", type=float, default=0.3, help="Share of visits held out for testing.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--iter", type=int, default=4000, help="Number of iterations for the sampler.")
    parser.add_argument("--burn", type=int, default=1500, help="Number of burn-in iterations for the sampler.")
    parser.add_argument("--chains", type=int, default=2, help="Number of chains for the sampler.")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples to load.")
    parser.add_argument("--setting", type=str, default="setting1", help="Setting to use for data loading.")
    parser.add_argument("--timestamp", type=str, default=None, help="Timestamp of the data file to load. If omitted, the latest timestamp is used.")
    parser.add_argument("--target", type=str, default="mortality", help="Target variable for prediction.")
    parser.add_argument("--quick", action="store_true", help="Run a quick test with fewer iterations.")
    parser.add_argument("--out", default=str( "Results" / "evaluation" / "calibration_slope.csv"),
                        help="Output file path for calibration slope results.")

    args = parser.parse_args()


    if args.quick:
        args.iter, args.burn, args.chains = 200, 80, 1

    random.seed(args.seed)
    # step 1: load data
    data_df = load_data(n_samples=args.n_samples, seed=args.seed, setting=args.setting, timestamp=args.timestamp)

    # if setting 2 then preprocess the data for setting 2
    if args.setting == "setting2":
        data_df = preprocess_setting2(data_df)

    # step 2: test train split
    X_train, X_test, y_train, y_test = train_test_split(data_df.drop(columns=[args.target]), data_df[args.target], test_size=args.test_frac, random_state=args.seed)
    
    # step 3: fit the model using the bootstrap method
    model = build_model(X_train, y_train, iter=args.iter, burn=args.burn, chains=args.chains)
    fit_model(model,X_train, y_train, iter=args.iter, burn=args.burn, chains=args.chains)

    # step 4: evaluate the model
    evaluate_model(X_test, y_test, out_path=args.out)



    

    main(args)
