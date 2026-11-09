"""
Step 3 - "DEPLOY": load whatever model currently has the 'production' alias and use it.
The code never names a version number - promoting a new model in the registry is enough
to change what gets served. Also saves the predictions, which monitor_drift.py checks.

Usage:  python predict.py                     (scores current.csv, or simulates it)
        python predict.py --file new_movies.csv
"""
import argparse
from pathlib import Path

import mlflow
import pandas as pd

HERE = Path(__file__).parent
mlflow.set_tracking_uri(mlflow.environment_variables.MLFLOW_TRACKING_URI.get()
                        or f"sqlite:///{HERE / 'mlflow.db'}")

parser = argparse.ArgumentParser()
parser.add_argument("--file", default=str(HERE / "current.csv"))
args = parser.parse_args()

if not Path(args.file).exists():
    # no production data yet -> create a simulated batch (see simulate_production.py)
    from simulate_production import make_batch
    make_batch(drift=False).to_csv(args.file, index=False)

batch = pd.read_csv(args.file)
model = mlflow.sklearn.load_model("models:/movie-blockbuster@production")
info = mlflow.MlflowClient().get_model_version_by_alias("movie-blockbuster", "production")

features = batch.drop(columns=["blockbuster"], errors="ignore")
batch["prediction"] = model.predict(features)
batch.to_csv(args.file, index=False)

print(f"Served by movie-blockbuster v{info.version}")
print(f"Scored {len(batch)} movies, {batch['prediction'].mean():.1%} predicted blockbusters")
