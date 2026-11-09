"""
Step 4 - MONITOR for drift with Evidently AI.

Compares two datasets:
    reference.csv -> what the model was trained on (written by train.py)
    current.csv   -> what the model is seeing in production now (predict.py / simulate_production.py)

For every column Evidently runs a statistical test (picked automatically from column type
and sample size) and says whether its distribution has changed. We also check
PREDICTION drift: is the model suddenly predicting a very different share of blockbusters?

Outputs:
    drift_report.html  -> interactive Evidently report (open in a browser)
    MLflow run         -> experiment "drift-monitoring": drift metrics + the HTML report
    exit code 1        -> if --fail-on-drift and drift is detected (CI uses this as the alarm)

Usage:  python monitor_drift.py
        python monitor_drift.py --fail-on-drift
"""
import argparse
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

HERE = Path(__file__).parent
DRIFT_SHARE = 0.3   # dataset drift alarm = 30% or more of the columns drifted

CATEGORICAL = ["decade", "genre", "prediction"]
NUMERICAL = ["year", "runtime_min", "budget_usd", "revenue_usd", "profit_usd",
             "roi", "rating", "vote_count", "popularity"]

mlflow.set_tracking_uri(mlflow.environment_variables.MLFLOW_TRACKING_URI.get()
                        or f"sqlite:///{HERE / 'mlflow.db'}")


def load_frames():
    model = mlflow.sklearn.load_model("models:/movie-blockbuster@production")
    reference = pd.read_csv(HERE / "reference.csv")
    current = pd.read_csv(HERE / "current.csv")
    # Score the reference too, so we can compare prediction distributions
    reference["prediction"] = model.predict(reference[NUMERICAL + ["decade", "genre"]])
    if "prediction" not in current:
        current["prediction"] = model.predict(current[NUMERICAL + ["decade", "genre"]])
    cols = CATEGORICAL + NUMERICAL
    return reference[cols], current[cols]


def run_report(reference, current):
    definition = DataDefinition(categorical_columns=CATEGORICAL, numerical_columns=NUMERICAL)
    report = Report([DataDriftPreset(drift_share=DRIFT_SHARE)])
    return report.run(
        current_data=Dataset.from_pandas(current, data_definition=definition),
        reference_data=Dataset.from_pandas(reference, data_definition=definition),
    )


def summarise(snapshot):
    """Pull the numbers we care about out of the Evidently result."""
    overall, columns = None, {}
    for m in snapshot.dict()["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith("DriftedColumnsCount"):
            overall = m["value"]                       # {"count": ..., "share": ...}
        elif cfg["type"].endswith("ValueDrift"):
            score, thr, method = m["value"], cfg["threshold"], cfg["method"]
            # p-value tests: drift if p < threshold.  Distance tests: drift if distance >= threshold
            drifted = score < thr if "p_value" in method else score >= thr
            columns[cfg["column"]] = {"score": score, "threshold": thr,
                                      "method": method, "drifted": drifted}
    return overall, columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-on-drift", action="store_true")
    args = ap.parse_args()

    reference, current = load_frames()
    snapshot = run_report(reference, current)
    report_path = HERE / "drift_report.html"
    snapshot.save_html(str(report_path))

    overall, columns = summarise(snapshot)
    dataset_drift = overall["share"] >= DRIFT_SHARE

    print(f"{'column':14} {'method':38} {'score':>8}  drift?")
    for col, r in columns.items():
        print(f"{col:14} {r['method']:38} {r['score']:8.4f}  {'YES' if r['drifted'] else 'no'}")
    print(f"\nDrifted columns: {int(overall['count'])}/{len(columns)} "
          f"({overall['share']:.0%})  ->  DATASET DRIFT: {'YES' if dataset_drift else 'no'}")
    print(f"Prediction drift: {'YES' if columns['prediction']['drifted'] else 'no'}")
    print(f"Report: {report_path}")

    # Keep a history of every drift check in MLflow, next to the training runs
    mlflow.set_experiment("drift-monitoring")
    with mlflow.start_run(run_name="drift-check"):
        mlflow.log_metric("drifted_columns", overall["count"])
        mlflow.log_metric("drift_share", overall["share"])
        mlflow.log_metric("dataset_drift", int(dataset_drift))
        for col, r in columns.items():
            mlflow.log_metric(f"drift_score_{col}", r["score"])
        mlflow.log_artifact(str(report_path))

    # In GitHub Actions, show a summary table on the run page
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as f:
            f.write("## Evidently drift check\n\n| column | score | drift |\n|---|---|---|\n")
            for col, r in columns.items():
                f.write(f"| {col} | {r['score']:.4f} | {'🔴' if r['drifted'] else '🟢'} |\n")
            f.write(f"\n**Dataset drift: {'YES' if dataset_drift else 'no'}** "
                    f"({overall['share']:.0%} of columns)\n")

    if dataset_drift and args.fail_on_drift:
        print("DRIFT DETECTED -> failing so someone looks at it / retrains.")
        sys.exit(1)


if __name__ == "__main__":
    main()
