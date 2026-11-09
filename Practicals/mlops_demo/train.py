
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "Data" / "movie_blockbusters.csv"

MODEL_NAME = "movie-blockbuster"
EXPERIMENT = "movie-blockbuster-practical"
MIN_F1 = 0.80          # quality gate - the best model must reach this to be deployed

# MLflow 3 saves sklearn models with "skops" (safer than pickle). skops refuses to load
# types it doesn't know unless we list them as trusted - these are all plain sklearn/numpy.
TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree",
                 "sklearn.compose._column_transformer._RemainderColsList"]

# Use the tracking server in MLFLOW_TRACKING_URI if set (e.g. a shared server),
# otherwise a local SQLite file next to this script.
mlflow.set_tracking_uri(mlflow.environment_variables.MLFLOW_TRACKING_URI.get()
                        or f"sqlite:///{HERE / 'mlflow.db'}")
mlflow.set_experiment(EXPERIMENT)


def load_data():
    df = pd.read_csv(DATA).drop(columns=["movie_id", "title"])
    X, y = df.drop(columns=["blockbuster"]), df["blockbuster"]
    return train_test_split(X, y, test_size=0.20, random_state=123, stratify=y)


def make_pipeline(X, model):
    cat_cols = list(X.select_dtypes(include="object").columns)
    num_cols = list(X.select_dtypes(include="number").columns)
    transformer = ColumnTransformer([
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")),
                          ("scaler", RobustScaler())]), num_cols),
    ])
    return Pipeline([("transformer", transformer), ("model", model)])


# name, model class, hyper-parameters to try
EXPERIMENTS = [
    ("logistic_regression", LogisticRegression, {"C": 1.0, "max_iter": 2000}),
    ("logistic_regression", LogisticRegression, {"C": 0.1, "max_iter": 2000}),
    ("decision_tree", DecisionTreeClassifier, {"max_depth": 5, "random_state": 0}),
    ("random_forest", RandomForestClassifier, {"n_estimators": 200, "max_depth": 10, "random_state": 0}),
    ("gradient_boosting", GradientBoostingClassifier, {"n_estimators": 150, "max_depth": 3, "random_state": 0}),
]


def main():
    X_train, X_test, y_train, y_test = load_data()

    # Save the training data as the REFERENCE for drift monitoring (monitor_drift.py)
    X_train.assign(blockbuster=y_train).to_csv(HERE / "reference.csv", index=False)

    for name, Model, params in EXPERIMENTS:
        with mlflow.start_run(run_name=name):
            pipe = make_pipeline(X_train, Model(**params)).fit(X_train, y_train)
            preds = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)[:, 1]

            metrics = {
                "f1": f1_score(y_test, preds),
                "roc_auc": roc_auc_score(y_test, proba),
                "precision": precision_score(y_test, preds),
                "recall": recall_score(y_test, preds),
            }
            mlflow.log_param("model_type", name)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            # Tag the run with the commit that produced it (set automatically in GitHub Actions)
            mlflow.set_tag("git_sha", os.environ.get("GITHUB_SHA", "local"))
            mlflow.sklearn.log_model(
                pipe, name="model",
                signature=infer_signature(X_test, preds),
                input_example=X_test.head(3),
                skops_trusted_types=TRUSTED_TYPES,
            )
            print(f"{name:20} f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

    # Best run of THIS training session (newest runs only)
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=len(EXPERIMENTS))
    best = runs.sort_values("metrics.f1", ascending=False).iloc[0]
    print(f"\nBest: {best['params.model_type']}  f1={best['metrics.f1']:.3f}")

    # ---------- QUALITY GATE ----------
    if best["metrics.f1"] < MIN_F1:
        print(f"GATE FAILED: f1 {best['metrics.f1']:.3f} < {MIN_F1}. Not registering.")
        sys.exit(1)

    # ---------- REGISTER + PROMOTE ----------
    version = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
    mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, "production", version.version)
    print(f"GATE PASSED: {MODEL_NAME} v{version.version} -> alias 'production'")


if __name__ == "__main__":
    main()
