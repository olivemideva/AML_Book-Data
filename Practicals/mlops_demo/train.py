"""
Step 1 - TRAIN the movie blockbuster classifier and save it for the API.

    python train.py

Trains 5 candidate pipelines (preprocessing + model), keeps the best one by F1,
and writes two files:
    model.joblib   -> the winning Pipeline + its name and score (loaded by app.py)
    reference.csv  -> the training data, used as the "normal" baseline in drift_monitoring.ipynb
"""
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "Data" / "movie_blockbusters.csv"
MODEL_PATH = HERE / "model.joblib"
MIN_F1 = 0.80          # quality gate - don't save a model worse than this


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


CANDIDATES = {
    "logistic_regression": LogisticRegression(C=1.0, max_iter=2000),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=0),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0),
    "logistic_regression_C0.1": LogisticRegression(C=0.1, max_iter=2000),
}


def main():
    X_train, X_test, y_train, y_test = load_data()
    X_train.assign(blockbuster=y_train).to_csv(HERE / "reference.csv", index=False)

    best = None
    for name, model in CANDIDATES.items():
        pipe = make_pipeline(X_train, model).fit(X_train, y_train)
        f1 = f1_score(y_test, pipe.predict(X_test))
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        print(f"{name:26} f1={f1:.3f}  roc_auc={auc:.3f}")
        if best is None or f1 > best["f1"]:
            best = {"name": name, "f1": f1, "pipeline": pipe}

    print(f"\nBest: {best['name']}  f1={best['f1']:.3f}")
    if best["f1"] < MIN_F1:
        print(f"GATE FAILED: f1 {best['f1']:.3f} < {MIN_F1}. Model not saved.")
        sys.exit(1)

    joblib.dump({"pipeline": best["pipeline"], "name": best["name"], "f1": round(best["f1"], 4),
                 "trained_at": datetime.now().isoformat(timespec="seconds")}, MODEL_PATH)
    print(f"Saved {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
