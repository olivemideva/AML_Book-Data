"""
Step 2 - TEST the model that is marked 'production' in the MLflow registry.
CI runs these with pytest; any failure stops the pipeline before deployment.
"""
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from sklearn.metrics import f1_score

HERE = Path(__file__).parent
mlflow.set_tracking_uri(mlflow.environment_variables.MLFLOW_TRACKING_URI.get()
                        or f"sqlite:///{HERE / 'mlflow.db'}")


@pytest.fixture(scope="module")
def model():
    return mlflow.sklearn.load_model("models:/movie-blockbuster@production")


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(HERE.parent.parent / "Data" / "movie_blockbusters.csv")
    return df.drop(columns=["movie_id", "title", "blockbuster"]), df["blockbuster"]


def test_model_loads(model):
    assert model is not None


def test_predictions_are_binary(model, data):
    X, _ = data
    assert set(model.predict(X.head(50))) <= {0, 1}


def test_probabilities_in_range(model, data):
    X, _ = data
    proba = model.predict_proba(X.head(50))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_f1_good_enough(model, data):
    X, y = data
    assert f1_score(y, model.predict(X)) >= 0.80


def test_unseen_genre_does_not_crash(model, data):
    X, _ = data
    row = X.head(1).assign(genre="Musical")      # category never seen in training
    assert model.predict(row)[0] in (0, 1)
