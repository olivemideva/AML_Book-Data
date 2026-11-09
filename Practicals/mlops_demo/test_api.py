"""Pytest checks on the FastAPI app (needs model.joblib: run train.py first)."""
from fastapi.testclient import TestClient

from app import app

SAMPLE = {"year": 1981, "decade": "1980s", "genre": "Sci-Fi", "runtime_min": 128,
          "budget_usd": 306333628.06, "revenue_usd": 1237478548.98, "profit_usd": 931144920.92,
          "roi": 3.04, "rating": 5.3, "vote_count": 49703, "popularity": 7.65}


def test_health():
    with TestClient(app) as client:              # 'with' runs the startup (model load)
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["model"] is not None


def test_predict():
    with TestClient(app) as client:
        body = client.post("/predict", json=SAMPLE).json()
        assert body["blockbuster"] in (0, 1) and 0 <= body["probability"] <= 1


def test_predict_batch():
    with TestClient(app) as client:
        r = client.post("/predict_batch", json=[SAMPLE, {**SAMPLE, "genre": "Drama"}])
        assert r.status_code == 200 and len(r.json()) == 2


def test_bad_input_rejected():
    with TestClient(app) as client:
        r = client.post("/predict", json={**SAMPLE, "rating": 42})   # rating must be 0-10
        assert r.status_code == 422
