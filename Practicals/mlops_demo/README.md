# Practical: Serving a Model with FastAPI + Simulating Drift Monitoring

We take the **movie blockbuster classifier** from `classification.ipynb`, serve it as a web API,
and then simulate what happens when production data drifts away from the training data.

```
train.py  ──>  model.joblib  ──>  app.py (FastAPI)            POST /predict -> {"blockbuster": 1, ...}
          └─>  reference.csv ──>  drift_monitoring.ipynb      Evidently: new data vs training -> drift alarm
```

| File | What it does |
|---|---|
| `train.py` | Trains 5 pipelines (preprocessing + model), keeps the best by F1 (quality gate: F1 ≥ 0.80), saves `model.joblib` and `reference.csv` (training data). |
| `app.py` | FastAPI app serving `model.joblib`: `/health`, `/predict`, `/predict_batch`, `/reload`. |
| `test_api.py` | Pytest checks on the API (valid input → prediction, bad input → 422). |
| `drift_monitoring.ipynb` | Simulates a normal week and a drifted week of production data and runs Evidently drift checks on both. |

## Run it

```bash
cd "Practicals/mlops_demo"
pip install -r requirements.txt

python train.py                 # -> model.joblib + reference.csv
uvicorn app:app --reload        # -> http://127.0.0.1:8000/docs  (try POST /predict)
pytest -q test_api.py           # test the API
jupyter notebook drift_monitoring.ipynb
```

## How inference works in `app.py`

1. **Contract (Pydantic):** `Movie` lists the 11 input columns with rules (e.g. `rating` 0–10). Bad requests get a **422** before they reach the model.
2. **Load once:** `model.joblib` is loaded at startup, not per request. After retraining, `POST /reload` picks up the new file without a restart.
3. **Score:** JSON → DataFrame → `pipeline.predict` / `predict_proba`. The saved Pipeline repeats the exact training preprocessing (impute, one-hot, scale), so there's no train/serve mismatch.
4. **Respond:** numpy values → plain Python → `{"blockbuster": 0|1, "probability": 0.97}`.

> **Discussion point:** every model scores F1 ≈ 1.0. That's too good to be true. `revenue_usd`,
> `profit_usd` and `roi` are only known *after* a film is released, so this is **target leakage**,
> and an API that needs them can't be used before release. Exercise: drop them in `train.py` and compare.

## CI

`.github/workflows/practicals-mlops.yml` runs `train.py` (the quality gate is just an exit code:
`sys.exit(1)` if F1 is too low) and then `pytest test_api.py` on every push/PR touching this folder.
