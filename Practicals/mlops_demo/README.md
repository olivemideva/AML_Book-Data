# Practical: MLflow + CI/CD + Drift Monitoring with Evidently AI

We take the **movie blockbuster classifier** from `classification.ipynb` and turn it into a
small MLOps pipeline:

```
 git push
    │
    ▼
┌──────────────── CI (every push / PR) ────────────────┐
│ train.py      train 5 models, log each to MLflow     │
│               QUALITY GATE: best F1 >= 0.80 ?         │──✗──> pipeline fails, nothing deployed
│               register best -> alias "production"     │
│ test_model.py pytest the production model             │──✗──> pipeline fails
└───────────────────────────┬──────────────────────────┘
                            ▼  (master only)
┌──────────────── CD ──────────────────────────────────┐
│ predict.py    load models:/movie-blockbuster@production
│               score this week's production batch     │
└───────────────────────────┬──────────────────────────┘
                            ▼  (also every Monday on a schedule)
┌──────────────── MONITOR ─────────────────────────────┐
│ monitor_drift.py  Evidently: production vs training  │──✗──> DRIFT ALARM (job fails, report uploaded)
│                   log drift metrics to MLflow        │
└──────────────────────────────────────────────────────┘
```

| File | What it does |
|---|---|
| `train.py` | Trains 5 pipelines (preprocessing + model), logs params/metrics/model to MLflow, applies the quality gate, registers the winner as `production`. Saves `reference.csv` (training data) for monitoring. |
| `test_model.py` | Pytest checks on whatever model is `production` (loads, binary output, F1 ≥ 0.80, unseen genre doesn't crash). |
| `predict.py` | The "deployment": loads `models:/movie-blockbuster@production` and scores a batch. |
| `simulate_production.py` | Makes a fake production batch (`current.csv`), with or without drift. |
| `monitor_drift.py` | Evidently drift report + MLflow logging + fails with exit code 1 on drift. |
| `../../.github/workflows/practicals-mlops.yml` | GitHub Actions pipeline that wires it all together. |

---

## 1. Run it locally

```bash
cd "Practicals/mlops_demo"
pip install -r requirements.txt

python train.py                       # train + track + gate + register
pytest -q test_model.py               # test the production model
python simulate_production.py         # this week's data (no drift)
python predict.py                     # serve it with the production model
python monitor_drift.py               # Evidently drift check -> drift_report.html

python simulate_production.py --drift # now the world changes...
python predict.py
python monitor_drift.py --fail-on-drift   # -> DATASET DRIFT: YES, exit code 1
```

Open the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# http://127.0.0.1:5000
```

What to show in the UI:
- **Experiments → movie-blockbuster-practical**: one run per model. Tick all runs → **Compare** to see F1/ROC-AUC side by side, sort by `f1`.
- Click a run → **Artifacts → model**: the saved pipeline, its *signature* (expected input columns/types) and an *input example*.
- **Models → movie-blockbuster**: the versions and which one has the `production` alias. Run `train.py` again and a new version appears, and the alias moves.
- **Experiments → drift-monitoring**: one run per drift check, with `drift_share` and the HTML report as an artifact. This is your monitoring history.

> **Discussion point:** every model scores F1 = 1.0. That's too good to be true. `revenue_usd`,
> `profit_usd` and `roi` are only known *after* a film is released and almost define
> "blockbuster", so this is **target leakage**. Exercise: drop them in `train.py`, push, and watch
> what happens to the scores and to the quality gate.

---

## 2. MLflow: the key ideas

| Concept | In this code | Why it matters |
|---|---|---|
| **Tracking URI** | `sqlite:///mlflow.db` (or env `MLFLOW_TRACKING_URI`) | Where runs are stored. Point it at a shared server and the whole team sees the same runs. |
| **Experiment** | `mlflow.set_experiment(...)` | A folder of related runs. |
| **Run** | `with mlflow.start_run():` | One training attempt: params + metrics + artifacts. |
| **log_params / log_metrics** | hyper-parameters, F1, ROC-AUC | Reproducibility and comparison. |
| **log_model + signature** | the whole sklearn `Pipeline` | Preprocessing travels *with* the model, so no train/serve mismatch. |
| **Model Registry** | `register_model(...)` | Versioned models (v1, v2, …). |
| **Alias** | `@production` | Deployment code never hard-codes a version; promoting or rolling back is just moving the alias. |
| **Tags** | `git_sha` | Links each run to the commit that produced it (set automatically in CI). |

---

## 3. CI/CD: how GitHub Actions connects to MLflow

The workflow `.github/workflows/practicals-mlops.yml` runs on every push/PR that touches this folder
(or the dataset), every Monday on a schedule, and on demand.

1. **CI job**: installs requirements, runs `train.py` then `pytest`.
   - The **quality gate is just an exit code**: `train.py` calls `sys.exit(1)` if the best F1 is too
     low. GitHub Actions treats any non-zero exit as failure, so later jobs **don't run**.
   - The MLflow store (`mlflow.db` + `mlruns/`) and `reference.csv` are uploaded as a **build
     artifact**, so the next jobs use *exactly* the model that passed.
2. **CD job** (master only, `needs: ci`): downloads the store and serves a batch with
   `models:/movie-blockbuster@production`. In a real system this step would build/push a Docker
   image or update an endpoint (see `fraud_mlops/` for that version).
3. **Monitor job** (`needs: cd`): runs Evidently on the production batch. If drift is found the job
   **fails** (red ❌, GitHub emails you), and the HTML report is uploaded either way. A drift table
   also appears on the run's **Summary** page.

**Try it:** GitHub → **Actions → practicals-mlops → Run workflow** → tick **simulate drift**.
CI and CD pass, then Monitor fails. Download `evidently-drift-report` to see why.

Where to look after a run:
- **Summary page** → the drift table and the artifacts (`mlflow-store`, `evidently-drift-report`).
- Download `mlflow-store`, unzip it into this folder, and run `mlflow ui` to browse the runs CI produced.

> In production you'd replace the SQLite file with a shared MLflow server (Databricks, a VM, etc.):
> add a repo secret `MLFLOW_TRACKING_URI` and pass it as `env:` in the workflow. No code
> changes are needed, because every script reads that variable.

---

## 4. Monitoring drift with Evidently AI

### Why monitor?
A model is trained on a snapshot of the world. After deployment the world keeps changing, and
the model **silently** gets worse. Often you don't have true labels yet (you only learn which
films were blockbusters months later), so you can't just measure accuracy. Instead you watch
the **inputs** and **outputs**:

| Type | Question | In this demo |
|---|---|---|
| **Data drift** (covariate shift) | Do the inputs look different from training? | budgets ↑60%, more Horror/Animation, lower ratings |
| **Prediction drift** | Is the model's output distribution changing? | the `prediction` column |
| **Concept drift** | Has the relationship X → y changed? | needs labels; check later with Evidently's `ClassificationPreset` |

### How Evidently decides "drift"
For each column it compares the **reference** distribution (training data) with the **current**
one (production) using a statistical test picked automatically:

| Column type (size of the **reference** data) | Test (default) | Drift if |
|---|---|---|
| Numerical, > 1000 rows | Wasserstein distance (normed) | distance ≥ 0.1 |
| Numerical, ≤ 1000 rows | Kolmogorov–Smirnov | p-value < 0.05 |
| Categorical, > 1000 rows | Jensen–Shannon distance | distance ≥ 0.1 |
| Categorical, ≤ 1000 rows | chi-squared (Z-test if binary) | p-value < 0.05 |

Our reference has 4000 rows, so you'll see Wasserstein and Jensen–Shannon in the output.
You can override the test per column (e.g. `DataDriftPreset(method="psi")`).

Then **dataset drift** = the share of drifted columns ≥ `DRIFT_SHARE` (we use 0.3).

### The code (Evidently ≥ 0.7 API)
```python
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

definition = DataDefinition(categorical_columns=[...], numerical_columns=[...])
report   = Report([DataDriftPreset(drift_share=0.3)])
snapshot = report.run(
    current_data=Dataset.from_pandas(current, data_definition=definition),
    reference_data=Dataset.from_pandas(reference, data_definition=definition),
)
snapshot.save_html("drift_report.html")   # visual report
snapshot.dict()                           # numbers -> MLflow / alerts / CI exit code
```

### Expected output
```
No drift  -> Drifted columns: 0/12 (0%)   -> DATASET DRIFT: no
Drifted   -> budget_usd, revenue_usd, profit_usd, rating, vote_count, genre flagged
             Drifted columns: 6/12 (50%)  -> DATASET DRIFT: YES   (exit code 1)
```
Notice `roi` did **not** drift even though budgets and revenues both went up 60%. The ratio
stayed the same. And `prediction` barely moved. Ask students why that matters: input drift
doesn't always mean the model is broken, so the next step after an alarm is investigation
(and retraining on fresh data if needed), not automatic panic.

### Reading `drift_report.html`
Open it in a browser. For every column you'll see the test used, the score, and a drift/no-drift
verdict, plus overlaid reference-vs-current distributions. Look at `budget_usd` (shifted right)
and `genre` (Horror/Animation bars grow).

### What happens after a drift alarm? (the loop closes)
1. Investigate: real change or a data bug (e.g. currency changed from USD to KES)?
2. If real: collect recent labelled data → retrain → **push**, and CI runs the gate again →
   new registry version → alias moves → CD deploys it. That's the full MLOps loop.

---

## Exercises
1. Change `MIN_F1` to `1.01` and push. Watch CI fail at the gate, with no deploy.
2. Remove the leaky columns (`revenue_usd`, `profit_usd`, `roi`). How much does F1 drop? Does the gate still pass?
3. Add `XGBoost` or a tuned `RandomForest` to `EXPERIMENTS` and compare in the MLflow UI.
4. In `simulate_production.py`, drift **only** `rating`. Does the dataset alarm fire? Should it?
   Try per-column alarms (e.g. fail if `prediction` drifts).
5. Add Evidently's `DataSummaryPreset()` to the report to also catch missing values / new categories.
