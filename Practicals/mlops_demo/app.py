
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"


# ---------------------------------------------------------------------------
# STEP A - Define the INPUT and OUTPUT contract (Pydantic)
# The raw columns the model was trained on (training data minus movie_id, title, blockbuster).
# FastAPI rejects bad requests with a 422 error BEFORE they reach the model.
# ---------------------------------------------------------------------------
class Movie(BaseModel):
    year: int = Field(..., ge=1900, le=2100)
    decade: Literal["1980s", "1990s", "2000s", "2010s", "2020s"]
    genre: str = Field(..., examples=["Sci-Fi"])   # free text: the encoder ignores unseen genres
    runtime_min: float = Field(..., gt=0)
    budget_usd: float = Field(..., ge=0)
    revenue_usd: float = Field(..., ge=0)
    profit_usd: float
    roi: float
    rating: float = Field(..., ge=0, le=10)
    vote_count: int = Field(..., ge=0)
    popularity: float = Field(..., ge=0)

    model_config = {"json_schema_extra": {"example": {
        "year": 1981, "decade": "1980s", "genre": "Sci-Fi", "runtime_min": 128,
        "budget_usd": 306333628.06, "revenue_usd": 1237478548.98, "profit_usd": 931144920.92,
        "roi": 3.04, "rating": 5.3, "vote_count": 49703, "popularity": 7.65}}}


class Prediction(BaseModel):
    blockbuster: int          # 0 or 1
    probability: float        # model's confidence that it IS a blockbuster


# ---------------------------------------------------------------------------
# STEP B - Load the model ONCE at startup (not on every request: loading is slow)
# ---------------------------------------------------------------------------
state = {}


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"{MODEL_PATH.name} not found - run `python train.py` first")
    state.update(joblib.load(MODEL_PATH))     # pipeline, name, f1, trained_at
    print(f"Loaded {state['name']} (f1={state['f1']}, trained {state['trained_at']})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()                  # runs when uvicorn starts
    yield                         # the API serves requests here


app = FastAPI(title="Movie Blockbuster Classifier", lifespan=lifespan)


# ---------------------------------------------------------------------------
# STEP C - The inference itself, shared by both endpoints
# ---------------------------------------------------------------------------
def score(movies: list[Movie]) -> list[Prediction]:
    # 1. JSON objects -> DataFrame with the column names the model expects
    df = pd.DataFrame([m.model_dump() for m in movies])
    # 2. The Pipeline imputes, one-hot encodes and scales, THEN predicts.
    #    The exact preprocessing from training, so no train/serve mismatch.
    model = state["pipeline"]
    labels = model.predict(df)
    probs = model.predict_proba(df)[:, 1]
    # 3. numpy types -> plain Python, wrapped in the output schema
    return [Prediction(blockbuster=int(y), probability=round(float(p), 4))
            for y, p in zip(labels, probs)]


# ---------------------------------------------------------------------------
# STEP D - Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Is the API up, and which model is it serving?"""
    return {"status": "ok", "model": state.get("name"), "f1": state.get("f1"),
            "trained_at": state.get("trained_at")}


@app.post("/predict", response_model=Prediction)
def predict(movie: Movie):
    """Score ONE movie."""
    try:
        return score([movie])[0]
    except Exception as e:                       # never leak a stack trace to the client
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict_batch", response_model=list[Prediction])
def predict_batch(movies: list[Movie]):
    """Score MANY movies in one request (one model call, faster than looping /predict)."""
    if not movies:
        raise HTTPException(status_code=400, detail="Send at least one movie")
    return score(movies)


@app.post("/reload")
def reload():
    """After re-running train.py, pick up the new model.joblib without restarting."""
    load_model()
    return {"status": "reloaded", "model": state["name"], "trained_at": state["trained_at"]}
