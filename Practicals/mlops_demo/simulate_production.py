
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "Data" / "movie_blockbusters.csv"


def make_batch(drift: bool, n: int = 1000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.read_csv(DATA).drop(columns=["movie_id", "title"]).sample(n, random_state=seed)

    if drift:
        money = ["budget_usd", "revenue_usd", "profit_usd"]
        df[money] = df[money] * rng.normal(1.6, 0.1, size=(n, 1))
        shift = rng.random(n) < 0.4
        df.loc[shift, "genre"] = rng.choice(["Horror", "Animation"], size=shift.sum())
        df["rating"] = (df["rating"] - 1.2 + rng.normal(0, 0.3, n)).clip(0, 10).round(1)
        df["vote_count"] = (df["vote_count"] * 2.5).astype(int)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--drift", action="store_true")
    a = p.parse_args()
    make_batch(a.drift).to_csv(HERE / "current.csv", index=False)
    print(f"Wrote current.csv ({'DRIFTED' if a.drift else 'no drift'})")
