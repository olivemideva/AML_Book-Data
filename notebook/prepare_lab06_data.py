import re
import numpy as np
import pandas as pd

SEED = 42
BASE = r"C:\Users\Lbundi\Desktop\Strathmore Lectures\AML YR 26\Data"

df_raw = pd.read_csv(BASE + r"\mobile_money_statements.csv")


def parse_amount(value):
    if pd.isna(value):
        return np.nan
    value = str(value)
    negative = bool(re.search(r"Dr|\(.*\)|^-", value))
    digits = re.sub(r"[^\d.]", "", value)
    if not digits:
        return 0.0
    amount = float(digits)
    return -amount if negative else amount


df_raw["amount_clean"] = df_raw["amount"].apply(parse_amount)

parsed = pd.Series(pd.NaT, index=df_raw.index, dtype="datetime64[ns]")
for fmt in ["%Y-%m-%d %H:%M:%S", "%b %d, %Y %I:%M %p", "%d/%m/%Y %H:%M"]:
    mask = parsed.isna()
    parsed.loc[mask] = pd.to_datetime(df_raw.loc[mask, "txn_time"], format=fmt, errors="coerce")
df_raw["txn_dt"] = parsed

df = df_raw.copy()

df["abs_amount"] = df["amount_clean"].abs()
df["log_amount"] = np.log1p(df["abs_amount"])
df["is_debit"] = (df["amount_clean"] < 0).astype(int)

df["balance_after"] = pd.to_numeric(df["balance_after"], errors="coerce").fillna(0)
df["log_balance"] = np.log1p(df["balance_after"].clip(lower=0))

rng = np.random.RandomState(SEED)
df["risk_score"] = (df["manual_review_score"] + rng.normal(0, 0.4, len(df))).clip(0, 1)

df["hour"] = df["txn_dt"].dt.hour
df["day_of_week"] = df["txn_dt"].dt.dayofweek
df["day_of_month"] = df["txn_dt"].dt.day
df["month"] = df["txn_dt"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

df["has_gps"] = df["gps_lat"].notna().astype(int)
df["gps_lat"] = df["gps_lat"].fillna(0)
df["gps_lon"] = df["gps_lon"].fillna(0)

for col, new_col in [("txn_type", "txn_type_enc"), ("region", "region_enc"), ("segment", "segment_enc")]:
    df[new_col] = df[col].astype("category").cat.codes

feature_cols = [
    "abs_amount", "log_amount", "is_debit",
    "balance_after", "log_balance", "risk_score",
    "hour", "day_of_week", "day_of_month", "month",
    "is_weekend", "is_night", "has_gps",
    "gps_lat", "gps_lon",
    "txn_type_enc", "region_enc", "segment_enc",
]

out = df[["txn_dt"] + feature_cols + ["is_fraud"]]
out.to_csv(BASE + r"\CleanedFeaturesFromLoan.csv", index=False)
print("Saved rows:", len(out))
print("Date range:", out["txn_dt"].min(), "to", out["txn_dt"].max())
print("Fraud rate:", f"{out['is_fraud'].mean():.2%}")
print("Missing txn_dt:", out["txn_dt"].isna().sum())
