"""Baseline pipeline for per‑country 7‑day protest prediction

Steps in this single script:
 1. Load raw WORLDREP CSV (one row per event)
 2. Filter rows that are protest / civil‑unrest (simple keyword rule)
 3. Aggregate to daily counts per country
 4. Build 7‑day horizon labels (binary)
 5. Generate lagged‑count features
 6. Train / validate / test split (time‑based)
 7. Fit Logistic Regression baseline
 8. Save metrics & model artifacts

Run:
    python baseline_pipeline.py --csv data/raw/worldrep.csv --outdir artifacts/

Dependencies: pandas, numpy, scikit‑learn, tqdm
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

PROTEST_KEYWORDS = [
    "protest", "demonstration", "strike", "rally", "boycott", "unrest", "riot",
]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to raw WORLDREP csv")
    ap.add_argument("--outdir", default="artifacts", help="Folder to write models/metrics")
    ap.add_argument("--horizon", type=int, default=7, help="Days ahead for positive label")
    return ap.parse_args()

def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expect DATE in yyyymmddHHMMSS
    df["timestamp"] = pd.to_datetime(df["DATE"], format="%Y%m%d%H%M%S", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    # country field – default to Country1
    df["country"] = df["Country1"].fillna(df["Country2"])
    return df

def flag_protest(text: str) -> bool:
    text_l = str(text).lower()
    return any(k in text_l for k in PROTEST_KEYWORDS)

def filter_protests(df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas(desc="flag protests")
    df["is_protest"] = df["CONTENT"].progress_apply(flag_protest)
    return df[df["is_protest"]]

def build_daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["country", "date"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    return daily

def add_missing_dates(daily: pd.DataFrame) -> pd.DataFrame:
    # ensure every (country, date) combo exists so lag features don’t break
    all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    countries = daily["country"].unique()
    full_idx = pd.MultiIndex.from_product([countries, all_dates], names=["country", "date"])
    daily_full = daily.set_index(["country", "date"]).reindex(full_idx, fill_value=0).reset_index()
    return daily_full

def build_labels(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    daily = daily.sort_values(["country", "date"])
    daily["future_sum"] = (
        daily.groupby("country")["count"]
        .transform(lambda s: s.rolling(window=horizon, min_periods=1).sum().shift(-horizon + 1))
    )
    daily["label"] = (daily["future_sum"] > 0).astype(int)
    return daily.drop(columns=["future_sum"])

def add_lag_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.sort_values(["country", "date"])
    for k in [1, 3, 7, 30]:
        daily[f"lag_{k}"] = daily.groupby("country")["count"].shift(k).fillna(0)
    daily["dow"] = pd.to_datetime(daily["date"]).dt.dayofweek
    return daily

def time_split(daily: pd.DataFrame):
    train_end = pd.Timestamp("2022-01-01")
    val_end = pd.Timestamp("2022-07-01")
    train = daily[daily["date"] < train_end]
    val = daily[(daily["date"] >= train_end) & (daily["date"] < val_end)]
    test = daily[daily["date"] >= val_end]
    return train, val, test

def train_logreg(train, val, features):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(train[features], train["label"])
    # validation PR‑AUC
    val_probs = pipe.predict_proba(val[features])[:, 1]
    pr_auc = average_precision_score(val["label"], val_probs)
    threshold = 0.5
    f1 = f1_score(val["label"], (val_probs >= threshold).astype(int))
    return pipe, pr_auc, f1

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading raw csv …")
    df = load_raw(args.csv)
    print(f"Loaded {len(df):,} events")

    print("Flagging protest rows …")
    df_p = filter_protests(df)
    print(f"Protest events: {len(df_p):,}")

    print("Building daily counts …")
    daily = build_daily_counts(df_p)
    daily = add_missing_dates(daily)
    daily = build_labels(daily, args.horizon)
    daily = add_lag_features(daily)

    features = [c for c in daily.columns if c.startswith("lag_") or c == "dow"]

    train, val, test = time_split(daily)
    print(
        f"Train/Val/Test sizes: {len(train):,} / {len(val):,} / {len(test):,}")

    model, pr_auc, f1 = train_logreg(train, val, features)
    print(f"Validation PR‑AUC = {pr_auc:.3f}, F1 = {f1:.3f}")

    print("Saving model …")
    joblib.dump(model, outdir / "logreg.pkl")
    daily.to_parquet(outdir / "daily_dataset.parquet")
    print("Done.")

if __name__ == "__main__":
    main()
