"""Train a LightGBM model for 7‑day protest prediction

Usage
-----
python train_lgbm.py \
    --dataset artifacts/daily_dataset.parquet \
    --outdir artifacts/

Outputs
-------
- lgbm.pkl            : trained model
- metrics.json        : PR‑AUC & F1 for train / val / test
- shap_summary.png    : feature‑importance plot

Dependencies: lightgbm>=4.0, shap, pandas, numpy, scikit‑learn, tqdm, joblib, matplotlib
"""
from __future__ import annotations
import argparse, json, pathlib
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm

# ---------------------------- helpers ---------------------------- #

def time_split(df: pd.DataFrame):
    train_end = pd.Timestamp("2022-01-01")
    val_end   = pd.Timestamp("2022-07-01")
    train = df[df["date"] <  train_end]
    val   = df[(df["date"] >= train_end) & (df["date"] < val_end)]
    test  = df[df["date"] >= val_end]
    return train, val, test


def calc_metrics(model, data: pd.DataFrame, feats: list[str]):
    y_prob = model.predict(data[feats])
    y_true = data["label"].values
    pr_auc = average_precision_score(y_true, y_prob)
    f1     = f1_score(y_true, (y_prob >= 0.5))
    return {"pr_auc": pr_auc, "f1": f1}

# ------------------------- main training ------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Parquet produced by baseline_pipeline.py")
    ap.add_argument("--outdir",   default="artifacts", help="Folder for outputs")
    ap.add_argument("--gpu",      action="store_true", help="Use GPU if available")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. load data & split
    df = pd.read_parquet(args.dataset)
    train, val, test = time_split(df)

    feats = [c for c in df.columns if c.startswith("lag_") or c == "dow"]
    print(f"Training on {len(feats)} features: {feats}")

    dtrain = lgb.Dataset(train[feats], label=train["label"], free_raw_data=False)
    dval   = lgb.Dataset(val[feats],   label=val["label"],   free_raw_data=False)

    params = dict(
        objective="binary",
        metric="aucpr",
        learning_rate=0.02,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        is_unbalance=True,
    )
    if args.gpu:
        params.update(device_type="gpu", boosting="gbdt")

    print("Training LightGBM …")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        valid_names=["val"],
        early_stopping_rounds=100,
        verbose_eval=200,
    )

    # 2. metrics
    metrics = {
        "train": calc_metrics(model, train, feats),
        "val"  : calc_metrics(model, val,   feats),
        "test" : calc_metrics(model, test,  feats),
    }
    print(json.dumps(metrics, indent=2))

    # 3. save model & metrics
    joblib.dump(model, outdir / "lgbm.pkl")
    with open(outdir / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

    # 4. SHAP plot
    try:
        import shap, matplotlib.pyplot as plt
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(val[feats])
        shap.summary_plot(shap_vals, val[feats], show=False)
        plt.tight_layout()
        plt.savefig(outdir / "shap_summary.png", dpi=200)
        plt.close()
        print("Saved shap_summary.png")
    except Exception as e:
        print(f"SHAP plot skipped: {e}")

if __name__ == "__main__":
    main()
