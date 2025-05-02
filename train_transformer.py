"""Temporal Transformer for 7‑day protest prediction

This script builds a sequence dataset from `daily_dataset.parquet` and trains a
small Transformer encoder that ingests 30‑day windows of lag features.

Run
---
python train_transformer.py \
    --dataset artifacts/daily_dataset.parquet \
    --outdir  artifacts/ \
    --epochs 10 \
    --batch_size 512 \
    --gpu                # optional, CUDA if available

Outputs
-------
- tform_model.pt       : trained PyTorch model
- tform_metrics.json   : PR‑AUC & F1 for train / val / test
- pr_curve_test.png    : Precision‑Recall curve on test split

Dependencies: torch>=2.1, torchmetrics, pandas, numpy, matplotlib, tqdm, scikit‑learn
"""
from __future__ import annotations
import argparse, json, pathlib, math
import pandas as pd
import numpy as np
from datetime import timedelta
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

SEQ_LEN = 30
HORIZON = 7
FEATS   = ["lag_1", "lag_3", "lag_7", "lag_30", "dow"]

# --------------------- helper: split identical to others -------------------- #

def time_split(df: pd.DataFrame):
    train_end = pd.Timestamp("2022-01-01")
    val_end   = pd.Timestamp("2022-07-01")
    train = df[df["date"] <  train_end]
    val   = df[(df["date"] >= train_end) & (df["date"] < val_end)]
    test  = df[df["date"] >= val_end]
    return train, val, test

# -------------------------- Dataset construction --------------------------- #
# --- replace the existing Dataset class with this one ----------------------
class ProtestSeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str):
        self.split = split
        self.countries = []
        self.country_feat = []   # list of np.ndarray (days, feats)
        self.country_label = []  # list of np.ndarray (days,)
        self.samples = []        # (country_idx, start_pos)

        # 1. build per‑country arrays once
        for cid, (country, sub) in enumerate(df.groupby("country", observed=True)):
            sub = sub.sort_values("date")
            feat_arr = sub[FEATS].to_numpy(dtype=np.float32)
            lab_arr  = sub["label"].to_numpy(dtype=np.float32)
            self.countries.append(country)
            self.country_feat.append(feat_arr)
            self.country_label.append(lab_arr)

            # 2. enumerate valid windows for that country
            for start in range(len(sub) - SEQ_LEN - HORIZON + 1):
                self.samples.append((cid, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cid, start = self.samples[idx]
        x = self.country_feat[cid][start : start + SEQ_LEN]          # (T, F)
        y = self.country_label[cid][start + SEQ_LEN + HORIZON - 1]   # scalar
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

# ------------------------------- Model ------------------------------------ #
class TemporalTransformer(nn.Module):
    def __init__(self, d_feats=5, d_model=32, nhead=4, nlayers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_feats, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.cls = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
    def forward(self, x):           # x: (B, T, F)
        x = self.input_proj(x)      # (B, T, d_model)
        h = self.encoder(x)         # (B, T, d_model)
        h_avg = h.mean(dim=1)       # (B, d_model)
        return torch.sigmoid(self.cls(h_avg)).squeeze(-1)

# --------------------------- Training utilities --------------------------- #

def batch_to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)

def run_epoch(model, loader, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, preds, gts = 0.0, [], []
    bce = nn.BCELoss()
    for batch in loader:
        x, y = batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            p = model(x)
            loss = bce(p, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * len(x)
        preds.append(p.detach().cpu().numpy())
        gts.append(y.cpu().numpy())
    preds = np.concatenate(preds); gts = np.concatenate(gts)
    return total_loss / len(loader.dataset), preds, gts

# ------------------------------- Main ------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir",  default="artifacts")
    ap.add_argument("--epochs",  type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # data
    df = pd.read_parquet(args.dataset)
    train_df, val_df, test_df = time_split(df)

    train_ds = ProtestSeqDataset(train_df, "train")
    val_ds   = ProtestSeqDataset(val_df,   "val")
    test_ds  = ProtestSeqDataset(test_df,  "test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    model = TemporalTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_auc = 0.0
    history = {}
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_pred, tr_gt = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_pred, val_gt = run_epoch(model, val_loader, optimizer, device, train=False)
        val_auc = average_precision_score(val_gt, val_pred)
        print(f"Epoch {epoch:02d}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_PR_AUC={val_auc:.3f}")
        history[epoch] = dict(train_loss=tr_loss, val_loss=val_loss, val_auc=val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), outdir / "tform_model.pt")

    # load best and evaluate on test
    model.load_state_dict(torch.load(outdir / "tform_model.pt"))
    _, test_pred, test_gt = run_epoch(model, test_loader, optimizer, device, train=False)
    pr_auc = average_precision_score(test_gt, test_pred)
    f1     = f1_score(test_gt, (test_pred >= 0.5))
    print(f"Test PR-AUC={pr_auc:.3f}  F1@0.5={f1:.3f}")

    # metrics json
    with open(outdir / "tform_metrics.json", "w") as fp:
        json.dump({"test_pr_auc": pr_auc, "test_f1": f1, "val_best_auc": best_val_auc}, fp, indent=2)

    # PR curve plot
    prec, recall, _ = precision_recall_curve(test_gt, test_pred)
    plt.step(recall, prec, where="post")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR curve – Temporal Transformer (test)")
    plt.tight_layout(); plt.savefig(outdir / "pr_curve_test.png", dpi=200)

if __name__ == "__main__":
    main()
