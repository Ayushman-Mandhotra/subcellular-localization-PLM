#!/usr/bin/env python3
"""Train a final model on all data and save it for prediction."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from plm_localization.config import Config
from plm_localization.data import load_dataset
from plm_localization.embeddings import compute_features
from plm_localization.model import PooledMLP
from plm_localization.utils import setup_logger, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.xlsx")
    p.add_argument("--save-dir", default="runs/final_model")
    p.add_argument("--backbone", choices=["prott5", "esm1b"], default="prott5")
    p.add_argument("--nterm", type=int, default=250)
    p.add_argument("--pooling", default="mean_prefix_50")
    p.add_argument("--seed", type=int, default=200)
    args = p.parse_args()

    logger = setup_logger("INFO")
    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(
        data_path=args.data,
        save_dir=args.save_dir,
        backbone=args.backbone,
        prott5_name="Rostlab/prot_t5_xl_uniref50",
        esm1b_name="facebook/esm1b_t33_650M_UR50S",
        last_k=4,
        nterm_max_res=args.nterm,
        pooling=args.pooling,
        use_homology_aware_cv=True,
        min_seq_id=0.30,
        k_folds=5,
        graphpart_threads=2,
        prefer_graphpart_partitions=True,
        seed=args.seed,
        lr=1e-3,
        weight_decay=1e-5,
        h1=512,
        h2=128,
        dropout=0.3,
        label_smoothing=0.05,
        batch_size=64,
        epochs=50,
        early_stop_patience=8,
        force_deterministic=False,
        device=device,
        fp16=(device == "cuda"),
        embed_batch_size=8,
        cache_embeddings=True,
    )

    set_seed(cfg.seed, cfg.force_deterministic)

    df, le = load_dataset(cfg.data_path)
    joblib.dump(le, os.path.join(cfg.save_dir, "label_encoder.pkl"))
    logger.info("Loaded %d sequences, %d classes: %s", len(df), len(le.classes_), list(le.classes_))

    logger.info("Computing embeddings for all %d sequences...", len(df))
    X = compute_features(df["sequence_clean"].astype(str).tolist(), cfg)
    y = df["label_enc"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(cfg.save_dir, "scaler.pkl"))

    tr_ds = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)

    model = PooledMLP(X_scaled.shape[1], len(le.classes_), cfg.h1, cfg.h2, cfg.dropout).to(torch.device(device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    logger.info("Training final model on all %d sequences for %d epochs...", len(df), cfg.epochs)
    for ep in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        if ep % 10 == 0:
            logger.info("Epoch %d/%d  loss=%.4f", ep, cfg.epochs, total_loss / len(tr_loader))

    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "model.pt"))
    logger.info("Saved model to %s", args.save_dir)
    logger.info("Files saved: model.pt, scaler.pkl, label_encoder.pkl")


if __name__ == "__main__":
    main()
