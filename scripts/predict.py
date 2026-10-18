#!/usr/bin/env python3
"""Predict subcellular localization for new protein sequences."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import os

import csv
import joblib
import numpy as np
import torch

from plm_localization.config import Config
from plm_localization.data import clean_and_map
from plm_localization.embeddings import compute_features
from plm_localization.model import PooledMLP
from plm_localization.utils import setup_logger


def load_model(model_dir: str, device: str):
    le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

    summary_path = os.path.join(model_dir, "summary.json")
    # Read input dim from scaler
    in_dim = scaler.mean_.shape[0]
    n_classes = len(le.classes_)

    model = PooledMLP(in_dim, n_classes, h1=512, h2=128, dropout=0.0)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
    model.to(device).eval()
    return model, scaler, le


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="runs/final_model", help="Folder with model.pt, scaler.pkl, label_encoder.pkl")
    p.add_argument("--sequence", type=str, default=None, help="A single protein sequence string")
    p.add_argument("--fasta", type=str, default=None, help="Path to a FASTA file with one or more sequences")
    p.add_argument("--backbone", choices=["prott5", "esm1b"], default="prott5")
    p.add_argument("--nterm", type=int, default=250)
    p.add_argument("--pooling", default="mean_prefix_50")
    p.add_argument("--probs-out", default=None, help="Save full per-class probabilities to this CSV")
    args = p.parse_args()

    logger = setup_logger("INFO")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.sequence is None and args.fasta is None:
        p.error("Provide either --sequence or --fasta")

    # Collect sequences
    seq_ids = []
    raw_seqs = []

    if args.sequence:
        seq_ids.append("input_sequence")
        raw_seqs.append(args.sequence)

    if args.fasta:
        current_id, current_seq = None, []
        with open(args.fasta) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id is not None:
                        seq_ids.append(current_id)
                        raw_seqs.append("".join(current_seq))
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
        if current_id is not None:
            seq_ids.append(current_id)
            raw_seqs.append("".join(current_seq))

    clean_seqs = [clean_and_map(s) for s in raw_seqs]

    cfg = Config(
        data_path="",
        save_dir=args.model_dir,
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
        seed=0,
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
        cache_embeddings=False,
    )

    logger.info("Embedding %d sequence(s)...", len(clean_seqs))
    X = compute_features(clean_seqs, cfg)

    model, scaler, le = load_model(args.model_dir, device)
    X_scaled = scaler.transform(X)

    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_idx = np.argmax(probs, axis=1)

    print(f"\n{'ID':<30} {'Prediction':<50} {'Confidence'}")
    print("-" * 95)
    for sid, idx, prob_row in zip(seq_ids, pred_idx, probs):
        label = le.classes_[idx]
        confidence = prob_row[idx]
        print(f"{sid:<30} {label:<50} {confidence:.1%}")

    if len(seq_ids) == 1:
        print("\nAll class probabilities:")
        for cls, prob in zip(le.classes_, probs[0]):
            print(f"  {cls:<50} {prob:.1%}")

    if args.probs_out:
        with open(args.probs_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence_id"] + list(le.classes_))
            for sid, prob_row in zip(seq_ids, probs):
                writer.writerow([sid] + [f"{p:.6f}" for p in prob_row])
        print(f"\nFull probabilities saved to {args.probs_out}")


if __name__ == "__main__":
    main()
