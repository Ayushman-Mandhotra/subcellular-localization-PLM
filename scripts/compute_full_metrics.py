#!/usr/bin/env python3
"""
Compute full evaluation metrics from a comparison CSV (model vs ASAFind).

Uses the Baldi et al. (2000) recommended metrics:
  - Matthews Correlation Coefficient (MCC)
  - Per-class sensitivity (recall), precision, F1
  - Macro-F1, Weighted-F1
  - Accuracy (with caveat about class imbalance)
  - Confusion matrix

USAGE:
    python compute_full_metrics.py --csv comparison.csv

The CSV must have columns: model_prediction, asafind_prediction
(from compare_predictions.py output).
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Comparison CSV from compare_predictions.py")
    ap.add_argument("--pred-col", default="model_prediction", help="Column name for predictions")
    ap.add_argument("--true-col", default="asafind_prediction", help="Column name for ground truth / reference")
    args = ap.parse_args()

    y_true, y_pred = [], []
    with open(args.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred = row[args.pred_col]
            true = row[args.true_col]
            if pred == "MISSING" or true == "MISSING":
                continue
            y_true.append(true)
            y_pred.append(pred)

    if not y_true:
        print("No matched sequences found.")
        sys.exit(1)

    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)

    print(f"Matched sequences: {n}")
    print(f"Labels: {labels}\n")

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print("  (Note: accuracy can be misleading under class imbalance)\n")

    # MCC
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")

    # F1 scores
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}\n")

    # Per-class report
    print("Per-class classification report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix (rows=true, cols=predicted):")
    header = f"{'':>55s} " + " ".join(f"{l[:12]:>12s}" for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row_str = " ".join(f"{cm[i, j]:12d}" for j in range(len(labels)))
        print(f"{label:>55s} {row_str}")


if __name__ == "__main__":
    main()
