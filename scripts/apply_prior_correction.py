#!/usr/bin/env python3
"""
Apply prior (class-imbalance) correction to model predictions, and optionally
compare the corrected predictions against an ASAFind .tab file.

BACKGROUND:
The model was trained on a class-balanced-ish 262-sequence reference set
(SP-only 35.1%, Plastid 27.1%, PPC 16.8%, noTP 11.1%, mTP 9.9%). Real
genome-wide data has very different class frequencies (e.g. mostly
Plastid/noTP, very little SP-only/PPC). This causes the model to
systematically over-predict the classes that were over-represented in
training. This script corrects for that using logit adjustment:

    corrected_score[c] = log(prob[c]) - log(train_prior[c]) + log(target_prior[c])

then re-picks the argmax. This is a standard technique for correcting a
label-shift mismatch between training and deployment distributions,
without retraining the model.

USAGE:
    Step 1: Run predict.py with --probs-out to get full probabilities:
        python predict.py --model-dir runs/final_model \
            --fasta Bacillariophyta_fasta_for_targetp2_dedup.fsa \
            --probs-out bacillariophyta_probs.csv

    Step 2: Run this script:
        python apply_prior_correction.py \
            --probs bacillariophyta_probs.csv \
            --asafind-tab Bacillariophyta_fasta_for_targetp2_dedup.fsa.tab \
            --target-prior-from-asafind \
            --out bacillariophyta_corrected.csv

    This will:
      - estimate the target class distribution from ASAFind's own predictions
        on this dataset (a pragmatic proxy for the true distribution)
      - apply the correction
      - compare corrected predictions against ASAFind's calls
      - print before/after agreement so you can see if it helped
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter


# Training class counts from data.xlsx (262 sequences, Gruber et al. 2025
# reference set). Hardcoded here since this is fixed and known.
TRAIN_COUNTS = {
    "Plastid": 71,
    "PPC": 44,
    "Not plastid, not ppc, signal peptide identified": 92,
    "TargetP-2.0: mTP": 26,
    "TargetP-2.0: noTP": 29,
}


def normalize_asafind(label: str) -> str:
    label = label.strip()
    if label.startswith("Plastid"):
        return "Plastid"
    if label == "Not plastid, signal peptide identified":
        return "Not plastid, not ppc, signal peptide identified"
    if "PPC" in label:
        return "PPC"
    return label


def load_probs(path: str):
    """Returns (seq_ids, classes, prob_matrix as dict[seq_id] -> dict[class]->prob)"""
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        classes = header[1:]
        data = {}
        for row in reader:
            sid = row[0]
            probs = {c: float(v) for c, v in zip(classes, row[1:])}
            data[sid] = probs
    return classes, data


def load_asafind_tab(path: str):
    results = {}
    with open(path, encoding="utf-8") as f:
        f.readline()  # header
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            cols = line.split("\t")
            seq_id = cols[0].strip()
            results[seq_id] = normalize_asafind(cols[-1].strip())
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="CSV of full per-class probabilities from predict.py --probs-out")
    ap.add_argument("--asafind-tab", default=None, help="ASAFind .tab file (for target-prior estimation and/or comparison)")
    ap.add_argument("--target-prior-from-asafind", action="store_true",
                     help="Estimate the target class distribution from ASAFind's own predictions on this dataset")
    ap.add_argument("--out", default="corrected_predictions.csv")
    args = ap.parse_args()

    classes, probs_data = load_probs(args.probs)
    print(f"Loaded probabilities for {len(probs_data)} sequences, classes: {classes}")

    train_total = sum(TRAIN_COUNTS.values())
    train_prior = {c: TRAIN_COUNTS[c] / train_total for c in classes}

    asafind_preds = None
    if args.asafind_tab:
        asafind_preds = load_asafind_tab(args.asafind_tab)
        print(f"Loaded {len(asafind_preds)} ASAFind predictions")

    if args.target_prior_from_asafind:
        if asafind_preds is None:
            raise SystemExit("--target-prior-from-asafind requires --asafind-tab")
        counts = Counter(asafind_preds.values())
        total = sum(counts.values())
        target_prior = {c: max(counts.get(c, 0), 1) / total for c in classes}
        # max(...,1) avoids log(0) if a class never appears in ASAFind's output
        print("\nEstimated target prior (from ASAFind's own predictions):")
        for c in classes:
            print(f"  {c:55s} train={train_prior[c]:.3f}  target={target_prior[c]:.3f}")
    else:
        target_prior = train_prior  # no correction, sanity fallback
        print("No target prior specified — no correction will be applied (target = train).")

    log_adjust = {c: math.log(target_prior[c]) - math.log(train_prior[c]) for c in classes}

    rows = []
    n_changed = 0
    for sid, probs in probs_data.items():
        orig_pred = max(probs, key=probs.get)
        adjusted = {c: math.log(max(probs[c], 1e-12)) + log_adjust[c] for c in classes}
        corrected_pred = max(adjusted, key=adjusted.get)
        if corrected_pred != orig_pred:
            n_changed += 1
        rows.append({
            "sequence_id": sid,
            "original_prediction": orig_pred,
            "corrected_prediction": corrected_pred,
            "changed": corrected_pred != orig_pred,
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence_id", "original_prediction", "corrected_prediction", "changed"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}")
    print(f"Predictions changed by correction: {n_changed}/{len(rows)} ({100*n_changed/len(rows):.1f}%)")

    if asafind_preds:
        n_matched = 0
        n_agree_before = 0
        n_agree_after = 0
        for r in rows:
            sid = r["sequence_id"]
            if sid not in asafind_preds:
                continue
            n_matched += 1
            if r["original_prediction"] == asafind_preds[sid]:
                n_agree_before += 1
            if r["corrected_prediction"] == asafind_preds[sid]:
                n_agree_after += 1

        print(f"\n=== Agreement with ASAFind ===")
        print(f"Matched sequences: {n_matched}")
        print(f"BEFORE correction: {n_agree_before}/{n_matched} ({100*n_agree_before/n_matched:.1f}%)")
        print(f"AFTER correction:  {n_agree_after}/{n_matched} ({100*n_agree_after/n_matched:.1f}%)")


if __name__ == "__main__":
    main()
