#!/usr/bin/env python3
"""
Compare your model's predictions (from predict.py) against ASAFind 2.0's
predictions for the same sequences.

USAGE:
    1. Run your model and save its printed output to a file:
         python predict.py --model-dir runs/final_model \
             --fasta Kareniaceae_fasta_for_targetp2.fsa > kareniaceae_predictions.txt

    2. Run this script:
         python compare_predictions.py \
             --model-output kareniaceae_predictions.txt \
             --asafind-tab Kareniaceae_fasta_for_targetp2_fsa.tab \
             --out comparison.csv
"""
from __future__ import annotations

import argparse
import re
import csv


def normalize_asafind(label: str) -> str:
    """Map ASAFind 2.0's output category text to the model's label scheme."""
    label = label.strip()
    if label.startswith("Plastid"):
        return "Plastid"
    if label == "Not plastid, signal peptide identified":
        return "Not plastid, not ppc, signal peptide identified"
    if "PPC" in label:
        return "PPC"
    # TargetP-2.0: noTP / TargetP-2.0: mTP already match the model's wording
    return label


def parse_model_output(path: str) -> dict[str, tuple[str, float]]:
    """Parse the printed table from predict.py.

    Each result line looks like:
        <ID (no whitespace)>   <Prediction label (may contain spaces)>   <NN.N%>
    """
    results: dict[str, tuple[str, float]] = {}
    line_re = re.compile(r"^(\S+)\s+(.+?)\s+([\d.]+)%\s*$")
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line or line.startswith("ID") or set(line.strip()) == {"-"}:
                continue
            if line.startswith("All class probabilities"):
                break  # stop at the per-class probability dump for single-seq mode
            m = line_re.match(line)
            if not m:
                continue
            seq_id, label, conf = m.group(1), m.group(2).strip(), float(m.group(3))
            results[seq_id] = (label, conf)
    return results


def parse_asafind_tab(path: str) -> dict[str, str]:
    """Parse the ASAFind/TargetP .tab file; take the last column as the final call."""
    results: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            cols = line.split("\t")
            seq_id = cols[0].strip()
            final_call = cols[-1].strip()
            results[seq_id] = normalize_asafind(final_call)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-output", required=True, help="Saved stdout from predict.py")
    ap.add_argument("--asafind-tab", required=True, help="ASAFind/TargetP .tab file")
    ap.add_argument("--out", default="comparison.csv", help="Output CSV path")
    args = ap.parse_args()

    model_preds = parse_model_output(args.model_output)
    asafind_preds = parse_asafind_tab(args.asafind_tab)

    print(f"Parsed {len(model_preds)} model predictions")
    print(f"Parsed {len(asafind_preds)} ASAFind predictions")

    all_ids = sorted(set(model_preds) | set(asafind_preds))
    rows = []
    n_matched = 0
    n_agree = 0

    for seq_id in all_ids:
        model_label, model_conf = model_preds.get(seq_id, ("MISSING", None))
        asafind_label = asafind_preds.get(seq_id, "MISSING")
        matched = model_label != "MISSING" and asafind_label != "MISSING"
        agree = matched and (model_label == asafind_label)
        if matched:
            n_matched += 1
            if agree:
                n_agree += 1
        rows.append({
            "sequence_id": seq_id,
            "model_prediction": model_label,
            "model_confidence": model_conf,
            "asafind_prediction": asafind_label,
            "agree": agree if matched else "N/A",
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}")
    print(f"Sequences present in both outputs: {n_matched}")
    if n_matched:
        print(f"Agreement: {n_agree}/{n_matched} ({100 * n_agree / n_matched:.1f}%)")

    # Simple per-class breakdown
    print("\nDisagreements (model vs ASAFind):")
    disagree_counts: dict[tuple[str, str], int] = {}
    for r in rows:
        if r["agree"] is False:
            key = (r["model_prediction"], r["asafind_prediction"])
            disagree_counts[key] = disagree_counts.get(key, 0) + 1
    for (m, a), count in sorted(disagree_counts.items(), key=lambda x: -x[1]):
        print(f"  model='{m}'  vs  asafind='{a}'   x{count}")


if __name__ == "__main__":
    main()
