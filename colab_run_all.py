#!/usr/bin/env python3
"""
========================================================================
Master Colab script: Train & evaluate the PLM subcellular localization
model using ONLY the 262 curated P. tricornutum sequences (data.xlsx).
========================================================================

Run this in Google Colab with a T4 GPU runtime.

STEPS:
  1. Clone repo & install dependencies
  2. Run 5-fold cross-validation (homology-aware) on data.xlsx
  3. Train the final model on all 262 sequences
  4. Predict on test organisms (upload FASTA files)
  5. Compare with ASAFind and compute full Baldi metrics

COLAB USAGE — paste these cells in order:

=== CELL 1: Setup ===
!git clone https://github.com/nkhthorat/subcellular-localization-PLM.git
%cd subcellular-localization-PLM
!pip install torch transformers sentencepiece protobuf scikit-learn openpyxl joblib

=== CELL 2: Cross-validation ===
!python3 scripts/run_experiment.py --data data.xlsx

=== CELL 3: Train final model ===
!python3 scripts/train_final.py --data data.xlsx --save-dir runs/final_model

=== CELL 4: Predict on test organism (example: Kareniaceae) ===
# Upload your FASTA file first, then:
!python3 scripts/predict.py \\
    --model-dir runs/final_model \\
    --fasta Kareniaceae_fasta_for_targetp2.fsa \\
    --probs-out kareniaceae_probs.csv \\
    > kareniaceae_predictions.txt
!cat kareniaceae_predictions.txt

=== CELL 5: Compare with ASAFind ===
# Upload the ASAFind .tab file, then:
!python3 scripts/compare_predictions.py \\
    --model-output kareniaceae_predictions.txt \\
    --asafind-tab Kareniaceae_fasta_for_targetp2_fsa.tab \\
    --out kareniaceae_comparison.csv

=== CELL 6: Full Baldi metrics ===
!python3 scripts/compute_full_metrics.py --csv kareniaceae_comparison.csv

=== CELL 7: Prior correction (optional) ===
!python3 scripts/apply_prior_correction.py \\
    --probs kareniaceae_probs.csv \\
    --asafind-tab Kareniaceae_fasta_for_targetp2_fsa.tab \\
    --target-prior-from-asafind \\
    --out kareniaceae_corrected.csv
"""

print(__doc__)
print("This file documents the Colab workflow.")
print("Copy the cells above into separate Colab cells and run them in order.")
print("Make sure to select a T4 GPU runtime (Runtime > Change runtime type > T4 GPU).")
