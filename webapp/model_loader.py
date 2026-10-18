"""
Lazy singleton loader for ProtT5 backbone + PooledMLP classifier.
Loads the ~3 GB ProtT5 model once and keeps it in memory.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Optional, Tuple

import joblib
import numpy as np
import torch

# Add parent dir so plm_localization package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plm_localization.config import Config
from plm_localization.data import clean_and_map
from plm_localization.embeddings import compute_features
from plm_localization.model import PooledMLP


class ModelSingleton:
    """Thread-safe lazy singleton for the prediction pipeline."""

    _instance: Optional["ModelSingleton"] = None
    _lock = threading.Lock()

    def __init__(self):
        raise RuntimeError("Use ModelSingleton.get() instead")

    @classmethod
    def get(cls, model_dir: str = None) -> "ModelSingleton":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = object.__new__(cls)
                    obj._init(model_dir)
                    cls._instance = obj
        return cls._instance

    def _init(self, model_dir: str):
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(__file__), "..", "runs", "final_model"
            )
        self.model_dir = os.path.abspath(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[ModelLoader] Loading from {self.model_dir} on {self.device}")

        # Load label encoder and scaler
        self.le = joblib.load(os.path.join(self.model_dir, "label_encoder.pkl"))
        self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.pkl"))

        # Load MLP head
        in_dim = self.scaler.mean_.shape[0]
        n_classes = len(self.le.classes_)
        self.model = PooledMLP(in_dim, n_classes, h1=512, h2=128, dropout=0.0)
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, "model.pt"),
                map_location=self.device,
            )
        )
        self.model.to(self.device).eval()

        # Build config for embedding pipeline
        self.cfg = Config(
            data_path="",
            save_dir=self.model_dir,
            backbone="prott5",
            prott5_name="Rostlab/prot_t5_xl_uniref50",
            esm1b_name="facebook/esm1b_t33_650M_UR50S",
            last_k=4,
            nterm_max_res=250,
            pooling="mean_prefix_50",
            use_homology_aware_cv=False,
            min_seq_id=0.30,
            k_folds=5,
            graphpart_threads=2,
            prefer_graphpart_partitions=True,
            seed=0,
            lr=1e-3,
            weight_decay=1e-5,
            h1=512,
            h2=128,
            dropout=0.0,
            label_smoothing=0.0,
            batch_size=64,
            epochs=0,
            early_stop_patience=0,
            force_deterministic=False,
            device=self.device,
            fp16=(self.device == "cuda"),
            embed_batch_size=8,
            cache_embeddings=False,
        )

        # Pre-warm: load ProtT5 backbone into memory now
        print("[ModelLoader] Pre-loading ProtT5 backbone (this takes ~30s)...")
        _ = compute_features(["MAAAAA"], self.cfg)
        print("[ModelLoader] Ready!")

    @property
    def class_names(self):
        return list(self.le.classes_)

    def predict(self, seq_ids: list[str], raw_seqs: list[str]) -> list[dict]:
        """
        Run prediction on a list of sequences.

        Returns list of dicts:
          {seq_id, sequence_preview, predicted_class, confidence, probabilities}
        """
        clean_seqs = [clean_and_map(s) for s in raw_seqs]
        X = compute_features(clean_seqs, self.cfg)
        X_scaled = self.scaler.transform(X)

        with torch.no_grad():
            logits = self.model(
                torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_idx = np.argmax(probs, axis=1)

        results = []
        for i, (sid, idx) in enumerate(zip(seq_ids, pred_idx)):
            label = self.le.classes_[idx]
            conf = float(probs[i, idx])
            prob_dict = {
                cls: float(probs[i, j])
                for j, cls in enumerate(self.le.classes_)
            }
            results.append(
                {
                    "seq_id": sid,
                    "sequence_preview": raw_seqs[i][:60] + ("..." if len(raw_seqs[i]) > 60 else ""),
                    "predicted_class": label,
                    "confidence": conf,
                    "probabilities": prob_dict,
                }
            )
        return results
