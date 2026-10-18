from __future__ import annotations
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

from .config import Config
from .pooling import pool_residue_embeddings

# Module-level backbone cache to avoid reloading ~3 GB model on every call
_cached_backbone: dict = {}

def to_spaced_tokens(seq: str) -> str:
    return " ".join(list(seq))

@torch.no_grad()
def compute_features(
    seqs_clean: List[str],
    cfg: Config,
) -> np.ndarray:
    """
    Returns pooled features for each sequence under cfg.backbone + cfg.pooling.

    mean_tail_250_500 requires embedding context up to 500 residues; otherwise embed_max_res=nterm_max_res.
    """
    device = torch.device(cfg.device)
    use_amp = (cfg.fp16 and device.type == "cuda")
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    embed_max_res = 500 if cfg.pooling == "mean_tail_250_500" else cfg.nterm_max_res

    if cfg.backbone == "prott5":
        cache_key = ("prott5", cfg.prott5_name, str(device))
        if cache_key in _cached_backbone:
            tok, model = _cached_backbone[cache_key]
        else:
            tok = T5Tokenizer.from_pretrained(cfg.prott5_name, do_lower_case=False)
            if device.type == "cuda":
                model = T5EncoderModel.from_pretrained(cfg.prott5_name, torch_dtype=torch.float16).to(device).eval()
            else:
                model = T5EncoderModel.from_pretrained(cfg.prott5_name).to(device).eval()
            _cached_backbone[cache_key] = (tok, model)

        eos_id = getattr(tok, "eos_token_id", None)
        pad_id = getattr(tok, "pad_token_id", None)

        feats: List[np.ndarray] = []
        for start in range(0, len(seqs_clean), cfg.embed_batch_size):
            chunk = [s[:embed_max_res] for s in seqs_clean[start:start + cfg.embed_batch_size]]
            spaced = [to_spaced_tokens(s) for s in chunk]

            toks = tok(spaced, return_tensors="pt", truncation=False, padding=True, add_special_tokens=True)
            toks = {k: v.to(device) for k, v in toks.items()}

            with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=use_amp):
                out = model(**toks, output_hidden_states=True, return_dict=True)

            if cfg.last_k > 1:
                hs = torch.stack(out.hidden_states[-cfg.last_k:], dim=0).mean(0)
            else:
                hs = out.last_hidden_state

            input_ids = toks["input_ids"]
            attn_mask = toks["attention_mask"].bool()

            for b in range(hs.size(0)):
                keep = attn_mask[b].clone()
                if eos_id is not None:
                    keep &= (input_ids[b] != eos_id)
                if pad_id is not None:
                    keep &= (input_ids[b] != pad_id)

                emb = hs[b][keep].detach().float().cpu()
                pooled = pool_residue_embeddings(emb, cfg.pooling).numpy().astype(np.float32)
                feats.append(pooled)

        return np.stack(feats).astype(np.float32)

    if cfg.backbone == "esm1b":
        tok = AutoTokenizer.from_pretrained(cfg.esm1b_name, do_lower_case=False)
        if device.type == "cuda":
            model = AutoModel.from_pretrained(cfg.esm1b_name, torch_dtype=torch.float16).to(device).eval()
        else:
            model = AutoModel.from_pretrained(cfg.esm1b_name).to(device).eval()

        feats: List[np.ndarray] = []
        for start in range(0, len(seqs_clean), cfg.embed_batch_size):
            chunk = [s[:embed_max_res] for s in seqs_clean[start:start + cfg.embed_batch_size]]
            toks = tok(chunk, return_tensors="pt", truncation=False, padding=True, add_special_tokens=True)
            toks = {k: v.to(device) for k, v in toks.items()}

            with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=use_amp):
                out = model(**toks, output_hidden_states=True, return_dict=True)

            if cfg.last_k > 1 and out.hidden_states is not None:
                hs = torch.stack(out.hidden_states[-cfg.last_k:], dim=0).mean(0)
            else:
                hs = out.last_hidden_state

            for b, seq in enumerate(chunk):
                L = len(seq)
                emb = hs[b, 1:1 + L, :].detach().float().cpu()  # discard CLS/EOS
                pooled = pool_residue_embeddings(emb, cfg.pooling).numpy().astype(np.float32)
                feats.append(pooled)

        return np.stack(feats).astype(np.float32)

    raise ValueError(f"Unknown backbone: {cfg.backbone}")
