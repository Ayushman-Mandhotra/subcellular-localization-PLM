#!/usr/bin/env python3
"""
Flask web application for PLM-based protein subcellular localization prediction.

Supports:
  - Single sequence input
  - FASTA file upload (batch prediction)
  - Optional ASAFind .tab comparison
  - CSV / JSON export of results
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import time
import traceback

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from model_loader import ModelSingleton

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload

# Store results in-memory keyed by session (for download)
_results_cache: dict = {}

# ── helpers ──────────────────────────────────────────────────────────────

def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into list of (id, sequence) tuples."""
    entries = []
    current_id, current_seq = None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                entries.append((current_id, "".join(current_seq)))
            current_id = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line)
    if current_id is not None:
        entries.append((current_id, "".join(current_seq)))
    return entries


ASAFIND_MAP = {
    "plastid": "Plastid",
    "ppc": "PPC",
    "not plastid / not ppc (sp present)": "Not plastid/not ppc/SP-identified",
    "not plastid/not ppc/sp-identified": "Not plastid/not ppc/SP-identified",
    "sp-only": "Not plastid/not ppc/SP-identified",
    "targetp-2.0: mtp": "TargetP-2.0: mTP",
    "mtp": "TargetP-2.0: mTP",
    "targetp-2.0: notp": "TargetP-2.0: noTP",
    "notp": "TargetP-2.0: noTP",
    "other": "TargetP-2.0: noTP",
}


def normalize_asafind_label(raw: str) -> str:
    """Normalize ASAFind label to match model's class names."""
    key = raw.strip().lower()
    return ASAFIND_MAP.get(key, raw.strip())


def parse_asafind_tab(text: str) -> dict[str, str]:
    """Parse ASAFind .tab file → {seq_id: normalized_label}."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        seq_id = parts[0].strip()
        label = parts[-1].strip()  # last column is usually the classification
        if seq_id and label:
            result[seq_id] = normalize_asafind_label(label)
    return result


# ── routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    loader = ModelSingleton.get()
    return render_template(
        "index.html",
        class_names=loader.class_names,
        device=loader.device,
    )


@app.route("/predict", methods=["POST"])
def predict():
    loader = ModelSingleton.get()

    seq_ids = []
    raw_seqs = []

    # Source 1: single sequence text
    single_seq = request.form.get("sequence", "").strip()
    if single_seq:
        # Check if it looks like FASTA
        if single_seq.startswith(">"):
            entries = parse_fasta(single_seq)
        else:
            entries = [("input_sequence", re.sub(r"\s+", "", single_seq))]
        for sid, seq in entries:
            seq_ids.append(sid)
            raw_seqs.append(seq)

    # Source 2: uploaded FASTA file
    fasta_file = request.files.get("fasta_file")
    if fasta_file and fasta_file.filename:
        text = fasta_file.read().decode("utf-8", errors="replace")
        entries = parse_fasta(text)
        for sid, seq in entries:
            seq_ids.append(sid)
            raw_seqs.append(seq)

    if not seq_ids:
        flash("Please provide at least one sequence or upload a FASTA file.", "error")
        return redirect(url_for("index"))

    try:
        t0 = time.time()
        results = loader.predict(seq_ids, raw_seqs)
        elapsed = time.time() - t0
    except Exception as e:
        traceback.print_exc()
        flash(f"Prediction error: {e}", "error")
        return redirect(url_for("index"))

    # Optional: compare with ASAFind .tab
    asafind_labels = {}
    tab_file = request.files.get("asafind_tab")
    if tab_file and tab_file.filename:
        tab_text = tab_file.read().decode("utf-8", errors="replace")
        asafind_labels = parse_asafind_tab(tab_text)

    # Attach ASAFind comparison if available
    agree_count = 0
    compared_count = 0
    for r in results:
        sid = r["seq_id"]
        if sid in asafind_labels:
            r["asafind_label"] = asafind_labels[sid]
            r["agree"] = r["predicted_class"] == asafind_labels[sid]
            compared_count += 1
            if r["agree"]:
                agree_count += 1

    # Cache for download
    cache_key = str(time.time())
    _results_cache[cache_key] = results

    return render_template(
        "results.html",
        results=results,
        n_sequences=len(results),
        elapsed=f"{elapsed:.1f}",
        class_names=loader.class_names,
        has_asafind=bool(asafind_labels),
        agree_count=agree_count,
        compared_count=compared_count,
        cache_key=cache_key,
    )


@app.route("/download/<cache_key>/<fmt>")
def download(cache_key, fmt):
    results = _results_cache.get(cache_key)
    if not results:
        flash("Results expired. Please re-run the prediction.", "error")
        return redirect(url_for("index"))

    loader = ModelSingleton.get()

    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        header = ["sequence_id", "predicted_class", "confidence"]
        header += loader.class_names
        if results[0].get("asafind_label"):
            header += ["asafind_label", "agree"]
        writer.writerow(header)

        for r in results:
            row = [r["seq_id"], r["predicted_class"], f"{r['confidence']:.4f}"]
            row += [f"{r['probabilities'][c]:.6f}" for c in loader.class_names]
            if r.get("asafind_label"):
                row += [r["asafind_label"], str(r.get("agree", ""))]
            writer.writerow(row)

        buf = io.BytesIO(output.getvalue().encode("utf-8"))
        return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="predictions.csv")

    elif fmt == "json":
        buf = io.BytesIO(json.dumps(results, indent=2).encode("utf-8"))
        return send_file(buf, mimetype="application/json", as_attachment=True, download_name="predictions.json")

    flash("Unknown format.", "error")
    return redirect(url_for("index"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for programmatic access."""
    loader = ModelSingleton.get()

    data = request.get_json(force=True)
    sequences = data.get("sequences", [])  # list of {id, sequence}

    if not sequences:
        return jsonify({"error": "No sequences provided"}), 400

    seq_ids = [s.get("id", f"seq_{i}") for i, s in enumerate(sequences)]
    raw_seqs = [s["sequence"] for s in sequences]

    try:
        results = loader.predict(seq_ids, raw_seqs)
        return jsonify({"results": results, "class_names": loader.class_names})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": ModelSingleton.get().device})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None, help="Path to model directory (default: runs/final_model)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Pre-load model
    ModelSingleton.get(args.model_dir)

    app.run(host=args.host, port=args.port, debug=args.debug)
