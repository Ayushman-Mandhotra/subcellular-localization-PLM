#!/bin/bash
set -e

# ============================================================
# PLM Protein Localization Web App — VM Deployment Script
# ============================================================
#
# PREREQUISITES:
#   - Ubuntu VM with GPU (CUDA drivers installed)
#   - Python 3.10+ with pip
#   - The trained model at runs/final_model/
#     (model.pt, scaler.pkl, label_encoder.pkl)
#
# USAGE:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# After running, the app will be available at:
#   http://<VM_IP>:5000
# ============================================================

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WEBAPP_DIR="${REPO_DIR}/webapp"
MODEL_DIR="${REPO_DIR}/runs/final_model"

echo "================================================"
echo "  PLM Localization Web App — Deployment"
echo "================================================"
echo ""
echo "Repo:     ${REPO_DIR}"
echo "Webapp:   ${WEBAPP_DIR}"
echo "Model:    ${MODEL_DIR}"
echo ""

# ── Step 1: Check model files ────────────────────────────
echo "[1/5] Checking model files..."
for f in model.pt scaler.pkl label_encoder.pkl; do
    if [ ! -f "${MODEL_DIR}/${f}" ]; then
        echo "ERROR: Missing ${MODEL_DIR}/${f}"
        echo ""
        echo "You need to train the model first. Run:"
        echo "  cd ${REPO_DIR}"
        echo "  python scripts/train_final.py --data data.xlsx --save-dir runs/final_model"
        exit 1
    fi
done
echo "  All model files found."

# ── Step 2: Create virtual environment ───────────────────
echo ""
echo "[2/5] Setting up Python virtual environment..."
if [ ! -d "${WEBAPP_DIR}/venv" ]; then
    python3 -m venv "${WEBAPP_DIR}/venv"
    echo "  Created venv at ${WEBAPP_DIR}/venv"
else
    echo "  venv already exists."
fi
source "${WEBAPP_DIR}/venv/bin/activate"

# ── Step 3: Install dependencies ─────────────────────────
echo ""
echo "[3/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -r "${WEBAPP_DIR}/requirements.txt" -q
echo "  Dependencies installed."

# ── Step 4: Check GPU ────────────────────────────────────
echo ""
echo "[4/5] Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU available: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
else:
    print('  WARNING: No GPU detected. Running on CPU (slower).')
"

# ── Step 5: Launch ────────────────────────────────────────
echo ""
echo "[5/5] Starting web server..."
echo ""
echo "================================================"
echo "  App will be available at:"
echo "  http://0.0.0.0:5000"
echo ""
echo "  Press Ctrl+C to stop."
echo "================================================"
echo ""

cd "${WEBAPP_DIR}"
exec gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 1 \
    --threads 4 \
    --timeout 300 \
    --preload \
    app:app
