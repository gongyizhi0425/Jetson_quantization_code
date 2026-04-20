#!/usr/bin/env bash
# ============================================================
# Deploy KV-Cache Experiment to Jetson Orin NX
# ============================================================
# Run on your HOST machine (not Jetson).
# Prerequisites:
#   - Jetson is accessible via SSH (key-based auth recommended)
#   - JetPack 6.x already flashed on Jetson
# ============================================================

set -euo pipefail

# ---- Configuration (edit these) ----
JETSON_USER="${JETSON_USER:-gyz}"
JETSON_HOST="${JETSON_HOST:-jetson-orin}"        # hostname or IP
JETSON_DIR="${JETSON_DIR:-/home/${JETSON_USER}/KV_cache_experiment}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"    # repo root

echo "========================================="
echo " Deploy KV-Cache Experiment to Jetson"
echo "========================================="
echo "  Source : $LOCAL_DIR"
echo "  Target : ${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}"
echo ""

# ---- Step 1: Sync files via rsync ----
echo "[1/4] Syncing project files..."
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude 'venv/' \
    --exclude '.venv/' \
    --exclude '*.pyc' \
    --exclude 'results/' \
    --exclude 'data/pubmedqa_cache/' \
    "$LOCAL_DIR/" \
    "${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}/"

echo ""

# ---- Step 2: Install base dependencies on Jetson ----
echo "[2/4] Installing Python dependencies on Jetson..."
ssh "${JETSON_USER}@${JETSON_HOST}" << 'REMOTE_SCRIPT'
set -e
cd ~/KV_cache_experiment

# Create venv if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -U pip

# Core dependencies (PyTorch should already be installed via NVIDIA wheel)
pip install -r requirements_jetson.txt

echo ""
echo "Verifying PyTorch..."
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
"
REMOTE_SCRIPT

echo ""

# ---- Step 3: Setup JupyterLab on Jetson ----
echo "[3/4] Setting up JupyterLab on Jetson..."
ssh "${JETSON_USER}@${JETSON_HOST}" << 'REMOTE_SCRIPT'
set -e
cd ~/KV_cache_experiment
source venv/bin/activate

pip install jupyterlab ipykernel
python3 -m ipykernel install --user --name kv_cache --display-name "KV Cache (venv)"

# Generate Jupyter config for remote access
jupyter lab --generate-config 2>/dev/null || true

echo ""
echo "JupyterLab installed. Start with:"
echo "  cd ~/KV_cache_experiment && source venv/bin/activate"
echo "  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
REMOTE_SCRIPT

echo ""

# ---- Step 4: Print instructions ----
echo "[4/4] Done! Next steps:"
echo ""
echo "  1. SSH into Jetson:"
echo "       ssh ${JETSON_USER}@${JETSON_HOST}"
echo ""
echo "  2. Start JupyterLab:"
echo "       cd ~/KV_cache_experiment"
echo "       source venv/bin/activate"
echo "       jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
echo ""
echo "  3. Open in browser from your host:"
echo "       http://${JETSON_HOST}:8888"
echo ""
echo "  4. Select kernel 'KV Cache (venv)' in each notebook"
echo ""
echo "  5. Run notebooks in order: 00 → 01 → 02 → 03 → 04 → 05 → 06"
echo ""
echo "  6. (Optional) Open a separate terminal for system monitoring:"
echo "       sudo jtop    # jetson-stats GPU/memory monitor"
echo ""
echo "  7. (Optional) Install vLLM and KIVI:"
echo "       bash ~/KV_cache_experiment/scripts/setup_vllm_jetson.sh"
echo "       bash ~/KV_cache_experiment/scripts/setup_kivi_jetson.sh"
echo "========================================="
