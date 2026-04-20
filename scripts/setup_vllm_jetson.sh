#!/usr/bin/env bash
# ============================================================
# Setup vLLM on Jetson Orin NX (JetPack 6.x / CUDA 12.x)
# ============================================================
#
# vLLM provides PagedAttention — the production-grade paged KV
# cache management with custom CUDA kernels.
#
# Option A: pip install (may work on JetPack 6.1+)
# Option B: Build from source (more reliable on Jetson)
# ============================================================

set -e

echo "========================================="
echo " vLLM Setup for Jetson Orin NX"
echo "========================================="

# ---- Check prerequisites ----
echo "[1/5] Checking prerequisites..."

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit:"
    echo "  sudo apt install nvidia-cuda-toolkit"
    exit 1
fi

python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "ERROR: PyTorch CUDA not available."
    echo "Install Jetson PyTorch wheel from NVIDIA first."
    exit 1
}

CUDA_VER=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
echo "  CUDA: $CUDA_VER"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  Arch: $(uname -m)"

# ---- Option A: pip install ----
echo ""
echo "[2/5] Attempting pip install..."
echo "  (This may fail on ARM64 — fall back to Option B)"

if pip install vllm 2>/dev/null; then
    echo "  vLLM installed via pip ✓"
    python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"
    exit 0
fi

echo "  pip install failed. Building from source..."

# ---- Option B: Build from source ----
echo ""
echo "[3/5] Cloning vLLM..."

VLLM_DIR="${HOME}/vllm-source"
if [ -d "$VLLM_DIR" ]; then
    echo "  $VLLM_DIR already exists, pulling latest..."
    cd "$VLLM_DIR" && git pull
else
    git clone https://github.com/vllm-project/vllm.git "$VLLM_DIR"
    cd "$VLLM_DIR"
fi

echo ""
echo "[4/5] Installing build dependencies..."
pip install -U pip setuptools wheel
pip install -r requirements-build.txt 2>/dev/null || true

echo ""
echo "[5/5] Building vLLM (this may take 30-60 minutes)..."

# Jetson Orin NX uses sm_87 (Ampere)
export TORCH_CUDA_ARCH_LIST="8.7"
export MAX_JOBS=4  # Orin NX has limited RAM for compilation

pip install -e . --no-build-isolation

echo ""
echo "========================================="
echo " vLLM installed successfully!"
echo "========================================="
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"
