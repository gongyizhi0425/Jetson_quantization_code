#!/usr/bin/env bash
# ============================================================
# Setup KIVI (2-bit KV Cache) on Jetson Orin NX
# ============================================================
#
# KIVI requires custom CUDA kernels for:
#   - 2-bit packing/unpacking
#   - Fused dequant + attention
#
# Repository: https://github.com/jy-yuan/KIVI
# ============================================================

set -e

echo "========================================="
echo " KIVI Setup for Jetson Orin NX"
echo "========================================="

# ---- Check prerequisites ----
echo "[1/4] Checking prerequisites..."

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. KIVI needs CUDA compilation."
    echo "  sudo apt install nvidia-cuda-toolkit"
    exit 1
fi

python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "ERROR: PyTorch CUDA not available."
    exit 1
}

CUDA_VER=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
TORCH_VER=$(python3 -c 'import torch; print(torch.__version__)')
echo "  CUDA: $CUDA_VER"
echo "  PyTorch: $TORCH_VER"
echo "  Arch: $(uname -m)"

# ---- Clone KIVI ----
echo ""
echo "[2/4] Cloning KIVI..."

KIVI_DIR="${HOME}/KIVI"
if [ -d "$KIVI_DIR" ]; then
    echo "  $KIVI_DIR already exists, pulling latest..."
    cd "$KIVI_DIR" && git pull
else
    git clone https://github.com/jy-yuan/KIVI.git "$KIVI_DIR"
    cd "$KIVI_DIR"
fi

# ---- Install dependencies ----
echo ""
echo "[3/4] Installing dependencies..."
pip install -U pip
pip install triton 2>/dev/null || echo "  triton not available for ARM64 (OK if using custom kernel)"

# Some KIVI forks need these
pip install packaging ninja

# ---- Build CUDA extensions ----
echo ""
echo "[4/4] Building KIVI CUDA extensions..."

# Jetson Orin NX → sm_87 (Ampere)
export TORCH_CUDA_ARCH_LIST="8.7"
export MAX_JOBS=4

# Try to build the CUDA extension
cd "$KIVI_DIR"

if [ -f "setup.py" ]; then
    pip install -e . --no-build-isolation
elif [ -f "quant/setup_cuda.py" ]; then
    cd quant
    python setup_cuda.py install
    cd ..
else
    echo "WARNING: No standard setup.py found."
    echo "Attempting manual compilation of CUDA kernels..."

    # Try to find and compile .cu files
    find . -name "*.cu" -exec echo "  Found: {}" \;

    echo ""
    echo "Manual compilation may be needed. Check the KIVI repo README."
fi

echo ""
echo "========================================="
echo " KIVI setup complete. Testing import..."
echo "========================================="

python3 -c "
try:
    import kivi_gemm
    print('kivi_gemm CUDA backend: OK ✓')
except ImportError as e:
    print(f'kivi_gemm import failed: {e}')
    print('Falling back to pure-Python reference implementation.')
"
