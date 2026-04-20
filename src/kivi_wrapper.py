"""
KIVI backend wrapper — tries the real CUDA kernel backend first,
falls back to the pure-Python reference implementation.

Real KIVI backend (https://github.com/jy-yuan/KIVI):
  - Custom CUDA kernels for 2-bit packing/unpacking
  - Fused dequant + attention kernel (much faster)
  - Requires compilation with nvcc on Jetson

Setup: See scripts/setup_kivi_jetson.sh
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---- Try to import real KIVI CUDA backend ----
_KIVI_CUDA_AVAILABLE = False
_kivi_real = None

try:
    import kivi_gemm  # the compiled CUDA extension from KIVI repo
    _KIVI_CUDA_AVAILABLE = True
    logger.info("KIVI CUDA backend loaded successfully")
except ImportError:
    logger.warning(
        "KIVI CUDA backend not found. Using pure-Python reference.\n"
        "For real performance, compile KIVI:\n"
        "  cd /path/to/KIVI && pip install -e .\n"
        "  See scripts/setup_kivi_jetson.sh"
    )

try:
    # Some KIVI forks expose the cache class differently
    from kivi import KVQuantCache as _RealKIVICache
    _kivi_real = _RealKIVICache
except ImportError:
    pass


def is_cuda_backend_available() -> bool:
    """Check if the real KIVI CUDA kernels are compiled and importable."""
    return _KIVI_CUDA_AVAILABLE


def get_backend_info() -> str:
    """Return a description of the active KIVI backend."""
    if _KIVI_CUDA_AVAILABLE:
        return "KIVI CUDA backend (compiled kernels)"
    return "KIVI pure-Python reference (no CUDA acceleration)"


# ---- Unified factory ----

def create_kivi_cache(
    residual_length: int = 128,
    group_size: int = 32,
    bits: int = 2,
    prefer_cuda: bool = True,
):
    """Create a KIVI cache instance.

    Tries the real CUDA backend first.  Falls back to the
    pure-Python implementation in ``src/kivi_cache.py``.

    Parameters
    ----------
    residual_length : int
        Recent tokens kept in FP16.
    group_size : int
        Quantization group size.
    bits : int
        Bit-width (default: 2).
    prefer_cuda : bool
        If True and CUDA backend is available, use it.

    Returns
    -------
    Cache object (real or reference)
    """
    if prefer_cuda and _kivi_real is not None:
        logger.info("Using real KIVI CUDA cache")
        return _kivi_real(
            bits=bits,
            group_size=group_size,
            residual_length=residual_length,
        )

    # Fallback to our pure-Python reference
    from .kivi_cache import KIVICache
    logger.info("Using pure-Python KIVI reference cache")
    return KIVICache(
        residual_length=residual_length,
        group_size=group_size,
        bits=bits,
    )
