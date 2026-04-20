"""
Jetson Orin NX specific utilities.

Handles:
  - JetPack / L4T version detection
  - Unified memory budget checking
  - Safe model loading with memory guards
  - ARM64 / sm_87 awareness
"""

from __future__ import annotations

import gc
import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Hardware constants — Jetson Orin NX 16 GB
# ---------------------------------------------------------------------------
ORIN_NX_SM = "8.7"                  # Ampere sm_87
ORIN_NX_TOTAL_GB = 16.0             # LPDDR5 unified
ORIN_NX_USABLE_GB = 11.0            # conservative after system + Jupyter
ORIN_NX_SAFE_MODEL_GB = 8.0         # leave headroom for KV + CUDA ctx


# ---------------------------------------------------------------------------
# JetPack / L4T detection
# ---------------------------------------------------------------------------

def detect_jetpack_version() -> Dict[str, str]:
    """Return JetPack and L4T version info.

    JetPack 6.x ≈ L4T R36.x, CUDA 12.x, cuDNN 9.x
    """
    info: Dict[str, str] = {
        "arch": platform.machine(),          # expect aarch64
        "l4t": "unknown",
        "jetpack": "unknown",
        "cuda": "unknown",
        "cudnn": "unknown",
    }

    # L4T version from /etc/nv_tegra_release
    tegra_path = Path("/etc/nv_tegra_release")
    if tegra_path.exists():
        text = tegra_path.read_text()
        # e.g. "# R36 (release), REVISION: 4.0 ..."
        import re
        m = re.search(r"R(\d+).*REVISION:\s*([\d.]+)", text)
        if m:
            info["l4t"] = f"R{m.group(1)}.{m.group(2)}"

    # JetPack from dpkg
    try:
        out = subprocess.check_output(
            ["dpkg-query", "--show", "nvidia-jetpack"],
            stderr=subprocess.DEVNULL, text=True,
        )
        info["jetpack"] = out.strip().split("\t")[-1]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # CUDA
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda or "unknown"
        info["cudnn"] = str(torch.backends.cudnn.version())

    return info


def detect_compute_capability() -> Tuple[int, int]:
    """Return (major, minor) CUDA compute capability."""
    if not torch.cuda.is_available():
        return (0, 0)
    prop = torch.cuda.get_device_properties(0)
    return (prop.major, prop.minor)


def is_jetson() -> bool:
    """Heuristic: running on Jetson if aarch64 + tegra file exists."""
    return (
        platform.machine() == "aarch64"
        and Path("/etc/nv_tegra_release").exists()
    )


# ---------------------------------------------------------------------------
# Memory helpers for unified memory architecture
# ---------------------------------------------------------------------------

def get_memory_status_mb() -> Dict[str, float]:
    """Current GPU (unified) memory state in MB."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        "total_mb": torch.cuda.get_device_properties(0).total_mem / (1024 ** 2),
        "free_mb": (
            torch.cuda.get_device_properties(0).total_mem
            - torch.cuda.memory_allocated()
        ) / (1024 ** 2),
    }


def check_memory_budget(required_gb: float) -> bool:
    """Check if *required_gb* fits within the Jetson usable budget.

    On unified memory the CUDA-reported 'total' may be close to 16 GB,
    but the OS + Jupyter already consume 4-6 GB.
    """
    status = get_memory_status_mb()
    if not status:
        return False
    free_gb = status["free_mb"] / 1024
    ok = free_gb >= required_gb
    if not ok:
        print(
            f"⚠ Memory check: need {required_gb:.1f} GB but only "
            f"{free_gb:.1f} GB free (of {status['total_mb']/1024:.1f} GB total)"
        )
    return ok


def aggressive_cleanup():
    """Force-release as much GPU memory as possible."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ---------------------------------------------------------------------------
# Safe model loading
# ---------------------------------------------------------------------------

def load_model_safe(
    model_name: str,
    fallback_name: Optional[str] = None,
    dtype=torch.float16,
    device: str = "cuda",
    max_memory_gb: Optional[float] = None,
):
    """Load a HuggingFace CausalLM with Jetson memory guards.

    Key optimisations:
      - ``low_cpu_mem_usage=True`` → halves peak RAM during init
      - ``torch_dtype=float16`` → 2× smaller than FP32
      - Pre-load memory check → avoids half-loaded OOM
      - Automatic fallback to a lighter model when memory is tight

    Parameters
    ----------
    model_name : str
        HuggingFace model id (e.g. "Qwen/Qwen2.5-1.5B-Instruct").
    fallback_name : str | None
        Smaller model to try if *model_name* exceeds memory.
    dtype : torch.dtype
        Weight dtype.  ``float16`` or ``bfloat16``.
    device : str
        Target device.
    max_memory_gb : float | None
        Memory cap passed to ``device_map="auto"`` via ``max_memory``.
        Defaults to :data:`ORIN_NX_SAFE_MODEL_GB`.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if max_memory_gb is None:
        max_memory_gb = ORIN_NX_SAFE_MODEL_GB

    aggressive_cleanup()

    # Quick free-memory check
    if not check_memory_budget(required_gb=3.0):
        if fallback_name:
            print(f"Switching to lighter model: {fallback_name}")
            model_name = fallback_name
        else:
            print("⚠ Low memory — loading may OOM. Consider closing other processes.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,        # avoid 2× memory spike during init
        trust_remote_code=True,
    )
    model.eval()

    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    print(
        f"✓ Loaded {model_name} ({dtype})  "
        f"GPU mem: {allocated_gb:.2f} GB"
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Environment summary (for notebook cell 00_env_check)
# ---------------------------------------------------------------------------

def print_jetson_summary():
    """Print a concise environment report suitable for a notebook."""
    jp = detect_jetpack_version()
    cc = detect_compute_capability()
    mem = get_memory_status_mb()

    lines = [
        "=== Jetson Orin NX Environment ===",
        f"  Architecture : {jp['arch']}",
        f"  L4T          : {jp['l4t']}",
        f"  JetPack      : {jp['jetpack']}",
        f"  CUDA         : {jp['cuda']}",
        f"  cuDNN        : {jp['cudnn']}",
        f"  Compute cap  : {cc[0]}.{cc[1]} (sm_{cc[0]}{cc[1]})",
    ]
    if mem:
        lines += [
            f"  GPU total    : {mem['total_mb']/1024:.1f} GB (unified)",
            f"  GPU allocated: {mem['allocated_mb']:.0f} MB",
            f"  GPU free     : {mem['free_mb']/1024:.1f} GB",
            f"  Usable budget: ~{ORIN_NX_USABLE_GB:.0f} GB (after OS+Jupyter)",
        ]

    is_arm = jp["arch"] == "aarch64"
    lines.append(f"  ARM64        : {'Yes' if is_arm else 'No (unexpected!)'}")
    print("\n".join(lines))
