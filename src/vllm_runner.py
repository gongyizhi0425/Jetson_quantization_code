"""
vLLM runner for PagedAttention experiments on Jetson Orin NX.

vLLM provides *real* PagedAttention with custom CUDA kernels that:
  - Manage KV cache as fixed-size blocks (pages) in GPU memory
  - Eliminate memory fragmentation
  - Enable efficient memory sharing for parallel sequences

Setup on Jetson (JetPack 6.x):
  See scripts/setup_vllm_jetson.sh

If vLLM is not installed, this module falls back to a manual
measurement approach (no real PagedAttention benefits).
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


@dataclass
class VLLMMetrics:
    """Metrics from a vLLM generation run."""
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    total_time_ms: float = 0.0
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    generated_text: str = ""


def check_vllm_available() -> bool:
    """Check if vLLM is importable."""
    return VLLM_AVAILABLE


def create_vllm_engine(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    gpu_memory_utilization: float = 0.60,
    max_model_len: int = 4096,
    block_size: int = 16,
    dtype: str = "float16",
) -> "LLM":
    """Create a vLLM LLM engine with PagedAttention.

    Parameters
    ----------
    model_name : str
        HuggingFace model name.
    gpu_memory_utilization : float
        Fraction of **reported** GPU memory vLLM can use.
        On Jetson 16GB unified memory, 0.60 ≈ 9.6 GB reported,
        but OS already uses ~4-6 GB, so real headroom is smaller.
        Keep this conservative to avoid system instability.
    max_model_len : int
        Maximum sequence length (input + output).
    block_size : int
        PagedAttention block size (tokens per page).
    dtype : str
        Model dtype.

    Returns
    -------
    vllm.LLM engine
    """
    if not VLLM_AVAILABLE:
        raise RuntimeError(
            "vLLM is not installed. See scripts/setup_vllm_jetson.sh\n"
            "Install with: pip install vllm  (or build from source on Jetson)"
        )

    engine = LLM(
        model=model_name,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        block_size=block_size,
        trust_remote_code=True,
        enforce_eager=True,  # disable CUDA graph on Jetson for stability
    )
    return engine


def run_vllm_benchmark(
    engine: "LLM",
    prompts: List[Dict[str, str]],
    max_new_tokens: int = 256,
    warmup_runs: int = 2,
) -> List[Dict]:
    """Run benchmark using vLLM engine (PagedAttention enabled).

    vLLM handles TTFT/TPOT differently — we measure wall-clock time
    and derive per-token latency.
    """
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,        # greedy
        top_p=1.0,
    )

    # Warmup
    warmup_texts = [prompts[0]["prompt"]] * warmup_runs
    _ = engine.generate(warmup_texts, SamplingParams(max_tokens=16, temperature=0))

    results = []
    for item in prompts:
        prompt_text = item["prompt"]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = engine.generate([prompt_text], sampling_params)

        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000.0

        output = outputs[0]
        n_input = len(output.prompt_token_ids)
        n_output = len(output.outputs[0].token_ids) if output.outputs else 0
        gen_text = output.outputs[0].text if output.outputs else ""

        # Approximate TTFT and TPOT from vLLM metrics if available
        ttft_ms = 0.0
        tpot_ms = 0.0
        if hasattr(output, "metrics") and output.metrics is not None:
            if hasattr(output.metrics, "first_token_time"):
                ttft_ms = (output.metrics.first_token_time - output.metrics.first_scheduled_time) * 1000
            if hasattr(output.metrics, "finished_time") and n_output > 1:
                decode_time = (output.metrics.finished_time - output.metrics.first_token_time) * 1000
                tpot_ms = decode_time / (n_output - 1)
        else:
            # Fallback: rough estimation
            if n_output > 0:
                ttft_ms = total_ms * (n_input / (n_input + n_output))
                tpot_ms = (total_ms - ttft_ms) / max(n_output - 1, 1)

        results.append({
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "total_time_ms": total_ms,
            "num_input_tokens": n_input,
            "num_output_tokens": n_output,
            "question": item.get("question", ""),
            "pubid": item.get("pubid", ""),
            "sample_text": gen_text[:200],
            "config": "GQA+PagedAttn(vLLM)",
        })

    return results


def get_vllm_cache_stats(engine: "LLM") -> Dict:
    """Extract PagedAttention memory stats from vLLM engine."""
    stats = {}
    try:
        scheduler = engine.llm_engine.scheduler
        if hasattr(scheduler, "block_manager"):
            bm = scheduler.block_manager
            stats["num_total_gpu_blocks"] = getattr(bm, "num_total_gpu_blocks", -1)
            stats["num_free_gpu_blocks"] = getattr(bm, "num_free_gpu_blocks", -1)
            if hasattr(bm, "block_size"):
                stats["block_size"] = bm.block_size
    except Exception:
        pass
    return stats
