"""
TTFT / TPOT measurement utilities for KV-cache experiments.

TTFT (Time To First Token): dominated by memory bandwidth during prefill.
TPOT (Time Per Output Token): reflects per-step KV-cache memory transfer cost.

Jetson Orin NX: Unified Memory architecture — CPU and GPU share 16 GB
LPDDR5.  With system services + Jupyter, only ~10-12 GB is available.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Constants — Jetson Orin NX memory budget
# ---------------------------------------------------------------------------

JETSON_TOTAL_GB = 16.0
JETSON_USABLE_GB = 11.0  # after system + Jupyter


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GenerationMetrics:
    """Metrics collected from a single generation run."""

    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    total_time_ms: float = 0.0
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    peak_memory_mb: float = 0.0
    model_weight_mb: float = 0.0        # separated from KV
    kv_cache_memory_mb: float = 0.0
    memory_fragmentation: float = 0.0   # 1 - allocated/reserved
    memory_utilization: float = 0.0     # peak / usable budget
    token_times_ms: List[float] = field(default_factory=list)
    generated_text: str = ""


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    cache_impl: Optional[Any] = None,
    device: str = "cuda",
) -> GenerationMetrics:
    """Run manual greedy generation and measure TTFT / TPOT.

    Parameters
    ----------
    model : PreTrainedModel
        Loaded HuggingFace causal-LM (e.g. Qwen2.5).
    tokenizer : PreTrainedTokenizer
        Corresponding tokenizer.
    prompt : str
        Input text.
    max_new_tokens : int
        Maximum tokens to generate.
    cache_impl : Cache | None
        Custom KV-cache object.  ``None`` → default ``DynamicCache``.
    device : str
        CUDA device.

    Returns
    -------
    GenerationMetrics
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    # 构建初始 attention_mask 和 position_ids
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

    # Snapshot model-only memory (before any KV allocation)
    torch.cuda.empty_cache()
    model_weight_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)

    # Reset memory tracking
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)

    token_times: List[float] = []
    generated_ids: List[int] = []
    past = cache_impl  # None or custom Cache

    # ---- Prefill --------------------------------------------------------
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past,
        use_cache=True,
    )
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

    torch.cuda.synchronize()
    ttft = (time.perf_counter() - t0) * 1000.0

    generated_ids.append(next_token.item())
    past = outputs.past_key_values
    cur_pos = seq_len  # 当前已处理的 token 位置

    mem_after_prefill = torch.cuda.memory_allocated(device)
    kv_cache_mem_mb = (mem_after_prefill - mem_before) / (1024 ** 2)

    # ---- Decode ---------------------------------------------------------
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens - 1):
        if next_token.item() == eos_id:
            break

        # 逐步扩展 attention_mask，显式传递 position_ids
        # 确保 RoPE 位置编码正确，避免自定义 cache 下位置漂移
        attention_mask = torch.ones(1, cur_pos + 1, dtype=torch.long, device=device)
        step_position_ids = torch.tensor([[cur_pos]], dtype=torch.long, device=device)

        torch.cuda.synchronize()
        t_step = time.perf_counter()

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=step_position_ids,
            past_key_values=past,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t_step) * 1000.0

        token_times.append(step_ms)
        generated_ids.append(next_token.item())
        past = outputs.past_key_values
        cur_pos += 1

    # ---- Collect memory stats -------------------------------------------
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    frag = 1.0 - allocated / reserved if reserved > 0 else 0.0

    # KV-cache-level memory if the cache exposes it
    if hasattr(past, "memory_usage_bytes"):
        kv_cache_mem_mb = past.memory_usage_bytes() / (1024 ** 2)

    # Memory utilization vs Jetson budget
    usable_mb = JETSON_USABLE_GB * 1024
    utilization = peak_mem / usable_mb if usable_mb > 0 else 0.0

    tpot = sum(token_times) / len(token_times) if token_times else 0.0

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return GenerationMetrics(
        ttft_ms=ttft,
        tpot_ms=tpot,
        total_time_ms=ttft + sum(token_times),
        num_input_tokens=input_ids.shape[1],
        num_output_tokens=len(generated_ids),
        peak_memory_mb=peak_mem,
        model_weight_mb=model_weight_mb,
        kv_cache_memory_mb=kv_cache_mem_mb,
        memory_fragmentation=frag,
        memory_utilization=utilization,
        token_times_ms=[ttft] + token_times,
        generated_text=text,
    )


# ---------------------------------------------------------------------------
# Batch runner (with warmup)
# ---------------------------------------------------------------------------

def run_benchmark(
    model,
    tokenizer,
    prompts: List[Dict[str, str]],
    cache_factory: Optional[Callable] = None,
    max_new_tokens: int = 256,
    warmup_runs: int = 2,
    num_runs: int = 1,
    device: str = "cuda",
) -> List[Dict]:
    """Run a full benchmark over *prompts* with optional warmup.

    Parameters
    ----------
    cache_factory : callable | None
        ``lambda: KIVICache(...)`` or ``None`` for default cache.
    num_runs : int
        Repeats per sample.  Default 1 (greedy decoding is deterministic,
        timing variance on Jetson is <2%, multiple runs waste compute).
    """
    # Warmup — 让 CUDA kernel 编译完成，避免首次运行的开销污染数据
    print(f"Running {warmup_runs} warmup cycles...")
    for i in range(warmup_runs):
        cache = cache_factory() if cache_factory else None
        _ = measure_generation(
            model, tokenizer, prompts[0]["prompt"],
            max_new_tokens=min(32, max_new_tokens),
            cache_impl=cache, device=device,
        )
        torch.cuda.empty_cache()
        gc.collect()
    print("Warmup complete. Starting benchmark...\n")

    results = []
    total = len(prompts)
    for idx, item in enumerate(prompts, 1):
        prompt_results: List[GenerationMetrics] = []
        for run_i in range(num_runs):
            cache = cache_factory() if cache_factory else None
            m = measure_generation(
                model, tokenizer, item["prompt"],
                max_new_tokens=max_new_tokens,
                cache_impl=cache, device=device,
            )
            prompt_results.append(m)
            torch.cuda.empty_cache()
            gc.collect()

        # Aggregate
        avg = _average_metrics(prompt_results)
        avg["question"] = item.get("question", "")
        avg["pubid"] = item.get("pubid", "")
        avg["generated_text"] = prompt_results[0].generated_text
        avg["sample_text"] = prompt_results[0].generated_text[:200]
        results.append(avg)

        print(f"  [{idx}/{total}] ttft={avg['ttft_ms']:.0f}ms  "
              f"tpot={avg['tpot_ms']:.1f}ms  "
              f"peak={avg['peak_memory_mb']:.0f}MB  "
              f"out={avg['num_output_tokens']}tok")

    return results


def _average_metrics(runs: List[GenerationMetrics]) -> Dict:
    n = len(runs)
    return {
        "ttft_ms": sum(r.ttft_ms for r in runs) / n,
        "tpot_ms": sum(r.tpot_ms for r in runs) / n,
        "total_time_ms": sum(r.total_time_ms for r in runs) / n,
        "num_input_tokens": runs[0].num_input_tokens,
        "num_output_tokens": round(sum(r.num_output_tokens for r in runs) / n),
        "peak_memory_mb": max(r.peak_memory_mb for r in runs),
        "model_weight_mb": runs[0].model_weight_mb,
        "kv_cache_memory_mb": sum(r.kv_cache_memory_mb for r in runs) / n,
        "memory_fragmentation": sum(r.memory_fragmentation for r in runs) / n,
        "memory_utilization": max(r.memory_utilization for r in runs),
    }


# ---------------------------------------------------------------------------
# OOM threshold detection
# ---------------------------------------------------------------------------

def find_oom_threshold(
    model,
    tokenizer,
    context_lengths: Optional[List[int]] = None,
    max_new_tokens: int = 32,
    cache_factory: Optional[Callable] = None,
    device: str = "cuda",
    memory_headroom_mb: float = 1500.0,
) -> Dict[str, Any]:
    """Find the maximum context length before OOM (or near-OOM).

    Generates dummy input of increasing length and catches CUDA OOM.
    Reports the last successful length and the length that caused OOM.

    Safety: before each probe, checks free GPU memory. If insufficient
    headroom remains, marks as 'skip' instead of risking a kernel crash.

    Parameters
    ----------
    context_lengths : list[int]
        Sequence lengths to probe.  Default: powers of 2 from 256 to 16384.
    memory_headroom_mb : float
        Minimum free memory (MB) required before attempting a probe.
        Default 1500 MB — prevents unrecoverable CUDA OOM.

    Returns
    -------
    dict with keys: max_safe_length, oom_length, results (per-length details)
    """
    if context_lengths is None:
        context_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384]

    # Build a dummy repeating token sequence
    dummy_token = tokenizer.encode("medical", add_special_tokens=False)[0]

    results = []
    max_safe = 0
    oom_at = None

    for seq_len in context_lengths:
        torch.cuda.empty_cache()
        gc.collect()

        # --- Safety check: skip if free memory is too low ---
        free_mb = (
            torch.cuda.get_device_properties(device).total_memory
            - torch.cuda.memory_allocated(device)
        ) / (1024 ** 2)
        if free_mb < memory_headroom_mb:
            oom_at = seq_len
            results.append({
                "context_length": seq_len,
                "status": f"skip (free={free_mb:.0f}MB < headroom={memory_headroom_mb:.0f}MB)",
                "peak_memory_mb": float("nan"),
                "utilization": float("nan"),
            })
            print(f"  ctx={seq_len}: skipped (only {free_mb:.0f} MB free)")
            break

        try:
            input_ids = torch.full(
                (1, seq_len), dummy_token, dtype=torch.long, device=device,
            )
            attention_mask = torch.ones_like(input_ids)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            cache = cache_factory() if cache_factory else None

            torch.cuda.reset_peak_memory_stats(device)

            # --- Prefill ---
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )

            # --- Decode a few tokens to stress KV cache ---
            past = outputs.past_key_values
            next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
            cur_pos = seq_len

            for _ in range(min(max_new_tokens, 16)):
                step_mask = torch.ones(1, cur_pos + 1, dtype=torch.long, device=device)
                step_pos = torch.tensor([[cur_pos]], device=device)

                outputs = model(
                    input_ids=next_tok,
                    attention_mask=step_mask,
                    position_ids=step_pos,
                    past_key_values=past,
                    use_cache=True,
                )
                next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
                past = outputs.past_key_values
                cur_pos += 1

            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            util = peak / (JETSON_USABLE_GB * 1024)

            results.append({
                "context_length": seq_len,
                "status": "ok",
                "peak_memory_mb": peak,
                "utilization": util,
            })
            max_safe = seq_len
            print(f"  ctx={seq_len}: OK  peak={peak:.0f} MB  util={util*100:.1f}%")

            del outputs, past, next_tok, input_ids, attention_mask, position_ids, cache
            torch.cuda.empty_cache()
            gc.collect()

        except torch.cuda.OutOfMemoryError:
            oom_at = seq_len
            results.append({
                "context_length": seq_len,
                "status": "OOM",
                "peak_memory_mb": float("nan"),
                "utilization": float("nan"),
            })
            print(f"  ctx={seq_len}: OOM!")
            torch.cuda.empty_cache()
            gc.collect()
            break

        except Exception as e:
            results.append({
                "context_length": seq_len,
                "status": f"error: {e}",
                "peak_memory_mb": float("nan"),
                "utilization": float("nan"),
            })
            print(f"  ctx={seq_len}: error: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            break

    return {
        "max_safe_length": max_safe,
        "oom_length": oom_at,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Memory budget helpers
# ---------------------------------------------------------------------------

def print_memory_budget(model=None, device: str = "cuda") -> Dict[str, float]:
    """Print current Jetson memory budget breakdown."""
    total_mb = JETSON_TOTAL_GB * 1024
    usable_mb = JETSON_USABLE_GB * 1024

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)

    weight_mb = 0.0
    if model is not None:
        weight_mb = sum(
            p.nelement() * p.element_size() for p in model.parameters()
        ) / (1024 ** 2)

    kv_budget_mb = usable_mb - weight_mb - 500  # ~500 MB CUDA overhead

    budget = {
        "total_physical_mb": total_mb,
        "usable_mb": usable_mb,
        "model_weights_mb": weight_mb,
        "cuda_overhead_mb": 500,
        "kv_budget_mb": kv_budget_mb,
        "currently_allocated_mb": allocated,
        "currently_reserved_mb": reserved,
    }

    print(f"{'='*50}")
    print(f"Jetson Orin NX Memory Budget")
    print(f"{'='*50}")
    print(f"Total physical    : {total_mb:,.0f} MB ({JETSON_TOTAL_GB} GB)")
    print(f"Usable (after OS) : {usable_mb:,.0f} MB ({JETSON_USABLE_GB} GB)")
    print(f"Model weights     : {weight_mb:,.0f} MB")
    print(f"CUDA overhead     : ~500 MB")
    print(f">>> KV Cache budget: {kv_budget_mb:,.0f} MB ({kv_budget_mb/1024:.1f} GB)")
    print(f"{'='*50}")
    print(f"Currently allocated: {allocated:,.0f} MB")
    print(f"Currently reserved : {reserved:,.0f} MB")

    return budget


# ---------------------------------------------------------------------------
# Theoretical KV-cache size
# ---------------------------------------------------------------------------

def compute_kv_cache_size_mb(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2,
    quant_bits: Optional[int] = None,
    group_size: int = 32,
) -> float:
    """Compute theoretical KV-cache size in MB."""
    if quant_bits:
        bytes_per_val = quant_bits / 8
        # Scales + zeros overhead (FP16 per group)
        num_groups = seq_len / group_size
        overhead_bytes = num_groups * 2 * 2  # scale + zero, FP16 each
        per_head = seq_len * head_dim * bytes_per_val + num_groups * head_dim * 4
    else:
        per_head = seq_len * head_dim * dtype_bytes

    total = 2 * num_layers * num_kv_heads * per_head  # 2 for K and V
    return total / (1024 ** 2)
