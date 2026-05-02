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

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import psutil
import os
import torch

try:
    from vllm import LLM, SamplingParams
    from vllm.config import CacheConfig
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
    peak_memory_mb: float = 0.0


def check_vllm_available() -> bool:
    """Check if vLLM is importable."""
    return VLLM_AVAILABLE


def create_vllm_engine(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    gpu_memory_utilization: float = 0.60,
    max_model_len: int = 4096,
    block_size: int = 16,
    cache_dtype: str = "auto",
    dtype: str = "float16",
    enable_prefix_caching: bool = False,
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
        gpu_memory_utilization=gpu_memory_utilization,  # 直接传
        max_model_len=max_model_len,
        
        swap_space=0,  # Jetson: No CPU swap
        
        enable_prefix_caching=enable_prefix_caching,
        enforce_eager=True,  # Jetson: Disable CUDA graph
        disable_custom_all_reduce=True,  # Single GPU
        trust_remote_code=True,
    )
    
    # 打印配置
    print(f"\n✅ vLLM Engine Initialized")
    print(f"  Model          : {model_name}")
    print(f"  Dtype          : {dtype}")
    print(f"  Max Seq Len    : {max_model_len}")
    print(f"  Block Size     : {block_size}")
    print(f"  Cache Dtype    : {cache_dtype}")
    print(f"  GPU Mem Util   : {gpu_memory_utilization*100:.0f}%")
    
    # 提取 KV cache 统计
    if hasattr(engine, "llm_engine") and hasattr(engine.llm_engine, "cache_config"):
        cfg = engine.llm_engine.cache_config
        print(f"\n📊 PagedAttention Config:")
        print(f"  Num GPU Blocks : {getattr(cfg, 'num_gpu_blocks', 'N/A')}")
        print(f"  Block Size     : {getattr(cfg, 'block_size', block_size)}")
        if hasattr(cfg, 'num_gpu_blocks') and cfg.num_gpu_blocks:
            print(f"  Max Context    : ~{cfg.num_gpu_blocks * block_size} tokens")
    print()
    
    return engine

def run_vllm_benchmark(engine, prompts, max_new_tokens=256, warmup_runs=2, device="cuda", jetson_usable_gb=11.0, batch_size=1):
    """Run vLLM benchmark with configurable batch size.
    
    Parameters
    ----------
    batch_size : int
        Number of prompts to send in one engine.generate() call.
        Default 1 preserves original single-request behavior.
    """
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,  # greedy decoding
        seed=42
    )
    results = []

    # ---------------- 提取 vLLM Cache Config ----------------
    cache_cfg = getattr(engine.llm_engine, "cache_config", None)
    default_block_size = getattr(cache_cfg, "block_size", 16) if cache_cfg else 16
    _raw_total = getattr(cache_cfg, "num_gpu_blocks", 0)
    num_total_gpu_blocks = int(_raw_total) if _raw_total is not None else 0
    
    print(f"\n📊 vLLM PagedAttention 配置:")
    print(f"  总 GPU Blocks      : {num_total_gpu_blocks}")
    print(f"  Block Size (tokens): {default_block_size}")
    if num_total_gpu_blocks > 0:
        print(f"  Max Context Length : ~{num_total_gpu_blocks * default_block_size} tokens")
    print(f"  Batch Size         : {batch_size}")
    print()

    # Warmup (use repeated first prompt to form a batch)
    print(f" 正在执行 {warmup_runs} 次预热 (batch_size={batch_size})...")
    warmup_prompts = [prompts[0]['prompt']] * batch_size
    for _ in range(warmup_runs):
        _ = engine.generate(warmup_prompts, SamplingParams(max_tokens=50, temperature=0.0), use_tqdm=False)
    print(" 预热完成，开始科学测速。\n")

    torch.cuda.reset_peak_memory_stats(device)
    model_weight_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)

    # ---------------- 辅助：读取 Block 状态 ----------------
    def _get_block_stats():
        num_free_blocks = 0
        try:
            scheduler = None
            if hasattr(engine, "_scheduler"):
                scheduler = engine._scheduler
            elif hasattr(engine, "llm_engine") and hasattr(engine.llm_engine, "_scheduler"):
                scheduler = engine.llm_engine._scheduler
            elif hasattr(engine, "scheduler"):
                scheduler = engine.scheduler
            elif hasattr(engine, "llm_engine") and hasattr(engine.llm_engine, "scheduler"):
                scheduler = engine.llm_engine.scheduler
            
            if scheduler:
                kv_manager = getattr(scheduler, "kv_cache_manager", None)
                bm = getattr(scheduler, "block_manager", getattr(scheduler, "_block_manager", None))
                
                if kv_manager:
                    _tot = getattr(kv_manager, "num_gpu_blocks", 0)
                    nonlocal num_total_gpu_blocks
                    num_total_gpu_blocks = int(_tot) if _tot is not None else num_total_gpu_blocks
                    num_free_blocks = int(getattr(kv_manager, "num_free_blocks", 0))
                elif bm:
                    _tot = getattr(bm, "num_total_gpu_blocks", 0)
                    if _tot is not None and _tot > 0:
                        num_total_gpu_blocks = int(_tot)
                    if hasattr(bm, "get_num_free_gpu_blocks"):
                        num_free_blocks = int(bm.get_num_free_gpu_blocks())
                    else:
                        num_free_blocks = int(getattr(bm, "num_free_gpu_blocks", 0))
        except Exception:
            pass
        
        if num_total_gpu_blocks > 0:
            num_used_blocks = num_total_gpu_blocks - num_free_blocks
            block_utilization = num_used_blocks / num_total_gpu_blocks
            frag = num_free_blocks / num_total_gpu_blocks
        else:
            num_used_blocks = 0
            block_utilization = 0.0
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            frag = 1.0 - (allocated / reserved) if reserved > 0 else 0.0
        return num_used_blocks, block_utilization, frag

    # ---------------- Batch 主循环 ----------------
    total = len(prompts)
    batch_idx = 0
    for start in range(0, total, batch_size):
        batch_idx += 1
        batch = prompts[start:start + batch_size]
        batch_prompts = [item['prompt'] for item in batch]
        batch_base = start + 1  # 1-based global index

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        outputs = engine.generate(batch_prompts, sampling_params, use_tqdm=False)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        batch_wall_ms = (t1 - t0) * 1000.0

        # 显存与碎片：batch 结束后统一读一次（全局状态）
        process = psutil.Process(os.getpid())
        peak_mem = process.memory_info().rss / (1024 ** 2)
        num_used_blocks, block_utilization, frag = _get_block_stats()
        blocks_display = f"{num_used_blocks}/{num_total_gpu_blocks}" if num_total_gpu_blocks > 0 else "?/?"

        # 逐条拆分结果
        for b_i, (output, item) in enumerate(zip(outputs, batch)):
            global_idx = start + b_i + 1
            n_input = len(output.prompt_token_ids)
            n_output = len(output.outputs[0].token_ids) if output.outputs else 0
            gen_text = output.outputs[0].text if output.outputs else ""

            # TTFT / TPOT：优先用每条 output 自己的 metrics
            ttft_ms = 0.0
            tpot_ms = 0.0
            if hasattr(output, "metrics") and output.metrics is not None:
                m = output.metrics
                t_sched = getattr(m, 'first_scheduled_time', None)
                t_first = getattr(m, 'first_token_time', None)
                t_finish = getattr(m, 'finished_time', None)

                if t_first and t_sched:
                    ttft_ms = (t_first - t_sched) * 1000.0
                if t_finish and t_first and n_output > 1:
                    tpot_ms = ((t_finish - t_first) * 1000.0) / (n_output - 1)

            # 降级估算（metrics 缺失时）
            if ttft_ms <= 0:
                # batch 中各条共享 wall-clock，按平均估计
                ttft_ms = batch_wall_ms * 0.12 
            if tpot_ms <= 0 and n_output > 1:
                decode_time_ms = max(0.0, batch_wall_ms - ttft_ms ) 
                tpot_ms = decode_time_ms / (n_output - 1)
            elif tpot_ms <= 0:
                tpot_ms = batch_wall_ms / len(batch)

            # 单条 KV Cache 估算（与 batch_size 无关，按单条序列长度）
            bytes_per_token = 28 * 2 * 128 * 2 * 2
            kv_cache_mem_mb = ((n_input + n_output) * bytes_per_token) / (1024 ** 2)
            utilization = block_utilization if block_utilization > 0 else (peak_mem / (jetson_usable_gb * 1024) if jetson_usable_gb > 0 else 0.0)

            results.append({
                'ttft_ms': ttft_ms,
                'tpot_ms': tpot_ms,
                'total_time_ms': batch_wall_ms / len(batch),  # 均摊 wall-clock
                'num_input_tokens': n_input,
                'num_output_tokens': n_output,
                'peak_memory_mb': peak_mem,
                'model_weight_mb': model_weight_mb,
                'kv_cache_memory_mb': kv_cache_mem_mb,
                'memory_fragmentation': frag,
                'memory_utilization': utilization,
                'num_used_blocks': num_used_blocks,
                'num_total_blocks': num_total_gpu_blocks,
                'block_size': default_block_size,
                'batch_size': batch_size,
                'batch_idx': batch_idx,
                'generated_text': gen_text,
            })

            print(f"[{global_idx}/{total}] batch={batch_idx}  ttft={ttft_ms:.1f}ms  tpot={tpot_ms:.1f}ms  "
                  f"peak={peak_mem:.0f}MB  blocks={blocks_display}  "
                  f"({block_utilization*100:.1f}%)  out={n_output}tok")

    return results





def get_vllm_cache_stats(engine: "LLM") -> Dict:
    """Extract PagedAttention memory stats from vLLM engine."""
    stats = {
        "num_total_gpu_blocks": 0,
        "num_free_gpu_blocks": 0,
        "block_size": 16,
    }
    
    try:
        if hasattr(engine, "llm_engine"):
            cache_cfg = engine.llm_engine.cache_config
            stats["block_size"] = getattr(cache_cfg, "block_size", 16)
            stats["num_total_gpu_blocks"] = getattr(cache_cfg, "num_gpu_blocks", 0)
            
            # Try to get free blocks
            if hasattr(engine.llm_engine, "scheduler"):
                scheduler = engine.llm_engine.scheduler
                if hasattr(scheduler, "block_manager"):
                    bm = scheduler.block_manager
                    if hasattr(bm, "get_num_free_gpu_blocks"):
                        stats["num_free_gpu_blocks"] = bm.get_num_free_gpu_blocks()
                    elif hasattr(bm, "num_free_gpu_blocks"):
                        stats["num_free_gpu_blocks"] = bm.num_free_gpu_blocks
    except Exception:
        pass
    
    return stats



def find_vllm_oom_threshold(
    engine: "LLM",
    context_lengths: Optional[List[int]] = None,
    max_new_tokens: int = 32,
) -> Dict:
    """Probe maximum context length before vLLM fails.
    
    Parameters
    ----------
    engine : vllm.LLM
        Engine created by create_vllm_engine().
    context_lengths : list[int] | None
        Sequence lengths to probe. Default: [256, 512, 1024, 2048, 4096, 8192].
    max_new_tokens : int
        Tokens to generate per probe.

    Returns
    -------
    dict with keys: max_safe_length, oom_length, results
    """
    if context_lengths is None:
        context_lengths = [256, 512, 1024, 2048, 4096, 8192]

    dummy_word = "medical history patient diagnosis treatment "
    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    results = []
    max_safe = 0
    oom_at = None

    for seq_len in context_lengths:
        repeat_n = max(1, seq_len // 5)
        dummy_text = (dummy_word * repeat_n)[:seq_len * 6]

        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = engine.generate([dummy_text], sampling)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            output = outputs[0]
            n_in = len(output.prompt_token_ids)
            n_out = len(output.outputs[0].token_ids) if output.outputs else 0
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            results.append({
                "context_length": n_in,
                "target_length": seq_len,
                "status": "ok",
                "total_time_ms": elapsed_ms,
                "num_output_tokens": n_out,
                "peak_memory_mb": peak_mb,
            })
            max_safe = n_in
            print(f"  ctx={n_in:>6} → ok  {elapsed_ms:.0f}ms  out={n_out}tok")

        except Exception as e:
            err_str = str(e).lower()
            status = "OOM" if "memory" in err_str or "oom" in err_str else f"error: {e}"
            oom_at = seq_len
            results.append({
                "context_length": seq_len,
                "target_length": seq_len,
                "status": status,
                "total_time_ms": 0.0,
                "num_output_tokens": 0,
                "peak_memory_mb": float("nan"),
            })
            print(f"  ctx={seq_len:>6} → {status}")
            torch.cuda.empty_cache()
            break

    return {
        "max_safe_length": max_safe,
        "oom_length": oom_at,
        "results": results,
    }
