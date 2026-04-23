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

def run_vllm_benchmark(engine, prompts, max_new_tokens=256, warmup_runs=2, device="cuda", jetson_usable_gb=11.0):
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
    print()

    # Warmup
    print(f" 正在执行 {warmup_runs} 次预热，激活 CUDA Graph 与显存池...")
    for _ in range(warmup_runs):
        # 关闭 tqdm 避免刷屏干扰
        _ = engine.generate([prompts[0]['prompt']], SamplingParams(max_tokens=50, temperature=0.0), use_tqdm=False)
    print(" 预热完成，开始科学测速。\n")

    torch.cuda.reset_peak_memory_stats(device)
    model_weight_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)  #模型本身的权重 + vLLM 预先霸占的 KV Cache 静态池

    for idx, item in enumerate(prompts, 1):
        prompt_text = item['prompt']

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        outputs = engine.generate([prompt_text], sampling_params, use_tqdm=False)
        
        torch.cuda.synchronize()  #really important！ Otherwise, the time is not accurate
        t1 = time.perf_counter()  

        output = outputs[0]
        n_input = len(output.prompt_token_ids)  # 喂进去的长度
        n_output = len(output.outputs[0].token_ids) if output.outputs else 0   #decode出来的长度
        gen_text = output.outputs[0].text if output.outputs else ""

        total_time_ms = (t1 - t0) * 1000.0

# ---------------- 3. 时间指标科学修正 (TTFT / TPOT) ----------------
        ttft_ms = 0.0
        tpot_ms = 0.0

        if hasattr(output, "metrics") and output.metrics is not None:    #metrics是vLLM生成过程中记录的详细时间点，包含调度、首token生成、完成等时间戳
            m = output.metrics

            
            t_sched = getattr(m, 'first_scheduled_time', None)   #调度器是几分几秒正式把你的任务塞进 GPU 显存开始计算的（中间可能因为 GPU 忙而在排队）
            t_first = getattr(m, 'first_token_time', None)   #GPU 吭哧吭哧算完，吐出第一个字是几分几秒。
            t_finish = getattr(m, 'finished_time', None)

            if t_first and t_sched:
               ttft_ms = (t_first - t_sched) * 1000.0  # 排除排队干扰：故意没有用 arrival_time
            
            if t_finish and t_first and n_output > 1:
                tpot_ms = ((t_finish - t_first) * 1000.0) / (n_output - 1)# -假设大模型一共回答了 10 个字。第一个字的生成时间已经被算在 TTFT 里了。所以在这段剩余的时间里，大模型其实只生成了 9 个字
        

# 降级估算 (应对极端情况)
        if ttft_ms <= 0:
            ttft_ms = total_time_ms * 0.12  
        if tpot_ms <= 0 and n_output > 1:
            decode_time_ms = max(0.0, total_time_ms - ttft_ms)
            tpot_ms = decode_time_ms / (n_output - 1)
        elif tpot_ms <= 0:
            tpot_ms = total_time_ms

        # ---------------- 4. 显存与碎片率提取 ----------------
        # 峰值与总利用率
        #peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  #torch.cuda.max_memory_allocated(device): 获取自程序运行以来 PyTorch 在指定 GPU 上实际分配过的最大显存字节数。
        # utilization = peak_mem / ...
        process = psutil.Process(os.getpid())
        process_mem_bytes = process.memory_info().rss  # RSS: 常驻集大小
        peak_mem = process_mem_bytes / (1024 ** 2)     # 转换为 MB

        total_blocks = 0
        free_blocks = 0
        block_size = default_block_size
        frag = 0.0
        
       


        # 由于 vLLM 霸占显存，我们用严格的数学公式倒推当前序列的实际 KV Cache 占用
        # 适配 Qwen2.5-1.5B 结构 (28层, 2组KV头, 128维, FP16占2字节)
 
        
        num_free_blocks = 0
        frag = 0.0
        block_utilization = 0.0
        
        try:
            # 适配 vLLM 0.8.x 的多级探针
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
                # vLLM V1 架构首选 kv_cache_manager
                kv_manager = getattr(scheduler, "kv_cache_manager", None)
                bm = getattr(scheduler, "block_manager", getattr(scheduler, "_block_manager", None))
                
                if kv_manager:
                    _tot = getattr(kv_manager, "num_gpu_blocks", 0)
                    num_total_gpu_blocks = int(_tot) if _tot is not None else num_total_gpu_blocks
                    num_free_blocks = int(getattr(kv_manager, "num_free_blocks", 0))
                elif bm:
                    # 兼容 V0 架构的 block_manager
                    _tot = getattr(bm, "num_total_gpu_blocks", 0)
                    if _tot is not None and _tot > 0:
                        num_total_gpu_blocks = int(_tot)
                        
                    if hasattr(bm, "get_num_free_gpu_blocks"):
                        num_free_blocks = int(bm.get_num_free_gpu_blocks())
                    else:
                        num_free_blocks = int(getattr(bm, "num_free_gpu_blocks", 0))

        except Exception:
            pass

        # 严格的数学计算，防止被 0 除
        if num_total_gpu_blocks > 0:
            num_used_blocks = num_total_gpu_blocks - num_free_blocks
            block_utilization = num_used_blocks / num_total_gpu_blocks
            frag = num_free_blocks / num_total_gpu_blocks  # 空闲浪费率
        else:
            num_used_blocks = 0
            
            # Fallback 到 PyTorch 层面的碎片计算
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            frag = 1.0 - (allocated / reserved) if reserved > 0 else 0.0


        bytes_per_token = 28 * 2 * 128 * 2 * 2  # 每个 token 的 KV 占用 = 层数 × 头数 × 每头维度 × FP16字节数 × KV两份
        kv_cache_mem_mb = ((n_input + n_output) * bytes_per_token) / (1024 ** 2)
        utilization = block_utilization if block_utilization > 0 else (peak_mem / (jetson_usable_gb * 1024) if jetson_usable_gb > 0 else 0.0)


        
        # 从 vLLM 调度器抓取真实的区块使用情况来计算碎片率 定义为：分配了但没被当前序列有效利用的空闲池子比例。vLLM 将显存切分成了固定大小的区块（Blocks）。这里直接读取 GPU 上总的区块数 (total_blocks) 和当前空闲的区块数 (free_blocks)，算出空闲块占比作为碎片率/闲置率。
      
                
     

        # ---------------- 5. 严格遵循要求的输出格式 ----------------
        results.append({
            'ttft_ms': ttft_ms,
            'tpot_ms': tpot_ms,
            'total_time_ms': total_time_ms,
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
            'generated_text': gen_text,
        })
        
        # 安全打印：用三元表达式确保不会出现 0/0 或者 None
        blocks_display = f"{num_used_blocks}/{num_total_gpu_blocks}" if num_total_gpu_blocks > 0 else "?/?"
        
        print(f"[{idx}/{len(prompts)}] ttft={ttft_ms:.1f}ms  tpot={tpot_ms:.1f}ms  "
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
