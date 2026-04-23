"""
Perplexity (PPL) evaluation for KV-cache quality degradation assessment.

PPL is the key metric for judging whether 2-bit KV-cache compression
still preserves logical coherence in medical reasoning scenarios.

Lower PPL = better.  A PPL increase > 5% from baseline signals
meaningful quality degradation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 1024,
    stride: int = 512,
    cache_impl: Optional[Any] = None,
    device: str = "cuda",
    batch_size: int = 1,
) -> Dict[str, float]:
    """Compute perplexity over a list of texts using sliding window.

    This measures how well the model predicts the next token.
    Higher PPL after quantization → quality degradation.

    Parameters
    ----------
    model : PreTrainedModel
        The language model being evaluated.
    tokenizer : PreTrainedTokenizer
        Corresponding tokenizer.
    texts : list[str]
        Evaluation texts (e.g. PubMedQA long_answer fields).
    max_length : int
        Maximum context window per chunk.
    stride : int
        Sliding window stride (overlap = max_length - stride).
    cache_impl : Cache | None
        Custom KV cache for the model. None = default.
    device : str
        CUDA device.

    Returns
    -------
    dict with keys: ppl, avg_loss, num_tokens
    """
    model.eval()
    nlls: List[float] = []
    total_tokens = 0

    for text in tqdm(texts, desc="Computing PPL"):
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.shape[1]

        # Sliding window
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            input_chunk = input_ids[:, begin:end]
            target_len = end - prev_end  # tokens we score

            # Build targets: -100 for context tokens, real ids for scored
            target_ids = input_chunk.clone()
            target_ids[:, :-target_len] = -100

            outputs = model(
                input_ids=input_chunk,
                labels=target_ids,
                past_key_values=None,  # fresh for each window
                use_cache=False,       # no cache during PPL eval
            )

            neg_log_likelihood = outputs.loss * target_len
            nlls.append(neg_log_likelihood.item())
            total_tokens += target_len
            prev_end = end

            if end == seq_len:
                break

    avg_nll = sum(nlls) / total_tokens
    ppl = math.exp(avg_nll)

    return {
        "ppl": ppl,
        "avg_loss": avg_nll,
        "num_tokens": total_tokens,
    }


@torch.no_grad()
def compute_ppl_with_kv_cache(model, tokenizer, texts, cache_factory=None, max_length=512, device="cuda"):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    loss_fn = torch.nn.CrossEntropyLoss()

    for text in tqdm(texts, desc="KIVI Decode PPL"):
        input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
        seq_len = input_ids.shape[1]
        prefill_len = min(128, seq_len // 2)
        if seq_len <= prefill_len + 1: continue

        # 初始化自定义的 Cache 对象
        past_kv = cache_factory() if cache_factory is not None else None

        # 1. 预填充阶段 (建立初始 KV Cache)
        outputs = model(input_ids[:, :prefill_len], past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        
        # 2. 自回归 Decode 阶段 (强制触发底层 2-bit 读取)
        for i in range(prefill_len, seq_len):
            outputs = model(input_ids[:, i-1:i], past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            
            loss = loss_fn(outputs.logits[0, -1, :], input_ids[0, i])
            total_loss += loss.item()
            total_tokens += 1

    return {"ppl": math.exp(total_loss / total_tokens), "num_tokens": total_tokens}