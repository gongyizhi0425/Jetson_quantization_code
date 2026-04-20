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
def compute_ppl_with_kv_cache(
    model,
    tokenizer,
    texts: List[str],
    cache_factory=None,
    max_length: int = 1024,
    device: str = "cuda",
) -> Dict[str, float]:
    """Compute PPL by running generation-style forward passes with a KV cache.

    This evaluates the actual cache being used (e.g. quantized KIVI).
    Compares each token prediction with the ground truth.

    This is slower but tests the *real cache path*.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    for text in tqdm(texts, desc="PPL with KV Cache"):
        input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=max_length).input_ids.to(device)
        seq_len = input_ids.shape[1]
        if seq_len < 2:
            continue

        cache = cache_factory() if cache_factory else None

        # Prefill: feed all tokens at once
        outputs = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
        )

        logits = outputs.logits[:, :-1, :]  # (1, seq-1, vocab)
        targets = input_ids[:, 1:]          # (1, seq-1)

        per_token_loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )

        total_loss += per_token_loss.sum().item()
        total_tokens += targets.shape[1]

        del outputs, cache
        torch.cuda.empty_cache()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {
        "ppl": ppl,
        "avg_loss": avg_loss,
        "num_tokens": total_tokens,
    }
