"""
PubMedQA dataset loading and prompt construction for medical-scenario KV-cache experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an experienced medical expert. Provide detailed, evidence-based "
    "reasoning before stating your conclusion."
)

USER_TEMPLATE = (
    "Based on the following medical research context, answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Provide step-by-step medical reasoning and your final answer (yes / no / maybe)."
)


def _build_chat_prompt(context: str, question: str, tokenizer) -> str:
    """Build a chat-format prompt using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]
    # apply_chat_template returns a string ready for the model
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _build_plain_prompt(context: str, question: str) -> str:
    """Fallback if tokenizer has no chat template."""
    return (
        f"<|system|>{SYSTEM_PROMPT}\n"
        f"<|user|>{USER_TEMPLATE.format(context=context, question=question)}\n"
        f"<|assistant|>"
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_pubmedqa(
    max_samples: int = 50,
    tokenizer=None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Load PubMedQA (pqa_labeled) and return formatted prompts.

    Parameters
    ----------
    max_samples : int
        How many samples to load (for benchmarking, 30-50 is plenty).
    tokenizer : PreTrainedTokenizer | None
        If provided, uses ``apply_chat_template`` for prompt formatting.
    cache_dir : str | None
        HuggingFace datasets cache directory.

    Returns
    -------
    list of dict
        Each dict has keys: prompt, question, reference_answer, final_decision, pubid.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        split="train",
        cache_dir=cache_dir,
    )

    prompts: List[Dict[str, str]] = []
    for i, sample in enumerate(ds):
        if i >= max_samples:
            break

        # Combine context sentences
        context = "\n".join(sample["context"]["contexts"])
        question = sample["question"]

        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            prompt = _build_chat_prompt(context, question, tokenizer)
        else:
            prompt = _build_plain_prompt(context, question)

        prompts.append({
            "prompt": prompt,
            "question": question,
            "reference_answer": sample["long_answer"],
            "final_decision": sample["final_decision"],
            "pubid": str(sample["pubid"]),
        })

    return prompts


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def split_by_token_length(
    prompts: List[Dict],
    tokenizer,
    bins: Optional[List[int]] = None,
) -> Dict[str, List[Dict]]:
    """Split prompts into length bins for TTFT-vs-length analysis.

    Default bins: short (<256), medium (256-512), long (>512).
    """
    if bins is None:
        bins = [256, 512]

    groups: Dict[str, List[Dict]] = {"short": [], "medium": [], "long": []}
    for item in prompts:
        n = len(tokenizer.encode(item["prompt"]))
        item["num_tokens"] = n
        if n < bins[0]:
            groups["short"].append(item)
        elif n < bins[1]:
            groups["medium"].append(item)
        else:
            groups["long"].append(item)
    return groups


def save_prompts(prompts: List[Dict], path: str) -> None:
    """Save preprocessed prompts to JSON for reproducibility."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)


def load_prompts(path: str) -> List[Dict]:
    """Load preprocessed prompts from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Unified loader — picks PubMedQA or synthetic EHR based on config
# ---------------------------------------------------------------------------

def load_dataset_auto(
    source: str = "ehr",
    max_samples: int = 50,
    tokenizer=None,
    strategy: str = "selective",
    retention_ratio: float = 0.5,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Load prompts from either PubMedQA or the synthetic EHR pipeline.

    Parameters
    ----------
    source : str
        ``"pubmedqa"`` (download from HF) or ``"ehr"`` (synthetic EHR, default).
    strategy : str
        For EHR source: ``"selective"`` (default) | ``"full_context"`` | ``"sliding_window"``.
    retention_ratio : float
        For EHR selective strategy.
    """
    if source == "pubmedqa":
        return load_pubmedqa(
            max_samples=max_samples,
            tokenizer=tokenizer,
            cache_dir=cache_dir,
        )

    from .ehr_bridge import load_ehr_qa
    return load_ehr_qa(
        max_samples=max_samples,
        tokenizer=tokenizer,
        strategy=strategy,
        retention_ratio=retention_ratio,
    )
