"""Text normalization and token counting helpers."""

from __future__ import annotations

import re
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def normalize_answer(text: str) -> str:
    return " ".join(tokenize(text))


def token_f1(prediction: str, answer: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(answer)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def keyword_overlap(a: str, b: str) -> int:
    return len(set(tokenize(a)) & set(tokenize(b)))
