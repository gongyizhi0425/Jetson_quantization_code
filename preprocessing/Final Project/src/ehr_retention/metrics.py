"""Experiment metrics."""

from __future__ import annotations

from .text import count_tokens, normalize_answer, token_f1


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def evidence_recall(retained_event_ids: list[str], gold_event_ids: list[str]) -> float:
    if not gold_event_ids:
        return 1.0
    retained = set(retained_event_ids)
    return sum(1 for event_id in gold_event_ids if event_id in retained) / len(gold_event_ids)


def compute_record_metrics(record: dict[str, object]) -> dict[str, object]:
    prediction = str(record.get("prediction", ""))
    answer = str(record.get("answer", ""))
    retained = [str(x) for x in record.get("retained_events", [])]
    gold = [str(x) for x in record.get("gold_evidence_event_ids", [])]
    return {
        "exact_match": exact_match(prediction, answer),
        "token_f1": token_f1(prediction, answer),
        "evidence_recall": evidence_recall(retained, gold),
        "answerable": float(prediction.lower().strip() != "unknown"),
        "input_tokens": int(record.get("input_tokens", 0)),
        "latency_ms": float(record.get("latency_ms", 0.0)),
        "peak_memory_mb": float(record.get("peak_memory_mb", 0.0)),
        "retention_ratio_actual": float(record.get("retention_ratio_actual", 0.0)),
    }


def context_tokens(context: str) -> int:
    return count_tokens(context)
