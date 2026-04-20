"""Context retention strategies for patient timelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .text import count_tokens, keyword_overlap, tokenize
from .timeline import event_to_line


EVENT_TYPE_WEIGHTS = {
    "diagnosis": 8.0,
    "adverse_event": 8.0,
    "hospitalization": 8.0,
    "medication_start": 7.0,
    "medication_stop": 7.0,
    "medication_change": 7.0,
    "procedure": 4.0,
    "abnormal_lab": 4.0,
    "specialist_visit": 3.0,
    "routine_followup": 0.5,
    "normal_lab": 0.25,
}

IMPORTANT_KEYWORDS = {
    "diagnosed",
    "diagnosis",
    "started",
    "stopped",
    "adjusted",
    "adverse",
    "hospitalized",
    "abnormal",
    "elevated",
    "bleeding",
    "hypoglycemia",
    "exacerbation",
}


@dataclass(frozen=True)
class RetentionConfig:
    strategy: str
    retention_ratio: float = 0.5
    top_k_events: int | None = None
    max_context_tokens: int | None = None
    scoring_mode: str = "keyword"


def score_event(
    event: dict[str, object],
    query: str = "",
    recency_rank: int = 0,
    total_events: int = 1,
    scoring_mode: str = "keyword",
) -> float:
    text = str(event.get("text", ""))
    event_type = str(event.get("event_type", ""))
    score = EVENT_TYPE_WEIGHTS.get(event_type, 1.0)

    if scoring_mode in {"keyword", "query_aware"}:
        words = set(tokenize(text))
        score += 0.5 * len(words & IMPORTANT_KEYWORDS)

    if scoring_mode == "query_aware" and query:
        score += 1.5 * keyword_overlap(query, text)

    if total_events > 1:
        score += 0.75 * (recency_rank / (total_events - 1))
    return score


def _apply_token_budget(events: list[dict[str, object]], max_context_tokens: int | None) -> list[dict[str, object]]:
    if max_context_tokens is None:
        return events
    kept: list[dict[str, object]] = []
    total = 0
    for event in events:
        line_tokens = count_tokens(event_to_line(event))
        if kept and total + line_tokens > max_context_tokens:
            break
        kept.append(event)
        total += line_tokens
    return kept


def select_events(
    events: Sequence[dict[str, object]],
    query: str,
    config: RetentionConfig,
) -> list[dict[str, object]]:
    if not events:
        return []

    if config.strategy == "full_context":
        return _apply_token_budget(list(events), config.max_context_tokens)

    if config.strategy == "sliding_window":
        selected = list(events)
        if config.top_k_events is not None:
            selected = selected[-config.top_k_events :]
        selected = list(reversed(selected))
        selected = _apply_token_budget(selected, config.max_context_tokens)
        return sorted(selected, key=lambda e: (str(e["date"]), str(e["event_id"])))

    if config.strategy not in {"selective", "selective_retention", "selective_query_aware"}:
        raise ValueError(f"Unknown strategy: {config.strategy}")

    scoring_mode = config.scoring_mode
    if config.strategy == "selective_query_aware":
        scoring_mode = "query_aware"

    total = len(events)
    scored = []
    for idx, event in enumerate(events):
        scored.append(
            (
                score_event(event, query, recency_rank=idx, total_events=total, scoring_mode=scoring_mode),
                str(event["date"]),
                str(event["event_id"]),
                event,
            )
        )
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

    k = config.top_k_events
    if k is None:
        k = max(1, int(round(total * config.retention_ratio)))
    selected = [item[3] for item in scored[:k]]
    selected = _apply_token_budget(selected, config.max_context_tokens)
    return sorted(selected, key=lambda e: (str(e["date"]), str(e["event_id"])))


def render_context(events: Sequence[dict[str, object]]) -> str:
    return "\n".join(event_to_line(event) for event in events)
