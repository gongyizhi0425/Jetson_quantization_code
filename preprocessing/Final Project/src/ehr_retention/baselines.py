"""Thin compatibility wrappers around retention strategies."""

from __future__ import annotations

from .retention import RetentionConfig, render_context, select_events


def build_context(
    timeline: dict[str, object],
    question: str,
    strategy: str,
    retention_ratio: float = 0.5,
    top_k_events: int | None = None,
    max_context_tokens: int | None = None,
    scoring_mode: str = "keyword",
) -> tuple[str, list[dict[str, object]]]:
    config = RetentionConfig(
        strategy=strategy,
        retention_ratio=retention_ratio,
        top_k_events=top_k_events,
        max_context_tokens=max_context_tokens,
        scoring_mode=scoring_mode,
    )
    selected = select_events(list(timeline["events"]), question, config)
    return render_context(selected), selected
