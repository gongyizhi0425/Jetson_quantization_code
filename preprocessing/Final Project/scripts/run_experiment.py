#!/usr/bin/env python3
"""Run one strategy over the QA set and write per-sample JSONL results."""

from __future__ import annotations

import argparse
import time
import tracemalloc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_retention.baselines import build_context
from ehr_retention.config import load_config
from ehr_retention.inference import run_inference
from ehr_retention.io_utils import read_jsonl, write_jsonl
from ehr_retention.metrics import context_tokens


def _index_by(records: list[dict[str, object]], key: str) -> dict[str, dict[str, object]]:
    return {str(record[key]): record for record in records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timelines", default=None)
    parser.add_argument("--qa", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--strategy", choices=["full_context", "sliding_window", "selective", "selective_retention", "selective_query_aware"], required=True)
    parser.add_argument("--retention-ratio", type=float, default=None)
    parser.add_argument("--top-k-events", type=int, default=None)
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--scoring-mode", choices=["type", "keyword", "query_aware"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    timelines_path = args.timelines or str(cfg.get("timelines", "data/processed/timelines.jsonl"))
    qa_path = args.qa or str(cfg.get("qa", "data/processed/qa.jsonl"))
    backend = args.backend or str(cfg.get("backend", "rule_local"))
    model = args.model or str(cfg.get("model", "rule_local"))
    retention_ratio = args.retention_ratio if args.retention_ratio is not None else float(cfg.get("retention_ratio", 0.5))
    top_k_events = args.top_k_events if args.top_k_events is not None else cfg.get("top_k_events")
    max_context_tokens = args.max_context_tokens if args.max_context_tokens is not None else cfg.get("max_context_tokens")
    scoring_mode = args.scoring_mode or str(cfg.get("scoring_mode", "keyword"))

    if isinstance(top_k_events, str):
        top_k_events = int(top_k_events)
    if isinstance(max_context_tokens, str):
        max_context_tokens = int(max_context_tokens)

    out = args.out or f"data/results/{args.strategy}_{backend}.jsonl"

    timelines = _index_by(list(read_jsonl(timelines_path)), "patient_id")
    samples = list(read_jsonl(qa_path))
    records = []
    for sample in samples:
        timeline = timelines[str(sample["patient_id"])]
        context, retained_events = build_context(
            timeline,
            str(sample["question"]),
            strategy=args.strategy,
            retention_ratio=retention_ratio,
            top_k_events=top_k_events,
            max_context_tokens=max_context_tokens,
            scoring_mode=scoring_mode,
        )

        tracemalloc.start()
        start = time.perf_counter()
        result = run_inference(
            backend=backend,
            model=model,
            question=str(sample["question"]),
            context=context,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        retained_ids = [str(event["event_id"]) for event in retained_events]
        records.append(
            {
                "sample_id": sample["sample_id"],
                "patient_id": sample["patient_id"],
                "task_type": sample["task_type"],
                "strategy": args.strategy,
                "question": sample["question"],
                "answer": sample["answer"],
                "prediction": result.prediction,
                "gold_evidence_event_ids": sample["gold_evidence_event_ids"],
                "input_tokens": context_tokens(context),
                "retained_events": retained_ids,
                "retained_event_count": len(retained_events),
                "total_event_count": len(timeline["events"]),
                "retention_ratio_actual": round(len(retained_events) / len(timeline["events"]), 4),
                "latency_ms": round(latency_ms, 4),
                "peak_memory_mb": round(peak_bytes / (1024 * 1024), 4),
                "backend": result.backend,
                "model": result.model,
            }
        )

    count = write_jsonl(out, records)
    print(f"Wrote {count} experiment records to {out}")


if __name__ == "__main__":
    main()
