"""Aggregate JSONL experiment outputs into summary rows."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .io_utils import ensure_parent, read_jsonl
from .metrics import compute_record_metrics


SUMMARY_FIELDS = [
    "strategy",
    "backend",
    "model",
    "samples",
    "exact_match",
    "token_f1",
    "evidence_recall",
    "answerable",
    "input_tokens",
    "latency_ms",
    "peak_memory_mb",
    "retention_ratio_actual",
]


def summarize_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (str(record.get("strategy")), str(record.get("backend")), str(record.get("model")))
        groups[key].append(record)

    rows = []
    for (strategy, backend, model), group in sorted(groups.items()):
        metric_rows = [compute_record_metrics(record) for record in group]
        row: dict[str, object] = {
            "strategy": strategy,
            "backend": backend,
            "model": model,
            "samples": len(group),
        }
        for field in SUMMARY_FIELDS[4:]:
            row[field] = round(mean(float(m[field]) for m in metric_rows), 4)
        rows.append(row)
    return rows


def summarize_files(paths: list[str | Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        records.extend(read_jsonl(path))
    return summarize_records(records)


def write_summary_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    out = ensure_parent(path)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
