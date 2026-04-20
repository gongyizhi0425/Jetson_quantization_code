#!/usr/bin/env python3
"""Generate report-ready SVG figures from the experiment summary CSV."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_retention.io_utils import ensure_parent


PALETTE = {
    "full_context": "#2f6f4e",
    "sliding_window": "#b33c2e",
    "selective": "#277da1",
    "selective_query_aware": "#6a4c93",
}

LABELS = {
    "full_context": "Full context",
    "sliding_window": "Sliding window",
    "selective": "Selective",
    "selective_query_aware": "Query-aware selective",
}


def read_summary(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, ValueError):
        return 0.0


def _label(strategy: str) -> str:
    return LABELS.get(strategy, strategy.replace("_", " ").title())


def _color(strategy: str) -> str:
    return PALETTE.get(strategy, "#555555")


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, Helvetica, sans-serif; fill: #202124; }",
        ".title { font-size: 20px; font-weight: 700; }",
        ".axis { stroke: #4b5563; stroke-width: 1.2; }",
        ".grid { stroke: #d1d5db; stroke-width: 1; }",
        ".label { font-size: 12px; }",
        ".small { font-size: 11px; fill: #4b5563; }",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" class="title">{html.escape(title)}</text>',
    ]


def _write(path: Path, lines: list[str]) -> None:
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_quality_efficiency(rows: list[dict[str, str]], out: Path) -> None:
    width, height = 820, 520
    left, right, top, bottom = 90, 40, 70, 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_tokens = max(_float(row, "input_tokens") for row in rows) * 1.08

    lines = _svg_header(width, height, "Quality vs Context Size")
    for i in range(6):
        y = top + plot_h - (plot_h * i / 5)
        val = i / 5
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="small">{val:.1f}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" class="axis"/>')
    lines.append(f'<text x="{width / 2:.1f}" y="{height - 36}" text-anchor="middle" class="label">Average input tokens</text>')
    lines.append(f'<text transform="translate(28 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" class="label">Exact match</text>')

    for row in rows:
        strategy = row["strategy"]
        x = left + plot_w * (_float(row, "input_tokens") / max_tokens)
        y = top + plot_h - plot_h * _float(row, "exact_match")
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{_color(strategy)}"/>')
        lines.append(f'<text x="{x + 12:.1f}" y="{y + 4:.1f}" class="label">{html.escape(_label(strategy))}</text>')
    lines.append("</svg>")
    _write(out, lines)


def make_evidence_recall(rows: list[dict[str, str]], out: Path) -> None:
    width, height = 860, 470
    left, right, top, bottom = 80, 30, 70, 130
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_gap = 18
    bar_w = (plot_w - bar_gap * (len(rows) - 1)) / len(rows)

    lines = _svg_header(width, height, "Evidence Retention Under Compression")
    for i in range(6):
        y = top + plot_h - plot_h * i / 5
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="small">{i / 5:.1f}</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" class="axis"/>')
    lines.append(f'<text transform="translate(28 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" class="label">Evidence recall</text>')

    for i, row in enumerate(rows):
        strategy = row["strategy"]
        x = left + i * (bar_w + bar_gap)
        value = _float(row, "evidence_recall")
        bar_h = plot_h * value
        y = top + plot_h - bar_h
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{_color(strategy)}"/>')
        lines.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="label">{value:.2f}</text>')
        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="small">'
            f'{html.escape(_label(strategy))}</text>'
        )
        lines.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 42}" text-anchor="middle" class="small">'
            f'{_float(row, "input_tokens"):.0f} tokens</text>'
        )
    lines.append("</svg>")
    _write(out, lines)


def make_efficiency_bars(rows: list[dict[str, str]], out: Path) -> None:
    width, height = 900, 500
    left, right, top, bottom = 90, 40, 70, 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_latency = max(_float(row, "latency_ms") for row in rows) * 1.2
    max_memory = max(_float(row, "peak_memory_mb") for row in rows) * 1.2
    group_w = plot_w / len(rows)
    bar_w = min(42, group_w / 4)

    lines = _svg_header(width, height, "Efficiency Metrics by Strategy")
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" class="axis"/>')
    lines.append(f'<text transform="translate(28 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" class="label">Normalized bar height</text>')
    lines.append(f'<rect x="{width - 250}" y="52" width="14" height="14" fill="#277da1"/><text x="{width - 230}" y="64" class="small">Latency ms</text>')
    lines.append(f'<rect x="{width - 140}" y="52" width="14" height="14" fill="#8f5f2a"/><text x="{width - 120}" y="64" class="small">Peak memory MB</text>')

    for i, row in enumerate(rows):
        center = left + group_w * i + group_w / 2
        latency = _float(row, "latency_ms")
        memory = _float(row, "peak_memory_mb")
        lat_h = plot_h * (latency / max_latency if max_latency else 0)
        mem_h = plot_h * (memory / max_memory if max_memory else 0)
        lines.append(f'<rect x="{center - bar_w - 3:.1f}" y="{top + plot_h - lat_h:.1f}" width="{bar_w:.1f}" height="{lat_h:.1f}" fill="#277da1"/>')
        lines.append(f'<rect x="{center + 3:.1f}" y="{top + plot_h - mem_h:.1f}" width="{bar_w:.1f}" height="{mem_h:.1f}" fill="#8f5f2a"/>')
        lines.append(f'<text x="{center - bar_w / 2 - 3:.1f}" y="{top + plot_h - lat_h - 7:.1f}" text-anchor="middle" class="small">{latency:.3f}</text>')
        lines.append(f'<text x="{center + bar_w / 2 + 3:.1f}" y="{top + plot_h - mem_h - 7:.1f}" text-anchor="middle" class="small">{memory:.4f}</text>')
        lines.append(f'<text x="{center:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="small">{html.escape(_label(row["strategy"]))}</text>')
    lines.append("</svg>")
    _write(out, lines)


def make_pipeline_diagram(out: Path) -> None:
    width, height = 980, 260
    labels = [
        "Synthetic EHR",
        "Patient timeline",
        "QA with gold evidence",
        "Context strategy",
        "Inference backend",
        "Quality + efficiency",
    ]
    lines = _svg_header(width, height, "Selective Retention Experiment Pipeline")
    x0, y, w, h, gap = 45, 105, 125, 58, 32
    for i, label in enumerate(labels):
        x = x0 + i * (w + gap)
        color = "#277da1" if label == "Context strategy" else "#2f6f4e"
        lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{color}" opacity="0.92"/>')
        lines.append(f'<text x="{x + w / 2}" y="{y + 25}" text-anchor="middle" fill="#ffffff" class="label">{html.escape(label.split()[0])}</text>')
        lines.append(f'<text x="{x + w / 2}" y="{y + 43}" text-anchor="middle" fill="#ffffff" class="label">{html.escape(" ".join(label.split()[1:]))}</text>')
        if i < len(labels) - 1:
            ax = x + w + 6
            ay = y + h / 2
            lines.append(f'<line x1="{ax}" y1="{ay}" x2="{ax + gap - 12}" y2="{ay}" stroke="#4b5563" stroke-width="2"/>')
            lines.append(f'<polygon points="{ax + gap - 12},{ay - 5} {ax + gap - 12},{ay + 5} {ax + gap - 3},{ay}" fill="#4b5563"/>')
    lines.append('<text x="490" y="215" text-anchor="middle" class="small">Full context vs sliding window vs selective retention vs query-aware selective retention</text>')
    lines.append("</svg>")
    _write(out, lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="data/results/summary.csv")
    parser.add_argument("--out-dir", default="data/results/figures")
    args = parser.parse_args()

    rows = read_summary(args.summary)
    order = ["full_context", "sliding_window", "selective", "selective_query_aware"]
    rows = sorted(rows, key=lambda row: order.index(row["strategy"]) if row["strategy"] in order else len(order))
    out_dir = Path(args.out_dir)

    make_quality_efficiency(rows, out_dir / "quality_vs_tokens.svg")
    make_evidence_recall(rows, out_dir / "evidence_recall_by_strategy.svg")
    make_efficiency_bars(rows, out_dir / "efficiency_metrics.svg")
    make_pipeline_diagram(out_dir / "pipeline_overview.svg")
    print(f"Wrote SVG figures to {out_dir}")


if __name__ == "__main__":
    main()
