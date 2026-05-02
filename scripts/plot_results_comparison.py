from __future__ import annotations

import csv
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

GROUP_FILES = {
    "Baseline": RESULTS_DIR / "baseline_gqa_new.csv",
    "PagedAttn": RESULTS_DIR / "gqa_paged_vllm (1).csv",
    "KIVI": RESULTS_DIR / "gqa_kivi_llama3.2.csv",
}

PPL_VALUES = {
    "Baseline": 12.1067,
    "PagedAttn": 12.1067,
    "KIVI": 20.68,
}

METRIC_SPECS = [
    ("ttft_ms", "TTFT", "ms", "{:.1f}"),
    ("tpot_ms", "TPOT", "ms", "{:.1f}"),
    ("kv_cache_memory_mb", "KV Cache Memory", "MB", "{:.1f}"),
    ("peak_memory_mb", "Peak Memory", "MB", "{:.1f}"),
    ("perplexity", "Perplexity", "", "{:.2f}"),
]

COLORS = {
    "Baseline": "#4C78A8",
    "PagedAttn": "#72B7B2",
    "KIVI": "#F58518",
}


def mean_from_csv(path: Path, column: str) -> float:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        values = [float(row[column]) for row in reader if row.get(column) not in (None, "")]
    if not values:
        raise ValueError(f"No usable values found in {path} column {column}")
    return fmean(values)


def build_summary() -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for group, csv_path in GROUP_FILES.items():
        summary[group] = {
            "ttft_ms": mean_from_csv(csv_path, "ttft_ms"),
            "tpot_ms": mean_from_csv(csv_path, "tpot_ms"),
            "kv_cache_memory_mb": mean_from_csv(csv_path, "kv_cache_memory_mb"),
            "peak_memory_mb": mean_from_csv(csv_path, "peak_memory_mb"),
            "perplexity": PPL_VALUES[group],
        }
    return summary


def human_tick(value: float, _pos: int | None = None) -> str:
    return f"{value:,.0f}"


def add_value_labels(ax, bars, fmt: str, unit: str) -> None:
    x_max = ax.get_xlim()[1]
    offset = x_max * 0.015
    for bar in bars:
        value = bar.get_width()
        label = fmt.format(value) + (f" {unit}" if unit else "")
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#1F2937",
            clip_on=False,
        )


def plot(summary: dict[str, dict[str, float]]) -> None:
    groups = list(summary.keys())

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#D0D7DE",
            "grid.color": "#D8E1EB",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=False)
    axes = axes.flatten()
    fig.suptitle(
        "Baseline vs PagedAttention vs KIVI\nAverage Inference Metrics from results/",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )

    y_positions = list(range(len(groups)))
    for idx, (metric_key, title, unit, fmt) in enumerate(METRIC_SPECS):
        ax = axes[idx]
        values = [summary[group][metric_key] for group in groups]
        colors = [COLORS[group] for group in groups]

        max_value = max(values)
        ax.set_xlim(0, max_value * 1.30)
        bars = ax.barh(y_positions, values, color=colors, height=0.58, edgecolor="white")
        ax.set_yticks(y_positions, labels=groups)
        ax.invert_yaxis()
        ax.set_title(title, pad=10, fontweight="bold")
        ax.xaxis.set_major_formatter(FuncFormatter(human_tick))
        ax.grid(axis="x", alpha=0.9)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        add_value_labels(ax, bars, fmt=fmt, unit=unit)

    summary_ax = axes[-1]
    summary_ax.axis("off")
    lines = [
        "Data source",
        "• Baseline: results/baseline_gqa_new.csv",
        "• PagedAttn: results/gqa_paged_vllm (1).csv",
        "• KIVI: results/gqa_kivi_llama3.2.csv",
        "",
        "Perplexity values",
        "• Baseline = 12.1067",
        "• PagedAttn = 12.1067",
        "• KIVI = 20.68",
    ]
    summary_ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=12,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#EFF6FF", "edgecolor": "#BFDBFE"},
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.4, h_pad=1.2)

    png_path = RESULTS_DIR / "metrics_comparison.png"
    svg_path = RESULTS_DIR / "metrics_comparison.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(summary: dict[str, dict[str, float]]) -> None:
    output_path = RESULTS_DIR / "metrics_comparison_summary.csv"
    fieldnames = ["group", "ttft_ms", "tpot_ms", "kv_cache_memory_mb", "peak_memory_mb", "perplexity"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group, metrics in summary.items():
            writer.writerow({"group": group, **metrics})


if __name__ == "__main__":
    summary_data = build_summary()
    save_summary_csv(summary_data)
    plot(summary_data)
    print("Saved:")
    print(f"  {RESULTS_DIR / 'metrics_comparison.png'}")
    print(f"  {RESULTS_DIR / 'metrics_comparison.svg'}")
    print(f"  {RESULTS_DIR / 'metrics_comparison_summary.csv'}")
