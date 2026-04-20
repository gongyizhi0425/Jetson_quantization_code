#!/usr/bin/env python3
"""Aggregate experiment JSONL files into a CSV summary."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_retention.evaluation import summarize_files, write_summary_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--out", default="data/results/summary.csv")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.results:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])
    rows = summarize_files(paths)
    write_summary_csv(args.out, rows)
    print(f"Wrote {len(rows)} summary rows to {args.out}")


if __name__ == "__main__":
    main()
