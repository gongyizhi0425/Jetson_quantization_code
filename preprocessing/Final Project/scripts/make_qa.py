#!/usr/bin/env python3
"""Generate longitudinal EHR QA samples from timelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_retention.io_utils import read_jsonl, write_jsonl
from ehr_retention.qa_generation import generate_qa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timelines", default="data/processed/timelines.jsonl")
    parser.add_argument("--out", default="data/processed/qa.jsonl")
    args = parser.parse_args()

    samples = generate_qa(list(read_jsonl(args.timelines)))
    count = write_jsonl(args.out, samples)
    print(f"Wrote {count} QA samples to {args.out}")


if __name__ == "__main__":
    main()
