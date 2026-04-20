#!/usr/bin/env python3
"""Generate deterministic synthetic EHR events."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_retention.io_utils import write_jsonl
from ehr_retention.synthetic import generate_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-level", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--timeline-length", choices=["short", "medium", "long"], default="medium")
    parser.add_argument("--out", default="data/processed/events.jsonl")
    args = parser.parse_args()

    count = write_jsonl(
        args.out,
        generate_events(
            patients=args.patients,
            seed=args.seed,
            noise_level=args.noise_level,
            timeline_length=args.timeline_length,
        ),
    )
    print(f"Wrote {count} synthetic EHR events to {args.out}")


if __name__ == "__main__":
    main()
