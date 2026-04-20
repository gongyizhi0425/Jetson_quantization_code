#!/usr/bin/env python3
"""Convert synthetic event JSONL into patient timeline JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_retention.io_utils import read_jsonl, write_jsonl
from ehr_retention.timeline import build_timelines, validate_timeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="data/processed/events.jsonl")
    parser.add_argument("--out", default="data/processed/timelines.jsonl")
    args = parser.parse_args()

    timelines = build_timelines(read_jsonl(args.events))
    for timeline in timelines:
        validate_timeline(timeline)
    count = write_jsonl(args.out, timelines)
    print(f"Wrote {count} patient timelines to {args.out}")


if __name__ == "__main__":
    main()
