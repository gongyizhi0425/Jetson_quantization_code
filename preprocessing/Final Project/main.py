"""Convenience entrypoint for the selective retention project."""

from __future__ import annotations

import subprocess
import sys


COMMANDS = [
    [sys.executable, "scripts/generate_data.py", "--patients", "10", "--seed", "42"],
    [sys.executable, "scripts/build_timelines.py"],
    [sys.executable, "scripts/make_qa.py"],
    [sys.executable, "scripts/run_experiment.py", "--config", "configs/default.yaml", "--strategy", "full_context"],
    [
        sys.executable,
        "scripts/run_experiment.py",
        "--config",
        "configs/default.yaml",
        "--strategy",
        "sliding_window",
        "--max-context-tokens",
        "180",
    ],
    [
        sys.executable,
        "scripts/run_experiment.py",
        "--config",
        "configs/default.yaml",
        "--strategy",
        "selective",
        "--retention-ratio",
        "0.5",
    ],
    [sys.executable, "scripts/eval_results.py", "--results", "data/results/*.jsonl", "--out", "data/results/summary.csv"],
]


def main() -> None:
    for command in COMMANDS:
        print("$", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
