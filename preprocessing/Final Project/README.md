# Memory-Efficient Context Management for Longitudinal Healthcare Agents

This project implements a reproducible **selective retention** experiment for Efficient Machine Learning. It uses deterministic synthetic longitudinal EHR data, compares full context and sliding-window baselines, and exports JSONL/CSV results that can be reused by a quantization pipeline.

No real restricted medical records, credentialed datasets, cloud APIs, or downloads are required for the default run.

## Project Structure

```text
data/raw/                 optional raw Synthea CSV files
data/processed/           generated events, timelines, QA samples
data/results/             experiment JSONL and summary CSV
src/ehr_retention/        reusable Python package
scripts/                  reproducible command-line pipeline
configs/default.yaml      default experiment configuration
tests/                    lightweight verification tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default pipeline itself uses only the Python standard library. `pytest` is only needed for tests.

## Reproduce the Main Pipeline

For the shortest command-only version, see `runThisInTerminal.md`.

```bash
python scripts/generate_data.py --patients 300 --seed 42 --out data/processed/events.jsonl
python scripts/build_timelines.py --events data/processed/events.jsonl --out data/processed/timelines.jsonl
python scripts/make_qa.py --timelines data/processed/timelines.jsonl --out data/processed/qa.jsonl

python scripts/run_experiment.py --config configs/default.yaml --strategy full_context
python scripts/run_experiment.py --config configs/default.yaml --strategy sliding_window --max-context-tokens 1024
python scripts/run_experiment.py --config configs/default.yaml --strategy selective --retention-ratio 0.5
python scripts/run_experiment.py --config configs/default.yaml --strategy selective_query_aware --retention-ratio 0.5

python scripts/eval_results.py --results "data/results/*.jsonl" --out data/results/summary.csv
python scripts/plot_results.py --summary data/results/summary.csv --out-dir data/results/figures
```

For a fast smoke run:

```bash
python main.py
```

## What Gets Measured

Quality metrics:

- exact match
- token-level F1
- evidence recall
- answerability rate

Efficiency metrics:

- input tokens
- actual retention ratio
- latency in milliseconds
- peak Python runtime memory in MB

The default `rule_local` backend is deterministic and CPU-only. It is mainly for reproducibility, smoke tests, and isolating context-retention behavior. The same JSONL QA set and prompts can be reused with Ollama, Transformers, Qwen, or a quantized model.

## Report Figures

Generate SVG figures after creating `data/results/summary.csv`:

```bash
python scripts/plot_results.py --summary data/results/summary.csv --out-dir data/results/figures
```

The script writes:

- `quality_vs_tokens.svg`
- `evidence_recall_by_strategy.svg`
- `efficiency_metrics.svg`
- `pipeline_overview.svg`

## Strategies

`full_context` keeps the complete patient timeline.

`sliding_window` keeps the most recent events or applies a token budget from the end of the timeline.

`selective` scores events before tokenization and keeps the top events under the configured retention budget.

`selective_query_aware` adds question overlap to the event score.

Default scoring:

```text
score(event, query) =
  event_type_weight
  + keyword_weight
  + recency_weight
  + query_overlap_weight
```

High-value events include diagnosis, medication start/change/stop, adverse event, and hospitalization. Low-value events include normal labs, routine follow-ups, and stable status notes.

Detailed retention rules are documented in `docs/method_rules.md`. The executable implementation is in `src/ehr_retention/retention.py`.

Summary of the four compared strategies:

| Strategy | Role | Retention basis | Token reduction mechanism |
|---|---|---|---|
| `full_context` | Upper-bound baseline | Keeps all events | No reduction |
| `sliding_window` | Naive compression baseline | Keeps the most recent events | Drops older history |
| `selective` | Proposed method | Keeps highest-scoring clinical events | Drops low-value routine events |
| `selective_query_aware` | Proposed extension | Keeps high-scoring events relevant to the question | Drops low-value and query-irrelevant events |

The selective score used by the proposed methods is:

```text
score(event, query) =
  event_type_weight
  + 0.5 * important_keyword_count
  + 0.75 * normalized_recency
  + 1.5 * query_overlap_count
```

For plain `selective`, `query_overlap_count` is not used. For `selective_query_aware`, it is enabled.

## Ablation Examples

```bash
for ratio in 0.3 0.5 0.7; do
  python scripts/run_experiment.py --config configs/default.yaml --strategy selective --retention-ratio "$ratio" --out "data/results/selective_r${ratio}.jsonl"
done

for noise in low medium high; do
  python scripts/generate_data.py --patients 300 --seed 42 --noise-level "$noise" --out "data/processed/events_${noise}.jsonl"
done
```

## Optional LLM / Quantization Integration

The per-sample result format is stable:

```json
{
  "sample_id": "...",
  "strategy": "selective",
  "question": "...",
  "answer": "...",
  "prediction": "...",
  "input_tokens": 123,
  "retained_events": ["P00001-E0001"],
  "latency_ms": 1.23,
  "peak_memory_mb": 0.04,
  "backend": "rule_local",
  "model": "rule_local"
}
```

Your quantization teammate can keep `data/processed/qa.jsonl` and `data/processed/timelines.jsonl` fixed, then replace only the backend/model while preserving the same strategy outputs.

## Tests

```bash
PYTHONPATH=src pytest -q
```

If `pytest` is not installed, use the standard library runner:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

The tests verify deterministic generation, sorted timelines, QA evidence, scoring sanity, and selective retention behavior.

## Notes for Report

Use this framing:

> Top-k selection is an implementation detail, while selective retention is a general pre-tokenization context compression framework.

The recommended report comparison is:

- Baselines: full context, sliding window
- Proposed method: selective retention
- Ablation: retention ratio, noise level, and scoring mode
- Metrics: QA quality plus token count, latency, and peak memory

Optional benchmark integration guidance is in `docs/benchmark_extension.md`.

Dataset recommendations and citation guidance are in `docs/dataset_references.md`.


RUN THE BELOW CODES IN TERMINAL:
cd "/Users/froggen/HKUST 5202作业/Final Project"

python3 scripts/generate_data.py --patients 300 --seed 42 --noise-level high --timeline-length long --out data/processed/events.jsonl

python3 scripts/build_timelines.py --events data/processed/events.jsonl --out data/processed/timelines.jsonl

python3 scripts/make_qa.py --timelines data/processed/timelines.jsonl --out data/processed/qa.jsonl

python3 scripts/run_experiment.py --config configs/default.yaml --strategy full_context --out data/results/full_context_rule_local.jsonl

python3 scripts/run_experiment.py --config configs/default.yaml --strategy sliding_window --top-k-events 31 --out data/results/sliding_window_rule_local.jsonl

python3 scripts/run_experiment.py --config configs/default.yaml --strategy selective --retention-ratio 0.5 --out data/results/selective_rule_local.jsonl

python3 scripts/run_experiment.py --config configs/default.yaml --strategy selective_query_aware --retention-ratio 0.5 --out data/results/selective_query_aware_rule_local.jsonl

python3 scripts/eval_results.py --results data/results/*.jsonl --out data/results/summary.csv

python3 scripts/plot_results.py --summary data/results/summary.csv --out-dir data/results/figures
