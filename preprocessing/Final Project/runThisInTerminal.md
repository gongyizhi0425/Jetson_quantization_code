# Run This In Terminal

```bash
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

cat data/results/summary.csv
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests
```
