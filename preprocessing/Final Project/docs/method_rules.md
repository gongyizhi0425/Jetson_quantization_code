# Context Retention Method Rules

This document defines the retention logic used in the experiments. The implementation is in `src/ehr_retention/retention.py`.

## Compared Strategies

| Strategy | Role | What It Keeps | Why Tokens Change |
|---|---|---|---|
| `full_context` | Upper-bound baseline | Every event in the patient timeline | No compression; highest token count |
| `sliding_window` | Naive compression baseline | Most recent events only | Drops older events until the event or token budget is met |
| `selective` | Proposed method | Highest-scoring medical events | Removes low-value events before tokenization |
| `selective_query_aware` | Proposed extension | Highest-scoring events with query relevance | Removes low-value events and boosts events related to the question |

## Full Context Baseline

Full context keeps the entire longitudinal timeline in chronological order.

Rule:

```text
retained_events = all patient events
```

Purpose:

- Measures the best possible evidence coverage.
- Usually has the highest token count, memory use, and prefill latency.
- Serves as an accuracy upper bound for the rule-based synthetic evaluation.

## Sliding Window Baseline

Sliding window keeps only the most recent part of the patient history.

Rule:

```text
retained_events = last K events
```

or, when using a token budget:

```text
retained_events = newest events that fit within max_context_tokens
```

Purpose:

- Represents a common long-context truncation baseline.
- Reduces tokens aggressively.
- Fails when the answer depends on older but clinically important events, such as first diagnosis or earlier medication changes.

In the current main experiment, we use:

```text
--top-k-events 31
```

This gives sliding window approximately the same retained event ratio as `selective` under `--retention-ratio 0.5`.

## Selective Retention

Selective retention is the proposed framework. It scores each event before tokenization and keeps only the highest-value events.

Rule:

```text
score(event) =
  event_type_weight
  + keyword_weight
  + recency_weight
```

Then:

```text
retained_events = top K events by score
```

where:

```text
K = round(total_events * retention_ratio)
```

For example, `--retention-ratio 0.5` keeps roughly 50% of each patient's events.

### Event Type Weights

| Event type | Weight | Reason |
|---|---:|---|
| `diagnosis` | 8.0 | Core longitudinal medical history |
| `adverse_event` | 8.0 | Critical safety signal |
| `hospitalization` | 8.0 | Critical clinical outcome |
| `medication_start` | 7.0 | Treatment initiation |
| `medication_stop` | 7.0 | Treatment discontinuation |
| `medication_change` | 7.0 | Treatment adjustment |
| `procedure` | 4.0 | Medium-value clinical action |
| `abnormal_lab` | 4.0 | Medium-value clinical signal |
| `specialist_visit` | 3.0 | Medium-value clinical context |
| `routine_followup` | 0.5 | Low-value repeated context |
| `normal_lab` | 0.25 | Low-value routine evidence |

### Keyword Weight

The implementation adds:

```text
0.5 * number_of_important_keywords_in_event_text
```

Important keywords:

```text
diagnosed, diagnosis, started, stopped, adjusted,
adverse, hospitalized, abnormal, elevated,
bleeding, hypoglycemia, exacerbation
```

These words capture clinically important actions and outcomes.

### Recency Weight

The implementation adds a small recency bonus:

```text
0.75 * recency_rank / (total_events - 1)
```

This prevents the method from ignoring recent information entirely, while still allowing older high-value diagnosis or medication events to outrank recent routine noise.

## Query-Aware Selective Retention

Query-aware selective retention extends selective retention by adding question relevance.

Rule:

```text
score(event, query) =
  event_type_weight
  + keyword_weight
  + recency_weight
  + query_overlap_weight
```

where:

```text
query_overlap_weight = 1.5 * number_of_shared_tokens(query, event_text)
```

Purpose:

- Keeps the general clinical importance logic.
- Adds dynamic relevance to the current question.
- For medication questions, medication events receive an extra boost.
- For hospitalization questions, adverse event and hospitalization lines receive an extra boost.

## What Is Innovative

The innovation is not simply taking top-k events. Top-k is only the final selection operator.

The contribution is the selective retention framework:

```text
longitudinal EHR timeline
→ event-level importance scoring before tokenization
→ controlled retention ratio
→ reduced prompt tokens
→ same downstream QA interface
```

Compared with sliding window, selective retention is not position-based. It can retain an old diagnosis or medication stop event even if many routine follow-ups appear after it.

Compared with full context, selective retention removes redundant low-value events and reduces tokens, memory, and latency.

Compared with generic top-k, the scoring function is clinically motivated through event types, medical keywords, recency, and optional query relevance.

## Implementation Location

Core rules:

```text
src/ehr_retention/retention.py
```

Important symbols:

```text
EVENT_TYPE_WEIGHTS
IMPORTANT_KEYWORDS
score_event()
select_events()
RetentionConfig
```

Experiment entrypoint:

```text
scripts/run_experiment.py
```

The script records retained event IDs, input tokens, latency, memory, and quality metrics into JSONL files under:

```text
data/results/
```
