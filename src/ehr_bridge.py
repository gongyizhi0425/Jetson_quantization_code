"""
Bridge between the selective-retention preprocessing pipeline and the
KV-cache experiment runner.

Reads the JSONL outputs from ``preprocessing/Final Project/`` and converts
them into the prompt format expected by the KV-cache notebooks.

Pipeline:
  preprocessing generates → events.jsonl → timelines.jsonl → qa.jsonl
  this module reads qa.jsonl + timelines.jsonl
  → applies selective retention (importance-based event filtering)
  → produces List[Dict] with {prompt, question, reference_answer, ...}
     identical schema to load_pubmedqa() output

Quick start (for KV-cache quantization side):
  >>> from src.ehr_bridge import get_selective_prompts
  >>> prompts = get_selective_prompts(max_samples=50)
  # prompts is a list of dicts, each with 'prompt' key ready for model input
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PREPROC_ROOT = Path(__file__).resolve().parents[1] / "preprocessing" / "Final Project"
_DEFAULT_QA = _PREPROC_ROOT / "data" / "processed" / "qa.jsonl"
_DEFAULT_TIMELINES = _PREPROC_ROOT / "data" / "processed" / "timelines.jsonl"


def _read_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Selective-retention context builder (standalone, no ehr_retention import)
# ---------------------------------------------------------------------------

_EVENT_TYPE_WEIGHTS = {
    "diagnosis": 8.0,
    "adverse_event": 8.0,
    "hospitalization": 8.0,
    "medication_start": 7.0,
    "medication_stop": 7.0,
    "medication_change": 7.0,
    "procedure": 4.0,
    "abnormal_lab": 4.0,
    "specialist_visit": 3.0,
    "routine_followup": 0.5,
    "normal_lab": 0.25,
}

_IMPORTANT_KEYWORDS = {
    "diagnosed", "diagnosis", "started", "stopped", "adjusted",
    "adverse", "hospitalized", "abnormal", "elevated", "bleeding",
    "hypoglycemia", "exacerbation",
}


def _tokenize_simple(text: str) -> List[str]:
    import re
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _score_event(event: Dict, query: str, idx: int, total: int) -> float:
    text = str(event.get("text", ""))
    etype = str(event.get("event_type", ""))
    score = _EVENT_TYPE_WEIGHTS.get(etype, 1.0)
    words = set(_tokenize_simple(text))
    score += 0.5 * len(words & _IMPORTANT_KEYWORDS)
    if query:
        score += 1.5 * len(set(_tokenize_simple(query)) & words)
    if total > 1:
        score += 0.75 * (idx / (total - 1))
    return score


def selective_retain(
    events: List[Dict],
    query: str,
    retention_ratio: float = 0.5,
) -> List[Dict]:
    """Apply selective retention: keep top-scored events."""
    if not events:
        return []
    total = len(events)
    scored = []
    for i, ev in enumerate(events):
        scored.append((_score_event(ev, query, i, total), ev))
    scored.sort(key=lambda x: x[0], reverse=True)
    k = max(1, int(round(total * retention_ratio)))
    selected = [item[1] for item in scored[:k]]
    return sorted(selected, key=lambda e: (str(e["date"]), str(e["event_id"])))


def render_context(events: List[Dict]) -> str:
    """Render selected events as a text block."""
    lines = []
    for ev in events:
        lines.append(
            f"{ev['date']} | {ev['event_type']} | {ev['text']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt templates (medical EHR style)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an experienced medical expert reviewing a patient's longitudinal "
    "electronic health record (EHR). Provide detailed, evidence-based reasoning "
    "before stating your conclusion."
)

_USER_TEMPLATE_FULL = (
    "The following is the patient's complete medical timeline.\n\n"
    "Timeline:\n{context}\n\n"
    "Question: {question}\n\n"
    "Based on the timeline above, provide your answer with "
    "step-by-step medical reasoning."
)

_USER_TEMPLATE_SELECTIVE = (
    "The following is a selectively retained subset of the patient's medical "
    "timeline. Low-value routine events have been removed to focus on clinically "
    "significant records.\n\n"
    "Timeline (selective retention, ratio={ratio:.0%}):\n{context}\n\n"
    "Question: {question}\n\n"
    "Based on the retained records, provide your answer with "
    "step-by-step medical reasoning."
)


def _build_chat_prompt(context: str, question: str, tokenizer, template: str) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": template.format(
            context=context, question=question, ratio=0.5,
        )},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def _build_plain_prompt(context: str, question: str, template: str) -> str:
    return (
        f"<|system|>{_SYSTEM_PROMPT}\n"
        f"<|user|>{template.format(context=context, question=question, ratio=0.5)}\n"
        f"<|assistant|>"
    )


# ---------------------------------------------------------------------------
# Main loaders
# ---------------------------------------------------------------------------

def load_ehr_qa(
    qa_path: Optional[str] = None,
    timelines_path: Optional[str] = None,
    max_samples: int = 50,
    tokenizer=None,
    strategy: str = "full_context",
    retention_ratio: float = 0.5,
) -> List[Dict[str, str]]:
    """Load synthetic EHR QA data and produce prompts for KV-cache experiments.

    Parameters
    ----------
    qa_path : str | None
        Path to qa.jsonl. Default: preprocessing/Final Project/data/processed/qa.jsonl
    timelines_path : str | None
        Path to timelines.jsonl.
    max_samples : int
        Maximum QA samples to load.
    tokenizer : PreTrainedTokenizer | None
        If provided, apply chat template.
    strategy : str
        "full_context" | "selective" | "sliding_window"
    retention_ratio : float
        For selective strategy, fraction of events to keep.

    Returns
    -------
    list of dict
        Same schema as load_pubmedqa():
        {prompt, question, reference_answer, final_decision, pubid,
         context_strategy, num_events_original, num_events_retained}
    """
    qa_file = Path(qa_path) if qa_path else _DEFAULT_QA
    tl_file = Path(timelines_path) if timelines_path else _DEFAULT_TIMELINES

    if not qa_file.exists():
        raise FileNotFoundError(
            f"QA data not found at {qa_file}.\n"
            "Run the preprocessing pipeline first:\n"
            "  cd preprocessing/'Final Project'\n"
            "  python main.py"
        )

    qa_samples = _read_jsonl(qa_file)
    timelines = {str(t["patient_id"]): t for t in _read_jsonl(tl_file)}

    if strategy == "selective":
        template = _USER_TEMPLATE_SELECTIVE
    else:
        template = _USER_TEMPLATE_FULL

    prompts: List[Dict[str, str]] = []
    for i, sample in enumerate(qa_samples):
        if i >= max_samples:
            break

        timeline = timelines.get(str(sample["patient_id"]))
        if timeline is None:
            continue

        events = list(timeline["events"])
        n_original = len(events)

        if strategy == "selective":
            events = selective_retain(events, sample["question"], retention_ratio)
        elif strategy == "sliding_window":
            k = max(1, int(round(n_original * retention_ratio)))
            events = events[-k:]

        context = render_context(events)

        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            prompt = _build_chat_prompt(context, sample["question"], tokenizer, template)
        else:
            prompt = _build_plain_prompt(context, sample["question"], template)

        prompts.append({
            "prompt": prompt,
            "question": sample["question"],
            "reference_answer": sample["answer"],
            "final_decision": sample.get("task_type", ""),
            "pubid": sample.get("sample_id", f"EHR-{i}"),
            "context_strategy": strategy,
            "num_events_original": str(n_original),
            "num_events_retained": str(len(events)),
        })

    return prompts


# ---------------------------------------------------------------------------
# Convenience one-call API for quantization / KV-cache side
# ---------------------------------------------------------------------------

def get_selective_prompts(
    max_samples: int = 50,
    retention_ratio: float = 0.5,
    tokenizer=None,
) -> List[Dict[str, str]]:
    """Get selective-retained EHR prompts — the only function you need to call.

    This is the simplified entry point for KV-cache experiments.
    It reads the preprocessing pipeline's generated data, applies
    selective retention (importance-based event filtering), and returns
    prompts ready to feed into measure_generation() / run_benchmark().

    Example
    -------
    >>> from src.ehr_bridge import get_selective_prompts
    >>> prompts = get_selective_prompts(max_samples=50)
    >>> for p in prompts:
    ...     # p["prompt"]            — full prompt string for the model
    ...     # p["question"]          — the medical question
    ...     # p["reference_answer"]  — ground-truth answer
    ...     pass

    Parameters
    ----------
    max_samples : int
        Number of QA samples to load. Default 50.
    retention_ratio : float
        Fraction of events to keep (0.5 = top 50% by importance).
    tokenizer : PreTrainedTokenizer | None
        If provided, uses ``apply_chat_template`` for prompt formatting.
        If None, uses a plain-text template.

    Returns
    -------
    list of dict
        Each dict has keys:
        - prompt: str — the complete prompt string
        - question: str — the medical question
        - reference_answer: str — ground-truth answer
        - context_strategy: "selective"
        - num_events_original: str — original event count
        - num_events_retained: str — retained event count after filtering
        - pubid: str — sample identifier
        - final_decision: str — QA task type
    """
    return load_ehr_qa(
        max_samples=max_samples,
        tokenizer=tokenizer,
        strategy="selective",
        retention_ratio=retention_ratio,
    )
