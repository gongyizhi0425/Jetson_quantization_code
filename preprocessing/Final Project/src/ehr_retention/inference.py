"""Inference backends.

`rule_local` is deterministic and requires no model. Optional LLM backends are
kept behind imports/subprocess calls so the default project remains lightweight.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass


@dataclass
class InferenceResult:
    prediction: str
    backend: str
    model: str


def _find_line(context: str, marker: str) -> str:
    for line in context.splitlines():
        if marker in line:
            return line
    return ""


def _extract_value_date(line: str) -> tuple[str, str]:
    date = line.split("|", 1)[0].strip() if "|" in line else ""
    text = line.split("|", 2)[-1].split("[event_id=", 1)[0].strip()
    return text, date


def rule_local_answer(question: str, context: str) -> str:
    question_lower = question.lower()
    if "diagnosis" in question_lower:
        line = _find_line(context, "| diagnosis |")
        if not line:
            return "unknown"
        text, date = _extract_value_date(line)
        match = re.search(r"Diagnosed with (.*?) after", text)
        condition = match.group(1) if match else text
        return f"{condition} on {date}"

    if "medication was stopped" in question_lower:
        line = _find_line(context, "| medication_stop |")
        if not line:
            return "unknown"
        text, date = _extract_value_date(line)
        match = re.search(r"Stopped (.*?) after", text)
        medication = match.group(1) if match else text
        return f"{medication} was stopped on {date}"

    if "hospitalized" in question_lower:
        hospital_line = _find_line(context, "| hospitalization |")
        adverse_line = _find_line(context, "| adverse_event |")
        if not hospital_line or not adverse_line:
            return "unknown"
        hospital_text, _ = _extract_value_date(hospital_line)
        adverse_text, _ = _extract_value_date(adverse_line)
        h_match = re.search(r"Hospitalized for (.*?)\.", hospital_text)
        a_match = re.search(r"Reported adverse event: (.*?) after", adverse_text)
        hospital_reason = h_match.group(1) if h_match else hospital_text
        adverse = a_match.group(1) if a_match else adverse_text
        return f"{hospital_reason} after {adverse}"

    return "unknown"


def run_inference(
    backend: str,
    model: str,
    question: str,
    context: str,
    prompt_template: str | None = None,
) -> InferenceResult:
    if backend == "rule_local":
        return InferenceResult(rule_local_answer(question, context), backend, model or "rule_local")

    prompt = (prompt_template or "Context:\n{context}\n\nQuestion: {question}\nAnswer:").format(
        context=context,
        question=question,
    )

    if backend == "ollama":
        completed = subprocess.run(
            ["ollama", "run", model, prompt],
            check=True,
            capture_output=True,
            text=True,
        )
        return InferenceResult(completed.stdout.strip(), backend, model)

    if backend == "transformers":
        raise RuntimeError(
            "The transformers backend is an integration hook. Use rule_local for "
            "reproducibility or connect your quantized model by replacing this hook."
        )

    raise ValueError(f"Unsupported backend: {backend}")


def to_json(result: InferenceResult) -> str:
    return json.dumps(result.__dict__, sort_keys=True)
