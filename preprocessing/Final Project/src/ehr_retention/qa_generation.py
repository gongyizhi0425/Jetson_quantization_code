"""Rule-based QA generation with gold answers and evidence event IDs."""

from __future__ import annotations


def _first(events: list[dict[str, object]], event_type: str) -> dict[str, object] | None:
    for event in events:
        if event.get("event_type") == event_type:
            return event
    return None


def _sample(
    patient_id: str,
    task_type: str,
    suffix: str,
    question: str,
    answer: str,
    evidence: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "sample_id": f"{patient_id}-{suffix}",
        "patient_id": patient_id,
        "task_type": task_type,
        "question": question,
        "answer": answer,
        "gold_evidence_event_ids": [str(e["event_id"]) for e in evidence],
    }


def generate_qa(timelines: list[dict[str, object]]) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for timeline in timelines:
        patient_id = str(timeline["patient_id"])
        events = list(timeline["events"])
        diagnosis = _first(events, "diagnosis")
        med_start = _first(events, "medication_start")
        med_change = _first(events, "medication_change")
        adverse = _first(events, "adverse_event")
        hospital = _first(events, "hospitalization")
        med_stop = _first(events, "medication_stop")

        if diagnosis is not None:
            answer = f"{diagnosis['value']} on {diagnosis['date']}"
            samples.append(
                _sample(
                    patient_id,
                    "diagnosis_history",
                    "DX",
                    "What diagnosis was first recorded for this patient, and on what date?",
                    answer,
                    [diagnosis],
                )
            )

        if med_start is not None and med_stop is not None:
            answer = f"{med_stop['value']} was stopped on {med_stop['date']}"
            evidence = [med_start]
            if med_change is not None:
                evidence.append(med_change)
            evidence.append(med_stop)
            samples.append(
                _sample(
                    patient_id,
                    "medication_tracking",
                    "MED",
                    "Which medication was stopped, and when was it stopped?",
                    answer,
                    evidence,
                )
            )

        if adverse is not None and hospital is not None:
            answer = f"{hospital['value']} after {adverse['value']}"
            samples.append(
                _sample(
                    patient_id,
                    "adverse_hospitalization",
                    "HOSP",
                    "Why was the patient hospitalized, and what adverse event preceded it?",
                    answer,
                    [adverse, hospital],
                )
            )
    return samples
