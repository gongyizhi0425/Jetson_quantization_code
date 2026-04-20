"""Deterministic synthetic longitudinal EHR generator.

The generated records are synthetic and contain no real patient data. They are
designed to create old-but-important evidence mixed with routine noise, which
makes sliding windows and selective retention meaningfully different.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Iterable


CONDITION_PROFILES = [
    {
        "condition": "type 2 diabetes",
        "medication": "metformin",
        "procedure": "HbA1c monitoring",
        "adverse": "hypoglycemia",
        "hospital_reason": "severe hypoglycemia",
        "abnormal_lab": "HbA1c elevated",
    },
    {
        "condition": "hypertension",
        "medication": "lisinopril",
        "procedure": "blood pressure monitoring",
        "adverse": "persistent cough",
        "hospital_reason": "hypertensive urgency",
        "abnormal_lab": "blood pressure elevated",
    },
    {
        "condition": "asthma",
        "medication": "albuterol inhaler",
        "procedure": "spirometry",
        "adverse": "tachycardia",
        "hospital_reason": "acute asthma exacerbation",
        "abnormal_lab": "peak flow reduced",
    },
    {
        "condition": "chronic kidney disease",
        "medication": "furosemide",
        "procedure": "renal function panel",
        "adverse": "dehydration",
        "hospital_reason": "acute kidney injury",
        "abnormal_lab": "creatinine elevated",
    },
    {
        "condition": "atrial fibrillation",
        "medication": "warfarin",
        "procedure": "INR monitoring",
        "adverse": "bleeding episode",
        "hospital_reason": "warfarin-associated bleeding",
        "abnormal_lab": "INR supratherapeutic",
    },
]

NOISE_NOTES = [
    "routine follow-up with stable symptoms",
    "normal lab panel reviewed",
    "preventive care counseling completed",
    "vaccination status reviewed",
    "diet and exercise counseling repeated",
    "stable chronic condition noted",
    "administrative insurance form updated",
]


def _event(
    patient_id: str,
    event_id: str,
    event_date: date,
    event_type: str,
    text: str,
    value: str = "",
    importance: str = "low",
) -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "event_id": event_id,
        "date": event_date.isoformat(),
        "event_type": event_type,
        "text": text,
        "value": value,
        "importance": importance,
    }


def _add_days(base: date, rng: random.Random, low: int, high: int) -> date:
    return base + timedelta(days=rng.randint(low, high))


def generate_events(
    patients: int,
    seed: int = 42,
    noise_level: str = "medium",
    timeline_length: str = "medium",
) -> Iterable[dict[str, object]]:
    rng = random.Random(seed)
    noise_counts = {"low": 8, "medium": 18, "high": 36}
    extra_followups = {"short": 3, "medium": 8, "long": 18}
    if noise_level not in noise_counts:
        raise ValueError(f"Unsupported noise_level: {noise_level}")
    if timeline_length not in extra_followups:
        raise ValueError(f"Unsupported timeline_length: {timeline_length}")

    for idx in range(patients):
        patient_id = f"P{idx + 1:05d}"
        profile = CONDITION_PROFILES[idx % len(CONDITION_PROFILES)]
        start = date(2018, 1, 1) + timedelta(days=(idx % 90))
        counter = 1

        def next_event(event_date: date, event_type: str, text: str, value: str, importance: str):
            nonlocal counter
            event_id = f"{patient_id}-E{counter:04d}"
            counter += 1
            return _event(patient_id, event_id, event_date, event_type, text, value, importance)

        diagnosis_date = _add_days(start, rng, 0, 30)
        yield next_event(
            diagnosis_date,
            "diagnosis",
            f"Diagnosed with {profile['condition']} after evaluation.",
            str(profile["condition"]),
            "high",
        )

        med_start = _add_days(diagnosis_date, rng, 20, 80)
        yield next_event(
            med_start,
            "medication_start",
            f"Started {profile['medication']} for {profile['condition']}.",
            str(profile["medication"]),
            "high",
        )

        procedure_date = _add_days(med_start, rng, 30, 90)
        yield next_event(
            procedure_date,
            "procedure",
            f"Completed {profile['procedure']} for monitoring.",
            str(profile["procedure"]),
            "medium",
        )

        abnormal_date = _add_days(procedure_date, rng, 20, 60)
        yield next_event(
            abnormal_date,
            "abnormal_lab",
            f"Abnormal finding: {profile['abnormal_lab']}.",
            str(profile["abnormal_lab"]),
            "medium",
        )

        med_change = _add_days(abnormal_date, rng, 20, 80)
        yield next_event(
            med_change,
            "medication_change",
            f"Adjusted {profile['medication']} after abnormal monitoring result.",
            str(profile["medication"]),
            "high",
        )

        adverse_date = _add_days(med_change, rng, 25, 100)
        yield next_event(
            adverse_date,
            "adverse_event",
            f"Reported adverse event: {profile['adverse']} after treatment.",
            str(profile["adverse"]),
            "high",
        )

        hospital_date = _add_days(adverse_date, rng, 5, 40)
        yield next_event(
            hospital_date,
            "hospitalization",
            f"Hospitalized for {profile['hospital_reason']}.",
            str(profile["hospital_reason"]),
            "high",
        )

        stop_date = _add_days(hospital_date, rng, 2, 20)
        yield next_event(
            stop_date,
            "medication_stop",
            f"Stopped {profile['medication']} after {profile['adverse']}.",
            str(profile["medication"]),
            "high",
        )

        total_noise = noise_counts[noise_level] + extra_followups[timeline_length]
        for n in range(total_noise):
            event_date = _add_days(start, rng, 1, 1800)
            note = rng.choice(NOISE_NOTES)
            yield next_event(
                event_date,
                "routine_followup" if "follow-up" in note or "stable" in note else "normal_lab",
                note,
                "",
                "low",
            )
